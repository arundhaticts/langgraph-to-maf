"""Module 5 -- IR builder: assemble the framework-neutral IR.

Combines the `ComponentInventory` (Module 4) and `ReadmeSections` (Module 3)
into the `IR` -- the single source of truth. No stage after this looks at the
original source again; no stage before code generation knows the target syntax.

Responsibilities (Section 11, Module 5):
- classify each node's role (entry / terminal / linear / branch / loop / aux / hitl)
- classify the overall orchestration pattern
  (linear / branch / loop / loop_with_exit / agent_driven)
- keep `temperature` as None when absent -- never invent a default
- write `ir.json` as a debug checkpoint before conversion begins
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
from typing import Optional

from converter.config import Config
from converter.contracts import (
    ComponentInventory,
    FlatEdge,
    FunctionSpec,
    GraphNode,
    GraphSpec,
    HitlPoint,
    IR,
    IRMetadata,
    LoopGuard,
    NodeRole,
    OrchestrationMode,
    OrchestrationPattern,
    ReadmeSections,
    RepoManifest,
    WorkflowSpec,
)

# Sentinel node names that are not real nodes.
_SENTINELS = frozenset({"START", "END", "__start__", "__end__"})

# Substrings that mark a human-in-the-loop node (heuristic; R-08 territory).
_HITL_MARKERS = ("hitl", "human", "approval", "approve", "interrupt", "review", "confirm")

DEFAULT_TARGET_FRAMEWORK = "maf"

# Target framework name -> PyPI distribution name, for version pinning (Phase 0).
_DISTRIBUTION_NAMES = {"maf": "agent-framework"}


# ---------------------------------------------------------------------------
# Graph analysis
# ---------------------------------------------------------------------------

def _real(name: Optional[str]) -> bool:
    return bool(name) and name not in _SENTINELS


def _build_adjacency(graph: GraphSpec) -> dict[str, set[str]]:
    """Directed adjacency over real nodes only (sentinels dropped)."""
    adjacency: dict[str, set[str]] = {n.name: set() for n in graph.nodes}
    for edge in graph.edges:
        if _real(edge.source) and _real(edge.target):
            adjacency.setdefault(edge.source, set()).add(edge.target)
    for cond in graph.conditional_edges:
        if not _real(cond.source):
            continue
        for target in cond.outcomes.values():
            if _real(target):
                adjacency.setdefault(cond.source, set()).add(target)
    return adjacency


def _nodes_in_cycle(adjacency: dict[str, set[str]]) -> set[str]:
    """Nodes that lie on at least one directed cycle (incl. self-loops)."""
    in_cycle: set[str] = set()
    for start in adjacency:
        # DFS from `start` looking for a path back to `start`.
        stack = list(adjacency.get(start, ()))
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == start:
                in_cycle.add(start)
                break
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency.get(node, ()))
    return in_cycle


def _leads_to_end(node_name: str, graph: GraphSpec) -> bool:
    for edge in graph.edges:
        if edge.source == node_name and edge.target in ("END", "__end__"):
            return True
    for cond in graph.conditional_edges:
        if cond.source == node_name and any(
            t in ("END", "__end__") for t in cond.outcomes.values()
        ):
            return True
    return False


def _is_hitl(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _HITL_MARKERS)


def _classify_roles(graph: GraphSpec, hitl_by_body: frozenset[str] = frozenset()) -> None:
    """Assign `role` to every node in place.

    Precedence: HITL > ENTRY > BRANCH > LOOP > TERMINAL > AUX > LINEAR.
    A node is HITL if its name matches a marker OR its body calls `interrupt(`
    (`hitl_by_body`) -- the reliable signal, since node names vary.
    """
    adjacency = _build_adjacency(graph)
    in_cycle = _nodes_in_cycle(adjacency)
    conditional_sources = {c.source for c in graph.conditional_edges}
    has_any_touch = {n.name: bool(adjacency.get(n.name)) for n in graph.nodes}
    incoming = {n.name: False for n in graph.nodes}
    for targets in adjacency.values():
        for t in targets:
            if t in incoming:
                incoming[t] = True

    for node in graph.nodes:
        name = node.name
        if _is_hitl(name) or name in hitl_by_body:
            node.role = NodeRole.HITL
        elif name == graph.entry_point:
            node.role = NodeRole.ENTRY
        elif name in conditional_sources:
            node.role = NodeRole.BRANCH
        elif name in in_cycle:
            node.role = NodeRole.LOOP
        elif _leads_to_end(name, graph) or not has_any_touch[name]:
            # Terminal: routes to END, or has no real successor.
            if not has_any_touch[name] and not incoming[name]:
                node.role = NodeRole.AUX  # fully isolated node
            else:
                node.role = NodeRole.TERMINAL
        else:
            node.role = NodeRole.LINEAR


def _classify_pattern(graph: GraphSpec) -> OrchestrationPattern:
    """Classify the overall workflow shape.

    3+ conditional outcomes or dynamic (router with no static mapping) => the
    orchestration is complex and routed to Tier 3, so it is `agent_driven`.
    """
    adjacency = _build_adjacency(graph)
    has_cycle = bool(_nodes_in_cycle(adjacency))
    has_conditionals = bool(graph.conditional_edges)

    max_outcomes = max(
        (len(c.outcomes) for c in graph.conditional_edges), default=0
    )
    dynamic = any(
        c.router and not c.outcomes for c in graph.conditional_edges
    )

    if dynamic or max_outcomes >= 3:
        return OrchestrationPattern.AGENT_DRIVEN
    if has_cycle and has_conditionals:
        return OrchestrationPattern.LOOP_WITH_EXIT
    if has_cycle:
        return OrchestrationPattern.LOOP
    if has_conditionals:
        return OrchestrationPattern.BRANCH
    return OrchestrationPattern.LINEAR


# ---------------------------------------------------------------------------
# Phase 0 -- target SDK version pinning
# ---------------------------------------------------------------------------

def detect_target_version(target_framework: str) -> Optional[str]:
    """Best-effort installed version of the target SDK (Phase 0 pin/verify).

    Returns the installed distribution version so the IR records exactly which
    target-SDK surface the conversion was built against, or None if the SDK is
    not installed in this environment.
    """
    dist = _DISTRIBUTION_NAMES.get(target_framework, target_framework)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------------------
# Phase 1 -- per-node data-flow (reads / writes / tool calls)
# ---------------------------------------------------------------------------

def _find_funcdef(source: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _dict_keys(node: ast.AST) -> set[str]:
    """String keys of a dict literal (for `return {"x": ...}` write detection)."""
    keys: set[str] = set()
    if isinstance(node, ast.Dict):
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _analyze_node(
    fn: Optional[FunctionSpec], state_names: set[str], tool_names: set[str]
) -> tuple[list[str], list[str], list[str]]:
    """Return (reads, writes, calls_tools) for one node function.

    - reads: state fields read via `state["x"]`, `state.get("x")`, `state.x`.
    - writes: state fields written via a returned dict literal or `state["x"] = `.
    - calls_tools: tool names invoked in the body.
    """
    if fn is None or not fn.source:
        return [], [], []
    funcdef = _find_funcdef(fn.source)
    if funcdef is None:
        return [], [], []
    state_var = fn.params[0].name if fn.params else None

    reads: set[str] = set()
    writes: set[str] = set()
    calls: set[str] = set()

    def _is_state(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == state_var

    for node in ast.walk(funcdef):
        if isinstance(node, ast.Subscript) and _is_state(node.value):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                # `state["x"] = ...` is a write; a bare `state["x"]` is a read.
                (writes if isinstance(getattr(node, "ctx", None), ast.Store) else reads).add(
                    key.value
                )
        elif isinstance(node, ast.Attribute) and _is_state(node.value):
            if node.attr in state_names:
                (writes if isinstance(node.ctx, ast.Store) else reads).add(node.attr)
        elif isinstance(node, ast.Call):
            fname = _callable_name(node)
            if fname in tool_names:
                calls.add(fname)
            # `state.get("x")` read.
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and _is_state(node.func.value)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                reads.add(node.args[0].value)
        elif isinstance(node, ast.Return) and node.value is not None:
            writes |= _dict_keys(node.value)

    # Not intersected with the declared schema on purpose: a read/write of an
    # undeclared field is a real inconsistency the Phase 2 validator must catch.
    return (sorted(reads), sorted(writes), sorted(calls))


def _callable_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call):
        return _callable_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# ---------------------------------------------------------------------------
# Phase 1 -- edge flattening, back-edge / loop-guard detection
# ---------------------------------------------------------------------------

def _back_edges(adjacency: dict[str, set[str]], entry: Optional[str]) -> set[tuple[str, str]]:
    """DFS back-edges (u -> v where v is an ancestor on the DFS stack)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in adjacency}
    back: set[tuple[str, str]] = set()

    def visit(u: str) -> None:
        color[u] = GRAY
        for v in adjacency.get(u, ()):
            if color.get(v, WHITE) == GRAY:
                back.add((u, v))
            elif color.get(v, WHITE) == WHITE:
                visit(v)
        color[u] = BLACK

    roots = [entry] if entry and entry in adjacency else []
    roots += [n for n in adjacency if n not in roots]
    for root in roots:
        if color.get(root, WHITE) == WHITE:
            visit(root)
    return back


def _flatten_edges(graph: GraphSpec, back: set[tuple[str, str]]) -> list[FlatEdge]:
    """Every edge / conditional outcome as one explicit {src,target,label} triple."""
    flat: list[FlatEdge] = []
    for edge in graph.edges:
        flat.append(
            FlatEdge(
                source=edge.source,
                target=edge.target,
                is_loop=(edge.source, edge.target) in back,
            )
        )
    for cond in graph.conditional_edges:
        for label, target in cond.outcomes.items():
            flat.append(
                FlatEdge(
                    source=cond.source,
                    target=target,
                    condition_label=label,
                    router=cond.router,
                    is_loop=(cond.source, target) in back,
                )
            )
    return flat


def _loop_guards(
    graph: GraphSpec,
    back: set[tuple[str, str]],
    inventory: ComponentInventory,
) -> list[LoopGuard]:
    """One guard per loop back-target: router, exit labels, and iteration cap."""
    const_names = set(inventory.config.constants)
    guards: list[LoopGuard] = []
    seen: set[str] = set()
    for _src, loop_node in back:
        if loop_node in seen:
            continue
        seen.add(loop_node)
        router: Optional[str] = None
        exit_labels: list[str] = []
        # The router that closes this loop is the conditional edge with a
        # back-edge outcome returning to `loop_node`.
        for cond in graph.conditional_edges:
            if any(
                target == loop_node and (cond.source, target) in back
                for target in cond.outcomes.values()
            ):
                router = cond.router
                exit_labels = [
                    label
                    for label, target in cond.outcomes.items()
                    if (cond.source, target) not in back
                ]
                break
        # Iteration cap: a config constant referenced in the loop body / router.
        # Prefer the loop node's own body, then the router; deterministic order.
        counter_const: Optional[str] = None
        bodies: list[str] = []
        node_fn = inventory.functions.get(loop_node)
        if node_fn:
            bodies.append((node_fn.body or "") + "\n" + (node_fn.source or ""))
        if router and router in inventory.functions:
            rfn = inventory.functions[router]
            bodies.append((rfn.body or "") + "\n" + (rfn.source or ""))
        for body in bodies:
            matches = sorted(c for c in const_names if c in body)
            if matches:
                counter_const = matches[0]
                break
        guards.append(
            LoopGuard(
                loop_node=loop_node,
                router=router,
                counter_const=counter_const,
                exit_labels=exit_labels,
            )
        )
    return guards


# ---------------------------------------------------------------------------
# Phase 1 -- HITL payload / resume contract
# ---------------------------------------------------------------------------

def _interrupt_payload(fn: Optional[FunctionSpec]) -> Optional[str]:
    """Verbatim source of the first `interrupt(...)` argument in a node body."""
    if fn is None or not fn.source:
        return None
    funcdef = _find_funcdef(fn.source)
    if funcdef is None:
        return None
    for node in ast.walk(funcdef):
        if (
            isinstance(node, ast.Call)
            and _callable_name(node) == "interrupt"
            and node.args
        ):
            try:
                return ast.unparse(node.args[0])
            except Exception:  # pragma: no cover - defensive
                return None
    return None


def _build_hitl_points(
    graph: GraphSpec, inventory: ComponentInventory
) -> list[HitlPoint]:
    points: list[HitlPoint] = []
    for node in graph.nodes:
        if node.role is not NodeRole.HITL:
            continue
        fn = inventory.functions.get(node.target_callable or "") or inventory.functions.get(
            node.name
        )
        payload = _interrupt_payload(fn)
        points.append(
            HitlPoint(
                node=node.name,
                payload=payload,
                resume_contract=(
                    "Workflow halts here; the value supplied on resume is returned "
                    "by interrupt() and the node continues with that human response."
                ),
            )
        )
    return points


# ---------------------------------------------------------------------------
# Phase 0 -- orchestration mode
# ---------------------------------------------------------------------------

_API_MARKERS = ("FastAPI", "APIRouter", "uvicorn", "Flask", "flask", "fastapi")


def _detect_entrypoint(inventory: ComponentInventory) -> str:
    """"api" if the source exposes a web app (FastAPI/Flask/uvicorn), else "cli"."""
    haystack: list[str] = list(inventory.imports) + list(inventory.preamble)
    for fn in inventory.functions.values():
        if fn.source:
            haystack.append(fn.source)
    blob = "\n".join(haystack)
    return "api" if any(marker in blob for marker in _API_MARKERS) else "cli"


def _classify_mode(graph: GraphSpec) -> OrchestrationMode:
    """SINGLE_AGENT for a trivial tool-using agent; GRAPH_WORKFLOW otherwise."""
    real_nodes = [n for n in graph.nodes if _real(n.name)]
    has_branches = bool(graph.conditional_edges)
    has_hitl = any(n.role is NodeRole.HITL for n in graph.nodes)
    adjacency = _build_adjacency(graph)
    has_cycle = bool(_nodes_in_cycle(adjacency))
    if not has_branches and not has_hitl and not has_cycle and len(real_nodes) <= 1:
        return OrchestrationMode.SINGLE_AGENT
    return OrchestrationMode.GRAPH_WORKFLOW


# ---------------------------------------------------------------------------
# IR assembly
# ---------------------------------------------------------------------------

def build_ir(
    inventory: ComponentInventory,
    readme: ReadmeSections | None = None,
    manifest: RepoManifest | None = None,
    config: Config | None = None,
    target_framework: str = DEFAULT_TARGET_FRAMEWORK,
) -> IR:
    """Assemble the `IR` from consolidated components + README sections."""
    config = config or Config()
    graph = inventory.graph

    # A node is HITL if its function body pauses via interrupt() -- the reliable
    # signal independent of node naming.
    hitl_by_body: set[str] = set()
    for node in graph.nodes:
        fn = inventory.functions.get(node.target_callable or "") or inventory.functions.get(node.name)
        if fn and fn.body and "interrupt(" in fn.body:
            hitl_by_body.add(node.name)

    # Classify graph structure (mutates node roles in place).
    _classify_roles(graph, frozenset(hitl_by_body))
    pattern = _classify_pattern(graph)

    # Phase 1: per-node data-flow (reads / writes / tool calls), in place.
    state_names = {f.name for f in inventory.state}
    tool_names = {t.name for t in inventory.tools}
    for node in graph.nodes:
        fn = inventory.functions.get(node.target_callable or "") or inventory.functions.get(
            node.name
        )
        node.reads, node.writes, node.calls_tools = _analyze_node(
            fn, state_names, tool_names
        )

    # Phase 1: flatten edges to explicit triples; detect loops + guards; HITL.
    adjacency = _build_adjacency(graph)
    back = _back_edges(adjacency, graph.entry_point)
    flat_edges = _flatten_edges(graph, back)
    loop_guards = _loop_guards(graph, back, inventory)
    hitl_points = _build_hitl_points(graph, inventory)

    workflow = WorkflowSpec(
        pattern=pattern,
        nodes=graph.nodes,
        edges=graph.edges,
        conditional_edges=graph.conditional_edges,
        entry_point=graph.entry_point,
        readme_description=readme.workflow_description if readme else None,
        flat_edges=flat_edges,
        loop_guards=loop_guards,
        hitl_points=hitl_points,
    )

    metadata = IRMetadata(
        description=readme.purpose if readme else None,
        source_framework=manifest.detected_framework if manifest else None,
        target_framework=target_framework,
        target_framework_version=detect_target_version(target_framework),
        orchestration_mode=_classify_mode(graph),
        llm_provider=inventory.config.llm_provider,
        checkpointer=inventory.checkpointer or inventory.config.checkpointer,
        entrypoint=_detect_entrypoint(inventory),
    )

    return IR(
        metadata=metadata,
        tools=inventory.tools,
        state=inventory.state,
        config=inventory.config,  # temperature already None if not found
        workflow=workflow,
        functions=inventory.functions,
        imports=inventory.imports,
        preamble=inventory.preamble,
        state_class_names=inventory.state_class_names,
        files=manifest.files if manifest else [],
        agent_specs=inventory.agent_specs,
        task_specs=inventory.task_specs,
    )


def write_ir_json(ir: IR, path: str = "ir.json") -> str:
    """Write the IR debug checkpoint to `path`; returns the path written."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ir.to_json_dict(), fh, indent=2, default=str)
    return path
