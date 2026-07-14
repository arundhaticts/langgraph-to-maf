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

import json
from typing import Optional

from converter.config import Config
from converter.contracts import (
    ComponentInventory,
    GraphNode,
    GraphSpec,
    IR,
    IRMetadata,
    NodeRole,
    OrchestrationPattern,
    ReadmeSections,
    RepoManifest,
    WorkflowSpec,
)

# Sentinel node names that are not real nodes.
_SENTINELS = frozenset({"START", "END", "__start__", "__end__"})

# Substrings that mark a human-in-the-loop node (heuristic; R-08 territory).
_HITL_MARKERS = ("hitl", "human", "approval", "interrupt")

DEFAULT_TARGET_FRAMEWORK = "maf"


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


def _classify_roles(graph: GraphSpec) -> None:
    """Assign `role` to every node in place.

    Precedence: HITL > ENTRY > BRANCH > LOOP > TERMINAL > AUX > LINEAR.
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
        if _is_hitl(name):
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

    # Classify graph structure (mutates node roles in place).
    _classify_roles(graph)
    pattern = _classify_pattern(graph)

    workflow = WorkflowSpec(
        pattern=pattern,
        nodes=graph.nodes,
        edges=graph.edges,
        conditional_edges=graph.conditional_edges,
        entry_point=graph.entry_point,
        readme_description=readme.workflow_description if readme else None,
    )

    metadata = IRMetadata(
        description=readme.purpose if readme else None,
        source_framework=manifest.detected_framework if manifest else None,
        target_framework=target_framework,
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
    )


def write_ir_json(ir: IR, path: str = "ir.json") -> str:
    """Write the IR debug checkpoint to `path`; returns the path written."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ir.to_json_dict(), fh, indent=2, default=str)
    return path
