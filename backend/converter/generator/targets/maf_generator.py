"""MAF (Microsoft Agent Framework) target generator.

Owns everything MAF-specific the output package needs: the true WorkflowBuilder
graph (one `@executor` per node, guarded edges, RequestInfoExecutor HITL,
checkpointing), the offline `agent_framework/` SDK stub so the package runs
without the real SDK, the smoke test, and the `main.py` entrypoint that drives
the workflow with `run_stream` + `send_responses_streaming`.

This is the reference `TargetGenerator`; a new target (LangGraph, CrewAI, ...)
is a sibling subclass with the same surface. The code here was extracted
verbatim from the original `code_generator.py` so MAF output is unchanged.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from converter.adapters.base import TargetAdapter
from converter.contracts import (
    IR,
    NodeRole,
    OrchestrationMode,
)
from converter.generator.targets.base import TargetGenerator

# ---------------------------------------------------------------------------
# Phases 4-5 -- true MAF WorkflowBuilder graph (executors + guarded edges)
# ---------------------------------------------------------------------------

_SENTINEL_NODES = frozenset({"START", "END", "__start__", "__end__"})


def _executor_name(node_name: str) -> str:
    return f"{node_name}_exec"


def _maf_back_edges(adjacency: dict[str, set[str]], entry: Optional[str]) -> set[tuple[str, str]]:
    """DFS back-edges (u -> v where v is an ancestor on the DFS stack)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in adjacency}
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


def _maf_edge_tuples(wf) -> list[tuple]:
    """(source, target, condition_label, router, is_loop) for the MAF graph.

    Prefers the IR's pre-flattened `flat_edges` (populated by build_ir); falls
    back to deriving them from edges/conditional_edges so a hand-built IR (or any
    caller that skips build_ir) still produces a correct graph.
    """
    if wf.flat_edges:
        return [
            (e.source, e.target, e.condition_label, e.router, e.is_loop)
            for e in wf.flat_edges
        ]
    raw: list[tuple] = []
    adjacency: dict[str, set[str]] = {}

    def _adj(src: str, tgt: str) -> None:
        if src not in _SENTINEL_NODES and tgt not in _SENTINEL_NODES:
            adjacency.setdefault(src, set()).add(tgt)

    for edge in wf.edges:
        raw.append((edge.source, edge.target, None, None))
        _adj(edge.source, edge.target)
    for cond in wf.conditional_edges:
        for label, tgt in cond.outcomes.items():
            raw.append((cond.source, tgt, label, cond.router))
            _adj(cond.source, tgt)
    back = _maf_back_edges(adjacency, wf.entry_point)
    return [(s, t, lbl, r, (s, t) in back) for (s, t, lbl, r) in raw]


def _pascal(name: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[_\W]+", name) if p)


def _maf_workflow_block(ir: IR, adapter: TargetAdapter) -> str:
    """Emit the true MAF graph: executors + guarded WorkflowBuilder + tools + HITL.

    Phase 4: each node becomes a pure `node(state)->State` fn (already emitted)
    wrapped in an `@executor` that `ctx.send_message(...)`s the new state; terminal
    nodes `ctx.yield_output(...)`. Phase 5: every conditional outcome becomes a
    guarded `.add_edge(src, tgt, condition=...)`, loop back-edges included.
    Phase 6: the converted tools are registered (AGENT_TOOLS / NODE_TOOLS) so none
    are orphaned, and a ChatAgent is wired for single-agent mode. Phase 7: each
    HITL node has a real RequestInfoExecutor pause path alongside an auto-approve
    fast-path (AUTO_APPROVE_HITL), and checkpointing is enabled when the source
    used a saver. The whole block is SDK-guarded so the offline `run()` above keeps
    the agent runnable when agent-framework is absent.
    """
    wf = ir.workflow
    if not wf or not wf.nodes:
        return ""
    ctx_class = adapter.context_class_name
    roles = {n.name: n.role for n in wf.nodes}
    nodes = [n for n in wf.nodes if roles.get(n.name) is not NodeRole.AUX]
    if not nodes:
        return ""
    real_names = {n.name for n in nodes}
    hitl_names = [n.name for n in nodes if roles.get(n.name) is NodeRole.HITL]
    hitl_payload = {h.node: h.payload for h in (wf.hitl_points or [])}

    edge_tuples = _maf_edge_tuples(wf)
    succ: dict[str, list[tuple]] = {}
    for tup in edge_tuples:
        succ.setdefault(tup[0], []).append(tup)

    # Phase 6: register EVERY converted tool (so none is emitted-but-never-wired),
    # plus the node->tool mapping for the tools each node actually calls.
    node_tools = {n.name: sorted(n.calls_tools) for n in nodes if n.calls_tools}
    all_tools = sorted({t.name for t in ir.tools})

    single_agent = ir.metadata.orchestration_mode is OrchestrationMode.SINGLE_AGENT
    has_checkpointer = bool(ir.metadata.checkpointer)

    imports = ["WorkflowBuilder", "WorkflowContext", "executor"]
    if hitl_names:
        imports += ["RequestInfoExecutor", "RequestInfoMessage"]
    if single_agent and all_tools:
        imports.append("ChatAgent")
    if has_checkpointer:
        imports.append("FileCheckpointStorage")

    out: list[str] = [
        "# --- True Microsoft Agent Framework workflow (Phases 4-7) ---",
        "# Generated from the IR graph: one @executor per node, guarded edges per",
        "# router outcome, tools registered, HITL via RequestInfoExecutor. Requires",
        "# the agent-framework SDK; when it is absent, run(ctx) above is the offline",
        "# fast-path over the same pure node functions.",
        "try:",
        "    from agent_framework import " + ", ".join(sorted(set(imports))),
        "    _HAVE_AGENT_FRAMEWORK = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_AGENT_FRAMEWORK = False",
    ]
    if hitl_names:
        out += [
            "",
            "",
            "# Phase 7 switch: True auto-approves (runs end-to-end, no human); False",
            "# triggers a genuine human pause via RequestInfoExecutor. Both are wired.",
            "AUTO_APPROVE_HITL = True",
        ]
    out += ["", "", "if _HAVE_AGENT_FRAMEWORK:", ""]

    if hitl_names:
        out.append("    from dataclasses import dataclass, field")
        out.append("")

    # Phase 6: register tools (executors call them directly per NODE_TOOLS).
    if all_tools:
        out.append("    # Phase 6: converted tools, registered so none are orphaned. Executors")
        out.append("    # call them directly per the IR node->tool mapping (NODE_TOOLS below);")
        out.append("    # AGENT_TOOLS also feeds a ChatAgent for single-agent mode.")
        out.append(f"    AGENT_TOOLS = [{', '.join(all_tools)}]")
        pairs = ", ".join(f'"{n}": [{", ".join(ts)}]' for n, ts in node_tools.items())
        out.append("    NODE_TOOLS = {" + pairs + "}")
        out.append("")

    # Phase 7: one human-approval request message per HITL node.
    for hn in hitl_names:
        cls = _pascal(hn) + "Request"
        out.append("    @dataclass")
        out.append(f"    class {cls}(RequestInfoMessage):")
        out.append(f'        """Human-approval request emitted by node \'{hn}\'."""')
        out.append("        # payload carries the approval context (source shape in MIGRATION_REPORT.md).")
        out.append("        payload: dict = field(default_factory=dict)")
        out.append("")

    # Executors (Phase 4 + Phase 7 HITL branch).
    for node in nodes:
        name = node.name
        exec_name = _executor_name(name)
        outs = succ.get(name, [])
        real_outs = [t for t in outs if t[1] not in _SENTINEL_NODES and t[1] in real_names]
        end_outs = [t for t in outs if t[1] in _SENTINEL_NODES]
        router = next((t[3] for t in outs if t[3]), None)
        out.append(f'    @executor(id="{name}")')
        out.append(
            f"    async def {exec_name}(state: {ctx_class}, "
            f"ctx: WorkflowContext[{ctx_class}]) -> None:"
        )
        out.append(f'        """Executor for node \'{name}\' (wraps the ported node fn)."""')
        if name in hitl_names:
            cls = _pascal(name) + "Request"
            out.append("        if AUTO_APPROVE_HITL:")
            out.append(f"            await ctx.send_message({name}(state))  # fast-path (auto-approve)")
            out.append("        else:")
            out.append(
                f'            await ctx.send_message({cls}(payload={{"node": "{name}"}}))'
                "  # genuine human pause"
            )
        else:
            out.append(f"        result = {name}(state)")
            if end_outs and real_outs and router and end_outs[0][2]:
                end_label = end_outs[0][2]
                out.append(f'        if {router}(result) == "{end_label}":')
                out.append("            await ctx.yield_output(result)")
                out.append("        else:")
                out.append("            await ctx.send_message(result)")
            elif real_outs and not (roles.get(name) is NodeRole.TERMINAL):
                out.append("        await ctx.send_message(result)")
            else:
                out.append("        await ctx.yield_output(result)")
        out.append("")

    # Phase 7: apply-executor + RequestInfoExecutor instance for HITL.
    for hn in hitl_names:
        out.append(f'    @executor(id="{hn}_apply")')
        out.append(
            f"    async def {hn}_apply_exec(response, "
            f"ctx: WorkflowContext[{ctx_class}]) -> None:"
        )
        out.append(f'        """Apply the human decision for \'{hn}\', then continue."""')
        out.append("        # `response` is the human reply gathered by RequestInfoExecutor.")
        out.append("        # Merge the decision into the checkpointed state and continue")
        out.append("        # (implement the real approval logic; see MIGRATION_REPORT.md).")
        out.append(f"        await ctx.send_message({hn}(response))")
        out.append("")
    if hitl_names:
        out.append('    _request_info_exec = RequestInfoExecutor(id="request_info")')
        out.append("")

    # Phase 6: single tool-using agent (ChatAgent) for single-agent mode.
    if single_agent and all_tools:
        out.append("    def build_agent():")
        out.append('        """Single tool-using agent: converted tools wired onto a ChatAgent."""')
        out.append("        # TODO: pass your chat client (LLM provider). Tools are wired here.")
        out.append("        return ChatAgent(chat_client=None, tools=AGENT_TOOLS)")
        out.append("")

    # build_workflow (Phase 5 edges + Phase 7 HITL wiring + checkpointing).
    entry = wf.entry_point if wf.entry_point in real_names else nodes[0].name
    out.append("    def build_workflow():")
    out.append('        """Assemble the MAF workflow graph (Phase 8 wiring)."""')
    out.append(
        f"        builder = WorkflowBuilder().set_start_executor({_executor_name(entry)})"
    )
    for source, target, label, router, is_loop in edge_tuples:
        if source in _SENTINEL_NODES or target in _SENTINEL_NODES:
            continue
        if source not in real_names or target not in real_names:
            continue
        if source in hitl_names:
            continue  # HITL sources wired in the HITL section below
        src_x = _executor_name(source)
        tgt_x = _executor_name(target)
        if label and router:
            suffix = "  # loop back-edge" if is_loop else ""
            out.append(
                f"        builder.add_edge({src_x}, {tgt_x}, "
                f'condition=lambda s, _r={router}, _l="{label}": '
                f"_r(s) == _l){suffix}"
            )
        else:
            out.append(f"        builder.add_edge({src_x}, {tgt_x})")
    for hn in hitl_names:
        succs = [
            t for (s, t, _l, _r, _lp) in edge_tuples
            if s == hn and t in real_names and t not in _SENTINEL_NODES
        ]
        hx = _executor_name(hn)
        out.append(f"        # HITL '{hn}': fast-path continues directly; pause routes via RequestInfoExecutor.")
        out.append("        if AUTO_APPROVE_HITL:")
        if succs:
            for sc in succs:
                out.append(f"            builder.add_edge({hx}, {_executor_name(sc)})")
        else:
            out.append("            pass")
        out.append("        else:")
        out.append(f"            builder.add_edge({hx}, _request_info_exec)")
        out.append(f"            builder.add_edge(_request_info_exec, {hn}_apply_exec)")
        if succs:
            for sc in succs:
                out.append(f"            builder.add_edge({hn}_apply_exec, {_executor_name(sc)})")
        else:
            out.append("            pass")
    if has_checkpointer:
        out.append("        # Phase 7: source used a checkpointer -> enable file-based checkpointing.")
        out.append('        builder = builder.with_checkpointing(FileCheckpointStorage("./checkpoints"))')
    out.append("        return builder.build()")
    out.append("")
    out.append(f"    async def run_workflow(state: {ctx_class}):")
    out.append('        """Run the true MAF workflow; returns its output events."""')
    out.append("        workflow = build_workflow()")
    out.append("        return await workflow.run(state)")
    out.append("# --- end MAF workflow ---")
    return "\n".join(out)


# A fixed, per-conversion-independent stub of the `agent-framework` SDK. Generated
# into the output as a local `agent_framework/` package so the converted MAF graph
# BUILDS AND RUNS without the real (internal) SDK. Python resolves the local
# package before site-packages, so `from agent_framework import ...` finds this.
# When the real SDK is installed, delete the folder -- it is a drop-in replacement.
_AGENT_FRAMEWORK_STUB = '''"""Local stub of the Microsoft Agent Framework (`agent-framework`) SDK.

Auto-generated by the Framework Conversion Utility. This is NOT the real SDK --
it implements just enough of the interface (pure-Python async) that the converted
agent\\'s MAF workflow builds and runs offline. Python resolves this local package
before site-packages, so `from agent_framework import ...` finds this stub. When
the real SDK is installed, delete this folder -- it is a drop-in replacement and
nothing else in the converted package changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# --- Events -------------------------------------------------------------
@dataclass
class WorkflowOutputEvent:
    data: Any = None


@dataclass
class WorkflowStatusEvent:
    state: Any = None


@dataclass
class RequestInfoEvent:
    request_id: str = ""
    data: Any = None


# --- Human-in-the-loop --------------------------------------------------
@dataclass
class RequestInfoMessage:
    """Base for HITL request payloads (the agent subclasses this)."""
    request_id: str = ""


class RequestInfoExecutor:
    """Stub: would gather human input. In this stub HITL never actually pauses."""

    def __init__(self, id: str = "request_info", **kwargs: Any) -> None:
        self.id = id


# --- Checkpointing ------------------------------------------------------
class FileCheckpointStorage:
    def __init__(self, path: str = "./checkpoints", **kwargs: Any) -> None:
        self.path = path


# --- Tool decorator -----------------------------------------------------
def ai_function(*args: Any, **kwargs: Any):
    """No-op tool decorator. Supports both `@ai_function` and `@ai_function(...)`."""
    if args and callable(args[0]) and not kwargs:
        return args[0]

    def _wrap(fn: Callable) -> Callable:
        return fn

    return _wrap


# --- Single-agent mode --------------------------------------------------
class ChatAgent:
    def __init__(self, chat_client: Any = None, tools: Any = None, **kwargs: Any) -> None:
        self.chat_client = chat_client
        self.tools = list(tools or [])
        self.kwargs = kwargs

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return None


# --- Executor + context -------------------------------------------------
class WorkflowContext:
    """Runtime context passed to each executor; also usable as a type hint."""

    def __init__(self) -> None:
        self.messages: list = []
        self.outputs: list = []

    def __class_getitem__(cls, item):  # allow WorkflowContext[State] annotations
        return cls

    async def send_message(self, message: Any) -> None:
        self.messages.append(message)

    async def yield_output(self, data: Any) -> None:
        self.outputs.append(data)


def executor(id: Any = None, **kwargs: Any):
    """Register an async function as a workflow node.

    Supports `@executor(id="x")` and bare `@executor`. Returns the function with
    an `.id` attribute so it can act as a node identity in the builder.
    """
    def _wrap(fn: Callable) -> Callable:
        fn.id = id if isinstance(id, str) else getattr(fn, "__name__", "executor")
        fn._is_executor = True
        return fn

    if callable(id):  # bare @executor
        return _wrap(id)
    return _wrap


# --- Workflow -----------------------------------------------------------
class _Workflow:
    def __init__(self, start, edges, checkpoint=None, max_steps: int = 10000) -> None:
        self._start = start
        self._edges = edges  # list of (source, target, condition-or-None)
        self._checkpoint = checkpoint
        self._max_steps = max_steps

    async def _drive(self, state: Any) -> list:
        events: list = []
        if self._start is None:
            return events
        queue = [(self._start, state)]
        steps = 0
        while queue and steps < self._max_steps:
            steps += 1
            node, message = queue.pop(0)
            ctx = WorkflowContext()
            await node(message, ctx)
            for out in ctx.outputs:
                events.append(WorkflowOutputEvent(data=out))
            for msg in ctx.messages:
                for source, target, condition in self._edges:
                    if source is node:
                        try:
                            ok = condition is None or bool(condition(msg))
                        except Exception:
                            ok = False
                        if ok:
                            queue.append((target, msg))
        return events

    async def run(self, state: Any) -> Any:
        events = await self._drive(state)
        return events[-1].data if events else state

    async def run_stream(self, state: Any):
        for event in await self._drive(state):
            yield event

    async def send_responses_streaming(self, responses: Any):
        # HITL never pauses in this stub, so there is nothing to resume.
        if False:  # pragma: no cover
            yield None
        return


class WorkflowBuilder:
    def __init__(self) -> None:
        self._start = None
        self._edges: list = []
        self._checkpoint = None

    def set_start_executor(self, ex):
        self._start = ex
        return self

    def add_edge(self, source, target, condition=None):
        self._edges.append((source, target, condition))
        return self

    def with_checkpointing(self, storage):
        self._checkpoint = storage
        return self

    def build(self):
        return _Workflow(self._start, self._edges, self._checkpoint)

    # --- LangGraph-style compatibility aliases --------------------------
    def set_entry_point(self, ex):
        self._start = ex
        return self

    def set_finish_point(self, ex):
        return self

    def add_node(self, name, fn=None):
        return self

    def add_conditional_edges(self, source, router, mapping):
        for label, target in dict(mapping).items():
            self._edges.append((source, target, (lambda m, _r=router, _l=label: _r(m) == _l)))
        return self

    def compile(self):
        return self.build()


__all__ = [
    "WorkflowBuilder",
    "WorkflowContext",
    "executor",
    "RequestInfoExecutor",
    "RequestInfoMessage",
    "RequestInfoEvent",
    "WorkflowOutputEvent",
    "WorkflowStatusEvent",
    "FileCheckpointStorage",
    "ChatAgent",
    "ai_function",
]
'''


def _smoke_test(adapter: TargetAdapter, target: str) -> str:
    """A minimal offline-safe smoke test for the converted package."""
    ctx = adapter.context_class_name
    return (
        f'"""Smoke test for the converted {target} agent (offline-safe)."""\n\n'
        f"from agent_context import {ctx}\n"
        "import orchestrator\n\n\n"
        "def test_state_constructs():\n"
        f"    assert {ctx}() is not None\n\n\n"
        "def test_offline_entrypoint_present():\n"
        '    """The offline fast-path run() must exist even without the MAF SDK."""\n'
        '    assert callable(getattr(orchestrator, "run", None))\n\n\n'
        "def test_state_advance_returns_new_instance():\n"
        f"    a = {ctx}()\n"
        "    b = a.advance()\n"
        f"    assert isinstance(b, {ctx}) and b is not a\n"
    )


def _entrypoint(ir: IR, adapter: TargetAdapter, target: str) -> str:
    """Phase 9 entrypoint.

    Drives the true MAF workflow with `run_stream` + `send_responses_streaming`
    (handling HITL RequestInfoEvents until a WorkflowOutputEvent), and falls back
    to the offline `run()` when agent-framework is absent. When the source exposed
    a web app, a FastAPI endpoint + uvicorn runner is emitted; otherwise a CLI.
    """
    ctx = adapter.context_class_name
    is_api = (ir.metadata.entrypoint or "cli") == "api"

    lines: list[str] = [
        f'"""Generated entrypoint for the converted {target} agent."""',
        "from __future__ import annotations",
        "",
        "import asyncio",
        f"from agent_context import {ctx}",
        "from orchestrator import run",
        "",
        "try:",
        "    from orchestrator import build_workflow",
        "    from agent_framework import WorkflowOutputEvent, RequestInfoEvent",
        "    _HAVE_AGENT_FRAMEWORK = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_AGENT_FRAMEWORK = False",
        "",
        "",
        "def _respond_to(event) -> dict:",
        '    """Human response for one HITL request. Default approves; wire real input here."""',
        '    return {"approved": True}',
        "",
        "",
        f"async def run_workflow_stream(state: {ctx}):",
        '    """Phase 9: drive the MAF workflow, resolving HITL requests until output."""',
        "    workflow = build_workflow()",
        "    result = None",
        "    stream = workflow.run_stream(state)",
        "    while True:",
        "        responses = {}",
        "        async for event in stream:",
        "            if isinstance(event, WorkflowOutputEvent):",
        "                result = event.data",
        "            elif isinstance(event, RequestInfoEvent):",
        "                responses[event.request_id] = _respond_to(event)",
        "        if not responses:",
        "            break",
        "        # Resume the halted workflow with the collected human responses.",
        "        stream = workflow.send_responses_streaming(responses)",
        "    return result",
        "",
        "",
        f"def run_agent(state: {ctx} | None = None):",
        '    """Run via the MAF workflow when the SDK is present, else offline run()."""',
        f"    state = state or {ctx}()",
        "    if _HAVE_AGENT_FRAMEWORK:",
        "        return asyncio.run(run_workflow_stream(state))",
        "    return run(state)  # offline fast-path",
        "",
        "",
    ]

    if is_api:
        lines += [
            "from fastapi import FastAPI",
            "import uvicorn",
            "",
            "app = FastAPI()",
            "",
            "",
            '@app.post("/run")',
            "async def run_endpoint(payload: dict | None = None):",
            '    """Run the agent; converted from the source API entrypoint."""',
            f"    state = {ctx}(**(payload or {{}}))",
            "    result = run_agent(state)",
            "    return result",
            "",
            "",
            "def main():",
            '    uvicorn.run(app, host="0.0.0.0", port=8000)',
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    else:
        lines += [
            "def main():",
            "    result = run_agent()",
            "    print(result)",
            "    return result",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]

    return "\n".join(lines)


class MAFTargetGenerator(TargetGenerator):
    """Emits Microsoft Agent Framework output (the reference target)."""

    name = "maf"

    def workflow_block(self, ir: IR, adapter: TargetAdapter) -> str:
        return _maf_workflow_block(ir, adapter)

    def sdk_stub_files(self) -> dict[str, str]:
        # Local agent_framework/ stub so the MAF graph builds and runs without
        # the real (internal) SDK. Python resolves this local package first.
        return {os.path.join("agent_framework", "__init__.py"): _AGENT_FRAMEWORK_STUB}

    def smoke_test(self, adapter: TargetAdapter, target: str) -> str:
        return _smoke_test(adapter, target)

    def entrypoint(self, ir: IR, adapter: TargetAdapter, target: str) -> str:
        return _entrypoint(ir, adapter, target)

    def orchestrator_must_tokens(self, has_workflow_block: bool) -> list[str]:
        must = ["def run"]
        if has_workflow_block:
            must += ["WorkflowBuilder", "@executor", "set_start_executor"]
        return must
