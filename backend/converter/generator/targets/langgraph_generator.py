"""LangGraph target generator.

Emits a real LangGraph `StateGraph`: one `add_node` per IR node, `set_entry_point`,
plain `add_edge`s, and `add_conditional_edges(source, router, {label: target})`
for branches. The block is import-guarded (`try: from langgraph.graph import ...`)
so the offline `run()` fast-path the core generator synthesizes keeps the package
runnable when LangGraph is not installed -- and the smoke test stays offline-safe.

LangGraph is a public pip package, so `sdk_stub_files()` is empty (no local stub).
This is the second `TargetGenerator`; it proves the Phase-2 seam by producing a
non-MAF target from the same neutral IR (any source -> LangGraph).
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter
from converter.contracts import IR, NodeRole
from converter.generator.targets.base import TargetGenerator

_SENTINEL_NODES = frozenset({"START", "END", "__start__", "__end__"})


def _langgraph_graph_block(ir: IR, adapter: TargetAdapter) -> str:
    wf = ir.workflow
    if not wf or not wf.nodes:
        return ""
    ctx = adapter.context_class_name
    roles = {n.name: n.role for n in wf.nodes}
    nodes = [n for n in wf.nodes if roles.get(n.name) is not NodeRole.AUX]
    if not nodes:
        return ""
    real = {n.name for n in nodes}
    has_checkpointer = bool(ir.metadata.checkpointer)
    has_hitl = bool(ir.workflow and ir.workflow.hitl_points)
    needs_checkpointer = has_checkpointer or has_hitl

    out: list[str] = [
        "# --- LangGraph StateGraph (generated from the IR) ---",
        "# One add_node per node, guarded edges per router outcome. Import-guarded so",
        "# the offline run(ctx) above still drives the agent when langgraph is absent.",
        "try:",
        "    from langgraph.graph import StateGraph, END",
    ]
    if has_hitl:
        out.append("    from langgraph.types import interrupt  # noqa: F401 -- used in node bodies")
    out += [
        "    _HAVE_LANGGRAPH = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_LANGGRAPH = False",
        "",
        "",
        "if _HAVE_LANGGRAPH:",
        "",
        "    def build_graph():",
        '        """Assemble and compile the LangGraph StateGraph."""',
        f"        builder = StateGraph({ctx})",
    ]
    for node in nodes:
        out.append(f'        builder.add_node("{node.name}", {node.name})')

    entry = wf.entry_point if wf.entry_point in real else nodes[0].name
    out.append(f'        builder.set_entry_point("{entry}")')

    # Plain edges (START handled by set_entry_point; END -> LangGraph END).
    for edge in wf.edges:
        src, tgt = edge.source, edge.target
        if src in _SENTINEL_NODES or src not in real:
            continue
        if tgt in _SENTINEL_NODES:
            out.append(f'        builder.add_edge("{src}", END)')
        elif tgt in real:
            out.append(f'        builder.add_edge("{src}", "{tgt}")')

    # Conditional edges -> add_conditional_edges(source, router, {label: target}).
    for cond in wf.conditional_edges:
        if cond.source not in real:
            continue
        router = cond.router or "route"
        pairs = []
        for label, tgt in cond.outcomes.items():
            dest = "END" if tgt in _SENTINEL_NODES else f'"{tgt}"'
            pairs.append(f'"{label}": {dest}')
        out.append(
            f'        builder.add_conditional_edges("{cond.source}", {router}, '
            f"{{{', '.join(pairs)}}})"
        )

    if needs_checkpointer:
        # HITL via interrupt() requires a checkpointer so the graph can resume.
        reason = "HITL interrupt() requires a checkpointer to resume" if has_hitl else "Source used a checkpointer"
        out.append(f"        # {reason}.")
        out.append("        from langgraph.checkpoint.memory import MemorySaver")
        out.append("        return builder.compile(checkpointer=MemorySaver())")
    else:
        out.append("        return builder.compile()")
    out.append("")
    out.append(f"    def run_graph(state: {ctx}):")
    out.append('        """Run the compiled LangGraph graph."""')
    out.append("        return build_graph().invoke(state)")
    out.append("# --- end LangGraph graph ---")
    return "\n".join(out)


def _smoke_test(adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    return (
        f'"""Smoke test for the converted {target} agent (offline-safe)."""\n\n'
        f"from agent_context import {ctx}\n"
        "import orchestrator\n\n\n"
        "def test_state_constructs():\n"
        f"    assert {ctx}() is not None\n\n\n"
        "def test_offline_entrypoint_present():\n"
        '    """The offline fast-path run() must exist even without langgraph."""\n'
        '    assert callable(getattr(orchestrator, "run", None))\n\n\n'
        "def test_state_advance_returns_new_instance():\n"
        f"    a = {ctx}()\n"
        "    b = a.advance()\n"
        f"    assert isinstance(b, {ctx}) and b is not a\n"
    )


def _entrypoint(ir: IR, adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    is_api = (ir.metadata.entrypoint or "cli") == "api"

    lines: list[str] = [
        f'"""Generated entrypoint for the converted {target} agent."""',
        "from __future__ import annotations",
        "",
        f"from agent_context import {ctx}",
        "from orchestrator import run",
        "",
        "try:",
        "    from orchestrator import build_graph",
        "    _HAVE_LANGGRAPH = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_LANGGRAPH = False",
        "",
        "",
        f"def run_agent(state: {ctx} | None = None):",
        '    """Run via the compiled LangGraph graph when available, else offline run()."""',
        f"    state = state or {ctx}()",
        "    if _HAVE_LANGGRAPH:",
        "        return build_graph().invoke(state)",
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
            "    return run_agent(state)",
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


class LangGraphTargetGenerator(TargetGenerator):
    """Emits LangGraph output (any source -> LangGraph)."""

    name = "langgraph"

    def workflow_block(self, ir: IR, adapter: TargetAdapter) -> str:
        return _langgraph_graph_block(ir, adapter)

    def sdk_stub_files(self) -> dict[str, str]:
        # langgraph is a public pip package -> no local SDK stub needed.
        return {}

    def smoke_test(self, adapter: TargetAdapter, target: str) -> str:
        return _smoke_test(adapter, target)

    def entrypoint(self, ir: IR, adapter: TargetAdapter, target: str) -> str:
        return _entrypoint(ir, adapter, target)

    def orchestrator_must_tokens(self, has_workflow_block: bool) -> list[str]:
        must = ["def run"]
        if has_workflow_block:
            must += ["StateGraph", "add_node", "set_entry_point"]
        return must
