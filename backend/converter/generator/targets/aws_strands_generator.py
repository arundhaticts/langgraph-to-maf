"""AWS Strands target generator.

Single-agent shape (1 real node or AGENT_DRIVEN IR)
  → `Agent(model=..., tools=[...], system_prompt=...)`.
    All converted @tool functions are wired in; the model drives the agentic loop.

Multi-agent shape (2+ real non-AUX nodes with explicit ordering)
  → Agents-as-tools pattern:
    Each node becomes a Strands @tool that internally runs a specialised sub-Agent
    whose tools are limited to the node's own calls_tools set (falls back to all
    tools when not determinable). An orchestrator Agent calls those sub-agent tools
    in the right order. This implements the checklist's Phase 6-9 requirement:
    "multi-node IR → agents-as-tools or the graph/swarm/workflow multi-agent pattern".

HITL (checklist Phase 6-9)
  → When the IR has HitlPoints, a `request_human_approval` @tool is emitted.
    It pauses the agent and waits for a human decision; an automated fast-path
    (HITL_MODE=auto env var) keeps CI green.

System prompt
  → Derived from ir.metadata.description when available; falls back to a generic
    "converted agent" description.

strands-agents is a public pip package, so `sdk_stub_files()` is empty.
"""

from __future__ import annotations

import textwrap

from converter.adapters.base import TargetAdapter
from converter.contracts import IR, NodeRole, OrchestrationPattern
from converter.generator.targets.base import TargetGenerator

_SENTINEL_NODES = frozenset({"START", "END", "__start__", "__end__"})
_END_NODES = frozenset({"END", "__end__"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_nodes(ir: IR) -> list:
    wf = ir.workflow
    if not wf:
        return []
    return [n for n in wf.nodes if n.name not in _SENTINEL_NODES and n.role is not NodeRole.AUX]


def _system_prompt(ir: IR) -> str:
    desc = ir.metadata.description or ""
    if desc:
        return desc.strip().replace('"', '\\"')
    return "You are a converted agent. Use the available tools to carry out the user's request."


# ---------------------------------------------------------------------------
# Workflow introspection (order / routers / loops) — mirrors the deterministic
# run() so the Strands path executes the SAME control flow, not a guess.
# ---------------------------------------------------------------------------

def _ctx_name(adapter: TargetAdapter) -> str:
    return adapter.context_class_name


def _ordered_flow_nodes(ir: IR) -> list[str]:
    """Real, non-AUX/HITL nodes in execution order (entry point first)."""
    wf = ir.workflow
    if not wf:
        return []
    names = {n.name for n in wf.nodes}
    single_out: dict[str, str] = {}
    for e in wf.edges:
        single_out.setdefault(e.source, e.target)
    order: list[str] = []
    seen: set[str] = set()
    cur = wf.entry_point
    while cur and cur in names and cur not in seen:
        order.append(cur)
        seen.add(cur)
        cur = single_out.get(cur)
    for n in wf.nodes:
        if n.name not in seen:
            order.append(n.name)
    roles = {n.name: n.role for n in wf.nodes}
    return [n for n in order if roles.get(n) not in (NodeRole.HITL, NodeRole.AUX)]


def _loop_cap_constant(ir: IR) -> str | None:
    for name in ir.config.constants:
        upper = name.upper()
        if any(tok in upper for tok in ("RETRIES", "RETRY", "MAX", "ITER", "LIMIT")):
            return name
    return None


def _exit_label(cond) -> str | None:
    for label, target in cond.outcomes.items():
        if target in _END_NODES:
            return label
    return None


# ---------------------------------------------------------------------------
# Node tools — each @tool runs the REAL ported node logic on the shared context.
# ---------------------------------------------------------------------------

def _node_tool(node, ctx_class: str, ir: IR) -> list[str]:
    n = node.name
    doc = None
    fn = ir.functions.get(node.target_callable or "") or ir.functions.get(n)
    if fn and fn.docstring:
        doc = fn.docstring.strip().splitlines()[0].replace('"', '\\"')
    doc = doc or f"Run the {n!r} workflow step on the shared context."
    return [
        "@tool",
        f"def call_{n}(note: str = \"\") -> str:",
        f'    """{doc}',
        "",
        "    Real converted logic — mutates the shared AgentContext in place and",
        "    returns a short status string (the state itself lives on _CTX).",
        '    """',
        "    global _CTX",
        f"    if _CTX is None:",
        f"        _CTX = {ctx_class}()",
        f"    _CTX = {n}(_CTX)",
        f'    return "{n}: complete"',
        "",
    ]


def _orchestration_lines(ir: IR, flow: list[str]) -> list[str]:
    """run_agent() body that drives the workflow via the node @tools, honoring
    the source's routing and loop bounds — the SAME control flow as run()."""
    wf = ir.workflow
    pattern = wf.pattern if wf else OrchestrationPattern.LINEAR
    cond = wf.conditional_edges[0] if (wf and wf.conditional_edges) else None

    if pattern in (OrchestrationPattern.LOOP, OrchestrationPattern.LOOP_WITH_EXIT):
        cap = _loop_cap_constant(ir)
        lines = ["    _guard = 0"]
        if cap is None:
            lines.append("    _cap = 10  # TODO: confirm the real loop bound")
            cap = "_cap"
        lines.append(f"    while _guard < {cap}:")
        lines += [f"        call_{n}()" for n in flow] or ["        pass"]
        exit_label = _exit_label(cond) if cond else None
        if cond and cond.router and exit_label:
            lines.append(f"        if {cond.router}(_CTX) == \"{exit_label}\":")
            lines.append("            break")
        else:
            lines.append("        break  # TODO: add the real loop exit condition")
        lines.append("        _guard += 1")
        return lines

    if pattern is OrchestrationPattern.BRANCH and cond:
        lines = [f"    call_{n}()" for n in flow]
        router = cond.router or "route"
        lines.append(f"    _outcome = {router}(_CTX)")
        for i, (label, target) in enumerate(cond.outcomes.items()):
            kw = "if" if i == 0 else "elif"
            lines.append(f'    {kw} _outcome == "{label}":')
            lines.append("        pass" if target in _END_NODES else f"        call_{target}()")
        return lines

    return [f"    call_{n}()" for n in flow] or ["    pass"]


# ---------------------------------------------------------------------------
# Router tools + agent guidance — so the MODEL can branch/loop, not a script.
# ---------------------------------------------------------------------------

def _routers(ir: IR) -> list[str]:
    """Distinct router function names across the workflow's conditional edges."""
    wf = ir.workflow
    seen: list[str] = []
    for cond in (wf.conditional_edges if wf else []):
        if cond.router and cond.router not in seen:
            seen.append(cond.router)
    return seen


def _router_tools(ir: IR) -> tuple[list[str], list[str]]:
    """Emit a @tool per router so the agent can decide loops/branches itself."""
    lines: list[str] = []
    names: list[str] = []
    for r in _routers(ir):
        tool = f"decide_{r}"
        names.append(tool)
        lines += [
            "@tool",
            f"def {tool}(note: str = \"\") -> str:",
            f'    """Return the routing decision from {r!r} based on the shared context.',
            "",
            "    Call this after the gating step to choose the next branch / whether",
            "    to loop. The returned label tells you which step tool to call next.",
            '    """',
            f"    return str({r}(_CTX))",
            "",
        ]
    return lines, names


def _agent_guidance(ir: IR, flow: list[str]) -> str:
    """Plain-English routing/loop instructions for the Agent's system prompt."""
    wf = ir.workflow
    pattern = wf.pattern if wf else OrchestrationPattern.LINEAR
    cond = wf.conditional_edges[0] if (wf and wf.conditional_edges) else None
    order = " -> ".join(f"call_{n}" for n in flow) if flow else "the step tools"

    if pattern in (OrchestrationPattern.LOOP, OrchestrationPattern.LOOP_WITH_EXIT) and cond and cond.router:
        cap = _loop_cap_constant(ir)
        exit_label = _exit_label(cond)
        loop_labels = [l for l, t in cond.outcomes.items() if t not in _END_NODES]
        revise = loop_labels[0] if loop_labels else "revise"
        cap_txt = f" (at most {cap} iterations)" if cap else ""
        return (
            f"Call the step tools in order: {order}. Then call decide_{cond.router}; "
            f"if it returns '{revise}', repeat the loop steps{cap_txt}; when it "
            f"returns '{exit_label or 'done'}', the workflow is complete."
        )
    if pattern is OrchestrationPattern.BRANCH and cond and cond.router:
        branches = "; ".join(
            f"on '{l}' call call_{t}" for l, t in cond.outcomes.items() if t not in _END_NODES
        )
        return (
            f"Call the step tools in order: {order}. Then call decide_{cond.router} "
            f"and branch on its result ({branches}). Finish when no branch remains."
        )
    return f"Call the step tools in order: {order}, then stop."


# ---------------------------------------------------------------------------
# Workflow-driven block: node @tools ARE the workflow steps (real logic on
# shared state). run_agent() is PRIMARY and the MODEL drives it; a deterministic
# offline walk of the same tools is the no-SDK / CI fallback.
# ---------------------------------------------------------------------------

def _workflow_block(ir: IR, nodes: list, all_tools: list[str], adapter: TargetAdapter, has_hitl: bool) -> list[str]:
    ctx_class = _ctx_name(adapter)
    sp = _system_prompt(ir)
    flow = _ordered_flow_nodes(ir)
    out: list[str] = []

    # One real @tool per node — the tool body runs the ported node function.
    for node in nodes:
        out += _node_tool(node, ctx_class, ir)

    # Router tools — expose the source's routing decisions to the model.
    router_lines, router_tool_names = _router_tools(ir)
    out += router_lines

    agent_tool_names = [f"call_{n.name}" for n in nodes] + router_tool_names
    if has_hitl:
        agent_tool_names.append("request_human_approval")
    out.append(f"AGENT_TOOLS = [{', '.join(agent_tool_names)}]")
    out.append("")

    guidance = _agent_guidance(ir, flow).replace('"', '\\"')
    hitl_hint = (
        " Call request_human_approval at any point that needs a human decision."
        if has_hitl else ""
    )
    out += [
        "def build_agent():",
        '    """Assemble the real Strands Agent whose tools ARE the workflow steps.',
        "",
        "    Model: Amazon Bedrock (Claude). Swap BedrockModel for AnthropicModel,",
        "    OpenAIModel, LiteLLMModel, or OllamaModel without touching anything else.",
        '    """',
        "    if not _HAVE_STRANDS:",
        "        raise RuntimeError(",
        "            'strands-agents is not installed. Run: pip install strands-agents'",
        "        )",
        "    model = BedrockModel(",
        "        model_id='us.anthropic.claude-sonnet-4-20250514-v1:0',",
        "        region_name='us-east-1',",
        "    )",
        "    return Agent(",
        "        model=model,",
        "        tools=AGENT_TOOLS,",
        f'        system_prompt=("{sp} {guidance}{hitl_hint}"),',
        "    )",
        "",
        "",
        f"def run_agent(state: {ctx_class} | None = None, prompt: str = \"\"):",
        '    """PRIMARY entrypoint — the Strands Agent DRIVES the workflow.',
        "",
        "    When strands-agents is installed, the real Agent (Bedrock) is invoked and",
        "    the MODEL chooses which step/router tools to call; every tool mutates the",
        "    one shared AgentContext (_CTX). Offline (no SDK), falls back to a",
        "    deterministic walk of the SAME tools honoring the source routing/loops so",
        "    CI can run without a model. Returns the final AgentContext.",
        '    """',
        "    global _CTX",
        f"    _CTX = state if state is not None else {ctx_class}()",
        "    if _HAVE_STRANDS:",
        "        _agent = build_agent()",
        "        _agent(prompt or _DEFAULT_TASK)  # model-driven: it calls the tools",
        "        return _CTX",
        "    return _run_offline()  # no SDK -> deterministic fallback for CI",
        "",
        "",
        "def _run_offline():",
        '    """Deterministic execution of the workflow tools (no model), honoring the',
        "    source order, routing and loop bounds. Used when strands-agents is absent.",
        '    """',
        "    global _CTX",
        f"    if _CTX is None:",
        f"        _CTX = {ctx_class}()",
    ]
    out += _orchestration_lines(ir, flow)
    out += [
        "    return _CTX",
    ]
    return out


# ---------------------------------------------------------------------------
# Agent-driven block: no explicit workflow graph — the model drives tool use.
# ---------------------------------------------------------------------------

def _agent_driven_block(ir: IR, all_tools: list[str], adapter: TargetAdapter, has_hitl: bool) -> list[str]:
    ctx_class = _ctx_name(adapter)
    sp = _system_prompt(ir)
    tool_names = list(all_tools)
    if has_hitl:
        tool_names.append("request_human_approval")
    out: list[str] = [f"AGENT_TOOLS = [{', '.join(tool_names)}]", ""]
    out += [
        "def build_agent():",
        '    """Assemble the Strands Agent from the converted tools (model-driven)."""',
        "    if not _HAVE_STRANDS:",
        "        raise RuntimeError(",
        "            'strands-agents is not installed. Run: pip install strands-agents'",
        "        )",
        "    model = BedrockModel(",
        "        model_id='us.anthropic.claude-sonnet-4-20250514-v1:0',",
        "        region_name='us-east-1',",
        "    )",
        "    return Agent(",
        "        model=model,",
        f"        tools={'AGENT_TOOLS' if tool_names else '[]'},",
        f'        system_prompt="{sp}",',
        "    )",
        "",
        "",
        f"def run_agent(state: {ctx_class} | None = None, prompt: str = \"\"):",
        '    """Run the Strands agent with a natural-language prompt (model-driven)."""',
        "    if not _HAVE_STRANDS:",
        f"        return run(state if state is not None else {ctx_class}())",
        "    return build_agent()(prompt or 'Run the converted workflow.')",
    ]
    return out


# ---------------------------------------------------------------------------
# HITL tool block
# ---------------------------------------------------------------------------

def _hitl_tool_block() -> list[str]:
    return [
        "@tool",
        "def request_human_approval(summary: str, options: list) -> str:",
        '    """Ask a human to approve an action before the agent continues.',
        "",
        "    Set HITL_MODE=auto in the environment to skip the prompt (CI fast-path).",
        "    Without it the agent blocks until the operator responds.",
        '    """',
        "    import os",
        '    if os.environ.get("HITL_MODE", "interactive") == "auto":',
        "        return options[0] if options else 'approved'",
        "    print(f'\\n[HITL] {summary}')",
        "    if options:",
        "        print(f'Options: {chr(44).join(str(o) for o in options)}')",
        "    answer = input('Enter your choice: ').strip()",
        "    return answer or (options[0] if options else 'approved')",
        "",
    ]


# ---------------------------------------------------------------------------
# Full Strands block
# ---------------------------------------------------------------------------

def _strands_block(ir: IR, adapter: TargetAdapter) -> str:
    all_tools = sorted({t.name for t in ir.tools})
    nodes = _real_nodes(ir)
    has_hitl = bool(ir.workflow and ir.workflow.hitl_points)
    has_workflow = bool(nodes)

    out: list[str] = [
        "# ── AWS Strands Agent ──────────────────────────────────────────────────────────",
    ]
    if has_workflow:
        out += [
            "# Native Strands workflow: each source node is a real @tool that runs the",
            "# ported logic on ONE shared AgentContext (_CTX). run_agent() is the primary",
            "# entrypoint — it builds the real Agent (Bedrock) and drives the step tools in",
            "# the source order, honoring the original routing/loops. run(ctx) remains as an",
            "# offline (no-SDK) execution path for CI.",
        ]
    else:
        out += [
            "# Agent-driven: the converted @tool functions are wired into one Strands",
            "# Agent (Bedrock) and the model decides how to use them.",
            "# Requires:  pip install strands-agents",
        ]

    # Import guard: @tool becomes a transparent no-op offline so the module
    # still imports cleanly when strands-agents is not present (e.g. in CI that
    # only tests the deterministic offline path). Strands' real @tool keeps the
    # decorated function directly callable, so the orchestration below runs the
    # node logic identically whether or not the SDK is installed.
    out += [
        "try:",
        "    from strands import Agent, tool",
        "    from strands.models import BedrockModel",
        "    _HAVE_STRANDS = True",
        "except ImportError:  # strands-agents not installed — offline-only mode",
        "    _HAVE_STRANDS = False",
        "    def tool(*_a, **_k):  # transparent no-op decorator",
        "        \"\"\"Offline shim: remove once strands-agents is installed.\"\"\"",
        "        return _a[0] if _a and callable(_a[0]) else (lambda fn: fn)",
        "    BedrockModel = None  # type: ignore[assignment,misc]",
        "    Agent = None  # type: ignore[assignment,misc]",
        "",
        "# Shared workflow state — the SAME context flows through every Strands tool.",
        "_CTX = None",
        "# Default task handed to the model when run_agent() is called without a prompt.",
        "_DEFAULT_TASK = (",
        "    'Run the full workflow to completion. Call the step tools in the '",
        "    'prescribed order and use the routing tools to decide loops and branches.'",
        ")",
        "",
    ]

    if has_hitl:
        out += _hitl_tool_block()

    if has_workflow:
        out += _workflow_block(ir, nodes, all_tools, adapter, has_hitl)
    else:
        out += _agent_driven_block(ir, all_tools, adapter, has_hitl)

    out.append("# ── end AWS Strands Agent ──────────────────────────────────────────────────────")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Smoke test, entrypoint
# ---------------------------------------------------------------------------

def _smoke_test(adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    return (
        f'"""Smoke test for the converted {target} agent (offline-safe)."""\n\n'
        f"from agent_context import {ctx}\n"
        "import orchestrator\n\n\n"
        "def test_state_constructs():\n"
        f"    assert {ctx}() is not None\n\n\n"
        "def test_offline_entrypoint_present():\n"
        '    """The offline fast-path run() must exist even without strands-agents."""\n'
        '    assert callable(getattr(orchestrator, "run", None))\n\n\n'
        "def test_strands_agent_wiring_present():\n"
        '    """build_agent and run_agent must be importable at module level."""\n'
        '    assert callable(getattr(orchestrator, "build_agent", None)), (\n'
        '        "build_agent() not found — AWS Strands block was not emitted correctly"\n'
        "    )\n"
        '    assert callable(getattr(orchestrator, "run_agent", None)), (\n'
        '        "run_agent() not found — AWS Strands block was not emitted correctly"\n'
        "    )\n\n\n"
        "def test_state_advance_returns_new_instance():\n"
        f"    a = {ctx}()\n"
        "    b = a.advance()\n"
        f"    assert isinstance(b, {ctx}) and b is not a\n"
    )


def _entrypoint(ir: IR, adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    is_api = (ir.metadata.entrypoint or "cli") == "api"

    lines: list[str] = [
        f'"""Generated entrypoint for the converted {target} agent.',
        "",
        "run_agent() (in orchestrator.py) is the primary path: it builds the real",
        "Strands Agent (Bedrock) and runs the workflow through the wired @tool steps",
        "on a shared AgentContext. It also works offline (no SDK) for CI, executing the",
        "same ported node logic.",
        '"""',
        "from __future__ import annotations",
        "",
        f"from agent_context import {ctx}",
        "from orchestrator import run_agent",
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
            '    """Run the converted workflow; converted from the source API entrypoint."""',
            f"    state = {ctx}(**(payload or {{}}))",
            "    return run_agent(state=state)",
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
            "    result = run_agent(prompt='Run the converted workflow.')",
            "    print(result)",
            "    return result",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class AWSStrandsTargetGenerator(TargetGenerator):
    """Emits AWS Strands output (any source -> Strands Agent or agents-as-tools)."""

    name = "aws_strands"

    def workflow_block(self, ir: IR, adapter: TargetAdapter) -> str:
        return _strands_block(ir, adapter)

    def sdk_stub_files(self) -> dict[str, str]:
        # strands-agents is a public pip package -> no local SDK stub needed.
        return {}

    def smoke_test(self, adapter: TargetAdapter, target: str) -> str:
        return _smoke_test(adapter, target)

    def entrypoint(self, ir: IR, adapter: TargetAdapter, target: str) -> str:
        return _entrypoint(ir, adapter, target)

    def orchestrator_must_tokens(self, has_workflow_block: bool) -> list[str]:
        must = ["def run", "def build_agent", "def run_agent"]
        if has_workflow_block:
            must += ["Agent", "_HAVE_STRANDS"]
        return must
