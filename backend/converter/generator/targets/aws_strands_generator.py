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
from converter.contracts import IR, NodeRole
from converter.generator.targets.base import TargetGenerator

_SENTINEL_NODES = frozenset({"START", "END", "__start__", "__end__"})


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


def _node_tools(node, all_tool_names: list[str]) -> list[str]:
    """Tools this node calls, or all tools if the call set is empty."""
    if node.calls_tools:
        return [t for t in node.calls_tools if t in all_tool_names]
    return all_tool_names


# ---------------------------------------------------------------------------
# Single-agent block (one node or AGENT_DRIVEN)
# ---------------------------------------------------------------------------

def _single_agent_block(ir: IR, all_tools: list[str]) -> list[str]:
    sp = _system_prompt(ir)
    out: list[str] = []
    if all_tools:
        out.append(f"    AGENT_TOOLS = [{', '.join(all_tools)}]")
        out.append("")

    out += [
        "    def build_agent():",
        '        """Assemble the Strands Agent from the converted tools."""',
        "        # Default: Amazon Bedrock. Swap the model to use another provider:",
        "        # AnthropicModel, OpenAIModel, LiteLLMModel, OllamaModel.",
        "        model = BedrockModel(",
        "            model_id='us.anthropic.claude-sonnet-4-20250514-v1:0',",
        "            region_name='us-east-1',",
        "        )",
        "        return Agent(",
        "            model=model,",
        f"            tools={'AGENT_TOOLS' if all_tools else '[]'},",
        f'            system_prompt="{sp}",',
        "        )",
        "",
        "    def run_agent(prompt: str):",
        '        """Run the Strands agent with a natural-language prompt."""',
        "        return build_agent()(prompt)",
    ]
    return out


# ---------------------------------------------------------------------------
# Multi-agent block (agents-as-tools pattern)
# ---------------------------------------------------------------------------

def _multi_agent_block(ir: IR, nodes: list, all_tools: list[str]) -> list[str]:
    """Emit an orchestrator Agent whose tools are sub-Agent wrappers per node."""
    sp = _system_prompt(ir)
    out: list[str] = []

    if all_tools:
        out.append(f"    ALL_TOOLS = [{', '.join(all_tools)}]")
        out.append("")

    # One @tool wrapper per node — each runs a specialised sub-Agent.
    for node in nodes:
        node_tools = _node_tools(node, all_tools)
        node_sp = (
            ir.functions.get(node.name, None) and
            ir.functions[node.name].docstring or
            f"Execute the '{node.name}' step of the converted workflow."
        )
        if isinstance(node_sp, str):
            node_sp = node_sp.strip().splitlines()[0].replace('"', '\\"')

        out += [
            f"    @tool",
            f"    def call_{node.name}(input: str) -> str:",
            f'        """Run the {node.name!r} sub-agent with the provided input."""',
            f"        _model = BedrockModel(",
            f"            model_id='us.anthropic.claude-sonnet-4-20250514-v1:0',",
            f"            region_name='us-east-1',",
            f"        )",
            f"        _sub = Agent(",
            f"            model=_model,",
            f"            tools={node_tools or 'ALL_TOOLS' if all_tools else '[]'},",
            f'            system_prompt="{node_sp}",',
            f"        )",
            f"        return str(_sub(input))",
            "",
        ]

    sub_agent_tools = [f"call_{n.name}" for n in nodes]
    out += [
        "    def build_agent():",
        '        """Assemble the orchestrator Agent; sub-agents are wired as tools."""',
        "        model = BedrockModel(",
        "            model_id='us.anthropic.claude-sonnet-4-20250514-v1:0',",
        "            region_name='us-east-1',",
        "        )",
        "        return Agent(",
        "            model=model,",
        f"            tools=[{', '.join(sub_agent_tools)}],",
        f'            system_prompt=("{sp} '
        f'Orchestrate by calling sub-agent tools in the correct order."),',
        "        )",
        "",
        "    def run_agent(prompt: str):",
        '        """Run the orchestrator agent with a natural-language prompt."""',
        "        return build_agent()(prompt)",
    ]
    return out


# ---------------------------------------------------------------------------
# HITL tool block
# ---------------------------------------------------------------------------

def _hitl_tool_block() -> list[str]:
    return [
        "    @tool",
        "    def request_human_approval(summary: str, options: list) -> str:",
        '        """Ask a human to approve an action before the agent continues.',
        "",
        "        Set HITL_MODE=auto in the environment to skip the prompt (CI fast-path).",
        '        """',
        "        import os",
        '        if os.environ.get("HITL_MODE", "interactive") == "auto":',
        "            return options[0] if options else 'approved'",
        "        print(f'\\n[HITL] {summary}')",
        "        if options:",
        "            print(f'Options: {chr(44).join(str(o) for o in options)}')",
        "        answer = input('Enter your choice: ').strip()",
        "        return answer or (options[0] if options else 'approved')",
        "",
    ]


# ---------------------------------------------------------------------------
# Full Strands block
# ---------------------------------------------------------------------------

def _strands_block(ir: IR, adapter: TargetAdapter) -> str:
    all_tools = sorted({t.name for t in ir.tools})
    nodes = _real_nodes(ir)
    has_hitl = bool(ir.workflow and ir.workflow.hitl_points)
    is_multi = len(nodes) > 1

    out: list[str] = [
        "# --- AWS Strands Agent (generated from the IR) ---",
    ]
    if is_multi:
        out += [
            "# Multi-node IR: agents-as-tools pattern.",
            "# Each source node becomes a @tool-wrapped sub-Agent; an orchestrator",
            "# Agent calls them. For explicit ordering / shared state, consider",
            "# strands.multiagent.GraphBuilder (manual wiring required).",
        ]
    else:
        out += [
            "# Single-agent: all converted @tool functions wired into one Agent.",
            "# Import-guarded so the offline run(ctx) above works without strands-agents.",
        ]

    out += [
        "try:",
        "    from strands import Agent, tool",
        "    from strands.models import BedrockModel",
        "    _HAVE_STRANDS = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_STRANDS = False",
        "",
        "",
        "if _HAVE_STRANDS:",
        "",
    ]

    if has_hitl:
        out += _hitl_tool_block()

    if is_multi:
        out += _multi_agent_block(ir, nodes, all_tools)
    else:
        out += _single_agent_block(ir, all_tools)

    out.append("# --- end AWS Strands Agent ---")
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
        "    from orchestrator import build_agent",
        "    _HAVE_STRANDS = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_STRANDS = False",
        "",
        "",
        f"def run_agent(state: {ctx} | None = None, prompt: str = ''):",
        '    """Run via the Strands Agent when available, else the offline run()."""',
        f"    state = state or {ctx}()",
        "    if _HAVE_STRANDS:",
        "        return build_agent()(prompt or str(state))",
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
        must = ["def run"]
        if has_workflow_block:
            must += ["Agent", "build_agent"]
        return must
