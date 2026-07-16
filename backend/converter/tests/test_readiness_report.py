"""Tests for the agent-specific READINESS report generator."""

from __future__ import annotations

from converter.config import Config, ConversionMode
from converter.engine import convert
from converter.generator import generate, generate_readiness_report
from converter.generator.readiness_report import collect_facts
from converter.tests.test_code_generator import _ported_ir
from converter.contracts import GraphNode, NodeRole, ToolSpec


def _ir_with_hitl_and_orphan():
    ir = _ported_ir()
    ir.workflow.nodes.append(GraphNode("hitl_approve", role=NodeRole.HITL))
    ir.tools.append(ToolSpec(name="unused_tool"))  # orphan (no node calls it)
    return ir


def test_collect_facts_is_agent_specific(tmp_path):
    ir = _ir_with_hitl_and_orphan()
    gen = generate(ir, convert(ir), str(tmp_path))
    facts = collect_facts(ir, convert(ir), gen, config=Config(), agent_name="MyAgent")
    assert facts["agent_name"] == "MyAgent"
    assert "hitl_approve" in facts["hitl_nodes"]
    assert "unused_tool" in facts["orphan_tools"]
    assert facts["target_framework"] == "maf"
    assert facts["state_field_count"] == len(ir.state)


def test_fallback_report_has_sections_and_agent_details(tmp_path):
    ir = _ir_with_hitl_and_orphan()
    gen = generate(ir, convert(ir), str(tmp_path))
    # Deterministic mode -> no LLM -> fallback report.
    md = generate_readiness_report(
        ir, convert(ir), gen, Config(mode=ConversionMode.DETERMINISTIC), agent_name="MyAgent"
    )
    assert "# Readiness Report - MyAgent" in md
    assert "## Remaining work" in md
    assert "## Accuracy by dimension" in md
    assert "## Key insight" in md
    # Agent-specific rows.
    assert "hitl_approve" in md
    assert "unused_tool" in md
    # Owner + time columns present.
    assert "Human via Claude Code" in md
    assert "agent_framework/ is a local stub" in md


def test_llm_path_used_when_client_available(tmp_path):
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))

    class _FakeResp:
        text = "# Readiness Report - LLM\n\n## Remaining work\n\n(llm authored)\n"

    class _FakeClient:
        def generate_content(self, prompt):
            assert "READINESS report" in prompt  # agent facts were sent
            return _FakeResp()

    md = generate_readiness_report(
        ir, convert(ir), gen, Config(mode=ConversionMode.HYBRID),
        agent_name="LLM", client=_FakeClient(),
    )
    assert "(llm authored)" in md


def test_llm_fence_stripping():
    from converter.generator.readiness_report import _strip_fences

    fenced = "```markdown\n# R\n\ncontent\n```"
    assert _strip_fences(fenced).startswith("# R")
    assert "```" not in _strip_fences(fenced)
