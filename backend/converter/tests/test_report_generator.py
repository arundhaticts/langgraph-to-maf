"""Tests for Module 9 -- migration report generator."""

from __future__ import annotations

import os

from converter.config import Config, ConversionMode
from converter.contracts import (
    IR,
    ConfigSpec,
    ConversionResult,
    ConversionUnit,
    GraphNode,
    IRMetadata,
    NodeRole,
    OrchestrationPattern,
    StateField,
    Tier,
    Tier3Result,
    ToolSpec,
    WorkflowSpec,
)
from converter.engine import convert
from converter.generator import build_report, render_report, write_report
from converter.generator.code_generator import GenerationResult


def _full_ir() -> IR:
    return IR(
        metadata=IRMetadata(description="Test Optimiser", target_framework="maf"),
        tools=[ToolSpec(name="read_tests")],
        state=[StateField("audit_log", "list", is_append_only=True)],
        config=ConfigSpec(constants={"MAX_GEN_RETRIES": 3}),
        workflow=WorkflowSpec(
            pattern=OrchestrationPattern.LINEAR,
            nodes=[
                GraphNode("read", role=NodeRole.ENTRY),
                GraphNode("hitl_approve", role=NodeRole.HITL),
            ],
            entry_point="read",
        ),
    )


def test_auto_converted_section():
    ir = _full_ir()
    report = build_report(ir, convert(ir), generated_date="2026-07-10")
    texts = [e.text for e in report.auto_converted]
    assert any("[R-01] read_tests -> ReadTestsTool" in t for t in texts)
    assert any("[R-11] audit_log" in t for t in texts)


def test_hitl_goes_to_manual_action():
    ir = _full_ir()
    report = build_report(ir, convert(ir))
    manual = [e.text for e in report.manual_action_required]
    assert any("[R-08] hitl_approve" in t for t in manual)
    # And its detail carries the wiring instruction.
    r08 = next(e for e in report.manual_action_required if "R-08" in e.text)
    assert "approval flow" in r08.detail


def test_tier3_below_threshold_goes_to_needs_review():
    wf = WorkflowSpec(pattern=OrchestrationPattern.AGENT_DRIVEN, readme_description="")
    ir = IR(metadata=IRMetadata(target_framework="maf"), workflow=wf)

    def low(ir_, workflow, cfg):
        return Tier3Result("agent_driven", "code", "3 outcomes not 2", 0.61)

    report = build_report(ir, convert(ir, tier3_resolver=low))
    assert len(report.needs_review) == 1
    entry = report.needs_review[0]
    assert "confidence: 0.61" in entry.text
    assert "3 outcomes" in entry.detail


def test_unresolved_workflow_goes_to_manual():
    wf = WorkflowSpec(pattern=OrchestrationPattern.AGENT_DRIVEN, readme_description="")
    ir = IR(metadata=IRMetadata(target_framework="maf"), workflow=wf)
    config = Config(mode=ConversionMode.DETERMINISTIC)  # no LLM
    report = build_report(ir, convert(ir, config=config))
    assert any("workflow" in e.text for e in report.manual_action_required)


def test_syntax_errors_go_to_manual():
    ir = _full_ir()
    gen = GenerationResult(
        output_root="/tmp/x", syntax_errors=["orchestrator.py"]
    )
    report = build_report(ir, convert(ir), gen)
    assert any("[SYNTAX] orchestrator.py" in e.text for e in report.manual_action_required)


def test_validation_warnings_go_to_needs_review():
    ir = _full_ir()
    gen = GenerationResult(
        output_root="/tmp/x", validation_warnings=["tools.py: missing kernel_function"]
    )
    report = build_report(ir, convert(ir), gen)
    assert any("[VALIDATION]" in e.text for e in report.needs_review)


def test_render_report_has_all_sections():
    ir = _full_ir()
    md = render_report(build_report(ir, convert(ir), generated_date="2026-07-10"))
    assert "# Migration Report - Test Optimiser" in md
    assert "Generated: 2026-07-10" in md
    assert "## Auto-converted" in md
    assert "## Needs review" in md
    assert "## Manual action required" in md


def test_empty_sections_show_placeholder():
    ir = IR(metadata=IRMetadata(target_framework="maf"),
            workflow=WorkflowSpec(pattern=OrchestrationPattern.LINEAR, entry_point="a"))
    md = render_report(build_report(ir, convert(ir)))
    assert "_Nothing flagged for review._" in md
    assert "_No manual action required._" in md


def test_write_report_creates_file(tmp_path):
    ir = _full_ir()
    path = write_report(build_report(ir, convert(ir)), str(tmp_path))
    assert os.path.basename(path) == "MIGRATION_REPORT.md"
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        assert "Migration Report" in fh.read()
