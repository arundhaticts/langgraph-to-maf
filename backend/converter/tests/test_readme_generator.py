"""Tests for Module 8 -- README generator."""

from __future__ import annotations

from converter.contracts import (
    IR,
    ConfigSpec,
    GraphNode,
    IRMetadata,
    NodeRole,
    OrchestrationPattern,
    StateField,
    ToolSpec,
    WorkflowSpec,
)
from converter.engine import convert
from converter.generator import build_readme


def _ir(**kw) -> IR:
    base = dict(
        metadata=IRMetadata(description="Optimises a test suite.", target_framework="maf"),
        tools=[ToolSpec(name="read_tests", docstring="Reads tests.")],
        state=[
            StateField("project_id", "str"),
            StateField("audit_log", "list", is_append_only=True),
        ],
        config=ConfigSpec(constants={"MAX_GEN_RETRIES": 3}, temperature=0.2),
        workflow=WorkflowSpec(
            pattern=OrchestrationPattern.LINEAR,
            nodes=[GraphNode("read", role=NodeRole.ENTRY)],
            entry_point="read",
            readme_description="Read the tests, then analyse.",
        ),
    )
    base.update(kw)
    return IR(**base)


def test_uses_maf_vocabulary():
    md = build_readme(_ir())
    assert "## Tools" in md            # true-MAF vocabulary
    assert "## Context" in md          # State -> Context


def test_lists_skills_as_plugins():
    md = build_readme(_ir())
    assert "`ReadTestsTool.read_tests`: Reads tests." in md


def test_context_fields_and_append_only():
    md = build_readme(_ir())
    assert "`project_id` (str)" in md
    assert "`audit_log` (list) — append-only" in md


def test_purpose_and_config_rendered():
    md = build_readme(_ir())
    assert "Optimises a test suite." in md
    assert "`MAX_GEN_RETRIES`" in md
    assert "temperature: `0.2`" in md


def test_hitl_stub_described():
    ir = _ir(
        workflow=WorkflowSpec(
            pattern=OrchestrationPattern.LINEAR,
            nodes=[
                GraphNode("read", role=NodeRole.ENTRY),
                GraphNode("hitl_approve", role=NodeRole.HITL),
            ],
            entry_point="read",
        )
    )
    md = build_readme(ir, convert(ir))
    assert "Human-in-the-loop" in md
    assert "HumanApprovalRequired" in md
    assert "hitl_approve" in md


def test_pattern_shown():
    md = build_readme(_ir())
    assert "linear" in md


def test_no_tools_shows_na():
    ir = _ir(tools=[])
    md = build_readme(ir)
    # Tools section present but empty -> N/A
    assert "## Tools\nN/A" in md
