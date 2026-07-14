"""Tests for Module docs -- INSTALL.md and ARCHITECTURE.md generators."""

from __future__ import annotations

import os

from converter.config import Config, ConversionMode
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
from converter.generator import build_architecture_md, build_install_md, write_docs


def _ir() -> IR:
    return IR(
        metadata=IRMetadata(source_framework="langgraph", target_framework="maf"),
        tools=[ToolSpec(name="read_tests", source_file="agent.py")],
        state=[
            StateField("coverage", "float"),
            StateField("audit_log", "list", is_append_only=True),
        ],
        config=ConfigSpec(constants={"MAX": 3}),
        imports=["import os", "from langchain_openai import ChatOpenAI"],
        workflow=WorkflowSpec(
            pattern=OrchestrationPattern.LOOP_WITH_EXIT,
            nodes=[
                GraphNode("generate", role=NodeRole.ENTRY),
                GraphNode("hitl_approve", role=NodeRole.HITL),
            ],
            entry_point="generate",
        ),
    )


def test_install_md_lists_deps_and_run_steps():
    md = build_install_md(_ir())
    assert "pip install semantic-kernel" in md
    assert "langchain-openai" in md          # mapped from langchain_openai import
    assert "from orchestrator import run" in md
    assert "AUTO-APPROVE" not in md          # install doc explains, doesn't dump code
    assert "hitl_approve" in md              # HITL section names the node


def test_architecture_md_has_layout_and_tiers():
    ir = _ir()
    md = build_architecture_md(ir, convert(ir))
    assert "# Architecture" in md
    assert "agent_context.py" in md
    assert "orchestrator.py" in md
    assert "Orchestration pattern:** loop_with_exit" in md
    assert "Tier 1" in md and "Tier 3" in md


def test_architecture_reflects_mode():
    ir = _ir()
    manual = build_architecture_md(ir, convert(ir), Config(mode=ConversionMode.DETERMINISTIC))
    llm = build_architecture_md(ir, convert(ir), Config(mode=ConversionMode.HYBRID))
    assert "manual (you implement)" in manual
    assert "LLM-assisted (you review)" in llm


def test_write_docs_creates_both_files(tmp_path):
    ir = _ir()
    written = write_docs(ir, convert(ir), str(tmp_path))
    assert set(written) == {"INSTALL.md", "ARCHITECTURE.md"}
    assert os.path.exists(os.path.join(str(tmp_path), "INSTALL.md"))
    assert os.path.exists(os.path.join(str(tmp_path), "ARCHITECTURE.md"))
