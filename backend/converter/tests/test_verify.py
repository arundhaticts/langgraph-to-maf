"""Tests for Phase 11 -- the acceptance gate (converter.verify)."""

from __future__ import annotations

import os

from converter.contracts import ToolSpec
from converter.engine import convert
from converter.generator import generate
from converter.generator.code_generator import _rewrite_path_anchors
from converter.verify import render_acceptance, verify_output
from converter.tests.test_code_generator import _ported_ir


def test_clean_output_passes_all_checks(tmp_path):
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    report = verify_output(ir, gen)
    assert report.passed, report.issues()
    names = {n for n, _ok, _d in report.checks}
    assert {
        "all_python_compiles",
        "no_source_framework_residue",
        "requirements_clean",
        "all_nodes_generated",
        "all_tools_generated",
        "target_shape_present",
    } <= names


def test_detects_missing_tool(tmp_path):
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    # Add a tool to the IR that was never generated -> coverage fails.
    ir.tools.append(ToolSpec(name="ghost_tool"))
    report = verify_output(ir, gen)
    assert not report.passed
    assert any("ghost_tool" in i for i in report.issues())


def test_detects_residue_and_dirty_requirements(tmp_path):
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    # Inject source-framework residue + a banned requirement.
    with open(os.path.join(gen.output_root, "orchestrator.py"), "a", encoding="utf-8") as fh:
        fh.write("\n# from langgraph.graph import StateGraph\n")
    with open(os.path.join(gen.output_root, "requirements.txt"), "a", encoding="utf-8") as fh:
        fh.write("langgraph\n")
    report = verify_output(ir, gen)
    assert not report.passed
    issues = " ".join(report.issues())
    assert "from langgraph" in issues
    assert "langgraph" in issues  # requirements_clean too


def test_render_acceptance_markdown():
    ir = _ported_ir()
    # Minimal: render should produce a header + one line per check.
    from converter.verify import AcceptanceReport

    r = AcceptanceReport()
    r.add("all_python_compiles", True)
    r.add("all_tools_generated", False, "tools with no plugin function: ['x']")
    md = render_acceptance(r)
    assert "# Acceptance Report - FAILED" in md
    assert "- [PASS] all_python_compiles" in md
    assert "- [FAIL] all_tools_generated -- tools with no plugin function: ['x']" in md


def test_is_secret_file():
    from converter.generator.code_generator import _is_secret_file

    assert _is_secret_file(".env")
    assert _is_secret_file("src/.env.production")
    assert _is_secret_file("keys/server.pem")
    assert _is_secret_file("deploy/id_rsa")
    assert _is_secret_file("config/credentials.json")
    assert not _is_secret_file("prompts/system.md")
    assert not _is_secret_file("config.py")


def test_orphan_tool_still_registered(tmp_path):
    # A tool no node calls must still be registered in AGENT_TOOLS (Phase 6).
    ir = _ported_ir()  # its 'read_tests' tool is not called by any node
    generate(ir, convert(ir), str(tmp_path))
    orch = open(os.path.join(str(tmp_path), "orchestrator.py"), encoding="utf-8").read()
    assert "AGENT_TOOLS = [read_tests]" in orch


def test_agent_framework_stub_generated_and_runnable(tmp_path):
    import ast

    from converter.verify import verify_runnable

    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    stub = os.path.join(gen.output_root, "agent_framework", "__init__.py")
    assert os.path.exists(stub)
    # Stub parses and exports the full MAF interface the converted code imports.
    text = open(stub, encoding="utf-8").read()
    ast.parse(text)
    for sym in (
        "class WorkflowBuilder", "class WorkflowContext", "def executor",
        "class RequestInfoExecutor", "class RequestInfoMessage",
        "class WorkflowOutputEvent", "class RequestInfoEvent",
        "class FileCheckpointStorage", "class ChatAgent", "def ai_function",
    ):
        assert sym in text, sym
    # The two acceptance subprocess checks pass against the generated output.
    results = {n: (ok, d) for n, ok, d in verify_runnable(gen.output_root)}
    assert results["stub_imports"][0], results["stub_imports"][1]
    assert results["graph_builds"][0], results["graph_builds"][1]


def test_path_anchor_rewrite():
    # Source-relative literals get the top package dir stripped.
    assert _rewrite_path_anchors('open("src/prompts/system.md")', "src") == 'open("prompts/system.md")'
    assert _rewrite_path_anchors("p = 'src/data/x.json'", "src") == "p = 'data/x.json'"
    # Non-source strings are untouched; None root is a no-op.
    assert _rewrite_path_anchors('open("prompts/x.md")', "src") == 'open("prompts/x.md")'
    assert _rewrite_path_anchors('open("srcish/x")', "src") == 'open("srcish/x")'
    assert _rewrite_path_anchors('open("src/x")', None) == 'open("src/x")'
