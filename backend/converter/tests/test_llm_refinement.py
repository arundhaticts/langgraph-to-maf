"""Tests for the Stage 11 LLM Refinement Pass."""

from __future__ import annotations

import io
import json
import os
import zipfile

import pytest

from converter.engine import convert
from converter.generator import generate
from converter.generator.llm_refinement import (
    RefinementResult,
    _build_prompt,
    _extract_remaining_tasks,
    _parse_response,
    _validate_python,
    run_llm_refinement,
    write_refinement_log,
)
from converter.tests.test_code_generator import _ported_ir
from service import convert_folder

# ---------------------------------------------------------------------------
# Unit tests for individual helpers
# ---------------------------------------------------------------------------

SAMPLE_READINESS = """\
# Readiness Report - TestAgent

## Remaining work

| Item | Owner | Action | Time |
|---|---|---|---|
| HITL node 'approve' defaults to auto-approve | Human via Claude Code | Implement the decision-merge | 1-2 hrs |
| Tool 'fetch' is defined but not called by any node | Human via Claude Code | Wire the tool | 15 min |
| agent_framework/ is a local stub | Human via Claude Code | pip install the real SDK | 30 min |

## Accuracy by dimension

| Dimension | Accuracy | Why |
|---|---|---|
| Deterministic logic | ~85% | Ported verbatim |

## Key insight

HITL needs real wiring.
"""


def test_extract_remaining_tasks_finds_all_rows():
    tasks = _extract_remaining_tasks(SAMPLE_READINESS)
    assert len(tasks) == 3
    assert any("approve" in t for t in tasks)
    assert any("fetch" in t for t in tasks)
    assert any("stub" in t for t in tasks)


def test_extract_remaining_tasks_no_outstanding():
    md = (
        "# Readiness Report\n\n## Remaining work\n\n"
        "| Item | Owner | Action | Time |\n|---|---|---|---|\n"
        "| No outstanding items detected | - | Ship it | - |\n"
    )
    tasks = _extract_remaining_tasks(md)
    # The task is parsed but the runner will detect "no outstanding".
    assert len(tasks) == 1
    assert "no outstanding" in tasks[0].lower()


def test_validate_python_accepts_valid():
    assert _validate_python("x = 1\ndef foo(): pass\n") is True


def test_validate_python_rejects_invalid():
    assert _validate_python("def (:\n  pass") is False


def test_parse_response_valid_json():
    payload = {
        "changes": [
            {"file": "orchestrator.py", "content": "x = 1\n", "summary": "Fixed stub"}
        ],
        "overall_summary": "Done.",
    }
    changes, summary = _parse_response(json.dumps(payload))
    assert len(changes) == 1
    assert changes[0]["file"] == "orchestrator.py"
    assert summary == "Done."


def test_parse_response_fenced_json():
    payload = {"changes": [], "overall_summary": "Nothing to do."}
    text = f"```json\n{json.dumps(payload)}\n```"
    changes, summary = _parse_response(text)
    assert changes == []
    assert summary == "Nothing to do."


def test_parse_response_bad_json():
    changes, summary = _parse_response("not json at all")
    assert changes == []
    assert "parsed" in summary.lower()


def test_build_prompt_contains_key_sections():
    prompt = _build_prompt(
        output_files={"orchestrator.py": "def run(): pass\n"},
        readiness_md=SAMPLE_READINESS,
        remaining_tasks=["HITL stub needs wiring"],
        framework_docs="# Docs\nUse WorkflowBuilder.",
    )
    assert "READINESS_REPORT.md" in prompt
    assert "HITL stub needs wiring" in prompt
    assert "orchestrator.py" in prompt
    assert "Target framework knowledge pack" in prompt


# ---------------------------------------------------------------------------
# Integration: no-op when LLM key absent
# ---------------------------------------------------------------------------

class _MockConfig:
    """Minimal Config stand-in for refinement tests."""
    allow_llm_fallback = True
    frameworks_dir = "frameworks"

    def llm_api_key(self):
        return ""  # no key -> refinement skipped

    def resolved_model(self):
        return "gemini-2.0-flash"


class _FakeIR:
    class _meta:
        target_framework = "maf"
        source_framework = "langgraph"
    metadata = _meta()


class _FakeConversion:
    units = []


class _FakeGeneration:
    output_root = ""
    validation_warnings = []
    syntax_errors = []


# ---------------------------------------------------------------------------
# Integration: the gate-closed repair loop (with a mock LLM client)
# ---------------------------------------------------------------------------

class _KeyedConfig:
    """Config stand-in with LLM enabled (a mock client bypasses the real key)."""
    allow_llm_fallback = True
    frameworks_dir = "frameworks"

    def llm_api_key(self):
        return "test-key"

    def resolved_model(self):
        return "gemini-2.0-flash"


class _Resp:
    def __init__(self, text):
        self.text = text


class _FnClient:
    """Mock Gemini client whose response is computed per-call by `fn(call_no, prompt)`."""
    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def generate_content(self, prompt):
        self.calls += 1
        return _Resp(self._fn(self.calls, prompt))


def _find_plugin_with(gen, needle: str):
    """Return (rel_path, content) of the first plugins/* file containing `needle`."""
    for rel in gen.written_files:
        norm = rel.replace("\\", "/")
        if not norm.startswith("plugins/") or not norm.endswith(".py"):
            continue
        content = open(os.path.join(gen.output_root, rel), encoding="utf-8").read()
        if needle in content:
            return rel, content
    raise AssertionError(f"no plugin file contains {needle!r}")


def test_refinement_loop_converges_after_repair(tmp_path):
    """Round 1 leaves the gate RED (removes a tool fn); round 2 restores it -> GREEN.

    Exercises the full closed loop against the REAL acceptance gate: apply patch,
    re-verify, feed the failure back, patch again, re-verify green, stop.
    """
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    tool = ir.tools[0].name  # 'read_tests'
    rel, original = _find_plugin_with(gen, f"def {tool}(")
    broken = original.replace(f"def {tool}(", f"def _disabled_{tool}(", 1)

    # Outstanding tasks so the loop engages even though the initial gate is green.
    (tmp_path / "READINESS_REPORT.md").write_text(SAMPLE_READINESS, encoding="utf-8")

    def _respond(call_no, _prompt):
        if call_no == 1:
            # Regress: drop the tool function -> all_tools_generated goes RED.
            return json.dumps({
                "changes": [{"file": rel.replace("\\", "/"), "content": broken,
                             "summary": "round1: (introduces a coverage gap)"}],
                "overall_summary": "round 1",
            })
        # Repair: restore the tool function -> gate GREEN.
        return json.dumps({
            "changes": [{"file": rel.replace("\\", "/"), "content": original,
                         "summary": "round2: restored the tool function"}],
            "overall_summary": "round 2 fixed it",
        })

    client = _FnClient(_respond)
    result = run_llm_refinement(
        ir=ir, conversion=convert(ir), generation=gen, config=_KeyedConfig(),
        output_root=str(tmp_path), agent_name="LoopAgent", client=client,
    )

    assert result.ran is True
    assert result.iterations == 2, f"expected 2 rounds, got {result.iterations}"
    assert result.gate_passed is True
    assert result.gate_issues_remaining == []
    assert client.calls == 2
    # The final on-disk content is the restored (green) version.
    final = open(os.path.join(gen.output_root, rel), encoding="utf-8").read()
    assert f"def {tool}(" in final


def test_refinement_loop_single_pass_stays_green(tmp_path):
    """One valid patch that keeps the gate green -> converges in a single round."""
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    tool = ir.tools[0].name
    rel, original = _find_plugin_with(gen, f"def {tool}(")
    patched = original + "\n# refined by the LLM pass\n"

    (tmp_path / "READINESS_REPORT.md").write_text(SAMPLE_READINESS, encoding="utf-8")

    def _respond(_call_no, _prompt):
        return json.dumps({
            "changes": [{"file": rel.replace("\\", "/"), "content": patched,
                         "summary": "added a clarifying comment"}],
            "overall_summary": "single pass",
        })

    client = _FnClient(_respond)
    result = run_llm_refinement(
        ir=ir, conversion=convert(ir), generation=gen, config=_KeyedConfig(),
        output_root=str(tmp_path), agent_name="OnePass", client=client,
    )

    assert result.ran is True
    assert result.gate_passed is True
    assert result.iterations == 1
    assert len(result.patches) == 1
    assert client.calls == 1


def test_refinement_skips_when_green_and_no_tasks(tmp_path):
    """Gate green + no outstanding tasks -> early return, LLM never called."""
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    md = (
        "# Readiness Report\n\n## Remaining work\n\n"
        "| Item | Owner | Action | Time |\n|---|---|---|---|\n"
        "| No outstanding items detected | - | Ship it | - |\n"
    )
    (tmp_path / "READINESS_REPORT.md").write_text(md, encoding="utf-8")

    def _boom(_call_no, _prompt):
        raise AssertionError("LLM should not be called when the gate is green and no tasks remain")

    client = _FnClient(_boom)
    result = run_llm_refinement(
        ir=ir, conversion=convert(ir), generation=gen, config=_KeyedConfig(),
        output_root=str(tmp_path), agent_name="NoOp", client=client,
    )

    assert result.ran is True
    assert result.gate_passed is True
    assert result.patches == []
    assert client.calls == 0


def test_refinement_records_manual_action_when_gate_stays_red(tmp_path):
    """If the LLM cannot make the gate green within the cap, remaining issues are
    recorded as manual action (never silently 'green')."""
    ir = _ported_ir()
    gen = generate(ir, convert(ir), str(tmp_path))
    tool = ir.tools[0].name
    rel, original = _find_plugin_with(gen, f"def {tool}(")
    broken = original.replace(f"def {tool}(", f"def _disabled_{tool}(", 1)

    (tmp_path / "READINESS_REPORT.md").write_text(SAMPLE_READINESS, encoding="utf-8")

    def _always_break(_call_no, _prompt):
        # Every round drops the tool fn -> gate never goes green.
        return json.dumps({
            "changes": [{"file": rel.replace("\\", "/"), "content": broken,
                         "summary": "keeps the coverage gap"}],
            "overall_summary": "cannot fix",
        })

    client = _FnClient(_always_break)
    result = run_llm_refinement(
        ir=ir, conversion=convert(ir), generation=gen, config=_KeyedConfig(),
        output_root=str(tmp_path), agent_name="StuckAgent", client=client,
        max_iterations=2,
    )

    assert result.ran is True
    assert result.gate_passed is False
    # First round applies the break; second round's identical content is a no-op
    # (unchanged), so the loop stops making progress -> issues surfaced as manual.
    assert result.gate_issues_remaining, "unresolved gate issues must be recorded"
    log = write_refinement_log(result, str(tmp_path), agent_name="StuckAgent")
    text = open(log, encoding="utf-8").read()
    assert "Manual action required" in text


def test_run_llm_refinement_skips_without_key(tmp_path):
    gen = _FakeGeneration()
    gen.output_root = str(tmp_path)
    # Write a dummy readiness report so the reader doesn't error.
    (tmp_path / "READINESS_REPORT.md").write_text(SAMPLE_READINESS, encoding="utf-8")
    result = run_llm_refinement(
        ir=_FakeIR(),
        conversion=_FakeConversion(),
        generation=gen,
        config=_MockConfig(),
        output_root=str(tmp_path),
    )
    assert result.ran is False
    assert result.patches == []


def test_write_refinement_log_skipped(tmp_path):
    result = RefinementResult(ran=False)
    path = write_refinement_log(result, str(tmp_path), agent_name="MyAgent")
    text = open(path, encoding="utf-8").read()
    assert "MyAgent" in text
    assert "skipped" in text.lower()


def test_write_refinement_log_with_patches(tmp_path):
    from converter.generator.llm_refinement import FilePatch
    result = RefinementResult(
        ran=True,
        patches=[FilePatch("orchestrator.py", "x=1\n", "Implemented HITL flow")],
        skipped=["bad.py (failed ast.parse — discarded)"],
        overall_summary="Fixed one file.",
    )
    path = write_refinement_log(result, str(tmp_path), agent_name="TestAgent")
    text = open(path, encoding="utf-8").read()
    assert "orchestrator.py" in text
    assert "Implemented HITL flow" in text
    assert "bad.py" in text
    assert "Fixed one file." in text


# ---------------------------------------------------------------------------
# Integration: refinement log appears in the zip output (manual mode)
# ---------------------------------------------------------------------------

README = """# Demo
## Purpose
A demo agent for testing the refinement pass.
## Tools
- `search`: searches
## Workflow
Sequential.
"""

LG_SOURCE = '''
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.tools import tool


class State(TypedDict):
    x: int


@tool
def search(query: str) -> str:
    """Search the web."""
    return "results"


def gather(state):
    """Gather inputs."""
    return {"x": state["x"] + 1}


def summarise(state):
    """Summarise findings."""
    return state


g = StateGraph(State)
g.add_node("gather", gather)
g.add_node("summarise", summarise)
g.set_entry_point("gather")
g.add_edge("gather", "summarise")
g.add_edge("summarise", END)
'''


def test_refinement_log_in_zip_output():
    """Stage 11 always writes REFINEMENT_LOG.md, even when the LLM is absent."""
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": LG_SOURCE},
    ]
    raw = convert_folder(files, "manual", target="maf", source="langgraph")
    zf = zipfile.ZipFile(io.BytesIO(raw))
    assert "REFINEMENT_LOG.md" in zf.namelist(), (
        f"REFINEMENT_LOG.md missing from zip. Got: {zf.namelist()}"
    )
    log_text = zf.read("REFINEMENT_LOG.md").decode("utf-8")
    # In manual mode (no LLM key) it must say it was skipped.
    assert "skipped" in log_text.lower() or "Refinement" in log_text
