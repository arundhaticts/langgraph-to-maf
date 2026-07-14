"""Tests for Module 4 -- component extractor."""

from __future__ import annotations

import os

import pytest

from converter.contracts import FileAction
from converter.extractor import extract_components
from converter.parser.readme_parser import parse_readme
from converter.scanner import scan_repo


def _write(root: str, rel_path: str, content: str) -> None:
    abs_path = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(abs_path) or root, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)


TOOLS_PY = '''
from langchain_core.tools import tool

@tool
def read_tests(path: str) -> list:
    """Read tests."""
    return []

@tool
def detect_flaky(runs: int) -> bool:
    """Detect flaky tests."""
    return False
'''

GRAPH_PY = '''
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from operator import add
from langchain_openai import ChatOpenAI

MAX_RETRIES = 3

class AgentState(TypedDict):
    project_id: str
    audit_log: Annotated[list, add]

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

g = StateGraph(AgentState)
g.add_node("read", read_node)
g.add_node("analyze", analyze_node)
g.set_entry_point("read")
g.add_edge("read", "analyze")
g.add_edge("analyze", END)
'''

HELPERS_PY = '''
def helper(x):
    return x * 2
'''

README = """# Agent
## Purpose
Test optimiser.
## Tools
- `read_tests`: Read tests
- `detect_flaky`: Detect flaky tests
## State
- `project_id` (str): id
- `audit_log` (list): log
## Workflow
Read then analyze.
"""


def _build_repo(root: str) -> None:
    _write(root, "README.md", README)
    _write(root, "tools.py", TOOLS_PY)
    _write(root, "graph.py", GRAPH_PY)
    _write(root, "helpers.py", HELPERS_PY)


def test_consolidates_across_files(tmp_path):
    root = str(tmp_path)
    _build_repo(root)
    manifest = scan_repo(root)
    readme = parse_readme(README)

    inv = extract_components(manifest, readme)

    assert {t.name for t in inv.tools} == {"read_tests", "detect_flaky"}
    assert {s.name for s in inv.state} == {"project_id", "audit_log"}
    assert {n.name for n in inv.graph.nodes} == {"read", "analyze"}
    assert inv.graph.entry_point == "read"
    assert inv.config.temperature == 0.2
    assert inv.config.constants["MAX_RETRIES"] == 3


def test_file_actions_assigned(tmp_path):
    root = str(tmp_path)
    _build_repo(root)
    manifest = scan_repo(root)

    extract_components(manifest, parse_readme(README))
    actions = {f.relative_path: f.file_action for f in manifest.files}

    assert actions["README.md"] is FileAction.REWRITE
    assert actions["graph.py"] is FileAction.REWRITE      # has graph + state
    assert actions["tools.py"] is FileAction.ADAPT        # tools only
    assert actions["helpers.py"] is FileAction.COPY_THROUGH


def test_no_files_left_unassigned(tmp_path):
    root = str(tmp_path)
    _build_repo(root)
    manifest = scan_repo(root)
    extract_components(manifest, parse_readme(README))
    assert all(f.file_action is not None for f in manifest.files)


def test_cross_reference_flags_drift(tmp_path):
    root = str(tmp_path)
    # README documents a tool the code lacks, and omits one the code has.
    readme_text = (
        "# Agent\n## Tools\n- `ghost_tool`: not in code\n"
        "## State\n- `project_id` (str): id\n"
    )
    _write(root, "README.md", readme_text)
    _write(root, "tools.py", TOOLS_PY)
    manifest = scan_repo(root)

    inv = extract_components(manifest, parse_readme(readme_text))
    joined = "\n".join(inv.warnings)
    assert "ghost_tool" in joined                 # documented, not in code
    assert "read_tests" in joined                 # in code, not documented


def test_syntax_error_file_is_skipped_with_warning(tmp_path):
    root = str(tmp_path)
    _write(root, "README.md", "# Agent\n")
    _write(root, "broken.py", "def (((\n")
    _write(root, "tools.py", TOOLS_PY)
    manifest = scan_repo(root)

    inv = extract_components(manifest)
    assert {t.name for t in inv.tools} == {"read_tests", "detect_flaky"}
    assert any("broken.py" in w for w in inv.warnings)


def test_works_without_readme_argument(tmp_path):
    root = str(tmp_path)
    _build_repo(root)
    manifest = scan_repo(root)
    inv = extract_components(manifest)  # readme=None
    assert len(inv.tools) == 2
