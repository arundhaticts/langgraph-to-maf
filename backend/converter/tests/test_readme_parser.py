"""Tests for Module 3 -- README parser."""

from __future__ import annotations

from converter.parser.readme_parser import parse_readme

FULL_README = """# Test Optimiser Agent

## Purpose
Optimises a project's test suite.

## Framework
LangGraph

## Tools
- `read_tests`: Reads test files from the repo
- `detect_flaky_tests` - Flags tests that fail intermittently

## Workflow
First read the tests, then analyse coverage. If coverage is below the
floor, loop back and generate more tests, up to 3 times.

### Notes
This subsection stays inside Workflow.

## State
- `project_id` (str): The project identifier
- `retry_count` (int): Number of retries so far
- `audit_log` (list): Append-only audit trail

## Configuration
Uses OPENAI_API_KEY and a temperature of 0.2.

## Dependencies
langgraph, langchain-openai
"""


def test_all_sections_parsed():
    r = parse_readme(FULL_README)
    assert r.purpose == "Optimises a project's test suite."
    assert r.framework == "LangGraph"
    assert r.configuration.startswith("Uses OPENAI_API_KEY")
    assert "langgraph" in r.dependencies
    assert r.missing_sections == []


def test_tool_bullets():
    r = parse_readme(FULL_README)
    tools = {t.name: t.description for t in r.tools}
    assert tools["read_tests"] == "Reads test files from the repo"
    assert tools["detect_flaky_tests"] == "Flags tests that fail intermittently"


def test_state_bullets_with_types():
    r = parse_readme(FULL_README)
    state = {s.name: s for s in r.state}
    assert state["project_id"].type == "str"
    assert state["retry_count"].type == "int"
    assert state["audit_log"].description == "Append-only audit trail"


def test_workflow_kept_verbatim_including_subsection():
    r = parse_readme(FULL_README)
    assert "loop back and generate more tests, up to 3 times." in r.workflow_description
    # level-3 subsection stays inside the workflow body, not split out
    assert "### Notes" in r.workflow_description
    assert "This subsection stays inside Workflow." in r.workflow_description


def test_missing_sections_warned_not_fatal():
    text = "# Agent\n\n## Purpose\nDo things.\n\n## Tools\n- `t`: a tool\n"
    r = parse_readme(text)
    assert r.purpose == "Do things."
    # Framework, Workflow, State, Configuration, Dependencies all missing
    assert set(r.missing_sections) == {
        "Framework",
        "Workflow",
        "State",
        "Configuration",
        "Dependencies",
    }


def test_case_insensitive_headers():
    text = "## purpose\nlower header works\n"
    r = parse_readme(text)
    assert r.purpose == "lower header works"


def test_raw_sections_available():
    r = parse_readme(FULL_README)
    assert "Purpose" in r.raw_sections
    assert r.raw_sections["Framework"] == "LangGraph"
