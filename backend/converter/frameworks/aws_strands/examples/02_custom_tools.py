"""
AWS Strands Example 02 — Custom tools with @tool.

NATIVE PATTERN (Strands):
    A tool is a plain Python function decorated with @tool (imported from
    `strands`). Strands introspects the function's TYPE HINTS and DOCSTRING
    (including the `Args:` section) to build the tool schema shown to the model.
    Register a tool simply by passing the function object in Agent(tools=[...]).

IR MAPPING:
    @tool function  -> a tool on the single agent node.
    NEVER model each tool as its own node — a lone Agent with N tools is still
    exactly ONE SINGLE_AGENT node.
"""
from __future__ import annotations

from strands import Agent, tool
from strands.models import BedrockModel


# ---------------------------------------------------------------------------
# Tool definitions — plain typed functions.
# The docstring description + Args become the tool/parameter descriptions.
# ---------------------------------------------------------------------------
@tool
def get_history(test_id: str, path: str | None = None) -> dict:
    """Return CI run stats for one test, or an empty dict if none exist.

    Args:
        test_id: The unique test identifier.
        path: Optional path to a CI-history JSON file.
    """
    history = _load_ci_history(path)
    return history.get(test_id, {})


@tool
def read_tests(suite_path: str) -> list[dict]:
    """Read and normalise test files from a suite path.

    Args:
        suite_path: Path to the test-suite directory or archive.
    """
    return _parse_tests(suite_path)


@tool
def validate_test(code: str) -> dict:
    """Validate that a Python test snippet compiles.

    Args:
        code: Python source code of the test function.
    """
    try:
        compile(code, "<string>", "exec")
        return {"valid": True, "error": ""}
    except SyntaxError as exc:
        return {"valid": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Agent with the tools attached. The model chooses which tool to call, and
# when, inside the agentic loop.
# ---------------------------------------------------------------------------
agent = Agent(
    model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
    system_prompt=(
        "You are a test-optimisation assistant. Use read_tests to load a "
        "suite, get_history to spot flaky/slow tests, and validate_test to "
        "check any code you generate."
    ),
    tools=[get_history, read_tests, validate_test],
)


def main() -> None:
    result = agent("Load ./uploads/suite/ and tell me which tests are flaky.")
    print(result.message)


# ---------------------------------------------------------------------------
# Stubs (replace with real implementations).
# ---------------------------------------------------------------------------
def _load_ci_history(path: str | None) -> dict:
    import json
    import pathlib

    if path and pathlib.Path(path).exists():
        return json.loads(pathlib.Path(path).read_text())
    return {}


def _parse_tests(suite_path: str) -> list[dict]:
    return [{"id": "test_stub", "name": "test_stub", "code": "def test_stub(): pass"}]


if __name__ == "__main__":
    main()
