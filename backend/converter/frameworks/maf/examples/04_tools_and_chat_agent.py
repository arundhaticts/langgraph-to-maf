"""
MAF Example 04 — Tools and ChatAgent.

SOURCE PATTERN (LangGraph / @tool):
    @tool
    def get_history(test_id: str, path: str | None = None) -> dict | None:
        \"\"\"Return CI run stats for one test.\"\"\"
        ...

    # In Semantic Kernel (wrong framework):
    class GetHistoryPlugin:
        @kernel_function(description="...")
        def get_history(self, test_id: str) -> dict | None: ...

TARGET PATTERN (MAF):
    A MAF tool is a plain typed function.
    Use @ai_function only to override the inferred name / description.
    Pass functions directly to ChatAgent(tools=[...]).
    Do NOT emit wrapper classes that are never instantiated.

AGENT-AS-TOOL:
    A sub-agent can itself be used as a tool by a supervisor agent.
"""
from __future__ import annotations

from typing import Annotated
from pydantic import Field
from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureOpenAIChatClient


# ---------------------------------------------------------------------------
# Tool definitions — plain typed functions
# The framework introspects the signature and docstring to build the schema.
# ---------------------------------------------------------------------------

@ai_function(description="Return CI run stats for one test, or None if no history.")
def get_history(
    test_id: Annotated[str, Field(description="The unique test identifier")],
    path: Annotated[str | None, Field(description="Path to an uploaded CI-history JSON file")] = None,
) -> dict | None:
    """Look up CI evidence (runs, fails, avg_seconds) for a single test."""
    return _load_ci_history(path).get(test_id)


@ai_function(description="Read and normalise test files from a suite path.")
def read_tests(
    suite_path: Annotated[str, Field(description="Path to the test suite directory or zip")],
) -> list[dict]:
    """Return a list of normalised test dicts from the suite."""
    return _parse_tests(suite_path)


@ai_function(description="Validate that a Python test code snippet compiles and runs.")
def validate_test(
    code: Annotated[str, Field(description="Python source code of the test function")],
) -> dict:
    """Return {valid: bool, error: str}."""
    try:
        compile(code, "<string>", "exec")
        return {"valid": True, "error": ""}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Basic ChatAgent — wraps a single LLM with the tools above
# ---------------------------------------------------------------------------
def build_agent() -> ChatAgent:
    return ChatAgent(
        chat_client=AzureOpenAIChatClient(),   # reads env: AZURE_OPENAI_ENDPOINT, KEY, DEPLOYMENT
        instructions=(
            "You are a test-optimisation assistant. "
            "Use get_history to identify flaky/slow tests, "
            "read_tests to load the suite, and validate_test to check generated code."
        ),
        tools=[get_history, read_tests, validate_test],
    )


# ---------------------------------------------------------------------------
# Multi-turn conversation with thread memory
# ---------------------------------------------------------------------------
async def multi_turn_example():
    agent = build_agent()
    thread = agent.get_new_thread()

    result1 = await agent.run("Load the suite at './uploads/suite/'", thread=thread)
    print("Turn 1:", result1.text)

    result2 = await agent.run("Which tests have a fail rate above 10%?", thread=thread)
    print("Turn 2:", result2.text)


# ---------------------------------------------------------------------------
# Agent-as-tool (sub-agent used by a supervisor)
# ---------------------------------------------------------------------------
async def supervisor_example():
    # Research agent — specialised sub-agent.
    research_agent = ChatAgent(
        chat_client=AzureOpenAIChatClient(),
        instructions="You retrieve CI history and summarise test quality.",
        tools=[get_history],
    )

    # Expose the research agent as a tool.
    research_tool = research_agent.as_tool(
        name="research_test_quality",
        description="Research CI history and quality signals for the test suite.",
    )

    # Supervisor agent uses the research tool alongside direct tools.
    supervisor = ChatAgent(
        chat_client=AzureOpenAIChatClient(),
        instructions="You optimise test suites. Use research_test_quality to gather evidence.",
        tools=[research_tool, validate_test],
    )

    result = await supervisor.run("Optimise the test suite at './uploads/suite/'")
    print("Supervisor result:", result.text)


# ---------------------------------------------------------------------------
# Streaming output
# ---------------------------------------------------------------------------
async def streaming_example():
    agent = build_agent()
    async for update in agent.run_stream("List the flaky tests in this suite."):
        print(update.text, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _load_ci_history(path: str | None) -> dict:
    import json, pathlib
    if path and pathlib.Path(path).exists():
        return json.loads(pathlib.Path(path).read_text())
    return {}

def _parse_tests(suite_path: str) -> list[dict]:
    return [{"id": "test_stub", "name": "test_stub", "code": "def test_stub(): pass"}]
