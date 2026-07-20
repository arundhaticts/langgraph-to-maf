"""
CrewAI Example 04 — Custom tools (both @tool and BaseTool).

SOURCE PATTERN (LangGraph / @tool):
    @tool
    def get_history(test_id: str) -> dict | None:
        \"\"\"Return CI run stats for one test.\"\"\"
        ...

    # Semantic Kernel (wrong framework — do not emit):
    class GetHistoryPlugin:
        @kernel_function(...)
        def get_history(self, test_id: str): ...

TARGET PATTERN (CrewAI):
    A CrewAI tool is an INSTANCE passed to Agent(tools=[...]).
    Two idiomatic ways:
      (a) @tool decorator from crewai.tools  -> the decorated object IS the tool.
      (b) subclass BaseTool, implement _run, then INSTANTIATE it.
    Prebuilt tools live in crewai_tools (e.g. SerperDevTool()).

    Pass INSTANCES, not classes. Do NOT emit tool classes that are never
    instantiated or never assigned to an agent.
"""
from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool, BaseTool


llm = LLM(model="openai/gpt-4o")


# ---------------------------------------------------------------------------
# (a) @tool decorator — quickest for a single function.
# The docstring becomes the tool description; the return should be a string.
# The decorated object is a ready-to-use tool instance.
# ---------------------------------------------------------------------------
@tool("CI History Lookup")
def get_history(test_id: str) -> str:
    """Return CI run stats (runs, fails, avg_seconds) for a single test id."""
    stats = _load_ci_history().get(test_id, {})
    return str(stats)


# ---------------------------------------------------------------------------
# (b) BaseTool subclass — when you want a typed args schema / reusable config.
# args_schema field names MUST match the _run parameter names.
# ---------------------------------------------------------------------------
class ValidateInput(BaseModel):
    code: str = Field(..., description="Python source code of the test function")


class ValidateTestTool(BaseTool):
    name: str = "Validate Test Code"
    description: str = "Check that a Python test snippet compiles. Returns valid/error."
    args_schema: Type[BaseModel] = ValidateInput

    def _run(self, code: str) -> str:
        try:
            compile(code, "<string>", "exec")
            return "valid=True"
        except SyntaxError as e:
            return f"valid=False error={e}"


# IMPORTANT: instantiate the BaseTool subclass before assigning it.
validate_tool = ValidateTestTool()


# ---------------------------------------------------------------------------
# Agent using both tools (instances, not classes).
# ---------------------------------------------------------------------------
analyst = Agent(
    role="Test Optimisation Assistant",
    goal="Identify flaky tests and validate any generated test code",
    backstory=(
        "You use CI history to find unreliable tests and validate that any "
        "generated test code actually compiles."
    ),
    tools=[get_history, validate_tool],   # both are INSTANCES
    llm=llm,
    verbose=True,
)

analyse_task = Task(
    description=(
        "Look up CI history for test id {test_id} and report whether it is flaky. "
        "If you propose replacement code, validate that it compiles."
    ),
    expected_output="A short verdict on flakiness plus any validation result.",
    agent=analyst,
)

crew = Crew(
    agents=[analyst],
    tasks=[analyse_task],
    process=Process.sequential,
    verbose=True,
)


def main():
    result = crew.kickoff(inputs={"test_id": "test_login"})
    print(result.raw)


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------
def _load_ci_history() -> dict:
    return {"test_login": {"runs": 100, "fails": 18, "avg_seconds": 2.4}}


if __name__ == "__main__":
    main()
