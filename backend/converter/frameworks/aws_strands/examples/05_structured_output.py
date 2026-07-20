"""
AWS Strands Example 05 — Structured output.

NATIVE PATTERN (Strands):
    agent.structured_output(PydanticModel, prompt) runs the agentic loop and
    returns a VALIDATED instance of the given Pydantic model instead of free
    text. There is also an async variant, structured_output_async(...).

IR MAPPING:
    Structured output is a property of the single agent node (its output is a
    typed schema) — not a separate step or node.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models import BedrockModel


# ---------------------------------------------------------------------------
# Define the output schema the model must fill.
# ---------------------------------------------------------------------------
class FlakyTest(BaseModel):
    test_id: str = Field(description="The test identifier")
    fail_rate: float = Field(description="Observed failure rate, 0.0-1.0")


class SuiteReport(BaseModel):
    score: float = Field(description="Overall suite health score, 0.0-1.0")
    flaky_tests: list[FlakyTest] = Field(default_factory=list)
    recommendation: str


@tool
def get_history(test_id: str) -> dict:
    """Return CI run stats for one test.

    Args:
        test_id: The unique test identifier.
    """
    return {"test_id": test_id, "runs": 100, "fails": 12}


agent = Agent(
    model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
    system_prompt="You analyse test suites and report quality signals.",
    tools=[get_history],
)


def main() -> None:
    # The return value is a fully validated SuiteReport instance.
    report: SuiteReport = agent.structured_output(
        SuiteReport,
        "Analyse the suite and identify flaky tests with their fail rates.",
    )
    print("Score:", report.score)
    for ft in report.flaky_tests:
        print(f"  {ft.test_id}: {ft.fail_rate:.0%}")
    print("Recommendation:", report.recommendation)


if __name__ == "__main__":
    main()
