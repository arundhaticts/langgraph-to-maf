"""
AWS Strands Example 06 — Agents as tools (hierarchical composition).

NATIVE PATTERN (Strands):
    Wrap a specialist Agent inside a @tool and give that tool to an
    orchestrator Agent. The orchestrator's model decides when to delegate by
    "calling" the specialist like any other tool. This is Strands' hierarchical
    multi-agent pattern — no explicit graph required.

IR MAPPING:
    Each Agent becomes its own node. The agent-as-tool wrapper becomes a
    tool edge / delegation note between the orchestrator node and the
    specialist node.
"""
from __future__ import annotations

from strands import Agent, tool
from strands.models import BedrockModel


MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


# ---------------------------------------------------------------------------
# Leaf tool used by the specialist agent.
# ---------------------------------------------------------------------------
@tool
def get_history(test_id: str) -> dict:
    """Return CI run stats for one test.

    Args:
        test_id: The unique test identifier.
    """
    return {"test_id": test_id, "runs": 100, "fails": 12}


# ---------------------------------------------------------------------------
# Specialist agent, exposed to the orchestrator as a single @tool.
# Note: a fresh specialist is created per call here for clarity; you could also
# build it once at module scope and reference it inside the tool.
# ---------------------------------------------------------------------------
@tool
def research_test_quality(query: str) -> str:
    """Delegate CI-history research to a specialist agent.

    Args:
        query: What to investigate about the test suite's quality.
    """
    specialist = Agent(
        model=BedrockModel(model_id=MODEL_ID),
        system_prompt="You retrieve CI history and summarise test quality.",
        tools=[get_history],
    )
    return str(specialist(query).message)


# ---------------------------------------------------------------------------
# Orchestrator agent — uses the specialist as one of its tools.
# ---------------------------------------------------------------------------
orchestrator = Agent(
    model=BedrockModel(model_id=MODEL_ID),
    system_prompt=(
        "You optimise test suites. When you need evidence about test quality, "
        "call research_test_quality to gather it before recommending changes."
    ),
    tools=[research_test_quality],
)


def main() -> None:
    result = orchestrator("Recommend which tests in ./uploads/suite/ to remove.")
    print(result.message)


if __name__ == "__main__":
    main()
