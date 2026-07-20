"""
AWS Strands Example 07 — Multi-agent orchestration (Swarm and Graph).

NATIVE PATTERN (Strands):
    Beyond a single agent, Strands offers opt-in multi-agent orchestration:
      - Swarm: self-organising handoff between agents via a shared handoff tool.
      - Graph (GraphBuilder): an author-declared DAG of AGENTS (not low-level
        steps); edges may carry conditions.
    These are the ONLY cases that map to more than one node in the IR.

IR MAPPING:
    Swarm([...])          -> multiple agent nodes + handoff orchestration notes
    GraphBuilder / Graph  -> multiple agent nodes + edges (multi-agent DAG)
    A lone Agent(tools=[...]) is still ONE node — do not inflate its tools.
"""
from __future__ import annotations

from strands import Agent
from strands.models import BedrockModel
from strands.multiagent import Swarm, GraphBuilder


MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


def _model() -> BedrockModel:
    return BedrockModel(model_id=MODEL_ID)


# Two specialist agents reused by both patterns below.
researcher = Agent(
    name="researcher",
    model=_model(),
    system_prompt="You gather CI history and quality signals for a test suite.",
)
writer = Agent(
    name="writer",
    model=_model(),
    system_prompt="You turn research findings into a concise optimisation plan.",
)


# ---------------------------------------------------------------------------
# 1) Swarm — agents hand off to each other until the task is complete.
# ---------------------------------------------------------------------------
def swarm_example() -> None:
    swarm = Swarm(
        [researcher, writer],
        max_handoffs=10,
        max_iterations=20,
    )
    result = swarm("Investigate ./uploads/suite/ and produce an optimisation plan.")
    print("Swarm result:", result)


# ---------------------------------------------------------------------------
# 2) Graph — an explicit DAG of agents with declared edges.
# ---------------------------------------------------------------------------
def graph_example() -> None:
    builder = GraphBuilder()
    builder.add_node(researcher, "research")
    builder.add_node(writer, "write")
    builder.add_edge("research", "write")   # research feeds the writer
    builder.set_entry_point("research")

    graph = builder.build()
    result = graph("Investigate ./uploads/suite/ and produce an optimisation plan.")
    print("Graph result:", result)


def main() -> None:
    graph_example()


if __name__ == "__main__":
    main()
