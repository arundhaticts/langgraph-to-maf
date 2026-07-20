"""
CrewAI Example 06 — Memory and knowledge sources.

SOURCE PATTERN (LangGraph):
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

TARGET PATTERN (CrewAI):
    CrewAI has NO per-step checkpoint / resume-from-checkpoint. Map a source
    checkpointer to Crew(memory=True), which enables:
      - short-term memory  (within the current run),
      - long-term memory   (persisted across runs, SQLite by default),
      - entity memory      (tracks entities encountered).
    NOTE: this is cross-run RECALL, not a resumable pause/resume checkpoint.

    Reference material the crew should consult is attached via
    knowledge_sources=[...] rather than baked into every prompt.
"""
from __future__ import annotations

from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


llm = LLM(model="openai/gpt-4o")


# ---------------------------------------------------------------------------
# Knowledge source — a policy the crew can retrieve from during execution.
# ---------------------------------------------------------------------------
quarantine_policy = StringKnowledgeSource(
    content=(
        "Test optimisation policy:\n"
        "1. Pinned tests are NEVER removed or quarantined.\n"
        "2. Quarantine is reversible; deletion is not.\n"
        "3. A test is flaky if its fail rate exceeds 10% over the last 100 runs.\n"
    )
)


analyst = Agent(
    role="Test Optimisation Assistant",
    goal="Recommend suite changes that respect the optimisation policy",
    backstory="A QA engineer who always follows the team's written policy.",
    llm=llm,
    verbose=True,
)

analyse_task = Task(
    description="Analyse the suite at {suite_path} and recommend quarantine actions.",
    expected_output="A policy-compliant list of quarantine recommendations.",
    agent=analyst,
)


# ---------------------------------------------------------------------------
# Crew with memory + knowledge enabled.
#   memory=True            -> short/long/entity memory (cross-run recall)
#   knowledge_sources=[..] -> retrievable reference material
# ---------------------------------------------------------------------------
crew = Crew(
    agents=[analyst],
    tasks=[analyse_task],
    process=Process.sequential,
    memory=True,
    knowledge_sources=[quarantine_policy],
    verbose=True,
)


def main():
    # First run learns; a later run can recall prior context thanks to memory=True.
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
    print(result.raw)


if __name__ == "__main__":
    main()
