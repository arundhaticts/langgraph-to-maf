"""
CrewAI Example 02 — Hierarchical process (manager delegation).

SOURCE PATTERN (supervisor / dynamic assignment):
    A supervisor node decides which worker handles each sub-task and in what
    order (LangGraph supervisor pattern, or an MAF agent-as-tool supervisor).

TARGET PATTERN (CrewAI):
    Process.hierarchical + a manager. The manager coordinates the workers,
    deciding which agent handles each task and delegating dynamically instead
    of following the static task-list order.

    Provide EITHER:
      - manager_llm=LLM(...)      -> CrewAI auto-creates a manager agent, or
      - manager_agent=<Agent>     -> you supply your own manager.

    Worker Tasks may omit agent= and let the manager assign them.
"""
from __future__ import annotations

from crewai import Agent, Task, Crew, Process, LLM


llm = LLM(model="openai/gpt-4o")


# ---------------------------------------------------------------------------
# Worker agents. Workers usually keep allow_delegation=False; the MANAGER
# is the one that delegates.
# ---------------------------------------------------------------------------
coverage_analyst = Agent(
    role="Coverage Analyst",
    goal="Measure test coverage and identify gaps",
    backstory="An expert in code coverage tooling and gap analysis.",
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

redundancy_analyst = Agent(
    role="Redundancy Analyst",
    goal="Find duplicate or overlapping tests that can be removed",
    backstory="A specialist in test-suite deduplication.",
    llm=llm,
    allow_delegation=False,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Tasks with NO explicit agent= — the manager assigns each to the best worker.
# ---------------------------------------------------------------------------
coverage_task = Task(
    description="Assess coverage for the suite at {suite_path} and list the gaps.",
    expected_output="A list of coverage gaps with severity.",
)

redundancy_task = Task(
    description="Identify redundant tests in the suite at {suite_path}.",
    expected_output="A list of redundant test ids with justification.",
)

summary_task = Task(
    description="Combine the coverage and redundancy findings into recommendations.",
    expected_output="A prioritised list of recommended suite changes.",
)


# ---------------------------------------------------------------------------
# Option A: let CrewAI build the manager from manager_llm.
# ---------------------------------------------------------------------------
crew = Crew(
    agents=[coverage_analyst, redundancy_analyst],
    tasks=[coverage_task, redundancy_task, summary_task],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4o"),   # auto-creates the manager agent
    verbose=True,
)


# ---------------------------------------------------------------------------
# Option B: supply your own manager agent instead (uncomment to use).
# A custom manager must be allowed to delegate.
# ---------------------------------------------------------------------------
def build_crew_with_custom_manager() -> Crew:
    manager = Agent(
        role="QA Lead",
        goal="Coordinate analysts to optimise the test suite",
        backstory="A pragmatic QA lead who delegates work and synthesises results.",
        llm=llm,
        allow_delegation=True,   # the manager delegates to the workers
        verbose=True,
    )
    return Crew(
        agents=[coverage_analyst, redundancy_analyst],
        tasks=[coverage_task, redundancy_task, summary_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def main():
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
    print(result.raw)


if __name__ == "__main__":
    main()
