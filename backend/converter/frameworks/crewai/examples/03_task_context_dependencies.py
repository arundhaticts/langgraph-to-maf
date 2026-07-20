"""
CrewAI Example 03 — Task context dependencies (a DAG, not just a line).

SOURCE PATTERN (LangGraph):
    graph.add_edge("intake", "coverage")
    graph.add_edge("intake", "redundancy")
    graph.add_edge("coverage", "report")
    graph.add_edge("redundancy", "report")   # report depends on BOTH branches

TARGET PATTERN (CrewAI):
    Express fan-in dependencies with Task(context=[...]). The context list names
    the specific upstream Task objects whose outputs are injected as context.
    Process.sequential still runs tasks in list order, but context= makes the
    real data dependencies explicit (a DAG edge, not just the previous task).
"""
from __future__ import annotations

from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel


llm = LLM(model="openai/gpt-4o")


class SuiteReport(BaseModel):
    coverage_score: float
    redundant_tests: list[str]
    recommendations: list[str]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
intake_agent = Agent(
    role="Intake Normaliser",
    goal="Load and normalise the test suite at {suite_path}",
    backstory="You standardise raw test files into a consistent structure.",
    llm=llm,
)

coverage_agent = Agent(
    role="Coverage Analyst",
    goal="Compute coverage and list gaps",
    backstory="A coverage-tooling expert.",
    llm=llm,
)

redundancy_agent = Agent(
    role="Redundancy Analyst",
    goal="Detect duplicate tests",
    backstory="A deduplication specialist.",
    llm=llm,
)

writer = Agent(
    role="Report Writer",
    goal="Synthesise all findings into one report",
    backstory="A writer who merges multiple analyses into a single report.",
    llm=llm,
)


# ---------------------------------------------------------------------------
# Tasks — note the context= wiring that forms the DAG.
# ---------------------------------------------------------------------------
intake_task = Task(
    description="Load and normalise the suite at {suite_path}.",
    expected_output="A normalised list of tests.",
    agent=intake_agent,
)

coverage_task = Task(
    description="Compute coverage for the normalised suite and list gaps.",
    expected_output="A coverage score and a list of gaps.",
    agent=coverage_agent,
    context=[intake_task],          # depends on intake
)

redundancy_task = Task(
    description="Find redundant tests in the normalised suite.",
    expected_output="A list of redundant test ids.",
    agent=redundancy_agent,
    context=[intake_task],          # also depends on intake (parallel branch)
)

report_task = Task(
    description="Combine coverage and redundancy findings into recommendations.",
    expected_output="A JSON object with coverage_score, redundant_tests, recommendations.",
    agent=writer,
    context=[coverage_task, redundancy_task],   # fan-in: depends on BOTH branches
    output_pydantic=SuiteReport,
)


# ---------------------------------------------------------------------------
# Crew — list order respects the dependency ordering; context= carries the DAG.
# ---------------------------------------------------------------------------
crew = Crew(
    agents=[intake_agent, coverage_agent, redundancy_agent, writer],
    tasks=[intake_task, coverage_task, redundancy_task, report_task],
    process=Process.sequential,
    verbose=True,
)


def main():
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
    print(result.pydantic)


if __name__ == "__main__":
    main()
