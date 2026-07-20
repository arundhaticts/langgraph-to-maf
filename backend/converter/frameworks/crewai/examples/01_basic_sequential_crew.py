"""
CrewAI Example 01 — Basic sequential Crew.

SOURCE PATTERN (LangGraph):
    graph = StateGraph(MyState)
    graph.add_node("analyse", analyse_fn)
    graph.add_node("report", report_fn)
    graph.add_edge("analyse", "report")
    graph.set_entry_point("analyse")
    app = graph.compile()
    result = app.invoke({"suite_path": "..."})

TARGET PATTERN (CrewAI):
    Nodes  -> Task objects, each executed by an Agent
    Edges  -> order of tasks in Crew(tasks=[...]) with Process.sequential
    State  -> data flows implicitly (prev task output) or explicitly via context=[...]
    invoke -> crew.kickoff(inputs={...})   ({placeholders} filled from inputs)
"""
from __future__ import annotations

from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Typed output for the final task (optional but recommended)
# ---------------------------------------------------------------------------
class Report(BaseModel):
    score: float
    gaps: list[str]


# ---------------------------------------------------------------------------
# The LLM — provider chosen by the model string (LiteLLM under the hood).
# Swap "openai/gpt-4o" for "azure/<deployment>", "anthropic/...", etc.
# ---------------------------------------------------------------------------
llm = LLM(model="openai/gpt-4o", temperature=0.2)


# ---------------------------------------------------------------------------
# Agents — each has the REQUIRED role / goal / backstory identity fields.
# An Agent does nothing until it is assigned to a Task inside a Crew.
# ---------------------------------------------------------------------------
analyst = Agent(
    role="Senior Test Analyst",
    goal="Find coverage gaps in the suite located at {suite_path}",
    backstory=(
        "You are a meticulous QA engineer who reads CI history and spots "
        "unreliable or missing test coverage."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Report Writer",
    goal="Turn raw analysis into a clear, actionable report",
    backstory="A technical writer who converts analysis into concise reports.",
    llm=llm,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Tasks — the "nodes". expected_output is REQUIRED.
# {suite_path} is interpolated from crew.kickoff(inputs=...).
# ---------------------------------------------------------------------------
analyse_task = Task(
    description="Analyse the test suite at {suite_path} and list coverage gaps.",
    expected_output="A bullet list of coverage gaps found in the suite.",
    agent=analyst,
)

report_task = Task(
    description=(
        "Using the analysis, produce a report with an overall coverage score "
        "(0.0-1.0) and the list of gaps."
    ),
    expected_output="A JSON object with fields: score (float) and gaps (list of strings).",
    agent=writer,
    context=[analyse_task],       # explicit dependency: inject analyse_task's output
    output_pydantic=Report,       # parse the result into the Report model
)


# ---------------------------------------------------------------------------
# Crew — tasks run in list order under Process.sequential.
# Every Task defined above MUST appear here, or it is dead code.
# ---------------------------------------------------------------------------
crew = Crew(
    agents=[analyst, writer],
    tasks=[analyse_task, report_task],
    process=Process.sequential,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
    print("Raw:", result.raw)
    print("Typed:", result.pydantic)   # a Report instance


if __name__ == "__main__":
    main()
