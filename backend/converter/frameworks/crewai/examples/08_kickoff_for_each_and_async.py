"""
CrewAI Example 08 — kickoff_for_each and async invocation.

SOURCE PATTERN:
    # batched invoke over many inputs
    results = [app.invoke({"suite_path": p}) for p in paths]
    # async invoke
    result = await app.ainvoke({"suite_path": p})

TARGET PATTERN (CrewAI):
    Build the Crew ONCE, then invoke it in different ways:
      crew.kickoff(inputs={...})                 -> single run
      crew.kickoff_for_each(inputs=[{...}, ...]) -> one run per input (data-parallel)
      await crew.kickoff_async(inputs={...})     -> single async run
      await crew.kickoff_for_each_async(inputs=[...]) -> async batch

    {placeholders} in task descriptions are filled from each input dict.
"""
from __future__ import annotations

import asyncio

from crewai import Agent, Task, Crew, Process, LLM


llm = LLM(model="openai/gpt-4o")


analyst = Agent(
    role="Test Analyst",
    goal="Analyse the suite at {suite_path} and score it",
    backstory="A QA engineer who quickly scores suites for quality.",
    llm=llm,
    verbose=True,
)

analyse_task = Task(
    description="Analyse the suite at {suite_path} and give it a quality score 0-1.",
    expected_output="A single float score between 0 and 1 with a one-line rationale.",
    agent=analyst,
)

# Build the crew once; reuse it for every invocation style below.
crew = Crew(
    agents=[analyst],
    tasks=[analyse_task],
    process=Process.sequential,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 1) Single synchronous run.
# ---------------------------------------------------------------------------
def run_single():
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite_a/"})
    print("single:", result.raw)


# ---------------------------------------------------------------------------
# 2) Batched run — one crew execution per input dict (data-parallel fan-out).
# ---------------------------------------------------------------------------
def run_for_each():
    suites = ["./uploads/suite_a/", "./uploads/suite_b/", "./uploads/suite_c/"]
    results = crew.kickoff_for_each(inputs=[{"suite_path": p} for p in suites])
    for path, result in zip(suites, results):
        print(f"{path} ->", result.raw)


# ---------------------------------------------------------------------------
# 3) Async run (and async batch).
# ---------------------------------------------------------------------------
async def run_async():
    result = await crew.kickoff_async(inputs={"suite_path": "./uploads/suite_a/"})
    print("async:", result.raw)

    suites = ["./uploads/suite_a/", "./uploads/suite_b/"]
    results = await crew.kickoff_for_each_async(
        inputs=[{"suite_path": p} for p in suites]
    )
    for path, result in zip(suites, results):
        print(f"async {path} ->", result.raw)


def main():
    run_single()
    run_for_each()
    asyncio.run(run_async())


if __name__ == "__main__":
    main()
