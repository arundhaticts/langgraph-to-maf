"""
CrewAI Example 05 — Human-in-the-loop (HITL).

SOURCE PATTERN (LangGraph):
    def approve(state):
        decision = interrupt(build_payload(state))   # pause for a human
        return {"approved": decision}

TARGET PATTERN (CrewAI):
    The native HITL primitive is Task(human_input=True). After the task's agent
    produces its output, execution PAUSES at the task boundary and the human is
    prompted to review / correct / approve; their feedback is fed back to the
    agent, which may revise before the crew continues.

NEVER do this:
    # auto-approve as the ONLY live path, real logic in comments
    approved = candidates            # <- wrong
    # decision = interrupt(payload)  # <- commented out = dead code
Either set human_input=True, or route through a Flow that captures a real
decision (see example 07).
"""
from __future__ import annotations

from crewai import Agent, Task, Crew, Process, LLM


llm = LLM(model="openai/gpt-4o")


analyst = Agent(
    role="Quarantine Analyst",
    goal="Propose which flaky tests to quarantine from the suite at {suite_path}",
    backstory=(
        "You are cautious: pinned tests are never removed, and quarantine is "
        "reversible. You justify every recommendation with CI evidence."
    ),
    llm=llm,
    verbose=True,
)


# ---------------------------------------------------------------------------
# The HITL task. human_input=True makes CrewAI pause after the agent responds
# so a human can review and approve or send corrections back to the agent.
# ---------------------------------------------------------------------------
propose_task = Task(
    description=(
        "Analyse the suite at {suite_path}. Propose a list of flaky tests to "
        "quarantine, with a one-line justification for each. Do NOT include "
        "pinned tests."
    ),
    expected_output="A list of test ids to quarantine, each with a justification.",
    agent=analyst,
    human_input=True,   # <- pause for human review/approval at this task boundary
)


# ---------------------------------------------------------------------------
# A follow-up task that consumes the (now human-approved) proposal.
# ---------------------------------------------------------------------------
apply_task = Task(
    description="Produce the final quarantine plan from the approved proposal.",
    expected_output="A confirmed quarantine plan ready to execute.",
    agent=analyst,
    context=[propose_task],
)


crew = Crew(
    agents=[analyst],
    tasks=[propose_task, apply_task],
    process=Process.sequential,
    verbose=True,
)


def main():
    # When kickoff reaches propose_task, CrewAI prompts the human on the console.
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
    print(result.raw)


if __name__ == "__main__":
    main()
