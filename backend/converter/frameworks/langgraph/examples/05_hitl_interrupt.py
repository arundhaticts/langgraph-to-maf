"""
LangGraph Example 05 — Human-in-the-loop with interrupt() + Command(resume=...).

Modern LangGraph HITL:
    - A node calls interrupt(payload). The graph PAUSES and surfaces the payload.
    - The caller inspects result["__interrupt__"], gets a human decision, then
      resumes with graph.invoke(Command(resume=decision), config).
    - The resume value becomes the RETURN value of interrupt(...) in the node.

REQUIRES: a checkpointer + config={"configurable": {"thread_id": ...}}.
          Without them the interrupt cannot persist or resume.

RULE: Keep an automated fast-path that returns the recommendation without
      pausing. NEVER implement HITL as hard-coded auto-approve with the real
      logic commented out.
"""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    run_mode: str            # "automated" or "interactive"
    candidates: list[str]
    recommended: list[str]
    approved: list[str]


# ---------------------------------------------------------------------------
# HITL node — interrupt() replaces a request/response pause.
# ---------------------------------------------------------------------------
def approve_removals(state: State) -> dict:
    if state["run_mode"] == "automated":
        # Automated fast-path: accept the recommendation, no human, no pause.
        return {"approved": state["recommended"]}

    # Interactive path: pause and surface the payload to the caller.
    human = interrupt({
        "question": "Approve these test removals?",
        "candidates": state["candidates"],
        "recommended": state["recommended"],
    })
    # Execution resumes HERE with `human` = the Command(resume=...) value.
    # None means the human accepted the recommendation.
    approved = human if human is not None else state["recommended"]
    return {"approved": approved}


def apply(state: State) -> dict:
    _remove(state["approved"])
    return {}


# ---------------------------------------------------------------------------
# Graph assembly — checkpointer is REQUIRED for interrupt/resume.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("approve_removals", approve_removals)
builder.add_node("apply", apply)
builder.add_edge(START, "approve_removals")
builder.add_edge("approve_removals", "apply")
builder.add_edge("apply", END)

graph = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Host-side pause / resume loop.
# ---------------------------------------------------------------------------
def run_with_hitl(candidates: list[str], recommended: list[str], run_mode: str):
    config = {"configurable": {"thread_id": "removal-run-1"}}
    inputs = {
        "run_mode": run_mode,
        "candidates": candidates,
        "recommended": recommended,
        "approved": [],
    }

    result = graph.invoke(inputs, config)

    # If the graph paused, "__interrupt__" carries the payload(s).
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decision = _get_human_decision(payload)     # your UI/API call
        result = graph.invoke(Command(resume=decision), config)

    return result


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _get_human_decision(payload: dict) -> list | None:
    """Replace with real UI/API. Return None to accept the recommendation."""
    print(f"[HITL] {payload['question']} recommended={payload['recommended']}")
    return None

def _remove(ids: list[str]) -> None:
    pass


def main():
    print(run_with_hitl(["t1", "t2"], ["t1"], run_mode="interactive"))


if __name__ == "__main__":
    main()
