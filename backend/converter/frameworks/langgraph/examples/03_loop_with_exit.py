"""
LangGraph Example 03 — Bounded loop (cycle) with a guaranteed exit.

A loop is just an edge that goes BACK to an earlier node, with a conditional
edge providing the exit. The exit guard is the iteration cap.

RULE: Always include a termination guard (iteration cap) so the cycle cannot
      run forever. recursion_limit in the invoke config is only a backstop and
      raises GraphRecursionError if hit.
"""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

MAX_ITERS = 5


class State(TypedDict):
    draft: str
    passed: bool
    iters: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def revise(state: State) -> dict:
    improved = _improve(state["draft"])
    return {"draft": improved, "iters": state["iters"] + 1}


def check(state: State) -> dict:
    return {"passed": _quality_ok(state["draft"])}


def finalize(state: State) -> dict:
    return {"draft": state["draft"].strip()}


# ---------------------------------------------------------------------------
# Loop router — go back to `revise` while failing AND under the cap; else exit.
# ---------------------------------------------------------------------------
def loop_or_exit(state: State) -> str:
    if state["passed"]:
        return "finalize"
    if state["iters"] >= MAX_ITERS:
        return "finalize"   # give up gracefully — never loop forever
    return "revise"


# ---------------------------------------------------------------------------
# Graph assembly: revise -> check -> (loop back to revise | finalize)
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("revise", revise)
builder.add_node("check", check)
builder.add_node("finalize", finalize)

builder.add_edge(START, "revise")
builder.add_edge("revise", "check")
builder.add_conditional_edges(
    "check",
    loop_or_exit,
    {"revise": "revise", "finalize": "finalize"},
)
builder.add_edge("finalize", END)

graph = builder.compile()


def main():
    final = graph.invoke({"draft": "rough draft", "passed": False, "iters": 0})
    print("iterations:", final["iters"], "result:", final["draft"])


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _improve(draft: str) -> str:
    return draft + " (improved)"

def _quality_ok(draft: str) -> bool:
    return draft.count("improved") >= 2


if __name__ == "__main__":
    main()
