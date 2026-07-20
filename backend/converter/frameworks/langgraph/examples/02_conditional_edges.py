"""
LangGraph Example 02 — Conditional edges (branching / routing).

add_conditional_edges runs a ROUTER function over the state and picks the next
node. The router returns a node name (or a key that the optional mapping dict
translates to a node name). It may also return END.

RULE: Every router you define MUST be wired into add_conditional_edges. A router
      function that is defined but never referenced is dead code and a bug.
"""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

MAX_RETRIES = 3


class State(TypedDict):
    candidates: list[dict]
    validation_passed: bool
    retries: int
    approved: list[dict]
    dropped: list[dict]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def validate(state: State) -> dict:
    passed = _run_validation(state["candidates"])
    return {"validation_passed": passed, "retries": state["retries"] + 1}


def approve(state: State) -> dict:
    return {"approved": state["candidates"]}


def drop(state: State) -> dict:
    return {"dropped": state["candidates"]}


def gap_gen(state: State) -> dict:
    # Regenerate candidates and try again.
    return {"candidates": _regenerate(state["candidates"])}


# ---------------------------------------------------------------------------
# Router — returns the NAME of the next node (one branch fires).
# ---------------------------------------------------------------------------
def route_after_validation(state: State) -> str:
    if state["validation_passed"]:
        return "approve"
    if state["retries"] >= MAX_RETRIES:
        return "drop"
    return "gap_gen"


# ---------------------------------------------------------------------------
# Graph assembly — the router is wired in via add_conditional_edges.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("validate", validate)
builder.add_node("approve", approve)
builder.add_node("drop", drop)
builder.add_node("gap_gen", gap_gen)

builder.add_edge(START, "validate")
builder.add_conditional_edges(
    "validate",
    route_after_validation,
    # Optional mapping: router return value -> node name. Here it is 1:1.
    {"approve": "approve", "drop": "drop", "gap_gen": "gap_gen"},
)
builder.add_edge("gap_gen", "validate")   # retry loops back through validate
builder.add_edge("approve", END)
builder.add_edge("drop", END)

graph = builder.compile()


def main():
    final = graph.invoke(
        {"candidates": [{"id": "t1"}], "validation_passed": False,
         "retries": 0, "approved": [], "dropped": []}
    )
    print("approved:", final["approved"], "dropped:", final["dropped"])


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _run_validation(candidates: list[dict]) -> bool:
    return True

def _regenerate(candidates: list[dict]) -> list[dict]:
    return candidates


if __name__ == "__main__":
    main()
