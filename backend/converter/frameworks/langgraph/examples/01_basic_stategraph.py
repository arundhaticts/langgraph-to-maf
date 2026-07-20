"""
LangGraph Example 01 — Basic linear StateGraph.

The core LangGraph model:
    - State is a schema (here a TypedDict). Nodes receive the FULL state.
    - Each node returns a PARTIAL dict of only the keys it changed.
    - Edges wire the control flow. START and END are sentinel node names.
    - builder.compile() produces a runnable exposing .invoke / .stream.

RULE: A node returns {changed_key: value}, NEVER the whole state object and
      never a bare value. Keys it does not return are left unchanged.
"""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# State schema — replaces a global state object. One shared shape for the graph.
# ---------------------------------------------------------------------------
class State(TypedDict):
    suite: list[dict]
    conventions: dict
    coverage_gaps: list[dict]
    score: float


# ---------------------------------------------------------------------------
# Nodes — plain functions: (state) -> partial update dict
# ---------------------------------------------------------------------------
def intake(state: State) -> dict:
    normalised = [_normalise(t) for t in state["suite"]]
    return {"suite": normalised, "conventions": _detect_conventions(normalised)}


def analyse(state: State) -> dict:
    gaps = _find_gaps(state["suite"])
    return {"coverage_gaps": gaps, "score": _score(state["suite"], gaps)}


def report(state: State) -> dict:
    # Terminal node still returns only what it changed.
    return {"score": round(state["score"], 3)}


# ---------------------------------------------------------------------------
# Graph assembly — add nodes, wire edges, compile.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("analyse", analyse)
builder.add_node("report", report)

builder.add_edge(START, "intake")
builder.add_edge("intake", "analyse")
builder.add_edge("analyse", "report")
builder.add_edge("report", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# Entry point — invoke returns the final merged state dict.
# ---------------------------------------------------------------------------
def main():
    final = graph.invoke(
        {"suite": [{"id": "test_login"}], "conventions": {}, "coverage_gaps": [], "score": 0.0}
    )
    print("score:", final["score"])


# ---------------------------------------------------------------------------
# Stubs (replace with real implementations)
# ---------------------------------------------------------------------------
def _normalise(t: dict) -> dict:
    return {**t, "entities": []}

def _detect_conventions(tests: list[dict]) -> dict:
    return {"framework": "pytest"}

def _find_gaps(tests: list[dict]) -> list[dict]:
    return []

def _score(tests: list[dict], gaps: list[dict]) -> float:
    return 1.0 - len(gaps) / max(len(tests), 1)


if __name__ == "__main__":
    main()
