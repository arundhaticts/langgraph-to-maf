"""
LangGraph Example 07 — State schemas and reducers (the append pattern).

Reducers control how a node's returned value is MERGED into state:
    - No reducer            -> last-write-wins (the value replaces the old one).
    - Annotated[list, operator.add] -> concatenate lists (append).
    - Annotated[list, add_messages] -> message-aware append/merge (chat history).

RULE: For a reducer key, a node returns ONLY the DELTA (the new items). The
      reducer concatenates it onto the existing value. Returning the whole
      accumulated list double-appends every step.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# State — mix of reducer keys (accumulate) and plain keys (replace).
# ---------------------------------------------------------------------------
class State(TypedDict):
    project_id: str                                  # plain: last-write-wins
    suite: list[dict]                                # plain: replaced wholesale
    audit_log: Annotated[list[dict], operator.add]   # append reducer
    tool_errors: Annotated[list[dict], operator.add] # append reducer
    messages: Annotated[list, add_messages]          # chat-aware append


def _audit(node: str, event: str, **kw) -> dict:
    return {"node": node, "event": event, **kw}


# ---------------------------------------------------------------------------
# Nodes — return only the delta for reducer keys.
# ---------------------------------------------------------------------------
def intake(state: State) -> dict:
    suite = _load_suite(state["project_id"])
    # `suite` is replaced; `audit_log` gets ONE new entry (reducer appends it).
    return {"suite": suite, "audit_log": [_audit("intake", "loaded", count=len(suite))]}


def coverage(state: State) -> dict:
    gaps = _find_gaps(state["suite"])
    updates: dict = {"audit_log": [_audit("coverage", "analysed", gaps=len(gaps))]}
    if not gaps:
        # Append to a DIFFERENT reducer key — again, only the new item.
        updates["tool_errors"] = [{"source": "coverage", "error": "no criteria file"}]
    return updates


def report(state: State) -> dict:
    return {"audit_log": [_audit("report", "complete", errors=len(state["tool_errors"]))]}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("coverage", coverage)
builder.add_node("report", report)
builder.add_edge(START, "intake")
builder.add_edge("intake", "coverage")
builder.add_edge("coverage", "report")
builder.add_edge("report", END)

graph = builder.compile()


def main():
    final = graph.invoke({
        "project_id": "proj-001",
        "suite": [],
        "audit_log": [],
        "tool_errors": [],
        "messages": [],
    })
    print("audit_log entries:", len(final["audit_log"]))
    print("tool_errors:", final["tool_errors"])


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _load_suite(project_id: str) -> list[dict]:
    return [{"id": "test_login"}, {"id": "test_logout"}]

def _find_gaps(suite: list[dict]) -> list[dict]:
    return []


if __name__ == "__main__":
    main()
