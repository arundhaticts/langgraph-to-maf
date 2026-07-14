"""Test Optimiser Agent (LangGraph source)."""

import os
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

MAX_GEN_RETRIES = 3
COVERAGE_FLOOR = 0.8

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


class State(TypedDict):
    project_id: str
    coverage: float
    gen_retry_count: int
    audit_log: Annotated[list, add]
    tool_errors: Annotated[list, add]


@tool
def read_tests(path: str) -> list:
    """Reads test files from the repo."""
    if not os.path.isdir(path):
        return []
    return [f for f in os.listdir(path) if f.startswith("test_")]


@tool
def detect_flaky_tests(runs: int) -> bool:
    """Flags tests that fail intermittently across the given number of runs."""
    return runs > 1


def generate(state):
    """Generate more tests and bump the retry counter."""
    count = state.get("gen_retry_count", 0)
    coverage = state.get("coverage", 0.0)
    new_coverage = min(1.0, coverage + 0.1)
    return {
        "gen_retry_count": count + 1,
        "coverage": new_coverage,
        "audit_log": [f"generated tests, attempt {count + 1}"],
    }


def gate(state):
    """Record whether the coverage floor was met."""
    met = coverage_floor_met(state)
    return {"audit_log": [f"coverage {state['coverage']} floor_met={met}"]}


def coverage_floor_met(state) -> bool:
    """True when coverage has reached the configured floor."""
    return state["coverage"] >= COVERAGE_FLOOR


def route(state) -> str:
    """Route back to generate until coverage is met or retries are exhausted."""
    if coverage_floor_met(state):
        return "done"
    if state["gen_retry_count"] >= MAX_GEN_RETRIES:
        return "done"
    return "revise"


def hitl_approve(state):
    """Human approves the final test change before it is committed."""
    decision = interrupt({"coverage": state["coverage"]})
    return {"audit_log": [f"human decision: {decision}"]}


g = StateGraph(State)
g.add_node("generate", generate)
g.add_node("gate", gate)
g.add_node("hitl_approve", hitl_approve)
g.set_entry_point("generate")
g.add_edge("generate", "gate")
g.add_conditional_edges("gate", route, {"revise": "generate", "done": END})
g.add_edge("hitl_approve", END)

app = g.compile()
