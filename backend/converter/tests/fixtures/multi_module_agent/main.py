"""Entrypoint -- builds and invokes the graph (must be rewired to converted modules)."""

from langgraph.types import Command

from src.config import COVERAGE_FLOOR
from src.graph import build_graph
from src.state import TestOptimiserState


def initial_state() -> dict:
    return {"project_id": "demo", "coverage": 0.0, "gen_retry_count": 0, "audit_log": []}


def run() -> dict:
    graph = build_graph()
    return graph.invoke(initial_state(), config={"configurable": {"thread_id": "1"}})


if __name__ == "__main__":
    print(run())
