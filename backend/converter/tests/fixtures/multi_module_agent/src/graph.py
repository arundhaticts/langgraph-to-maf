"""Graph wiring -- StateGraph + edges + conditional edges + MemorySaver checkpointer."""

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.state import TestOptimiserState
from src.nodes.intake import intake_node
from src.nodes.prioritisation import (
    coverage_floor_gate,
    prioritisation_node,
    revise_node,
)
from src.nodes.validation import (
    drop_failing_node,
    gap_generation_node,
    route_after_validation,
    validation_node,
)
from src.hitl.interrupts import hitl_removals_node


def build_graph(checkpointer=None):
    g = StateGraph(TestOptimiserState)
    g.add_node("intake", intake_node)
    g.add_node("prioritisation", prioritisation_node)
    g.add_node("revise", revise_node)
    g.add_node("hitl_removals", hitl_removals_node)
    g.add_node("gap_gen", gap_generation_node)
    g.add_node("validation", validation_node)
    g.add_node("drop_failing", drop_failing_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "prioritisation")
    g.add_conditional_edges(
        "prioritisation", coverage_floor_gate,
        {"revise": "revise", "approve_ranking": "hitl_removals"},
    )
    g.add_conditional_edges(
        "revise", coverage_floor_gate,
        {"revise": "revise", "approve_ranking": "hitl_removals"},
    )
    g.add_edge("hitl_removals", "gap_gen")
    g.add_edge("gap_gen", "validation")
    g.add_conditional_edges(
        "validation", route_after_validation,
        {"approve_tests": END, "gap_gen": "gap_gen", "drop_failing": "drop_failing"},
    )
    g.add_edge("drop_failing", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
