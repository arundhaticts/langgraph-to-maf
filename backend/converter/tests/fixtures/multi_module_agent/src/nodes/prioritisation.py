"""Prioritisation node + the coverage-floor gate router + revise node."""

from src.config import COVERAGE_FLOOR, MAX_REVISE_ITERS


def prioritisation_node(state) -> dict:
    """Bump coverage a little and record the attempt."""
    coverage = state.get("coverage", 0.0)
    return {"coverage": min(1.0, coverage + 0.1)}


def coverage_floor_gate(state) -> str:
    """Router: loop to revise until the coverage floor is met (bounded)."""
    if state["coverage"] >= COVERAGE_FLOOR:
        return "approve_ranking"
    if state.get("gen_retry_count", 0) >= MAX_REVISE_ITERS:
        return "approve_ranking"
    return "revise"


def revise_node(state) -> dict:
    """Revise: nudge coverage and count the iteration."""
    return {
        "coverage": min(1.0, state["coverage"] + 0.1),
        "gen_retry_count": state.get("gen_retry_count", 0) + 1,
    }
