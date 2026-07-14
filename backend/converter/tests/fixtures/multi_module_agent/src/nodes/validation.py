"""Validation node + a 3-outcome router (the genuinely complex part)."""

from src.config import MAX_GEN_RETRIES


def gap_generation_node(state) -> dict:
    """Generate candidate tests."""
    return {"audit_log": [{"event": "generated"}]}


def validation_node(state) -> dict:
    """Validate generated tests."""
    return {"audit_log": [{"event": "validated"}]}


def route_after_validation(state) -> str:
    """Router with THREE outcomes -> complex orchestration (Tier 3 territory)."""
    if state.get("valid", True):
        return "approve_tests"
    if state.get("gen_retry_count", 0) < MAX_GEN_RETRIES:
        return "gap_gen"
    return "drop_failing"


def drop_failing_node(state) -> dict:
    """Drop the failing drafts and continue."""
    return {"audit_log": [{"event": "dropped failing"}]}
