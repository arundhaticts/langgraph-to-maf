"""HITL nodes -- use interrupt() (must become HumanApprovalRequired, no interrupt)."""

from langgraph.types import interrupt


def is_protected(test_id: str, state) -> bool:
    """Helper that ALSO reads state -- must be uniformly rewritten to ctx."""
    return test_id in state.get("risk_areas", [])


def hitl_removals_node(state) -> dict:
    """Pause for human approval of removals."""
    decision = interrupt({"removals": state.get("approved_removals", [])})
    return {"audit_log": [{"event": "human decision", "decision": decision}]}
