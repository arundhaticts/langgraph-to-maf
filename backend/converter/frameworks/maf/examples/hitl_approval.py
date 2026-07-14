"""Example MAF HITL approval node.

Reference output the Tier 3 LLM should emulate when converting an
`interrupt()`-based human-in-the-loop step. `HumanApprovalRequired` is defined in
the generated `orchestrator.py`.
"""
from __future__ import annotations

from agent_context import AgentContext


def hitl_approve(ctx: AgentContext) -> AgentContext:
    # The source paused here with interrupt(); model it as an approval boundary.
    if not getattr(ctx, "approved", False):
        raise HumanApprovalRequired({"coverage": ctx.coverage})  # noqa: F821
    ctx.audit_log.append("human approved the change")
    return ctx
