"""Example MAF orchestration: generation/validation loop with a router exit.

Reference output the Tier 3 LLM should emulate for `loop_with_exit` patterns.
"""
from __future__ import annotations

from agent_context import AgentContext
from config import COVERAGE_FLOOR, MAX_GEN_RETRIES


def generate(ctx: AgentContext) -> AgentContext:
    ctx.coverage = min(1.0, (ctx.coverage or 0.0) + 0.1)
    ctx.gen_retry_count = (ctx.gen_retry_count or 0) + 1
    ctx.audit_log.append(f"generated tests, attempt {ctx.gen_retry_count}")
    return ctx


def gate(ctx: AgentContext) -> AgentContext:
    ctx.audit_log.append(f"coverage={ctx.coverage}")
    return ctx


def route(ctx: AgentContext) -> str:
    if ctx.coverage >= COVERAGE_FLOOR:
        return "done"
    if ctx.gen_retry_count >= MAX_GEN_RETRIES:
        return "done"
    return "revise"


def run(ctx: AgentContext) -> AgentContext:
    guard = 0
    while guard < MAX_GEN_RETRIES:
        ctx = generate(ctx)
        ctx = gate(ctx)
        if route(ctx) == "done":
            break
        guard += 1
    return ctx
