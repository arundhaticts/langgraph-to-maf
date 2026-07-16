"""
MAF Example 02 — Conditional edges (router functions).

SOURCE PATTERN (LangGraph):
    def route_after_validation(state) -> str:
        if state["validation_passed"]:
            return "approve"
        if state["retries"] >= MAX:
            return "drop"
        return "gap_gen"

    graph.add_conditional_edges("validation", route_after_validation, {
        "approve":  "approve_tests",
        "drop":     "drop_failing",
        "gap_gen":  "gap_gen",
    })

TARGET PATTERN (MAF):
    Router function -> one .add_edge(..., condition=lambda m: ...) per branch.
    The lambda is a predicate on the message the source executor emitted.
    Every branch gets its own edge call — never a single routing dict.
"""
from __future__ import annotations

from pydantic import BaseModel
from agent_framework import WorkflowBuilder, executor, WorkflowContext

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Message models
# ---------------------------------------------------------------------------
class GeneratedTests(BaseModel):
    tests: list[dict]
    retries: int = 0


class ValidationResult(BaseModel):
    tests: list[dict]
    passed: bool
    retries: int
    errors: list[str] = []


class ApprovedTests(BaseModel):
    tests: list[dict]


class DroppedTests(BaseModel):
    tests: list[dict]
    reason: str = "exceeded retry budget"


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
@executor(id="gap_gen")
async def gap_gen(msg: GeneratedTests, ctx: WorkflowContext[GeneratedTests]) -> None:
    drafts = _draft_tests(msg.tests)
    await ctx.send_message(GeneratedTests(tests=drafts, retries=msg.retries + 1))


@executor(id="validation")
async def validation(msg: GeneratedTests, ctx: WorkflowContext[ValidationResult]) -> None:
    results, all_ok = [], True
    for t in msg.tests:
        ok, err = _validate(t)
        results.append({**t, "valid": ok, "error": err})
        if not ok:
            all_ok = False
    await ctx.send_message(
        ValidationResult(tests=results, passed=all_ok, retries=msg.retries)
    )


@executor(id="approve_tests")
async def approve_tests(msg: ValidationResult, ctx: WorkflowContext[None, ApprovedTests]) -> None:
    await ctx.yield_output(ApprovedTests(tests=[t for t in msg.tests if t["valid"]]))


@executor(id="drop_failing")
async def drop_failing(msg: ValidationResult, ctx: WorkflowContext[None, DroppedTests]) -> None:
    await ctx.yield_output(DroppedTests(tests=msg.tests))


# ---------------------------------------------------------------------------
# Workflow — conditional edges replace the router function
#
# LangGraph router:
#   if passed -> "approve"
#   elif retries >= MAX -> "drop"
#   else -> "gap_gen"
#
# MAF translation: one edge per branch, condition is a lambda on the message.
# ---------------------------------------------------------------------------
workflow = (
    WorkflowBuilder()
    .set_start_executor(gap_gen)
    .add_edge(gap_gen, validation)
    # Branch 1: validation passed -> approve
    .add_edge(
        validation, approve_tests,
        condition=lambda m: m.passed,
    )
    # Branch 2: failed AND retry budget exhausted -> drop
    .add_edge(
        validation, drop_failing,
        condition=lambda m: not m.passed and m.retries >= MAX_RETRIES,
    )
    # Branch 3: failed AND retries remain -> loop back to gap_gen
    .add_edge(
        validation, gap_gen,
        condition=lambda m: not m.passed and m.retries < MAX_RETRIES,
    )
    .build()
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _draft_tests(existing: list[dict]) -> list[dict]:
    return existing or [{"id": "test_stub", "code": "def test_stub(): pass"}]

def _validate(test: dict) -> tuple[bool, str]:
    try:
        compile(test.get("code", ""), "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)
