"""
MAF Example 01 — Basic linear workflow.

SOURCE PATTERN (LangGraph):
    graph = StateGraph(MyState)
    graph.add_node("intake", intake_fn)
    graph.add_node("analyse", analyse_fn)
    graph.add_node("report", report_fn)
    graph.add_edge("intake", "analyse")
    graph.add_edge("analyse", "report")
    graph.set_entry_point("intake")
    app = graph.compile()
    result = app.invoke({"tests": [...]})

TARGET PATTERN (MAF):
    Nodes  -> Executors (class or @executor function)
    State  -> Typed pydantic messages passed along edges
    invoke -> await workflow.run(InputModel(...))
"""
from __future__ import annotations

from pydantic import BaseModel
from agent_framework import WorkflowBuilder, executor, WorkflowContext


# ---------------------------------------------------------------------------
# Message models (replace TypedDict state)
# ---------------------------------------------------------------------------
class RawInput(BaseModel):
    tests: list[dict]


class NormalisedSuite(BaseModel):
    tests: list[dict]
    conventions: dict = {}


class AnalysisResult(BaseModel):
    suite: list[dict]
    coverage_gaps: list[dict]
    score: float


class Report(BaseModel):
    summary: dict
    gaps: list[dict]


# ---------------------------------------------------------------------------
# Executors (replace node functions)
# Each receives a typed message; emits typed message to the next executor.
# Never returns a bare dict.
# ---------------------------------------------------------------------------
@executor(id="intake")
async def intake(msg: RawInput, ctx: WorkflowContext[NormalisedSuite]) -> None:
    normalised = [_normalise(t) for t in msg.tests]
    conventions = _detect_conventions(normalised)
    await ctx.send_message(NormalisedSuite(tests=normalised, conventions=conventions))


@executor(id="analyse")
async def analyse(msg: NormalisedSuite, ctx: WorkflowContext[AnalysisResult]) -> None:
    gaps = _find_gaps(msg.tests)
    score = _score(msg.tests, gaps)
    await ctx.send_message(AnalysisResult(suite=msg.tests, coverage_gaps=gaps, score=score))


@executor(id="report")
async def report(msg: AnalysisResult, ctx: WorkflowContext[None, Report]) -> None:
    # Terminal executor — yield_output instead of send_message.
    await ctx.yield_output(Report(summary={"score": msg.score}, gaps=msg.coverage_gaps))


# ---------------------------------------------------------------------------
# Workflow assembly (replace StateGraph + compile)
# ---------------------------------------------------------------------------
workflow = (
    WorkflowBuilder()
    .set_start_executor(intake)
    .add_edge(intake, analyse)
    .add_edge(analyse, report)
    .build()
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    result = await workflow.run(RawInput(tests=[{"id": "test_login", "name": "test_login"}]))
    print(result)


# ---------------------------------------------------------------------------
# Stubs (replace with real implementations)
# ---------------------------------------------------------------------------
def _normalise(t: dict) -> dict:
    return {**t, "entities": []}

def _detect_conventions(tests: list[dict]) -> dict:
    return {"framework": "pytest"}

def _find_gaps(tests: list[dict]) -> list[dict]:
    return []

def _score(tests: list[dict], gaps: list[dict]) -> float:
    return 1.0 - len(gaps) / max(len(tests), 1)
