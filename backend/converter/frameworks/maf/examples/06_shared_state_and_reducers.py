"""
MAF Example 06 — Shared state and append-reducer replacement.

SOURCE PATTERN (LangGraph):
    class MyState(TypedDict):
        audit_log: Annotated[list[dict], add]   # append reducer
        tool_errors: Annotated[list[dict], add]  # append reducer
        project_id: str

    def node_a(state) -> dict:
        return {"audit_log": [{"node": "a", "event": "done"}]}  # partial dict

TARGET PATTERN (MAF):
    - TypedDict graph state  -> typed pydantic message per edge
    - Annotated[list, add]   -> plain list field + .append()
    - partial dict return    -> typed message via ctx.send_message(...)
    - cross-cutting state    -> ctx.set_shared_state / ctx.get_shared_state

RULE: Every executor emits the SAME message type the next executor expects.
      Never return a bare partial dict.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from agent_framework import WorkflowBuilder, executor, WorkflowContext


# ---------------------------------------------------------------------------
# Message models — replace TypedDict
#
# Append-reducer fields (Annotated[list, add]) become plain list fields.
# Each executor appends to the list in the message it received, then
# forwards the updated message. No operator.add, no Annotated wrapper.
# ---------------------------------------------------------------------------
class PipelineState(BaseModel):
    project_id: str
    suite: list[dict] = Field(default_factory=list)
    # These were Annotated[list, add] reducers in LangGraph:
    audit_log: list[dict] = Field(default_factory=list)
    tool_errors: list[dict] = Field(default_factory=list)
    # Analysis results accumulated as the pipeline progresses:
    coverage_gaps: list[dict] = Field(default_factory=list)
    flakiness_flags: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper — build a standard audit entry
# ---------------------------------------------------------------------------
def _audit(node: str, event: str, **kwargs) -> dict:
    return {"node": node, "event": event, **kwargs}


# ---------------------------------------------------------------------------
# Executors — each receives PipelineState, appends its entries, forwards on
# ---------------------------------------------------------------------------
@executor(id="intake")
async def intake(msg: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
    suite = _load_suite(msg.project_id)

    # Append to the list in-place, then forward the whole updated message.
    # This replaces the LangGraph pattern of returning {"audit_log": [entry]}.
    msg.suite = suite
    msg.audit_log.append(_audit("intake", "loaded", count=len(suite)))

    await ctx.send_message(msg)


@executor(id="coverage_analysis")
async def coverage_analysis(msg: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
    gaps = _find_gaps(msg.suite)

    msg.coverage_gaps = gaps
    msg.audit_log.append(_audit("coverage", "analysed", gaps=len(gaps)))

    if not gaps:
        msg.tool_errors.append({"source": "coverage", "error": "no criteria file found", "action": "skipped"})

    await ctx.send_message(msg)


@executor(id="flakiness_analysis")
async def flakiness_analysis(msg: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
    flags = _detect_flaky(msg.suite)

    msg.flakiness_flags = flags
    msg.audit_log.append(_audit("flakiness", "flagged", count=len(flags)))

    await ctx.send_message(msg)


@executor(id="report")
async def report(msg: PipelineState, ctx: WorkflowContext[None, PipelineState]) -> None:
    msg.audit_log.append(_audit("report", "complete",
                                gaps=len(msg.coverage_gaps),
                                flaky=len(msg.flakiness_flags),
                                errors=len(msg.tool_errors)))
    # Yield the final accumulated state as the workflow output.
    await ctx.yield_output(msg)


# ---------------------------------------------------------------------------
# Shared state — for values that truly need cross-executor access
# (not needed for this linear pipeline, but shown for reference)
# ---------------------------------------------------------------------------
@executor(id="example_shared_state_writer")
async def _writer(msg: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
    # Store a value that any downstream executor can read by key.
    await ctx.set_shared_state("project_id", msg.project_id)
    await ctx.send_message(msg)


@executor(id="example_shared_state_reader")
async def _reader(msg: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
    project_id = await ctx.get_shared_state("project_id")
    msg.audit_log.append(_audit("reader", "got_project_id", project_id=project_id))
    await ctx.send_message(msg)


# ---------------------------------------------------------------------------
# Workflow assembly
# ---------------------------------------------------------------------------
workflow = (
    WorkflowBuilder()
    .set_start_executor(intake)
    .add_edge(intake, coverage_analysis)
    .add_edge(coverage_analysis, flakiness_analysis)
    .add_edge(flakiness_analysis, report)
    .build()
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    result = await workflow.run(PipelineState(project_id="proj-001"))
    print("audit_log entries:", len(result.audit_log))
    print("tool_errors:", result.tool_errors)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _load_suite(project_id: str) -> list[dict]:
    return [{"id": "test_login"}, {"id": "test_logout"}]

def _find_gaps(suite: list[dict]) -> list[dict]:
    return []

def _detect_flaky(suite: list[dict]) -> list[dict]:
    return []
