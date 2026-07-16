"""
MAF Example 05 — Checkpointing (durable pause / resume).

SOURCE PATTERN (LangGraph):
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "run-123"}}
    result = app.invoke(input_data, config=config)

TARGET PATTERN (MAF):
    from agent_framework import FileCheckpointStorage
    storage = FileCheckpointStorage(dir_path="./.checkpoints")
    workflow = WorkflowBuilder()...with_checkpointing(storage).build()

    # Resume from a saved checkpoint:
    await workflow.run_from_checkpoint(checkpoint_id, checkpoint_storage=storage)

WHY IT MATTERS:
    Checkpointing is essential for any workflow that includes HITL pauses —
    the workflow may be interrupted for minutes or hours while a human reviews.
    Without checkpointing a resumed run starts from scratch.
"""
from __future__ import annotations

from pydantic import BaseModel
from agent_framework import (
    WorkflowBuilder,
    executor, WorkflowContext,
    FileCheckpointStorage,
    RequestInfoExecutor, RequestInfoMessage, RequestResponse,
)
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Message models
# ---------------------------------------------------------------------------
class WorkInput(BaseModel):
    project_id: str
    suite_path: str


class StageOneResult(BaseModel):
    project_id: str
    tests: list[dict]


@dataclass
class ReviewRequest(RequestInfoMessage):
    checkpoint: str = "review_tests"
    tests: list = None   # type: ignore[assignment]

    def __post_init__(self):
        if self.tests is None:
            self.tests = []


class ReviewedResult(BaseModel):
    approved_tests: list[dict]
    project_id: str


class FinalReport(BaseModel):
    project_id: str
    summary: dict


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
@executor(id="stage_one")
async def stage_one(msg: WorkInput, ctx: WorkflowContext[StageOneResult]) -> None:
    tests = _load_tests(msg.suite_path)
    await ctx.send_message(StageOneResult(project_id=msg.project_id, tests=tests))


request_info = RequestInfoExecutor(id="request_info")


@executor(id="request_review")
async def request_review(msg: StageOneResult, ctx: WorkflowContext) -> None:
    await ctx.send_message(
        ReviewRequest(tests=msg.tests),
        target_id=request_info.id,
    )


@executor(id="apply_review")
async def apply_review(
    reply: RequestResponse[ReviewRequest, list],
    ctx: WorkflowContext[ReviewedResult],
) -> None:
    approved = reply.data if reply.data is not None else reply.original_request.tests
    # Grab project_id from shared state (set by stage_one or an upstream executor).
    project_id = await ctx.get_shared_state("project_id") or ""
    await ctx.send_message(ReviewedResult(approved_tests=approved, project_id=project_id))


@executor(id="finalise")
async def finalise(msg: ReviewedResult, ctx: WorkflowContext[None, FinalReport]) -> None:
    await ctx.yield_output(
        FinalReport(project_id=msg.project_id, summary={"approved": len(msg.approved_tests)})
    )


# ---------------------------------------------------------------------------
# Workflow with checkpointing enabled
# ---------------------------------------------------------------------------
storage = FileCheckpointStorage(dir_path="./.checkpoints")

workflow = (
    WorkflowBuilder()
    .set_start_executor(stage_one)
    .add_edge(stage_one, request_review)
    .add_executor(request_info)
    .add_edge(request_info, apply_review)
    .add_edge(apply_review, finalise)
    .with_checkpointing(storage)        # ← enables durable pause/resume
    .build()
)


# ---------------------------------------------------------------------------
# Orchestration — first run, pause, resume
# ---------------------------------------------------------------------------
async def run_with_checkpointing(project_id: str, suite_path: str):
    """
    Runs the workflow, handles any HITL pauses, and returns the final output.
    Checkpointing means the workflow survives a process restart between pause and resume.
    """
    input_data = WorkInput(project_id=project_id, suite_path=suite_path)

    # First pass — workflow runs until it needs human input, then pauses.
    events = [e async for e in workflow.run_stream(input_data)]
    pending = RequestInfoExecutor.pending_requests(events)

    # Save the checkpoint ID so we can resume even after a restart.
    checkpoint_id = _extract_checkpoint_id(events)
    _persist_checkpoint_id(checkpoint_id)

    while pending:
        # Hand the requests to a human (UI, API, email, etc.).
        responses = {}
        for req in pending:
            responses[req.request_id] = _get_human_response(req)

        # Resume from the saved checkpoint with the human's responses.
        events = [
            e async for e in workflow.send_responses_streaming(responses)
        ]
        pending = RequestInfoExecutor.pending_requests(events)

    return _extract_output(events)


async def resume_from_saved_checkpoint(checkpoint_id: str):
    """
    Resume a workflow that was paused (e.g. after a process restart).
    checkpoint_id was saved during the first run.
    """
    events = [
        e async for e in workflow.run_from_checkpoint(
            checkpoint_id, checkpoint_storage=storage
        )
    ]
    return _extract_output(events)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _load_tests(path: str) -> list[dict]:
    return [{"id": "test_login", "code": "def test_login(): pass"}]

def _get_human_response(req) -> list | None:
    print(f"[HITL] Review {len(req.tests)} tests. Press enter to approve all.")
    input()
    return None   # None = accept all

def _extract_checkpoint_id(events) -> str:
    for e in events:
        if hasattr(e, "checkpoint_id"):
            return e.checkpoint_id
    return ""

def _persist_checkpoint_id(checkpoint_id: str):
    import pathlib
    pathlib.Path(".last_checkpoint").write_text(checkpoint_id)

def _extract_output(events):
    for e in events:
        if hasattr(e, "output"):
            return e.output
    return None
