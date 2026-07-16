"""
MAF Example 03 — Human-in-the-loop (HITL) with request/response.

SOURCE PATTERN (LangGraph):
    def hitl_removals(state):
        payload = build_payload(state)
        if state["run_mode"] == "automated":
            return {"approved_removals": payload["recommended"]}
        approved = _decision(interrupt(payload), payload["recommended"])
        return {"approved_removals": approved}

TARGET PATTERN (MAF):
    interrupt(payload)   ->  send a RequestInfoMessage to a RequestInfoExecutor
    resume value         ->  arrives as RequestResponse.data on the resume executor

    The workflow truly pauses and emits a request event. External code supplies a
    response and calls workflow.send_responses_streaming(responses) to resume.
    An automated fast-path approves the recommendation without pausing.

NEVER do this:
    # auto-approve as the ONLY live path, real logic in comments
    ctx.approved = payload["recommended"]   # ← wrong
    # raise HumanApprovalRequired(...)       # ← commented out = dead code
"""
from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel
from agent_framework import (
    WorkflowBuilder,
    Executor, handler, WorkflowContext,
    RequestInfoExecutor, RequestInfoMessage, RequestResponse,
)


# ---------------------------------------------------------------------------
# Message models
# ---------------------------------------------------------------------------
class RemovalCandidates(BaseModel):
    run_mode: str   # "automated" or "interactive"
    candidates: list[dict]
    recommended: list[str]


@dataclass
class RemovalApprovalRequest(RequestInfoMessage):
    """Sent to the human approver."""
    checkpoint: str = "approve_removals"
    candidates: list = None       # type: ignore[assignment]
    recommended: list = None      # type: ignore[assignment]
    note: str = "Quarantine is reversible; pinned tests are never removed."

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []
        if self.recommended is None:
            self.recommended = []


class ApprovedRemovals(BaseModel):
    approved: list[str]
    mode: str


# ---------------------------------------------------------------------------
# Shared RequestInfoExecutor instance
# ---------------------------------------------------------------------------
request_info = RequestInfoExecutor(id="request_info")


# ---------------------------------------------------------------------------
# HITL executor — the interrupt() replacement
# ---------------------------------------------------------------------------
class HitlRemovals(Executor):
    """
    Automated path  → approves the recommendation immediately, no pause.
    Interactive path → sends a RequestInfoMessage; workflow pauses here.
    """

    @handler
    async def run(self, msg: RemovalCandidates, ctx: WorkflowContext) -> None:
        if msg.run_mode == "automated":
            # Fast-path: accept the recommended removals, no human needed.
            await ctx.send_message(
                ApprovedRemovals(approved=msg.recommended, mode="automated")
            )
            return

        # Interactive path: pause and ask the human.
        await ctx.send_message(
            RemovalApprovalRequest(
                candidates=msg.candidates,
                recommended=msg.recommended,
            ),
            target_id=request_info.id,
        )


# ---------------------------------------------------------------------------
# Resume executor — processes the human's reply
# ---------------------------------------------------------------------------
class ApplyRemovalDecision(Executor):
    """
    Called when the workflow resumes after the human responds.
    reply.data  = the human's selection (list of test ids, or None to accept default).
    reply.original_request = the RemovalApprovalRequest we sent originally.
    """

    @handler
    async def run(
        self,
        reply: RequestResponse[RemovalApprovalRequest, list],
        ctx: WorkflowContext[None, ApprovedRemovals],
    ) -> None:
        human_selection = reply.data
        recommended = reply.original_request.recommended
        # None means the human accepted the recommendation.
        approved = human_selection if human_selection is not None else recommended
        await ctx.yield_output(ApprovedRemovals(approved=approved, mode="interactive"))


# ---------------------------------------------------------------------------
# Workflow assembly
# ---------------------------------------------------------------------------
hitl_removals_exec = HitlRemovals()
apply_decision_exec = ApplyRemovalDecision()

workflow = (
    WorkflowBuilder()
    .set_start_executor(hitl_removals_exec)
    # Automated branch: HitlRemovals sends ApprovedRemovals directly → done.
    # Interactive branch: HitlRemovals sends to request_info (pause) → resume → ApplyRemovalDecision.
    .add_executor(request_info)
    .add_edge(request_info, apply_decision_exec)
    .build()
)


# ---------------------------------------------------------------------------
# Host-side pause / resume loop
# ---------------------------------------------------------------------------
async def run_with_hitl(candidates: list[dict], recommended: list[str], run_mode: str):
    """
    Drives the workflow, collecting any human-approval requests and supplying
    responses. In automated mode the workflow completes in a single pass.
    """
    input_data = RemovalCandidates(
        run_mode=run_mode,
        candidates=candidates,
        recommended=recommended,
    )

    responses = None
    while True:
        if responses is None:
            events = [e async for e in workflow.run_stream(input_data)]
        else:
            events = [e async for e in workflow.send_responses_streaming(responses)]

        # Collect any pending human-approval requests from the event stream.
        pending = RequestInfoExecutor.pending_requests(events)
        if not pending:
            # No pending requests → workflow finished.
            break

        # In a real integration, present each request to the human via UI/API.
        responses = {}
        for req in pending:
            human_answer = _get_human_decision(req)   # ← your UI/API call here
            responses[req.request_id] = human_answer

    # Extract the final output from the event stream.
    return _extract_output(events)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _get_human_decision(request: RemovalApprovalRequest) -> list | None:
    """Replace with real UI/API call. Return None to accept the recommendation."""
    print(f"[HITL] Approve removals? Recommended: {request.recommended}")
    return None   # accept recommendation

def _extract_output(events):
    for event in events:
        if hasattr(event, "output"):
            return event.output
    return None
