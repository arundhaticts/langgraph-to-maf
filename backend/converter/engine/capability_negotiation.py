"""Phase 5b — Capability Negotiation.

After the IR is frozen and before code generation begins, this module checks
every construct present in the IR against the target framework's capability
matrix. For each construct it produces a `CapabilityNegotiation` record:

  DIRECT      — native idiom; the deterministic generator handles it.
  LOSSY       — emulated with approximation; surfaces as needs_review.
  UNSUPPORTED — no target idiom; a stub is emitted; surfaces as
                manual_action_required.

The output feeds:
  1. The migration report (needs_review / manual_action_required sections).
  2. The LLM Refinement Pass (Stage 11) prompt — the LLM sees which constructs
     are lossy/unsupported and tries to improve the emulated output.
  3. The readiness report — accuracy ceiling per construct is derived here.

Usage
-----
    from converter.engine.capability_negotiation import negotiate

    results = negotiate(ir, target_adapter)
    for r in results:
        if r.support == ConstructSupport.LOSSY:
            report.needs_review.append(...)
        elif r.support == ConstructSupport.UNSUPPORTED:
            report.manual_action_required.append(...)
"""

from __future__ import annotations

from typing import Callable

from converter.adapters.base import TargetAdapter
from converter.contracts import (
    CapabilityNegotiation,
    ConstructSupport,
    ConstructType,
    IR,
)


# ---------------------------------------------------------------------------
# Per-construct presence detectors
# ---------------------------------------------------------------------------

def _has_tools(ir: IR) -> bool:
    return bool(ir.tools)


def _has_state_typed(ir: IR) -> bool:
    return bool(ir.state)


def _has_state_shared(ir: IR) -> bool:
    # State shared across multiple real nodes is only meaningful for multi-node graphs.
    if not ir.state:
        return False
    real_nodes = [
        n for n in (ir.workflow.nodes if ir.workflow else [])
        if n.name not in ("START", "END", "__start__", "__end__")
    ]
    return len(real_nodes) > 1


def _has_conditional_edges(ir: IR) -> bool:
    return bool(ir.workflow and ir.workflow.conditional_edges)


def _has_loops(ir: IR) -> bool:
    return bool(
        ir.workflow
        and any(e.is_loop for e in ir.workflow.flat_edges)
    )


def _has_hitl(ir: IR) -> bool:
    return bool(ir.workflow and ir.workflow.hitl_points)


def _has_checkpointing(ir: IR) -> bool:
    return bool(ir.metadata.checkpointer)


def _has_multi_agent(ir: IR) -> bool:
    return bool(ir.agent_specs) and len(ir.agent_specs) > 1


def _has_agent_roles(ir: IR) -> bool:
    return any(
        spec.role or spec.goal or spec.backstory
        for spec in ir.agent_specs
    )


# Ordered so the report reads naturally: tools → state → control flow → HITL → agents
_PRESENCE_CHECKS: list[tuple[ConstructType, "Callable[[IR], bool]", str]] = [
    (
        ConstructType.TOOLS,
        _has_tools,
        "{count} tool(s) defined",
    ),
    (
        ConstructType.STATE_TYPED,
        _has_state_typed,
        "{count} typed state field(s)",
    ),
    (
        ConstructType.STATE_SHARED,
        _has_state_shared,
        "state shared across multiple graph nodes",
    ),
    (
        ConstructType.CONDITIONAL_EDGES,
        _has_conditional_edges,
        "{count} conditional edge(s) / router(s)",
    ),
    (
        ConstructType.LOOPS,
        _has_loops,
        "{count} back-edge(s) / loop(s)",
    ),
    (
        ConstructType.HITL,
        _has_hitl,
        "{count} human-in-the-loop point(s)",
    ),
    (
        ConstructType.CHECKPOINTING,
        _has_checkpointing,
        "checkpointer configured ({name})",
    ),
    (
        ConstructType.MULTI_AGENT,
        _has_multi_agent,
        "{count} distinct agent(s)",
    ),
    (
        ConstructType.AGENT_ROLES,
        _has_agent_roles,
        "agents with role/goal/backstory",
    ),
]


# Emulation and manual-action notes per construct × support level.
_EMULATION_NOTES: dict[ConstructType, str] = {
    ConstructType.STATE_TYPED: (
        "State is approximated as task context / session state rather than a "
        "shared TypedDict. Field names are preserved but reducers are dropped."
    ),
    ConstructType.STATE_SHARED: (
        "Shared mutable state is emulated via task output passing or agent "
        "memory. Explicit field-level reads/writes from the source graph are lost."
    ),
    ConstructType.CONDITIONAL_EDGES: (
        "Conditional edges are emulated using the target's flow/routing API "
        "(e.g. Flows @router for CrewAI). Complex multi-outcome routers may be "
        "collapsed to a simpler branching pattern."
    ),
    ConstructType.LOOPS: (
        "Loop back-edges are emulated; the exit guard is preserved where the "
        "target has a routing API, otherwise the loop body includes a while-guard."
    ),
    ConstructType.HITL: (
        "HITL is approximated by the target's available primitive "
        "(e.g. human_input=True on a Task, or a blocking tool call). "
        "True suspend/resume is not available — a fast-path auto-approve is kept."
    ),
    ConstructType.MULTI_AGENT: (
        "Multiple agents are emulated via agents-as-tools or a sequential "
        "hand-off pattern. Parallel execution and shared orchestration state "
        "between agents are not guaranteed."
    ),
    ConstructType.AGENT_ROLES: (
        "Agent role/goal/backstory are stored as system-prompt text rather than "
        "first-class framework fields. CrewAI-style role enforcement is unavailable."
    ),
}

_MANUAL_ACTION_NOTES: dict[ConstructType, str] = {
    ConstructType.CHECKPOINTING: (
        "The target framework has no native checkpointing. Add an external "
        "persistence layer (e.g. database or file storage) and call it manually "
        "at desired save-points. The generated stub shows where to insert this."
    ),
    ConstructType.CONDITIONAL_EDGES: (
        "The target framework has no deterministic branching API — the model "
        "decides routing. Wrap conditional logic in a tool the agent calls, and "
        "trust the model to invoke the correct branch. Verify output carefully."
    ),
}


# ---------------------------------------------------------------------------
# Main negotiation function
# ---------------------------------------------------------------------------

def negotiate(ir: IR, target_adapter: TargetAdapter) -> list[CapabilityNegotiation]:
    """Check every IR construct against the target capability matrix.

    Only constructs that are PRESENT in the IR are reported. DIRECT constructs
    with no issue are omitted to keep the report concise (consumers can infer
    a missing entry means DIRECT / no issue).

    Returns a list ordered by ConstructType declaration order.
    """
    matrix = target_adapter.capability_matrix()
    results: list[CapabilityNegotiation] = []

    for construct, presence_fn, detail_template in _PRESENCE_CHECKS:
        if not presence_fn(ir):
            continue

        support = matrix.get(construct, ConstructSupport.DIRECT)

        # Build a human-readable detail string.
        count_map = {
            ConstructType.TOOLS: len(ir.tools),
            ConstructType.STATE_TYPED: len(ir.state),
            ConstructType.STATE_SHARED: len(ir.state),
            ConstructType.CONDITIONAL_EDGES: len(ir.workflow.conditional_edges) if ir.workflow else 0,
            ConstructType.LOOPS: sum(1 for e in ir.workflow.flat_edges if e.is_loop) if ir.workflow else 0,
            ConstructType.HITL: len(ir.workflow.hitl_points) if ir.workflow else 0,
            ConstructType.CHECKPOINTING: 0,
            ConstructType.MULTI_AGENT: len(ir.agent_specs),
            ConstructType.AGENT_ROLES: len(ir.agent_specs),
        }
        detail = detail_template.format(
            count=count_map.get(construct, 0),
            name=ir.metadata.checkpointer or "checkpointer",
        )

        emulation_note = _EMULATION_NOTES.get(construct) if support == ConstructSupport.LOSSY else None
        manual_action = _MANUAL_ACTION_NOTES.get(construct) if support == ConstructSupport.UNSUPPORTED else None

        results.append(CapabilityNegotiation(
            construct=construct,
            support=support,
            detail=detail,
            emulation_note=emulation_note,
            manual_action=manual_action,
        ))

    return results


def negotiation_summary(results: list[CapabilityNegotiation]) -> str:
    """One-line text summary of the negotiation outcome for logs / report headers."""
    direct = sum(1 for r in results if r.support == ConstructSupport.DIRECT)
    lossy = sum(1 for r in results if r.support == ConstructSupport.LOSSY)
    unsupported = sum(1 for r in results if r.support == ConstructSupport.UNSUPPORTED)
    total = len(results)
    if total == 0:
        return "No IR constructs detected — nothing to negotiate."
    parts = [f"{direct}/{total} direct"]
    if lossy:
        parts.append(f"{lossy} lossy (emulated)")
    if unsupported:
        parts.append(f"{unsupported} unsupported (stub)")
    return ", ".join(parts) + "."
