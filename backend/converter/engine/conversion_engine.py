"""Module 6 -- Conversion engine: Tier 1/2/3 orchestration.

Consumes the IR and produces a `ConversionResult` (a list of `ConversionUnit`
decisions). The tiers run in order and only escalate when the cheaper tier
cannot resolve a construct:

    Tier 1 (structure)  ->  Tier 2 (README prose)  ->  Tier 3 (LLM)

Tools, state, config, HITL, and checkpointer always resolve in Tier 1. Only the
*orchestration shape* can escalate. In Deterministic mode (Approach 1) the LLM
fallback is off, so an unresolved shape becomes a manual-action unit instead of
an LLM call -- same engine, different `Config`.
"""

from __future__ import annotations

from typing import Callable, Optional

from converter.adapters import get_target_adapter
from converter.adapters.base import TargetAdapter
from converter.config import Config
from converter.contracts import (
    IR,
    ConversionResult,
    ConversionUnit,
    NodeRole,
    OrchestrationPattern,
    Tier,
    Tier3Result,
)
from converter.engine import tier1_rules
from converter.engine.tier2_readme import classify_from_readme
from converter.engine.tier3_llm import resolve_hitl, resolve_with_llm

# Injectable Tier 3 resolvers (real Gemini in prod, fakes in tests).
Tier3Resolver = Callable[..., Optional[Tier3Result]]
HitlResolver = Callable[..., Optional[Tier3Result]]

# Patterns Tier 1 handles directly (mirrors tier1_rules).
_TIER1_RESOLVABLE = {
    OrchestrationPattern.LINEAR,
    OrchestrationPattern.LOOP,
    OrchestrationPattern.LOOP_WITH_EXIT,
}


def _resolve_workflow(
    ir: IR,
    config: Config,
    tier3_resolver: Tier3Resolver,
) -> ConversionUnit:
    """Resolve the orchestration through Tier 1 -> 2 -> 3."""
    workflow = ir.workflow

    # Tier 1: pure structure.
    unit = tier1_rules.workflow_tier1(ir)
    if unit is not None:
        return unit

    # Tier 2: keyword-match the verbatim README workflow prose.
    if workflow is not None:
        readme_pattern = classify_from_readme(workflow.readme_description)
        if readme_pattern in _TIER1_RESOLVABLE or (
            readme_pattern is OrchestrationPattern.BRANCH
        ):
            return ConversionUnit(
                rule_id="R-TIER2",
                tier=Tier.TIER2,
                source_ref="workflow",
                target_ref=f"orchestrator ({readme_pattern.value})",
                reasoning=(
                    f"structure ambiguous; README prose classified as "
                    f"{readme_pattern.value}"
                ),
            )

    # Tier 3: LLM with framework docs as context.
    result = tier3_resolver(ir, workflow, config) if workflow is not None else None
    if result is not None:
        below = result.confidence < config.tier3_confidence_threshold
        return ConversionUnit(
            rule_id="R-TIER3",
            tier=Tier.TIER3,
            source_ref="workflow",
            target_ref=f"orchestrator ({result.pattern})",
            generated_code=result.generated_code,
            reasoning=result.reasoning,
            confidence=result.confidence,
            needs_review=below,
        )

    # Nothing resolved it (deterministic mode, or LLM unavailable/failed).
    return ConversionUnit(
        rule_id=None,
        tier=Tier.UNRESOLVED,
        source_ref="workflow",
        target_ref=None,
        reasoning=(
            f"orchestration pattern '{workflow.pattern.value if workflow else 'unknown'}' "
            "could not be resolved deterministically and the LLM fallback was "
            "unavailable"
        ),
        manual_action="Convert the orchestration by hand.",
    )


def _resolve_hitl_nodes(
    ir: IR, config: Config, hitl_resolver: HitlResolver
) -> list[ConversionUnit]:
    """R-08: HITL nodes -> Tier 3 (Gemini) approval flow, else flagged stub.

    Either way the migration report carries the original logic so the flow can be
    completed after review.
    """
    units: list[ConversionUnit] = []
    if not ir.workflow:
        return units

    for node in ir.workflow.nodes:
        if node.role is not NodeRole.HITL:
            continue
        source = ir.functions.get(node.target_callable or "") or ir.functions.get(node.name)
        original = (source.body if source and source.body else "").strip()

        result = hitl_resolver(ir, node.name, source, config)
        if result is not None and result.generated_code:
            units.append(
                ConversionUnit(
                    rule_id="R-08",
                    tier=Tier.TIER3,
                    source_ref=node.name,
                    target_ref="HITL approval flow (Gemini)",
                    generated_code=result.generated_code,
                    reasoning=result.reasoning
                    or "Tier 3 generated the approval flow; review before deploying.",
                    confidence=result.confidence,
                    needs_review=True,  # approval flows always warrant human review
                )
            )
        else:
            units.append(
                ConversionUnit(
                    rule_id="R-08",
                    tier=Tier.TIER1,
                    source_ref=node.name,
                    target_ref="HumanApprovalRequired stub",
                    reasoning=(
                        f"HITL interrupt() has no direct equivalent. Original logic:\n"
                        f"{original}" if original else
                        "HITL interrupt() has no direct equivalent."
                    ),
                    manual_action="Implement the approval flow (Tier 3 unavailable).",
                )
            )
    return units


def convert(
    ir: IR,
    config: Config | None = None,
    adapter: TargetAdapter | None = None,
    tier3_resolver: Tier3Resolver | None = None,
    hitl_resolver: HitlResolver | None = None,
) -> ConversionResult:
    """Run the full conversion, returning every decision as a `ConversionUnit`."""
    config = config or Config()
    adapter = adapter or get_target_adapter(ir.metadata.target_framework or "maf")
    tier3_resolver = tier3_resolver or resolve_with_llm
    hitl_resolver = hitl_resolver or resolve_hitl

    units: list[ConversionUnit] = []
    units.extend(tier1_rules.tools_rules(ir, adapter))
    units.extend(tier1_rules.state_rules(ir, adapter))
    units.extend(tier1_rules.config_rules(ir))
    units.extend(_resolve_hitl_nodes(ir, config, hitl_resolver))
    units.extend(tier1_rules.checkpointer_rule(ir))
    units.append(_resolve_workflow(ir, config, tier3_resolver))

    return ConversionResult(units=units)
