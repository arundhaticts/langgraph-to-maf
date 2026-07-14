"""Tier 1 -- deterministic conversion rules (R-01 .. R-15).

Each function inspects the IR and emits `ConversionUnit`s for the parts that have
a clean, predictable mapping. No LLM, no README prose -- pure structure. See the
rule table in Section 12 of the build plan.

The rules record *decisions* (which rule fired, source -> target, any flag). The
actual code is emitted later by the generator (Module 7) from templates; only
Tier 3 units carry `generated_code` to be stitched in.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter
from converter.contracts import (
    IR,
    ConversionUnit,
    OrchestrationPattern,
    Tier,
)

# Orchestration patterns Tier 1 can resolve deterministically.
_TIER1_PATTERNS = {
    OrchestrationPattern.LINEAR: ("R-04", "linear spine -> sequential calls"),
    OrchestrationPattern.LOOP: ("R-06", "loop -> while with counter guard"),
    OrchestrationPattern.LOOP_WITH_EXIT: ("R-06", "loop-with-exit -> while + exit check"),
}

# State fields with these names get the reducer-specific rules (R-11/R-12).
_REDUCER_RULES = {"audit_log": "R-11", "tool_errors": "R-12"}


def tools_rules(ir: IR, adapter: TargetAdapter) -> list[ConversionUnit]:
    """R-01: each @tool -> a plugin class method."""
    units: list[ConversionUnit] = []
    for tool in ir.tools:
        units.append(
            ConversionUnit(
                rule_id="R-01",
                tier=Tier.TIER1,
                source_ref=tool.name,
                target_ref=adapter.plugin_class_name(tool.name),
                reasoning=f"@tool '{tool.name}' -> {adapter.plugin_class_name(tool.name)}",
            )
        )
    return units


def state_rules(ir: IR, adapter: TargetAdapter) -> list[ConversionUnit]:
    """R-02 / R-11 / R-12: TypedDict fields -> context dataclass fields."""
    units: list[ConversionUnit] = []
    for field in ir.state:
        if field.is_append_only:
            rule_id = _REDUCER_RULES.get(field.name, "R-02")
            reasoning = (
                f"append-only '{field.name}': reducer removed, direct .append() used"
            )
        else:
            rule_id = "R-02"
            reasoning = f"state field '{field.name}' ({field.type}) -> dataclass field"
        units.append(
            ConversionUnit(
                rule_id=rule_id,
                tier=Tier.TIER1,
                source_ref=field.name,
                target_ref=f"{adapter.context_class_name}.{field.name}",
                reasoning=reasoning,
            )
        )
    return units


def config_rules(ir: IR) -> list[ConversionUnit]:
    """R-14: config constants carried over; R-10: LLM invocation adapted."""
    units: list[ConversionUnit] = []
    for name in ir.config.constants:
        units.append(
            ConversionUnit(
                rule_id="R-14",
                tier=Tier.TIER1,
                source_ref=name,
                target_ref=name,
                reasoning="config constant carried over unchanged",
            )
        )
    if ir.config.llm_kwargs:
        units.append(
            ConversionUnit(
                rule_id="R-10",
                tier=Tier.TIER1,
                source_ref="llm",
                target_ref="chat client",
                reasoning="LLM instantiation adapted to target invocation wrapper",
            )
        )
    return units


def checkpointer_rule(ir: IR) -> list[ConversionUnit]:
    """R-15: MemorySaver / SqliteSaver -> persistence TODO stub (flagged).

    The frozen IR does not carry a dedicated checkpointer field, so detection is
    a best-effort scan of config constants for a known saver reference.
    """
    savers = ("MemorySaver", "SqliteSaver", "checkpointer", "checkpoint")
    for name, value in ir.config.constants.items():
        haystack = f"{name} {value}".lower()
        if any(s.lower() in haystack for s in savers):
            return [
                ConversionUnit(
                    rule_id="R-15",
                    tier=Tier.TIER1,
                    source_ref=name,
                    target_ref="# TODO: wire persistence layer",
                    reasoning="checkpointer removed",
                    manual_action=(
                        "Add persistence layer for paused runs if deploying over HTTP."
                    ),
                )
            ]
    return []


def workflow_tier1(ir: IR) -> ConversionUnit | None:
    """R-04/R-05/R-06/R-07: resolve the orchestration if Tier 1 can.

    Returns a unit if the pattern is deterministically resolvable, else None
    (the engine then tries Tier 2, then Tier 3).
    """
    if not ir.workflow:
        return None
    pattern = ir.workflow.pattern

    # Simple 2-outcome branch -> if/elif (R-05).
    if pattern is OrchestrationPattern.BRANCH:
        max_outcomes = max(
            (len(c.outcomes) for c in ir.workflow.conditional_edges), default=0
        )
        if max_outcomes <= 2:
            return ConversionUnit(
                rule_id="R-05",
                tier=Tier.TIER1,
                source_ref="workflow",
                target_ref="orchestrator (if/elif)",
                reasoning="simple 2-outcome branch -> if/elif block",
            )
        return None  # 3+ outcomes -> not Tier 1

    if pattern in _TIER1_PATTERNS:
        rule_id, reasoning = _TIER1_PATTERNS[pattern]
        return ConversionUnit(
            rule_id=rule_id,
            tier=Tier.TIER1,
            source_ref="workflow",
            target_ref="orchestrator",
            reasoning=reasoning,
        )

    return None  # AGENT_DRIVEN and anything else -> Tier 2/3
