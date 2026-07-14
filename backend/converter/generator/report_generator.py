"""Module 9 -- Migration report generator.

Builds the `MigrationReport` (frozen contract) from the conversion decisions and
the generation result, then writes `MIGRATION_REPORT.md`. Three sections
(Section 14 of the build plan):

- **Auto-converted**       -- one line per Tier 1/2 rule that fired cleanly
- **Needs review**         -- Tier 3 decisions below the confidence threshold
- **Manual action required** -- R-08/R-15 stubs, unresolved patterns, and any
                                `ast.parse()` failures from the generator

The `generated` date is passed in (never computed here) so runs are reproducible.
"""

from __future__ import annotations

import os
from typing import Optional

from converter.config import Config
from converter.contracts import (
    IR,
    ConversionResult,
    MigrationReport,
    ReportEntry,
    Tier,
)
from converter.generator.code_generator import GenerationResult


def build_report(
    ir: IR,
    conversion: ConversionResult,
    generation: Optional[GenerationResult] = None,
    config: Config | None = None,
    agent_name: str | None = None,
    generated_date: str = "",
) -> MigrationReport:
    """Assemble the three-section migration report."""
    config = config or Config()
    resolved_name = agent_name or ir.metadata.description or "Converted Agent"
    report = MigrationReport(agent_name=resolved_name, generated=generated_date)

    for unit in conversion.units:
        arrow_target = unit.target_ref or "(unresolved)"
        rule = unit.rule_id or "R-TIER3"

        # Manual action (R-08 stubs, R-15 stubs, unresolved workflow).
        if unit.manual_action:
            detail = unit.manual_action
            # Carry the original source logic so the follow-up (Claude Code /
            # LLM) has full context to implement it after review.
            if unit.reasoning:
                detail = f"{detail}\n  {unit.reasoning}"
            report.manual_action_required.append(
                ReportEntry(
                    text=f"[{rule}] {unit.source_ref} -> {arrow_target}",
                    detail=detail,
                )
            )
            continue

        # Tier 3 below threshold -> needs review.
        if unit.needs_review:
            score = f"{unit.confidence:.2f}" if unit.confidence is not None else "n/a"
            report.needs_review.append(
                ReportEntry(
                    text=f"[{rule}] {unit.source_ref} -> {arrow_target} "
                    f"(confidence: {score})",
                    detail=unit.reasoning,
                )
            )
            continue

        # Everything else that resolved (Tier 1/2, and confident Tier 3) is auto.
        if unit.tier is not Tier.UNRESOLVED:
            report.auto_converted.append(
                ReportEntry(
                    text=f"[{rule}] {unit.source_ref} -> {arrow_target}",
                    detail=unit.reasoning,
                )
            )

    # ast.parse() failures from the generator -> manual action.
    if generation is not None:
        for rel_path in generation.syntax_errors:
            report.manual_action_required.append(
                ReportEntry(
                    text=f"[SYNTAX] {rel_path} failed ast.parse()",
                    detail="Generated file is not valid Python; review the banner at the top.",
                )
            )
        # Target-framework validation warnings -> needs review.
        for warning in generation.validation_warnings:
            report.needs_review.append(ReportEntry(text="[VALIDATION] " + warning))

    return report


def render_report(report: MigrationReport) -> str:
    """Render the `MigrationReport` as markdown."""
    lines: list[str] = [f"# Migration Report - {report.agent_name}"]
    if report.generated:
        lines.append(f"Generated: {report.generated}")
    lines.append("")

    def section(header: str, entries: list[ReportEntry], empty: str) -> None:
        lines.append(f"## {header}")
        if not entries:
            lines.append(empty)
        else:
            for entry in entries:
                lines.append(entry.text)
                if entry.detail:
                    lines.append(f"  {entry.detail}")
        lines.append("")

    section("Auto-converted", report.auto_converted, "_Nothing auto-converted._")
    section("Needs review", report.needs_review, "_Nothing flagged for review._")
    section(
        "Manual action required",
        report.manual_action_required,
        "_No manual action required._",
    )

    return "\n".join(lines).rstrip() + "\n"


def write_report(report: MigrationReport, output_path: str) -> str:
    """Write MIGRATION_REPORT.md into the output folder; returns the path."""
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, "MIGRATION_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(report))
    return path
