"""Approach 3 (Hybrid) pipeline.

Deterministic for clean mappings, LLM only for the hard parts. This is the
implemented approach and it now runs end-to-end: scan -> parse README -> extract
components -> build IR -> convert (Tier 1/2/3) -> generate output repo -> write
README -> write migration report.

Approach 1 (Deterministic) will reuse this class with `allow_llm_fallback=False`;
Approach 2 (Full LLM) is a separate `ConversionPipeline` that bypasses the
parser/IR. Both produce the same `MigrationReport` contract, so nothing
downstream of the pipeline changes -- see `converter/main.py::PIPELINE_REGISTRY`.
"""

from __future__ import annotations

import os
from datetime import date

from converter.config import Config
from converter.contracts import MigrationReport
from converter.engine import convert
from converter.extractor import extract_components
from converter.generator import (
    build_readme,
    build_report,
    generate_from_paths,
    generate_readiness_report,
    validate_metrics,
    write_docs,
    write_readiness_metrics,
    write_readiness_report,
    write_readme,
    write_report,
)
from converter.engine.capability_negotiation import negotiate, negotiation_summary
from converter.generator.llm_refinement import run_llm_refinement, write_refinement_log
from converter.ir import build_ir, validate_ir, write_ir_json
from converter.adapters import get_target_adapter
from converter.parser import parse_readme_file
from converter.pipeline.base import ConversionPipeline
from converter.scanner import scan_repo
from converter.verify import verify_output, verify_runnable, write_acceptance


class HybridPipeline(ConversionPipeline):
    """Full input-folder -> output-folder conversion, hybrid strategy."""

    def __init__(self, config: Config, allow_llm_fallback: bool | None = None) -> None:
        super().__init__(config)
        # Approach 1 flips this off to run the deterministic-only variant.
        self.allow_llm_fallback = (
            config.allow_llm_fallback
            if allow_llm_fallback is None
            else allow_llm_fallback
        )

    def run(self, input_path: str, output_path: str) -> MigrationReport:
        # Stage 1 -- Module 1: scan + validate. Raises ScannerError on a bad repo.
        manifest = scan_repo(input_path, self.config)

        # Stage 2 -- Module 3: README parsing (verbatim workflow kept for Tier 2).
        readme = parse_readme_file(
            os.path.join(manifest.input_root, manifest.readme_path), self.config
        )

        # Stage 3 -- Module 4: parse every .py and consolidate; assigns file_action.
        inventory = extract_components(manifest, readme, self.config)

        # Stage 4 -- Module 5: assemble the IR and write the ir.json checkpoint.
        ir = build_ir(
            inventory, readme, manifest, self.config,
            target_framework=self.config.target_framework,
        )
        # Honour an explicit --source override (else keep auto-detected).
        if self.config.source_framework:
            ir.metadata.source_framework = self.config.source_framework
        write_ir_json(ir, os.path.join(os.getcwd(), "ir.json"))

        # Stage 4.5 -- Phase 2: validate the IR BEFORE generating anything.
        # Findings are non-fatal; they are surfaced to the migration report so a
        # human sees source inconsistencies instead of silent broken output.
        ir_issues = validate_ir(ir)

        # Stage 4.6 -- Phase 5b: capability negotiation.
        # Check every IR construct against the target framework's capability
        # matrix. LOSSY → needs_review; UNSUPPORTED → manual_action_required.
        # This runs before code generation so the generator can tag stubs.
        target_adapter_obj = get_target_adapter(self.config.target_framework or "maf")
        cap_results = negotiate(ir, target_adapter_obj)
        cap_summary = negotiation_summary(cap_results)
        for cap in cap_results:
            from converter.contracts import ConstructSupport
            if cap.support == ConstructSupport.LOSSY:
                ir_issues.append(
                    f"[CAP-LOSSY] {cap.construct.value}: {cap.detail}. "
                    + (cap.emulation_note or "Emulated.")
                )
            elif cap.support == ConstructSupport.UNSUPPORTED:
                ir_issues.append(
                    f"[CAP-UNSUPPORTED] {cap.construct.value}: {cap.detail}. "
                    + (cap.manual_action or "Manual implementation required.")
                )

        # Stage 5 -- Module 6: Tier 1/2/3 conversion engine.
        conversion = convert(ir, self.config)

        # Stage 6 -- Module 7: render + write the output repo.
        generation = generate_from_paths(
            ir, conversion, manifest.input_root, output_path, self.config
        )
        # Route IR-validation findings into the report's "needs review" section.
        for issue in ir_issues:
            generation.validation_warnings.append(f"[IR] {issue}")

        # Stage 6.5 -- Phase 11: acceptance gate. Diff the emitted package against
        # the IR + scan for residue; write ACCEPTANCE.md and surface any failure.
        acceptance = verify_output(ir, generation)
        # Optionally run the subprocess runnable-checks (stub imports + build_workflow).
        if self.config.validate_output:
            for check in verify_runnable(
                generation.output_root,
                target=self.config.target_framework or "maf",
            ):
                acceptance.add(*check)
        write_acceptance(acceptance, generation.output_root)
        for issue in acceptance.issues():
            generation.validation_warnings.append(f"[ACCEPTANCE] {issue}")

        # The input folder name reads best as the agent's title; fall back to
        # the README purpose if the folder name is unavailable.
        agent_name = (
            os.path.basename(manifest.input_root.rstrip(os.sep))
            or ir.metadata.description
            or "Converted Agent"
        )

        # Stage 7 -- Module 8: output README with target vocabulary.
        write_readme(
            build_readme(ir, conversion, self.config, agent_name=agent_name),
            generation.output_root,
        )

        # Stage 8 -- Module 9: migration report.
        report = build_report(
            ir,
            conversion,
            generation,
            self.config,
            agent_name=agent_name,
            generated_date=date.today().isoformat(),
        )
        write_report(report, generation.output_root)

        # Stage 9 -- INSTALL.md + ARCHITECTURE.md so the output is self-describing.
        write_docs(ir, conversion, generation.output_root, self.config)

        # Stage 10 -- agent-specific READINESS report (LLM-authored; deterministic
        # fallback when no key). What's left, who fixes it, time, accuracy. The
        # numeric metrics are computed deterministically and written as a JSON
        # sidecar the web service reads directly (no Markdown re-parsing).
        readiness_md, readiness_metrics = generate_readiness_report(
            ir, conversion, generation, self.config,
            acceptance=acceptance, agent_name=agent_name,
        )
        problems = validate_metrics(readiness_metrics)
        if problems:
            raise ValueError(
                "Readiness metric validation failed: " + "; ".join(problems)
            )
        write_readiness_report(readiness_md, generation.output_root)
        write_readiness_metrics(readiness_metrics, generation.output_root)

        # Stage 11 -- LLM Refinement Pass (gate-closed repair loop).
        # Feeds the generated code + READINESS_REPORT.md + the ACTUAL acceptance-
        # gate failures back to the LLM, applies validated patches, and re-runs the
        # gate -- looping until it is green or the iteration cap is hit. Gracefully
        # skipped when no API key is set or the LLM returns nothing. Never blocks.
        refinement = run_llm_refinement(
            ir, conversion, generation, self.config,
            output_root=generation.output_root,
            agent_name=agent_name,
        )
        write_refinement_log(refinement, generation.output_root, agent_name=agent_name)

        # If the refinement loop changed any files, the ACCEPTANCE.md written at
        # Stage 6.5 is stale -- re-run the gate and rewrite it so the shipped
        # acceptance report reflects the final (repaired) code.
        if refinement.ran and refinement.patches:
            acceptance = verify_output(ir, generation)
            write_acceptance(acceptance, generation.output_root)

        # main.py prints the output path; the pipeline just returns the report.
        return report
