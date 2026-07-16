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
    write_docs,
    write_readiness_report,
    write_readme,
    write_report,
)
from converter.ir import build_ir, validate_ir, write_ir_json
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
            for check in verify_runnable(generation.output_root):
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
        # fallback when no key). What's left, who fixes it, time, accuracy.
        write_readiness_report(
            generate_readiness_report(
                ir, conversion, generation, self.config,
                acceptance=acceptance, agent_name=agent_name,
            ),
            generation.output_root,
        )

        # main.py prints the output path; the pipeline just returns the report.
        return report
