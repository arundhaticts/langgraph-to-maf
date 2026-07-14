"""CLI entry point + pipeline dispatch.

`PIPELINE_REGISTRY` maps a `ConversionMode` to the pipeline that implements it.
Adding Approach 1 or 2 is a one-line registration here plus the pipeline class --
no other module changes, because everything speaks the frozen contracts.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from converter.config import Config, ConversionMode
from converter.pipeline.base import ConversionPipeline
from converter.pipeline.hybrid_pipeline import HybridPipeline
from converter.scanner import ScannerError

# mode -> factory(config) -> ConversionPipeline
PIPELINE_REGISTRY: dict[ConversionMode, Callable[[Config], ConversionPipeline]] = {
    ConversionMode.HYBRID: lambda cfg: HybridPipeline(cfg),
    # Approach 1 -- reuse the hybrid pipeline with the LLM fallback disabled:
    ConversionMode.DETERMINISTIC: lambda cfg: HybridPipeline(
        cfg, allow_llm_fallback=False
    ),
    # Approach 2 -- FULL_LLM: register a FullLlmPipeline here once written.
}


def build_pipeline(config: Config) -> ConversionPipeline:
    factory = PIPELINE_REGISTRY.get(config.mode)
    if factory is None:
        raise SystemExit(
            f"Approach '{config.mode.value}' is not implemented yet. "
            f"Available: {', '.join(m.value for m in PIPELINE_REGISTRY)}."
        )
    return factory(config)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert-agent",
        description="Convert an AI agent from one framework to another via a "
        "neutral IR (LangGraph -> MAF).",
    )
    parser.add_argument(
        "--input", required=True, help="Path to the source agent folder."
    )
    parser.add_argument(
        "--output", required=True, help="Destination folder for the converted agent."
    )
    parser.add_argument(
        "--mode",
        default=ConversionMode.HYBRID.value,
        choices=[m.value for m in ConversionMode],
        help="Conversion approach (default: hybrid).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source framework (default: auto-detect from imports).",
    )
    parser.add_argument(
        "--target",
        default="maf",
        help="Target framework (default: maf). Needs an adapter or a frameworks/<name> knowledge pack.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Config(
        mode=ConversionMode(args.mode),
        source_framework=args.source,
        target_framework=args.target,
    )

    pipeline = build_pipeline(config)
    try:
        report = pipeline.run(args.input, args.output)
    except ScannerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The pipeline writes the report to disk; it is also returned for tests.
    print(f"Converted agent written to: {args.output}")
    print(f"Migration report: {report.agent_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
