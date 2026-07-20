"""Target code generators -- one `TargetGenerator` per output framework.

`get_target_generator(name)` resolves the generator for a target framework. MAF
is the built-in reference; it is also the default fallback for any target that
has no dedicated generator yet (e.g. an uploaded pack driving a
`DynamicTargetAdapter`), which preserves today's behaviour until per-target
generators (LangGraph, CrewAI, ...) are added in later phases.
"""

from __future__ import annotations

from converter.generator.targets.aws_strands_generator import AWSStrandsTargetGenerator
from converter.generator.targets.base import TargetGenerator
from converter.generator.targets.crewai_generator import CrewAITargetGenerator
from converter.generator.targets.langgraph_generator import LangGraphTargetGenerator
from converter.generator.targets.maf_generator import MAFTargetGenerator

# Registry: target framework name -> generator class. Add a row to onboard a
# built-in target generator (paired with its TargetAdapter + templates).
TARGET_GENERATORS: dict[str, type[TargetGenerator]] = {
    "maf": MAFTargetGenerator,
    "langgraph": LangGraphTargetGenerator,
    "crewai": CrewAITargetGenerator,
    "aws_strands": AWSStrandsTargetGenerator,
}

# Fallback used when a target has no dedicated generator yet. MAF today, because
# the original generator emitted MAF constructs for every target.
_DEFAULT_GENERATOR: type[TargetGenerator] = MAFTargetGenerator


def get_target_generator(name: str | None) -> TargetGenerator:
    """Resolve the code generator for target `name` (MAF fallback)."""
    cls = TARGET_GENERATORS.get(name or "", _DEFAULT_GENERATOR)
    return cls()


__all__ = [
    "TargetGenerator",
    "MAFTargetGenerator",
    "LangGraphTargetGenerator",
    "CrewAITargetGenerator",
    "AWSStrandsTargetGenerator",
    "TARGET_GENERATORS",
    "get_target_generator",
]
