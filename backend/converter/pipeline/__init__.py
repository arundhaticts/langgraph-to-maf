"""Pipeline strategies -- one per approach, all behind `ConversionPipeline`."""

from converter.pipeline.base import ConversionPipeline
from converter.pipeline.hybrid_pipeline import HybridPipeline

__all__ = ["ConversionPipeline", "HybridPipeline"]
