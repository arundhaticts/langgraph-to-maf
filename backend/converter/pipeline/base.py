"""Pipeline strategy interface.

Each approach (Deterministic / Full LLM / Hybrid) is a `ConversionPipeline`.
They all take the same inputs and produce the same outputs (a written output
folder + a `MigrationReport`), which is what lets `main.py` swap between them
purely by `ConversionMode`.

Adding Approach 1 or 2 means writing one new class here (or reusing
`HybridPipeline` with a flag) and registering it in `main.PIPELINE_REGISTRY`.
No core module changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from converter.config import Config
from converter.contracts import MigrationReport


class ConversionPipeline(ABC):
    """A full input-folder -> output-folder conversion strategy."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def run(self, input_path: str, output_path: str) -> MigrationReport:
        """Convert the agent at `input_path`, writing to `output_path`.

        Returns the migration report (also written to disk by the pipeline).
        Must not execute the converted agent -- syntax validation only.
        """
        raise NotImplementedError
