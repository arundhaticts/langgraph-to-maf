"""`TargetGenerator` -- the code-emission strategy for one target framework.

Section 16 ("any framework -> any framework"): the `TargetAdapter` owns naming
and vocabulary (what a tool/context is *called*); the `TargetGenerator` owns the
framework-specific CODE the output package needs -- the orchestration graph, the
runnable entrypoint, the offline SDK stub, the smoke test, and the validation
tokens that prove the emitted framework constructs are present.

`code_generator.py` is otherwise framework-agnostic: it ports node/helper
bodies, curates imports, renders the context dataclass and plugin modules, then
delegates every target-specific block to the generator resolved for the chosen
target. Adding a target framework = one `TargetGenerator` subclass (+ its
templates), no changes to the core generator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from converter.adapters.base import TargetAdapter
from converter.contracts import IR


class TargetGenerator(ABC):
    """Emits the framework-specific code blocks for one target framework."""

    name: str

    @abstractmethod
    def workflow_block(self, ir: IR, adapter: TargetAdapter) -> str:
        """The framework's orchestration graph (executors + edges + HITL wiring).

        Returned as source that is appended to the orchestrator module. Return
        "" when the IR has no graph to wire (e.g. a single node) -- the offline
        `run()` synthesized by the core generator is then the only driver.
        """
        raise NotImplementedError

    @abstractmethod
    def sdk_stub_files(self) -> dict[str, str]:
        """Offline SDK stub files to write into the output, keyed by relative
        path. Empty when the target SDK is a public pip package (no stub needed).
        """
        raise NotImplementedError

    @abstractmethod
    def smoke_test(self, adapter: TargetAdapter, target: str) -> str:
        """A minimal offline-safe smoke test for the converted package."""
        raise NotImplementedError

    @abstractmethod
    def entrypoint(self, ir: IR, adapter: TargetAdapter, target: str) -> str:
        """The runnable `main.py` that drives the converted agent."""
        raise NotImplementedError

    def orchestrator_must_tokens(self, has_workflow_block: bool) -> list[str]:
        """Tokens the emitted orchestrator MUST contain (target-construct check).

        Always requires the offline `def run`; a framework whose `workflow_block`
        emits real graph constructs adds them when the block is present.
        """
        return ["def run"]

    def extra_files(self, ir: IR, adapter: TargetAdapter) -> dict[str, str]:
        """Extra non-Python files the target package needs, keyed by relative path.

        Default: none. A framework whose agents are prompt-driven (CrewAI's
        role/goal/backstory, task descriptions) overrides this to emit editable
        `prompts/` templates so the production prompts are first-class artifacts
        rather than string literals buried in the orchestrator.
        """
        return {}
