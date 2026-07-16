"""Source and target adapter interfaces.

Framework knowledge for the deterministic tiers (1 & 2) lives behind these
adapters, not in the core pipeline. A `SourceAdapter` owns detection and any
source-specific reading hints; a `TargetAdapter` owns the idioms, naming, and
vocabulary the generator needs to emit the target framework.

Adding a framework = one adapter (+ a Tier 3 knowledge pack), no engine changes.
This is the seam that makes "any framework -> any framework" (Section 16) reachable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from converter.contracts import ToolSpec


class SourceAdapter(ABC):
    """Reads a specific source framework (LangGraph today)."""

    name: str

    @abstractmethod
    def import_signatures(self) -> tuple[str, ...]:
        """Top-level module names that identify this framework in imports."""
        raise NotImplementedError

    def source_packages(self) -> tuple[str, ...]:
        """pip packages specific to this SOURCE framework, dropped from the
        converted agent's requirements (they are replaced by the target's)."""
        return ()


class TargetAdapter(ABC):
    """Emits a specific target framework (MAF today).

    Holds the deterministic (Tier 1/2) idioms: naming, imports, vocabulary, the
    tool decorator, and runtime dependencies. Nothing framework-specific is
    hardcoded in the generator -- it reads these off the adapter, so a new
    target is added here (or via the doc-driven Tier 3 path), not in the core.
    """

    name: str
    # README heading vocabulary, e.g. {"tools_heading": "Skills"} (R-13).
    README_VOCAB: dict[str, str]

    @abstractmethod
    def plugin_class_name(self, tool_name: str) -> str:
        """Target class name for a source tool (e.g. read_tests -> ReadTestsPlugin)."""
        raise NotImplementedError

    @abstractmethod
    def method_name(self, tool_name: str) -> str:
        """Target method name for a source tool."""
        raise NotImplementedError

    @property
    @abstractmethod
    def context_class_name(self) -> str:
        """Name of the generated context/state dataclass."""
        raise NotImplementedError

    # --- idioms the generator reads instead of hardcoding a framework ---

    def tool_style(self) -> str:
        """How tools are emitted: 'function' (decorated plain functions, e.g. MAF
        @ai_function) or 'plugin_class' (a class wrapping methods). Function-style
        avoids dead wrapper classes that are never registered."""
        return "plugin_class"

    def tool_decorator(self) -> str:
        """Decorator applied to plugin/skill methods."""
        return "kernel_function"

    def tool_decorator_import(self) -> str:
        """Import line providing `tool_decorator()` (with a runtime fallback shim
        added by the generator)."""
        return "from semantic_kernel.functions import kernel_function"

    def runtime_requirements(self) -> tuple[str, ...]:
        """pip packages the converted agent needs for THIS target framework."""
        return ()


def to_pascal_case(snake: str) -> str:
    """`detect_flaky_tests` -> `DetectFlakyTests`. Leaves camelCase words intact."""
    parts = [p for p in snake.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or snake
