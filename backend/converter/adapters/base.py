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

from converter.contracts import (
    AgentSpec,
    ConstructSupport,
    ConstructType,
    GraphSpec,
    SourceVocabulary,
    TaskSpec,
    ToolSpec,
)


class SourceAdapter(ABC):
    """Reads a specific source framework (LangGraph today).

    Owns everything the parser needs to know about ONE source framework:
    detection (does this repo look like us?), the call/name `vocabulary()` the
    AST parser matches against, and which pip packages to drop from the output.
    Adding a source framework = one adapter, no parser/engine changes.
    """

    name: str

    @abstractmethod
    def import_signatures(self) -> tuple[str, ...]:
        """Top-level module names that identify this framework in imports."""
        raise NotImplementedError

    def source_packages(self) -> tuple[str, ...]:
        """pip packages specific to this SOURCE framework, dropped from the
        converted agent's requirements (they are replaced by the target's)."""
        return ()

    def vocabulary(self) -> SourceVocabulary:
        """The call/name tables the code parser matches against for this
        framework. The default is the LangGraph vocabulary; a framework whose
        idioms differ (CrewAI, AutoGen, Strands, ...) overrides this."""
        return SourceVocabulary()

    def detect(self, imported_roots: set[str]) -> float:
        """Confidence in [0.0, 1.0] that a repo importing `imported_roots` is
        written in this framework. Default: 1.0 if any `import_signatures()`
        module was imported, else 0.0. Frameworks needing finer signals (e.g.
        distinguishing sub-packages) may override with a graded score."""
        return 1.0 if any(sig in imported_roots for sig in self.import_signatures()) else 0.0

    def extract_agents(self, source: str) -> list[AgentSpec]:
        """Parse agent definitions from one source file.

        Only relevant for role-based frameworks (CrewAI, Strands) where agents
        are constructed with explicit role/goal/backstory or system_prompt kwargs.
        Returns [] for graph-based frameworks (LangGraph, MAF) whose agents are
        implicit in the graph nodes.
        """
        return []

    def extract_tasks(self, source: str) -> list[TaskSpec]:
        """Parse task definitions from one source file (CrewAI model only).

        Returns [] for all non-CrewAI frameworks.
        """
        return []

    def extract_graph(self, source: str) -> GraphSpec | None:
        """Custom whole-file graph extraction for frameworks whose graph is NOT
        assembled with graph-builder method calls (`.add_node`/`.add_edge`).

        LangGraph's graph IS builder-method calls, so it returns None here and
        the default vocabulary-driven parser (`code_parser.extract_graph`) runs.
        A framework like CrewAI, whose graph is `Crew(tasks=[Task(...), ...])`
        constructor calls, overrides this to parse that shape into a `GraphSpec`.
        Return None for a file that has no graph so the default parser still runs.
        """
        return None


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

    def tool_decorator_call(self, description_repr: str) -> str:
        """The decorator EXPRESSION applied to a tool function (without the `@`).

        Default matches MAF's `@ai_function(description=...)`. A framework whose
        decorator does not take a `description` kwarg (e.g. LangChain's `@tool`,
        which reads the docstring) overrides this to return a bare name."""
        return f"{self.tool_decorator()}(description={description_repr})"

    def runtime_requirements(self) -> tuple[str, ...]:
        """pip packages the converted agent needs for THIS target framework."""
        return ()

    def capability_matrix(self) -> dict[ConstructType, ConstructSupport]:
        """Which IR constructs this target handles natively vs emulated vs unsupported.

        The default (optimistic) assumes every construct is DIRECT. Each concrete
        target adapter overrides only the constructs that are LOSSY or UNSUPPORTED.
        Used by Phase 5b capability negotiation before code generation.
        """
        return {ct: ConstructSupport.DIRECT for ct in ConstructType}


def to_pascal_case(snake: str) -> str:
    """`detect_flaky_tests` -> `DetectFlakyTests`. Leaves camelCase words intact."""
    parts = [p for p in snake.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or snake
