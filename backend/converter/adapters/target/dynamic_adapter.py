"""Dynamic target adapter — driven entirely by an uploaded framework pack.

This is the "any target framework" seam. Instead of a hand-written adapter class,
it reads the framework's `vocabulary.json` (uploaded through the UI and stored in
`frameworks/<name>/`) and fulfils the `TargetAdapter` contract from it. Drop a
pack folder in and the converter can target that framework with no code change.

Two `vocabulary.json` shapes are accepted:

- **flat** (simple):
    {"conventions": {"context_class": "...", "tool_decorator": "@dec",
                     "tool_decorator_import": "...", "plugin_class_suffix": "..."},
     "vocabulary": {"tools_heading": "...", "state_heading": "..."},
     "runtime_requirements": ["pkg"]}

- **rich** (structured, e.g. the bundled MAF pack): dependencies live under
  `dependencies.required`, headings/naming are inferred with safe defaults.

Anything absent falls back to a sensible default, so a minimal pack still works.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter, to_pascal_case
from converter.contracts import ConstructSupport, ConstructType


class DynamicTargetAdapter(TargetAdapter):
    """A `TargetAdapter` whose idioms come from a parsed `vocabulary.json`."""

    def __init__(self, name: str, vocab: dict) -> None:
        self.name = name
        self._vocab = vocab or {}
        conv = self._vocab.get("conventions", {}) or {}
        voc = self._vocab.get("vocabulary", {}) or {}

        self._context_class = conv.get("context_class") or "AgentContext"
        self._plugin_suffix = conv.get("plugin_class_suffix") or "Plugin"

        # Tool decorator: strip a leading '@' if the pack wrote "@decorator".
        raw_decorator = conv.get("tool_decorator") or ""
        self._tool_decorator = raw_decorator.lstrip("@").strip()
        self._tool_decorator_import = conv.get("tool_decorator_import") or ""

        # README heading vocabulary (Tools/State by default).
        self.README_VOCAB = {
            "tools_heading": voc.get("tools_heading") or "Tools",
            "state_heading": voc.get("state_heading") or "State",
        }

        self._runtime_requirements = self._resolve_requirements()
        self._reject_tokens = self._resolve_reject_tokens()
        self._capability_matrix = self._resolve_capability_matrix()

    # -- schema resolution -------------------------------------------------

    def _resolve_requirements(self) -> tuple[str, ...]:
        """Runtime pip deps, from either schema shape."""
        flat = self._vocab.get("runtime_requirements")
        if isinstance(flat, (list, tuple)):
            return tuple(str(x) for x in flat if x)
        deps = self._vocab.get("dependencies", {}) or {}
        required = deps.get("required")
        if isinstance(required, (list, tuple)):
            return tuple(str(x) for x in required if x)
        return ()

    def _resolve_reject_tokens(self) -> tuple[str, ...]:
        """Substrings that must NOT appear in generated output (pack guardrail)."""
        reject = self._vocab.get("reject_list", {}) or {}
        items = reject.get("items")
        if isinstance(items, (list, tuple)):
            return tuple(str(x) for x in items if x)
        return ()

    def _resolve_capability_matrix(self) -> dict[ConstructType, ConstructSupport]:
        """Build the capability matrix from the pack, exactly like a hand-written
        adapter (e.g. MAF) declares one.

        The pack declares support under a `capabilities` object in vocabulary.json,
        keyed by ConstructType value with a ConstructSupport value, e.g.::

            "capabilities": {
                "tools": "direct",
                "hitl": "lossy",
                "checkpointing": "unsupported"
            }

        Any construct the pack does not mention defaults to DIRECT (optimistic),
        so a minimal pack still works while a thorough pack drives real Phase-5b
        capability negotiation (LOSSY -> needs_review, UNSUPPORTED -> manual).
        """
        declared = self._vocab.get("capabilities", {}) or {}
        matrix: dict[ConstructType, ConstructSupport] = {}
        for construct in ConstructType:
            raw = declared.get(construct.value)
            support = ConstructSupport.DIRECT
            if isinstance(raw, str):
                try:
                    support = ConstructSupport(raw.strip().lower())
                except ValueError:
                    support = ConstructSupport.DIRECT
            matrix[construct] = support
        return matrix

    # -- TargetAdapter interface ------------------------------------------

    def plugin_class_name(self, tool_name: str) -> str:
        return f"{to_pascal_case(tool_name)}{self._plugin_suffix}"

    def method_name(self, tool_name: str) -> str:
        return tool_name

    @property
    def context_class_name(self) -> str:
        return self._context_class

    def tool_decorator(self) -> str:
        # Fall back to a no-op-ish decorator name so the generated shim stays valid
        # even for packs that use plain function tools (no decorator).
        return self._tool_decorator or "tool"

    def tool_decorator_import(self) -> str:
        # A non-empty statement is required inside the generated try-block.
        return self._tool_decorator_import or "pass  # no tool-decorator import for this framework"

    def runtime_requirements(self) -> tuple[str, ...]:
        return self._runtime_requirements

    def capability_matrix(self) -> dict[ConstructType, ConstructSupport]:
        return self._capability_matrix

    # -- extras used by the generator's validation (optional) -------------

    @property
    def reject_tokens(self) -> tuple[str, ...]:
        return self._reject_tokens
