"""MAF (Microsoft Agent Framework) target adapter.

Emits true MAF idioms for the deterministic tiers: tools are function-style
`@ai_function` (from `agent_framework`), not Semantic Kernel plugin classes, and
the runtime dependency is `agent-framework`. HITL / complex orchestration idioms
come from the Tier-3 knowledge pack in `frameworks/maf/`.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter, to_pascal_case


class MAFTargetAdapter(TargetAdapter):
    name = "maf"

    README_VOCAB = {
        "tools_heading": "Tools",
        "state_heading": "Context",
    }

    def plugin_class_name(self, tool_name: str) -> str:
        return f"{to_pascal_case(tool_name)}Tool"

    def method_name(self, tool_name: str) -> str:
        return tool_name

    @property
    def context_class_name(self) -> str:
        return "AgentContext"

    def tool_style(self) -> str:
        # MAF tools are plain typed functions decorated with @ai_function.
        return "function"

    def tool_decorator(self) -> str:
        return "ai_function"

    def tool_decorator_import(self) -> str:
        return "from agent_framework import ai_function"

    def runtime_requirements(self) -> tuple[str, ...]:
        return ("agent-framework",)
