"""MAF (Microsoft Agent Framework) target adapter.

Holds the deterministic Tier 1/2 idioms the generator needs: plugin/method
naming, the context class name, and README vocabulary (R-13: Tools -> Skills).
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter, to_pascal_case


class MAFTargetAdapter(TargetAdapter):
    name = "maf"

    # R-13: MAF calls tools "Skills".
    README_VOCAB = {
        "tools_heading": "Skills",
        "state_heading": "Context",
    }

    def plugin_class_name(self, tool_name: str) -> str:
        return f"{to_pascal_case(tool_name)}Plugin"

    def method_name(self, tool_name: str) -> str:
        return tool_name

    @property
    def context_class_name(self) -> str:
        return "AgentContext"

    def tool_decorator(self) -> str:
        return "kernel_function"

    def tool_decorator_import(self) -> str:
        return "from semantic_kernel.functions import kernel_function"

    def runtime_requirements(self) -> tuple[str, ...]:
        return ("semantic-kernel",)
