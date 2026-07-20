"""AWS Strands target adapter.

Emits Strands idioms for the deterministic tiers: tools are plain `@tool`
functions (from `strands`, reads schema from type hints + docstring Args:
section -- bare decorator, no `description=` arg), and the runtime dep is
`strands-agents`.  The agent graph (Agent/tools construction) is emitted by
`AWSStrandsTargetGenerator`.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter, to_pascal_case
from converter.contracts import ConstructSupport, ConstructType


class AWSStrandsTargetAdapter(TargetAdapter):
    name = "aws_strands"

    README_VOCAB = {
        "tools_heading": "Tools",
        "state_heading": "State",
    }

    def plugin_class_name(self, tool_name: str) -> str:
        return f"{to_pascal_case(tool_name)}Tool"

    def method_name(self, tool_name: str) -> str:
        return tool_name

    @property
    def context_class_name(self) -> str:
        return "AgentContext"

    def tool_style(self) -> str:
        return "function"

    def tool_decorator(self) -> str:
        return "tool"

    def tool_decorator_import(self) -> str:
        return "from strands import tool"

    def tool_decorator_call(self, description_repr: str) -> str:
        # Strands @tool reads the schema from type hints + docstring -- bare form.
        return self.tool_decorator()

    def runtime_requirements(self) -> tuple[str, ...]:
        return ("strands-agents",)

    def capability_matrix(self) -> dict[ConstructType, ConstructSupport]:
        D, L, U = ConstructSupport.DIRECT, ConstructSupport.LOSSY, ConstructSupport.UNSUPPORTED
        return {
            ConstructType.TOOLS:              D,  # @tool (strands), type-hint + docstring schema
            ConstructType.STATE_TYPED:        L,  # conversation/session state, not TypedDict
            ConstructType.STATE_SHARED:       L,  # agent memory, not explicit shared dict
            ConstructType.CONDITIONAL_EDGES:  U,  # model-driven; no deterministic branching API
            ConstructType.LOOPS:              L,  # model drives iteration; no explicit exit guard
            ConstructType.HITL:               L,  # via a blocking tool call (no native pause)
            ConstructType.CHECKPOINTING:      U,  # no native persist + resume primitive
            ConstructType.MULTI_AGENT:        L,  # agents-as-tools pattern; not first-class
            ConstructType.AGENT_ROLES:        L,  # system_prompt only; no role/goal/backstory
        }
