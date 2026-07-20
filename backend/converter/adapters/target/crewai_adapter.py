"""CrewAI target adapter.

Emits CrewAI idioms for the deterministic tiers: tools are function-style `@tool`
(from `crewai.tools`, which reads the name/description from the function + docstring,
so a bare decorator -- not `@tool(description=...)`), and the runtime dep is
`crewai`. The crew graph (Agent/Task/Crew construction) is emitted by
`CrewAITargetGenerator`.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter, to_pascal_case
from converter.contracts import ConstructSupport, ConstructType


class CrewAITargetAdapter(TargetAdapter):
    name = "crewai"

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
        return "from crewai.tools import tool"

    def tool_decorator_call(self, description_repr: str) -> str:
        # CrewAI's @tool reads the name/description from the function; bare form.
        return self.tool_decorator()

    def runtime_requirements(self) -> tuple[str, ...]:
        return ("crewai",)

    def capability_matrix(self) -> dict[ConstructType, ConstructSupport]:
        D, L, U = ConstructSupport.DIRECT, ConstructSupport.LOSSY, ConstructSupport.UNSUPPORTED
        return {
            ConstructType.TOOLS:              D,  # @tool (crewai.tools), docstring-driven
            ConstructType.STATE_TYPED:        L,  # task context only; no shared TypedDict
            ConstructType.STATE_SHARED:       L,  # implicit via task output passing
            ConstructType.CONDITIONAL_EDGES:  L,  # Flows @router, limited branch count
            ConstructType.LOOPS:              L,  # Flows with @router back to @start
            ConstructType.HITL:               L,  # human_input=True on Task (basic prompt)
            ConstructType.CHECKPOINTING:      U,  # no native persist + resume
            ConstructType.MULTI_AGENT:        D,  # Agent + Task + Crew is the core model
            ConstructType.AGENT_ROLES:        D,  # role / goal / backstory first-class
        }
