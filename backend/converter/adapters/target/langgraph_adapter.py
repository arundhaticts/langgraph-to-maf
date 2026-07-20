"""LangGraph target adapter.

Emits LangGraph idioms for the deterministic tiers: tools are function-style
`@tool` (from `langchain_core.tools`, which reads the docstring as the
description -- so no `description=` kwarg), the state object is `AgentState`, and
the runtime deps are `langgraph` + `langchain-core`. The orchestration graph
(StateGraph wiring) is emitted by `LangGraphTargetGenerator`.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter, to_pascal_case
from converter.contracts import ConstructSupport, ConstructType


class LangGraphTargetAdapter(TargetAdapter):
    name = "langgraph"

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
        # Idiomatic LangGraph state object name.
        return "AgentState"

    def tool_style(self) -> str:
        return "function"

    def tool_decorator(self) -> str:
        return "tool"

    def tool_decorator_import(self) -> str:
        return "from langchain_core.tools import tool"

    def tool_decorator_call(self, description_repr: str) -> str:
        # LangChain's @tool takes the description from the docstring; a bare
        # decorator, not @tool(description=...).
        return self.tool_decorator()

    def runtime_requirements(self) -> tuple[str, ...]:
        return ("langgraph", "langchain-core")

    def capability_matrix(self) -> dict[ConstructType, ConstructSupport]:
        D, L = ConstructSupport.DIRECT, ConstructSupport.LOSSY
        return {
            ConstructType.TOOLS:              D,  # @tool (langchain_core), docstring-driven
            ConstructType.STATE_TYPED:        D,  # TypedDict + Annotated reducers
            ConstructType.STATE_SHARED:       D,  # all nodes share one state dict
            ConstructType.CONDITIONAL_EDGES:  D,  # add_conditional_edges native
            ConstructType.LOOPS:              D,  # back-edges + DFS cycle detection
            ConstructType.HITL:               D,  # interrupt() + checkpointer resume
            ConstructType.CHECKPOINTING:      D,  # MemorySaver / SqliteSaver / Postgres
            ConstructType.MULTI_AGENT:        L,  # via subgraphs, not first-class roles
            ConstructType.AGENT_ROLES:        L,  # no role/goal/backstory primitive
        }
