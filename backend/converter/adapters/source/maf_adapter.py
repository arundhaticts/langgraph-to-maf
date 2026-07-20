"""MAF (Microsoft Agent Framework / agent_framework) source adapter.

MAF graphs are built with WorkflowBuilder + add_node/add_edge-style calls, but
the builder class is called `WorkflowBuilder` and decorators are `@executor` /
`@handler` rather than the LangGraph defaults.  Tools are decorated with
`@ai_function` (from `agent_framework`).  State is carried via `WorkflowContext`
(a dict-like object), not a TypedDict.

The vocabulary here tells the standard vocabulary-driven parser exactly what to
look for, so `extract_graph()` does NOT need to be overridden -- the parser's
graph-method detection handles `add_node`/`add_edge` equivalents.
"""

from __future__ import annotations

from converter.adapters.base import SourceAdapter
from converter.contracts import SourceVocabulary


class MAFSourceAdapter(SourceAdapter):
    name = "maf"

    def import_signatures(self) -> tuple[str, ...]:
        return ("agent_framework",)

    def source_packages(self) -> tuple[str, ...]:
        return ("agent-framework",)

    def vocabulary(self) -> SourceVocabulary:
        return SourceVocabulary(
            tool_decorators=frozenset({"ai_function"}),
            graph_methods=frozenset(
                {
                    "add_node",
                    "add_edge",
                    "add_conditional_edges",
                    "set_entry_point",
                    "set_finish_point",
                }
            ),
            llm_constructors=frozenset(
                {"OpenAI", "AzureOpenAI", "AzureOpenAIChatClient", "Anthropic", "ChatAgent"}
            ),
            llm_constructor_prefixes=("Chat",),
            checkpointer_constructors=frozenset(
                {"MemorySaver", "FileCheckpointStorage", "SqliteCheckpointStorage"}
            ),
            sentinels={
                "START": "START",
                "__start__": "START",
                "END": "END",
                "__end__": "END",
            },
            state_base_classes=frozenset({"TypedDict", "BaseModel"}),
            dropped_import_roots=frozenset({"agent_framework"}),
            graph_tokens=(
                "WorkflowBuilder",
                ".add_node",
                ".add_edge",
                ".add_conditional_edges",
                ".set_entry_point",
                ".compile(",
                "@executor",
                "@handler",
            ),
        )
