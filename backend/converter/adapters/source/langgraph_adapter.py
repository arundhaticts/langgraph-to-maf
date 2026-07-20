"""LangGraph source adapter.

Owns LangGraph detection, its parser `vocabulary()`, and the pip packages to
drop from the converted output. LangGraph is the reference source, so its
vocabulary is the `SourceVocabulary` default -- this adapter states it
explicitly rather than relying on the default, so the file reads as the template
every other source adapter (CrewAI, AutoGen, AWS Strands, ...) copies.
"""

from __future__ import annotations

from converter.adapters.base import SourceAdapter
from converter.contracts import SourceVocabulary


class LangGraphSourceAdapter(SourceAdapter):
    name = "langgraph"

    def import_signatures(self) -> tuple[str, ...]:
        return ("langgraph",)

    def source_packages(self) -> tuple[str, ...]:
        # Dropped from the converted agent's requirements (replaced by the target).
        return ("langgraph", "langgraph-checkpoint-sqlite", "langchain-core", "langchain")

    def vocabulary(self) -> SourceVocabulary:
        # StateGraph.add_node/add_edge/add_conditional_edges, @tool, TypedDict
        # state, MemorySaver/SqliteSaver checkpointers -- these ARE the
        # SourceVocabulary defaults, stated here as the reference example.
        return SourceVocabulary(
            tool_decorators=frozenset({"tool"}),
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
                {"init_chat_model", "OpenAI", "AzureOpenAI", "Anthropic", "AnthropicLLM"}
            ),
            llm_constructor_prefixes=("Chat",),
            checkpointer_constructors=frozenset(
                {
                    "MemorySaver",
                    "SqliteSaver",
                    "AsyncSqliteSaver",
                    "PostgresSaver",
                    "AsyncPostgresSaver",
                }
            ),
            sentinels={
                "START": "START",
                "__start__": "START",
                "END": "END",
                "__end__": "END",
            },
            state_base_classes=frozenset({"TypedDict"}),
            dropped_import_roots=frozenset({"langgraph"}),
            graph_tokens=(
                "StateGraph",
                "MessageGraph",
                ".add_node",
                ".add_edge",
                ".add_conditional_edges",
                ".set_entry_point",
                ".set_finish_point",
                ".compile(",
            ),
        )
