"""LangGraph source adapter."""

from __future__ import annotations

from converter.adapters.base import SourceAdapter


class LangGraphSourceAdapter(SourceAdapter):
    name = "langgraph"

    def import_signatures(self) -> tuple[str, ...]:
        return ("langgraph",)

    def source_packages(self) -> tuple[str, ...]:
        # Dropped from the converted agent's requirements (replaced by the target).
        return ("langgraph", "langgraph-checkpoint-sqlite", "langchain-core", "langchain")
