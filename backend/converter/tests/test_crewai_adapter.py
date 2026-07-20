"""Tests for the CrewAI source adapter (first non-LangGraph source)."""

from __future__ import annotations

import os

from converter.adapters import (
    CrewAISourceAdapter,
    detect_source_framework,
    get_source_adapter,
    list_source_frameworks,
)
from converter.extractor import extract_components
from converter.scanner import scan_repo


# A sequential crew: research -> write -> END.
SEQUENTIAL_CREW = '''
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


@tool("search")
def search(query: str) -> str:
    """Search the web."""
    return "results"


researcher = Agent(role="Researcher", goal="find facts", tools=[search])
writer = Agent(role="Writer", goal="write copy")

research = Task(description="Research the topic", agent=researcher)
write = Task(description="Write the article", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research, write], process=Process.sequential)
crew.kickoff()
'''

# A dependency crew: intake + gather both feed synthesize.
DEP_CREW = '''
from crewai import Agent, Task, Crew

intake = Task(description="Intake", agent=a)
gather = Task(description="Gather", agent=a)
synthesize = Task(description="Synthesize", agent=a, context=[intake, gather])

crew = Crew(agents=[a], tasks=[intake, gather, synthesize])
'''

README = """# Crew Demo
## Purpose
A crewai agent.
## Tools
- `search`: searches
## Workflow
Research then write.
"""


def _edges(graph):
    return {(e.source, e.target) for e in graph.edges}


def test_registry_lists_crewai_as_source():
    assert "crewai" in list_source_frameworks()
    assert isinstance(get_source_adapter("crewai"), CrewAISourceAdapter)


def test_detect_crewai_from_imports():
    assert detect_source_framework({"crewai", "os"}) == "crewai"


def test_vocabulary_has_no_graph_methods_but_keeps_tool():
    vocab = CrewAISourceAdapter().vocabulary()
    assert vocab.graph_methods == frozenset()          # graph parsed by adapter
    assert "tool" in vocab.tool_decorators
    assert "crewai" in vocab.dropped_import_roots


def test_extract_graph_sequential():
    graph = CrewAISourceAdapter().extract_graph(SEQUENTIAL_CREW)
    assert [n.name for n in graph.nodes] == ["research", "write"]
    assert graph.entry_point == "research"
    assert _edges(graph) == {("research", "write"), ("write", "END")}


def test_extract_graph_with_context_dependencies():
    graph = CrewAISourceAdapter().extract_graph(DEP_CREW)
    assert set(n.name for n in graph.nodes) == {"intake", "gather", "synthesize"}
    # intake/gather feed synthesize; synthesize is terminal.
    assert ("intake", "synthesize") in _edges(graph)
    assert ("gather", "synthesize") in _edges(graph)
    assert ("synthesize", "END") in _edges(graph)
    # intake and gather have no incoming deps -> one of them is the entry point.
    assert graph.entry_point in ("intake", "gather")


def test_non_crew_file_returns_none():
    # A plain tools file (no Crew/Task) -> adapter declines, default parser runs.
    assert CrewAISourceAdapter().extract_graph("x = 1\n") is None


def test_end_to_end_extract_components(tmp_path):
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "crew.py").write_text(SEQUENTIAL_CREW, encoding="utf-8")

    manifest = scan_repo(str(tmp_path))
    assert manifest.detected_framework == "crewai"

    inventory = extract_components(manifest)
    names = [n.name for n in inventory.graph.nodes]
    assert names == ["research", "write"]
    # @tool function was extracted despite the CrewAI-specific graph.
    assert "search" in {t.name for t in inventory.tools}
