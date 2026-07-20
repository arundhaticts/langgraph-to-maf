"""Tests for LangGraph as a conversion TARGET (first non-MAF output path)."""

from __future__ import annotations

import ast
import io
import zipfile

from converter.adapters import get_target_adapter, list_frameworks
from converter.adapters.target.langgraph_adapter import LangGraphTargetAdapter
from converter.generator.targets import LangGraphTargetGenerator, get_target_generator
from service import convert_folder

README = """# Demo
## Purpose
A demo agent.
## Tools
- `search`: searches
## Workflow
Sequential.
"""

CREW = '''
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


@tool("search")
def search(query: str) -> str:
    """Search the web."""
    return "results"


researcher = Agent(role="Researcher", goal="find facts", tools=[search])
writer = Agent(role="Writer", goal="write")

research = Task(description="Research", agent=researcher)
write = Task(description="Write", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research, write], process=Process.sequential)
'''

# LangGraph source with a conditional (looping) edge.
LG_SOURCE = '''
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    x: int


def check(state):
    return {"x": state["x"] + 1}


def route(state):
    return "again" if state["x"] < 3 else "done"


g = StateGraph(State)
g.add_node("check", check)
g.set_entry_point("check")
g.add_conditional_edges("check", route, {"again": "check", "done": END})
'''


def _zip(files, source):
    z, _ = convert_folder(files, "manual", target="langgraph", source=source)
    return zipfile.ZipFile(io.BytesIO(z))


def _assert_all_python_valid(zf):
    for name in zf.namelist():
        if name.endswith(".py"):
            ast.parse(zf.read(name).decode("utf-8"))


def test_registry_langgraph_is_targetable():
    assert "langgraph" in list_frameworks()
    assert isinstance(get_target_adapter("langgraph"), LangGraphTargetAdapter)
    assert isinstance(get_target_generator("langgraph"), LangGraphTargetGenerator)


def test_crewai_to_langgraph_emits_stategraph():
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/crew.py", "content": CREW},
    ]
    zf = _zip(files, source="crewai")
    orch = zf.read("orchestrator.py").decode()

    assert "StateGraph(AgentState)" in orch
    assert 'builder.add_node("research", research)' in orch
    assert 'builder.add_node("write", write)' in orch
    assert 'builder.set_entry_point("research")' in orch
    assert 'builder.add_edge("write", END)' in orch
    # No MAF residue leaked into the LangGraph output.
    assert "WorkflowBuilder" not in orch
    assert "@executor" not in orch

    main = zf.read("main.py").decode()
    assert "build_graph" in main and "invoke" in main

    reqs = zf.read("requirements.txt").decode()
    assert "langgraph" in reqs
    assert "crewai" not in reqs  # source package dropped

    _assert_all_python_valid(zf)


def test_langgraph_source_conditional_edge_to_langgraph():
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": LG_SOURCE},
    ]
    zf = _zip(files, source="langgraph")
    orch = zf.read("orchestrator.py").decode()
    assert 'builder.add_conditional_edges("check", route,' in orch
    assert '"again": "check"' in orch
    assert '"done": END' in orch
    _assert_all_python_valid(zf)


def test_langgraph_tools_use_bare_decorator_and_offline_safe_shim():
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/crew.py", "content": CREW},
    ]
    zf = _zip(files, source="crewai")
    plugin = zf.read("plugins/crew.py").decode()
    # LangChain's @tool is bare (reads the docstring), not @tool(description=...).
    assert "@tool" in plugin
    assert "@tool(description=" not in plugin
    assert "from langchain_core.tools import tool" in plugin
    # Offline shim tolerates bare decorator usage.
    assert "callable(args[0])" in plugin
