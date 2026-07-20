"""Tests for CrewAI as a conversion TARGET (any source -> CrewAI)."""

from __future__ import annotations

import ast
import io
import zipfile

from converter.adapters import get_target_adapter, list_frameworks
from converter.adapters.target.crewai_adapter import CrewAITargetAdapter
from converter.generator.targets import CrewAITargetGenerator, get_target_generator
from service import convert_folder

README = """# Demo
## Purpose
A demo agent.
## Tools
- `search`: searches
## Workflow
Sequential.
"""

# LangGraph source: two linear nodes + a @tool.
LG_SOURCE = '''
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.tools import tool


class State(TypedDict):
    x: int


@tool
def search(query: str) -> str:
    """Search the web."""
    return "results"


def gather(state):
    """Gather the inputs."""
    return {"x": state["x"] + 1}


def summarise(state):
    """Summarise the findings."""
    return state


g = StateGraph(State)
g.add_node("gather", gather)
g.add_node("summarise", summarise)
g.set_entry_point("gather")
g.add_edge("gather", "summarise")
g.add_edge("summarise", END)
'''

STRANDS_SOURCE = '''
from strands import Agent, tool


@tool
def word_count(text: str) -> int:
    """Count words."""
    return len(text.split())


assistant = Agent(tools=[word_count], system_prompt="Be helpful.")
result = assistant("hi")
'''


def _zip(files, source):
    z, _ = convert_folder(files, "manual", target="crewai", source=source)
    return zipfile.ZipFile(io.BytesIO(z))


def _assert_all_python_valid(zf):
    for name in zf.namelist():
        if name.endswith(".py"):
            ast.parse(zf.read(name).decode("utf-8"))


def test_registry_crewai_is_targetable():
    assert "crewai" in list_frameworks()
    assert isinstance(get_target_adapter("crewai"), CrewAITargetAdapter)
    assert isinstance(get_target_generator("crewai"), CrewAITargetGenerator)


def test_langgraph_to_crewai_emits_crew_and_tasks():
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": LG_SOURCE},
    ]
    zf = _zip(files, source="langgraph")
    orch = zf.read("orchestrator.py").decode()

    assert "from crewai import Agent, Task, Crew, Process" in orch
    assert "gather_task = Task(" in orch
    assert "summarise_task = Task(" in orch
    assert "process=Process.sequential" in orch
    assert "AGENT_TOOLS = [search]" in orch
    # No MAF / LangGraph residue in the CrewAI output.
    assert "WorkflowBuilder" not in orch and "StateGraph" not in orch

    main = zf.read("main.py").decode()
    assert "build_crew" in main and "kickoff" in main

    reqs = zf.read("requirements.txt").decode()
    assert "crewai" in reqs

    _assert_all_python_valid(zf)


def test_strands_to_crewai_single_task():
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": STRANDS_SOURCE},
    ]
    zf = _zip(files, source="aws_strands")
    orch = zf.read("orchestrator.py").decode()
    assert "assistant_task = Task(" in orch
    assert "process=Process.sequential" in orch
    _assert_all_python_valid(zf)


# A node reached by BOTH a forward edge AND a router loopback -- the case that
# used to drop the forward @listen (fatal: the flow tail never ran).
BRANCH_LOOP_SOURCE = '''
from langgraph.graph import StateGraph, END
from typing import TypedDict


class State(TypedDict):
    retries: int
    passed: bool


def prep(state):
    """Prepare."""
    return state


def gap_gen(state):
    """Generate gaps."""
    return {"retries": state.get("retries", 0) + 1}


def validation(state):
    """Validate."""
    return {"passed": state.get("retries", 0) >= 2}


def assemble(state):
    """Assemble the report."""
    return state


def route(state):
    """Route after validation."""
    return "assemble" if state.get("passed") else "gap_gen"


g = StateGraph(State)
g.add_node("prep", prep)
g.add_node("gap_gen", gap_gen)
g.add_node("validation", validation)
g.add_node("assemble", assemble)
g.set_entry_point("prep")
g.add_edge("prep", "gap_gen")
g.add_edge("gap_gen", "validation")
g.add_conditional_edges("validation", route, {"gap_gen": "gap_gen", "assemble": "assemble"})
g.add_edge("assemble", END)
'''


def test_branch_loop_wires_union_of_triggers():
    """gap_gen is a forward-edge target AND a router loopback target -- it must
    listen on BOTH events (deduped), or the forward path dies after prep."""
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": BRANCH_LOOP_SOURCE},
    ]
    zf = _zip(files, source="langgraph")
    orch = zf.read("orchestrator.py").decode()

    # The gap_gen method must listen on BOTH the forward edge (prep) AND the
    # router loopback (gap_gen), each exactly once (deduped per node).
    lines = orch.splitlines()
    gap_def = next(i for i, l in enumerate(lines) if l.strip() == "def gap_gen(self, event=None):")
    decos = [l.strip() for l in lines[:gap_def] if l.strip().startswith("@")][-3:]
    assert "@listen('prep')" in decos
    assert "@listen('gap_gen')" in decos
    assert decos.count("@listen('gap_gen')") == 1  # not duplicated on this node
    # The router calls the real converted router function.
    assert "_run_router('route'" in orch
    # No unresolved-graph warnings for a fully-connected source graph.
    assert "GRAPH VALIDATION WARNINGS" not in orch

    # The entrypoint threads the built state into the Flow (never an empty ctx).
    main = zf.read("main.py").decode()
    assert "build_flow(state).kickoff()" in main

    _assert_all_python_valid(zf)


def test_crewai_tools_use_bare_decorator():
    files = [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": LG_SOURCE},
    ]
    zf = _zip(files, source="langgraph")
    # The plugin module for a langgraph-source agent lands at plugins/agent.py.
    plugin = zf.read("plugins/agent.py").decode()
    assert "from crewai.tools import tool" in plugin
    assert "@tool" in plugin and "@tool(description=" not in plugin
