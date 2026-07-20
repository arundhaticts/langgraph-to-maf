"""Cross-framework regression matrix — every source × every target must stay green.

Purpose: guarantee that improving one target (e.g. CrewAI) cannot silently regress
another (e.g. MAF). Any new framework adapter added to the registries must extend the
parametrize marks here so it is automatically covered.
"""

from __future__ import annotations

import ast
import io
import zipfile

import pytest

from service import convert_folder

README = """# Regression Agent
## Purpose
A canonical multi-step agent used for regression testing.
## Tools
- `fetch`: fetches a resource
## Workflow
Linear: gather then summarise.
"""

# ── Source agents ─────────────────────────────────────────────────────────────

LANGGRAPH_AGENT = '''
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.tools import tool


class State(TypedDict):
    result: str


@tool
def fetch(url: str) -> str:
    """Fetch a resource."""
    return "data"


def gather(state):
    """Gather inputs."""
    return {"result": "gathered"}


def summarise(state):
    """Summarise findings."""
    return state


g = StateGraph(State)
g.add_node("gather", gather)
g.add_node("summarise", summarise)
g.set_entry_point("gather")
g.add_edge("gather", "summarise")
g.add_edge("summarise", END)
'''

AWS_STRANDS_AGENT = '''
from strands import Agent, tool


@tool
def fetch(url: str) -> str:
    """Fetch a resource."""
    return "data"


assistant = Agent(tools=[fetch], system_prompt="Be helpful.")
result = assistant("gather and summarise")
'''

MAF_AGENT = '''
from agent_framework import WorkflowBuilder, Executor, handler, WorkflowContext, ai_function


@ai_function(description="Fetch a resource.")
def fetch(url: str) -> str:
    return "data"


class GatherNode(Executor):
    @handler
    async def run(self, message: dict, ctx: WorkflowContext) -> None:
        await ctx.send_message({"result": "gathered"})


class SummariseNode(Executor):
    @handler
    async def run(self, message: dict, ctx: WorkflowContext) -> None:
        await ctx.yield_output({"done": True})


workflow = (
    WorkflowBuilder()
    .add_node(GatherNode())
    .add_node(SummariseNode())
    .add_edge(GatherNode, SummariseNode)
    .set_entry_point(GatherNode)
    .build()
)
'''

CREWAI_AGENT = '''
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


@tool
def fetch(url: str) -> str:
    """Fetch a resource."""
    return "data"


researcher = Agent(role="Researcher", goal="Gather data.", backstory="Expert.")
writer = Agent(role="Writer", goal="Summarise findings.", backstory="Expert.")

gather_task = Task(description="Gather inputs.", agent=researcher)
summarise_task = Task(description="Summarise findings.", agent=writer,
                      context=[gather_task])

crew = Crew(
    agents=[researcher, writer],
    tasks=[gather_task, summarise_task],
    process=Process.sequential,
)
'''

# ── Helpers ───────────────────────────────────────────────────────────────────

_ALL_TARGETS = ["maf", "langgraph", "crewai", "aws_strands"]


def _convert(agent_src: str, source: str, target: str) -> zipfile.ZipFile:
    files = [
        {"path": "agent/README.md", "content": README},
        {"path": "agent/agent.py", "content": agent_src},
    ]
    raw = convert_folder(files, "manual", target=target, source=source)
    return zipfile.ZipFile(io.BytesIO(raw))


def _assert_valid_python(zf: zipfile.ZipFile) -> None:
    for name in zf.namelist():
        if name.endswith(".py"):
            src = zf.read(name).decode("utf-8")
            try:
                ast.parse(src)
            except SyntaxError as exc:
                raise AssertionError(f"SyntaxError in {name}: {exc}") from exc


def _assert_core_files(zf: zipfile.ZipFile) -> None:
    names = set(zf.namelist())
    assert "orchestrator.py" in names, f"orchestrator.py missing; got: {sorted(names)}"
    assert "main.py" in names, f"main.py missing"
    assert "requirements.txt" in names, f"requirements.txt missing"


# ── Parametrized matrix ───────────────────────────────────────────────────────

@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_langgraph_source_all_targets(target: str):
    """LangGraph → every target must produce valid, complete output."""
    zf = _convert(LANGGRAPH_AGENT, source="langgraph", target=target)
    _assert_core_files(zf)
    _assert_valid_python(zf)


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_aws_strands_source_all_targets(target: str):
    """AWS Strands → every target must produce valid, complete output."""
    zf = _convert(AWS_STRANDS_AGENT, source="aws_strands", target=target)
    _assert_core_files(zf)
    _assert_valid_python(zf)


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_maf_source_all_targets(target: str):
    """MAF → every target must produce valid, complete output."""
    zf = _convert(MAF_AGENT, source="maf", target=target)
    _assert_core_files(zf)
    _assert_valid_python(zf)


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_crewai_source_all_targets(target: str):
    """CrewAI → every target must produce valid, complete output."""
    zf = _convert(CREWAI_AGENT, source="crewai", target=target)
    _assert_core_files(zf)
    _assert_valid_python(zf)


# ── Per-target idiom guards ────────────────────────────────────────────────────
# These confirm that each target's own vocabulary appears in the output, not
# residue from any other framework. They run against the LangGraph source (most
# comprehensive parser) so that they isolate target-generator quality only.

def test_maf_output_contains_maf_idioms():
    zf = _convert(LANGGRAPH_AGENT, source="langgraph", target="maf")
    orch = zf.read("orchestrator.py").decode()
    assert "WorkflowBuilder" in orch or "ChatAgent" in orch, "No MAF orchestration in output"
    assert "StateGraph" not in orch, "LangGraph residue in MAF output"
    assert "from crewai" not in orch, "CrewAI residue in MAF output"
    assert "from strands" not in orch, "Strands residue in MAF output"


def test_langgraph_output_contains_langgraph_idioms():
    zf = _convert(LANGGRAPH_AGENT, source="langgraph", target="langgraph")
    orch = zf.read("orchestrator.py").decode()
    assert "StateGraph" in orch or "def run" in orch, "No LangGraph orchestration in output"
    assert "WorkflowBuilder" not in orch, "MAF residue in LangGraph output"
    assert "from crewai" not in orch, "CrewAI residue in LangGraph output"
    assert "from strands" not in orch, "Strands residue in LangGraph output"


def test_crewai_output_contains_crewai_idioms():
    zf = _convert(LANGGRAPH_AGENT, source="langgraph", target="crewai")
    orch = zf.read("orchestrator.py").decode()
    assert "from crewai import" in orch, "No CrewAI import in output"
    assert "Task(" in orch or "Flow" in orch, "No CrewAI Task or Flow in output"
    assert "WorkflowBuilder" not in orch, "MAF residue in CrewAI output"
    assert "StateGraph" not in orch, "LangGraph residue in CrewAI output"


def test_aws_strands_output_contains_strands_idioms():
    zf = _convert(LANGGRAPH_AGENT, source="langgraph", target="aws_strands")
    orch = zf.read("orchestrator.py").decode()
    assert "from strands import" in orch or "strands" in orch.lower(), \
        "No Strands idiom in output"
    assert "WorkflowBuilder" not in orch, "MAF residue in Strands output"
    assert "from crewai" not in orch, "CrewAI residue in Strands output"


# ── Requirements-file guards ───────────────────────────────────────────────────

@pytest.mark.parametrize("source,target,must_have,must_not_have", [
    ("langgraph", "maf",         ["agent-framework"], ["langgraph", "crewai", "strands-agents"]),
    ("langgraph", "crewai",      ["crewai"],          ["langgraph", "agent-framework", "strands-agents"]),
    ("langgraph", "aws_strands", ["strands-agents"],  ["langgraph", "crewai", "agent-framework"]),
    ("aws_strands", "maf",       ["agent-framework"], ["strands-agents", "crewai", "langgraph"]),
    ("maf",       "crewai",      ["crewai"],          ["agent-framework", "langgraph", "strands-agents"]),
])
def test_requirements_has_only_target_framework(source, target, must_have, must_not_have):
    """requirements.txt must list the target SDK and must not list any source SDK."""
    zf = _convert(
        {
            "langgraph": LANGGRAPH_AGENT,
            "aws_strands": AWS_STRANDS_AGENT,
            "maf": MAF_AGENT,
            "crewai": CREWAI_AGENT,
        }[source],
        source=source,
        target=target,
    )
    reqs = zf.read("requirements.txt").decode().lower()
    for pkg in must_have:
        assert pkg.lower() in reqs, f"'{pkg}' missing from requirements.txt (target={target})"
    for pkg in must_not_have:
        assert pkg.lower() not in reqs, \
            f"Source package '{pkg}' leaked into requirements.txt (source={source}, target={target})"
