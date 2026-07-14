"""Tests for Module 2 -- code parser."""

from __future__ import annotations

from converter.parser.code_parser import (
    extract_config,
    extract_functions,
    extract_graph,
    extract_imports,
    extract_preamble,
    extract_state,
    extract_tools,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_extract_simple_tool():
    src = '''
from langchain_core.tools import tool

@tool
def read_tests(path: str, limit: int = 10) -> list[str]:
    """Read test files from the repo."""
    return []
'''
    tools = extract_tools(src, source_file="tools/repo_reader.py")
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "read_tests"
    assert t.docstring == "Read test files from the repo."
    assert t.returns == "list[str]"
    assert t.source_file == "tools/repo_reader.py"
    assert [p.name for p in t.params] == ["path", "limit"]
    limit = t.params[1]
    assert limit.annotation == "int"
    assert limit.default == "10"


def test_extract_tool_with_call_decorator_and_attribute():
    src = '''
import langchain_core.tools as lc

@lc.tool("named")
def detect_flaky_tests(runs):
    pass

@tool(return_direct=True)
def other():
    pass
'''
    names = {t.name for t in extract_tools(src)}
    assert names == {"detect_flaky_tests", "other"}


def test_non_tool_functions_ignored():
    src = '''
def helper(x):
    return x

@staticmethod
def not_a_tool():
    pass
'''
    assert extract_tools(src) == []


def test_async_tool_and_self_skipped():
    src = '''
class Toolbox:
    @tool
    async def scan(self, target: str):
        """Scan."""
'''
    tools = extract_tools(src)
    assert len(tools) == 1
    assert [p.name for p in tools[0].params] == ["target"]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def test_extract_linear_graph():
    src = '''
from langgraph.graph import StateGraph, START, END

g = StateGraph(State)
g.add_node("read", read_fn)
g.add_node("analyze", analyze_fn)
g.set_entry_point("read")
g.add_edge("read", "analyze")
g.add_edge("analyze", END)
'''
    graph = extract_graph(src)
    node_names = {n.name for n in graph.nodes}
    assert node_names == {"read", "analyze"}
    assert graph.entry_point == "read"
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("read", "analyze") in edges
    assert ("analyze", "END") in edges


def test_add_node_single_callable_infers_name():
    src = '''
g.add_node(planner)
'''
    graph = extract_graph(src)
    assert graph.nodes[0].name == "planner"
    assert graph.nodes[0].target_callable == "planner"


def test_entry_point_from_start_edge():
    src = '''
from langgraph.graph import START
g.add_edge(START, "first")
'''
    graph = extract_graph(src)
    assert graph.entry_point == "first"


def test_conditional_edges():
    src = '''
g.add_conditional_edges(
    "gate",
    route_fn,
    {"revise": "generate", "done": END},
)
'''
    graph = extract_graph(src)
    assert len(graph.conditional_edges) == 1
    ce = graph.conditional_edges[0]
    assert ce.source == "gate"
    assert ce.router == "route_fn"
    assert ce.outcomes == {"revise": "generate", "done": "END"}


def test_set_finish_point():
    src = 'g.set_finish_point("last")'
    graph = extract_graph(src)
    assert ("last", "END") in {(e.source, e.target) for e in graph.edges}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_extract_state_with_append_only():
    src = '''
from typing import Annotated, TypedDict
from operator import add

class AgentState(TypedDict):
    project_id: str
    retry_count: int
    audit_log: Annotated[list, add]
    tool_errors: Annotated[list[str], add]
'''
    fields = extract_state(src)
    by_name = {f.name: f for f in fields}
    assert by_name["project_id"].type == "str"
    assert by_name["project_id"].is_append_only is False
    assert by_name["audit_log"].is_append_only is True
    assert by_name["tool_errors"].is_append_only is True


def test_state_operator_add_attribute():
    src = '''
import operator
from typing import Annotated, TypedDict

class S(TypedDict):
    log: Annotated[list, operator.add]
'''
    fields = extract_state(src)
    assert fields[0].is_append_only is True


def test_no_typeddict_returns_empty():
    assert extract_state("x = 1\n") == []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_extract_config_llm_and_env():
    src = '''
import os
from langchain_openai import ChatOpenAI

MAX_GEN_RETRIES = 3
COVERAGE_FLOOR = 0.8

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
key = os.getenv("OPENAI_API_KEY")
region = os.environ.get("AWS_REGION")
token = os.environ["HF_TOKEN"]
'''
    cfg = extract_config(src)
    assert cfg.llm_kwargs["model"] == "gpt-4o"
    assert cfg.temperature == 0.2
    assert set(cfg.env_vars) == {"OPENAI_API_KEY", "AWS_REGION", "HF_TOKEN"}
    assert cfg.constants["MAX_GEN_RETRIES"] == 3
    assert cfg.constants["COVERAGE_FLOOR"] == 0.8


def test_temperature_none_when_absent():
    src = 'from x import ChatAnthropic\nllm = ChatAnthropic(model="claude")\n'
    cfg = extract_config(src)
    assert cfg.temperature is None
    assert cfg.llm_kwargs["model"] == "claude"


# ---------------------------------------------------------------------------
# Bodies, functions, imports, preamble (logic porting inputs)
# ---------------------------------------------------------------------------

def test_tool_body_captured():
    src = '''
@tool
def read_tests(path: str) -> list:
    """Reads."""
    import os
    return os.listdir(path)
'''
    tool = extract_tools(src)[0]
    assert tool.body is not None
    assert "os.listdir(path)" in tool.body
    # docstring is stripped from the body (emitted separately).
    assert '"""Reads."""' not in tool.body


def test_extract_functions_with_bodies():
    src = '''
def generate(state):
    return {"x": 1}

async def fetch(url):
    return url

def route(state) -> str:
    return "done"
'''
    funcs = extract_functions(src, source_file="agent.py")
    assert set(funcs) == {"generate", "fetch", "route"}
    assert funcs["generate"].first_param == "state"
    assert funcs["fetch"].is_async is True
    assert funcs["route"].returns == "str"
    assert funcs["generate"].source_file == "agent.py"


def test_extract_imports_drops_langgraph():
    src = (
        "import os\n"
        "from langgraph.graph import StateGraph, END\n"
        "from langchain_openai import ChatOpenAI\n"
        "import langgraph\n"
    )
    imports = extract_imports(src)
    joined = "\n".join(imports)
    assert "import os" in joined
    assert "ChatOpenAI" in joined
    assert "langgraph" not in joined


def test_extract_preamble_keeps_setup_drops_graph_and_constants():
    src = (
        "MAX = 3\n"
        "llm = ChatOpenAI(model='x')\n"
        "g = StateGraph(State)\n"
        "g.add_node('a', a)\n"
        "app = g.compile()\n"
    )
    preamble = extract_preamble(src)
    joined = "\n".join(preamble)
    assert "llm = ChatOpenAI(model='x')" in joined
    assert "MAX" not in joined            # ALL_CAPS constant -> config.py
    assert "StateGraph" not in joined     # graph wiring dropped
    assert "compile" not in joined
