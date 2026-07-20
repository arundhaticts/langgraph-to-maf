"""Tests for the AWS Strands source adapter (single-agent tool-calling loop)."""

from __future__ import annotations

import ast
import io
import zipfile

from converter.adapters import (
    AWSStrandsSourceAdapter,
    detect_source_framework,
    get_source_adapter,
    list_source_frameworks,
)
from converter.contracts import OrchestrationMode
from converter.extractor import extract_components
from converter.scanner import scan_repo
from service import convert_folder

STRANDS_AGENT = '''
from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator


@tool
def word_count(text: str) -> int:
    """Count the words in some text."""
    return len(text.split())


model = BedrockModel(model_id="anthropic.claude-3")
assistant = Agent(model=model, tools=[word_count, calculator], system_prompt="Be helpful.")
result = assistant("How many words is this?")
'''

README = """# Strands Demo
## Purpose
A strands agent.
## Tools
- `word_count`: counts words
## Workflow
Single agent tool-calling loop.
"""


def _files():
    return [
        {"path": "a/README.md", "content": README},
        {"path": "a/agent.py", "content": STRANDS_AGENT},
    ]


def test_registry_lists_aws_strands_as_source():
    assert "aws_strands" in list_source_frameworks()
    assert isinstance(get_source_adapter("aws_strands"), AWSStrandsSourceAdapter)


def test_detect_strands_from_imports():
    assert detect_source_framework({"strands", "strands_tools"}) == "aws_strands"


def test_extract_graph_single_agent_node():
    graph = AWSStrandsSourceAdapter().extract_graph(STRANDS_AGENT)
    assert [n.name for n in graph.nodes] == ["assistant"]
    assert graph.entry_point == "assistant"
    assert {(e.source, e.target) for e in graph.edges} == {("assistant", "END")}


def test_non_agent_file_returns_none():
    assert AWSStrandsSourceAdapter().extract_graph("x = 1\n") is None


def test_end_to_end_single_agent_mode(tmp_path):
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "agent.py").write_text(STRANDS_AGENT, encoding="utf-8")

    manifest = scan_repo(str(tmp_path))
    assert manifest.detected_framework == "aws_strands"

    inventory = extract_components(manifest)
    assert [n.name for n in inventory.graph.nodes] == ["assistant"]
    assert "word_count" in {t.name for t in inventory.tools}


def test_strands_to_maf_wires_single_agent():
    z = convert_folder(_files(), "manual", target="maf", source="aws_strands")
    zf = zipfile.ZipFile(io.BytesIO(z))
    orch = zf.read("orchestrator.py").decode()
    # SINGLE_AGENT mode -> a ChatAgent is built with the converted tools.
    assert "build_agent" in orch and "ChatAgent" in orch
    assert "word_count" in orch
    for name in zf.namelist():
        if name.endswith(".py"):
            ast.parse(zf.read(name).decode("utf-8"))


def test_strands_to_langgraph_single_node():
    z = convert_folder(_files(), "manual", target="langgraph", source="aws_strands")
    zf = zipfile.ZipFile(io.BytesIO(z))
    orch = zf.read("orchestrator.py").decode()
    assert 'builder.add_node("assistant", assistant)' in orch
    assert 'builder.set_entry_point("assistant")' in orch
    assert 'builder.add_edge("assistant", END)' in orch
    for name in zf.namelist():
        if name.endswith(".py"):
            ast.parse(zf.read(name).decode("utf-8"))
