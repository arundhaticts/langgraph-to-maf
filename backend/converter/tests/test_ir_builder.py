"""Tests for Module 5 -- IR builder."""

from __future__ import annotations

import json
import os

from converter.contracts import (
    ComponentInventory,
    ConditionalEdge,
    ConfigSpec,
    GraphEdge,
    GraphNode,
    GraphSpec,
    IRMetadata,
    NodeRole,
    OrchestrationPattern,
    ReadmeSections,
    StateField,
    ToolSpec,
)
from converter.ir import build_ir, write_ir_json


def _inv(graph: GraphSpec, **kw) -> ComponentInventory:
    return ComponentInventory(graph=graph, **kw)


# ---------------------------------------------------------------------------
# Orchestration pattern classification
# ---------------------------------------------------------------------------

def test_linear_pattern():
    graph = GraphSpec(
        nodes=[GraphNode("a"), GraphNode("b")],
        edges=[GraphEdge("a", "b"), GraphEdge("b", "END")],
        entry_point="a",
    )
    ir = build_ir(_inv(graph))
    assert ir.workflow.pattern is OrchestrationPattern.LINEAR


def test_branch_pattern():
    graph = GraphSpec(
        nodes=[GraphNode("gate"), GraphNode("x"), GraphNode("y")],
        conditional_edges=[
            ConditionalEdge("gate", "route", {"go": "x", "stop": "y"})
        ],
        entry_point="gate",
    )
    ir = build_ir(_inv(graph))
    assert ir.workflow.pattern is OrchestrationPattern.BRANCH


def test_loop_pattern():
    graph = GraphSpec(
        nodes=[GraphNode("gen"), GraphNode("check")],
        edges=[GraphEdge("gen", "check"), GraphEdge("check", "gen")],
        entry_point="gen",
    )
    ir = build_ir(_inv(graph))
    assert ir.workflow.pattern is OrchestrationPattern.LOOP


def test_loop_with_exit_pattern():
    graph = GraphSpec(
        nodes=[GraphNode("gen"), GraphNode("gate")],
        edges=[GraphEdge("gen", "gate")],
        conditional_edges=[
            ConditionalEdge("gate", "route", {"revise": "gen", "done": "END"})
        ],
        entry_point="gen",
    )
    ir = build_ir(_inv(graph))
    assert ir.workflow.pattern is OrchestrationPattern.LOOP_WITH_EXIT


def test_agent_driven_on_three_outcomes():
    graph = GraphSpec(
        nodes=[GraphNode("gate"), GraphNode("a"), GraphNode("b"), GraphNode("c")],
        conditional_edges=[
            ConditionalEdge("gate", "route", {"a": "a", "b": "b", "c": "c"})
        ],
        entry_point="gate",
    )
    ir = build_ir(_inv(graph))
    assert ir.workflow.pattern is OrchestrationPattern.AGENT_DRIVEN


def test_agent_driven_on_dynamic_router():
    graph = GraphSpec(
        nodes=[GraphNode("gate"), GraphNode("a")],
        conditional_edges=[ConditionalEdge("gate", "route", {})],  # no static map
        entry_point="gate",
    )
    ir = build_ir(_inv(graph))
    assert ir.workflow.pattern is OrchestrationPattern.AGENT_DRIVEN


# ---------------------------------------------------------------------------
# Node role classification
# ---------------------------------------------------------------------------

def test_node_roles():
    graph = GraphSpec(
        nodes=[
            GraphNode("read"),
            GraphNode("gate"),
            GraphNode("generate"),
            GraphNode("finish"),
            GraphNode("hitl_approve"),
        ],
        edges=[
            GraphEdge("read", "gate"),
            GraphEdge("generate", "gate"),
            GraphEdge("finish", "END"),
        ],
        conditional_edges=[
            ConditionalEdge("gate", "route", {"revise": "generate", "done": "finish"})
        ],
        entry_point="read",
    )
    build_ir(_inv(graph))
    roles = {n.name: n.role for n in graph.nodes}
    assert roles["read"] is NodeRole.ENTRY
    assert roles["gate"] is NodeRole.BRANCH
    assert roles["generate"] is NodeRole.LOOP        # gate->generate->gate cycle
    assert roles["finish"] is NodeRole.TERMINAL
    assert roles["hitl_approve"] is NodeRole.HITL


def test_isolated_node_is_aux():
    graph = GraphSpec(
        nodes=[GraphNode("main"), GraphNode("orphan")],
        edges=[GraphEdge("main", "END")],
        entry_point="main",
    )
    build_ir(_inv(graph))
    roles = {n.name: n.role for n in graph.nodes}
    assert roles["orphan"] is NodeRole.AUX


# ---------------------------------------------------------------------------
# Metadata, config, and serialisation
# ---------------------------------------------------------------------------

def test_metadata_and_readme_wired():
    graph = GraphSpec(nodes=[GraphNode("a")], edges=[GraphEdge("a", "END")], entry_point="a")
    readme = ReadmeSections(purpose="Do things.", workflow_description="Read then stop.")
    ir = build_ir(_inv(graph), readme=readme, target_framework="maf")
    assert ir.metadata.description == "Do things."
    assert ir.metadata.target_framework == "maf"
    assert ir.workflow.readme_description == "Read then stop."


def test_temperature_stays_none_when_absent():
    graph = GraphSpec(nodes=[GraphNode("a")])
    ir = build_ir(_inv(graph, config=ConfigSpec()))
    assert ir.config.temperature is None


def test_to_json_dict_is_serialisable():
    graph = GraphSpec(
        nodes=[GraphNode("a")],
        edges=[GraphEdge("a", "END")],
        conditional_edges=[],
        entry_point="a",
    )
    inv = _inv(
        graph,
        tools=[ToolSpec(name="t")],
        state=[StateField(name="x", type="str", is_append_only=True)],
    )
    ir = build_ir(inv)
    d = ir.to_json_dict()
    # Round-trips through JSON and enums are stringified.
    dumped = json.dumps(d)
    assert '"pattern": "linear"' in dumped
    assert d["tools"][0]["name"] == "t"
    assert d["state"][0]["is_append_only"] is True


def test_write_ir_json_checkpoint(tmp_path):
    graph = GraphSpec(nodes=[GraphNode("a")], edges=[GraphEdge("a", "END")], entry_point="a")
    ir = build_ir(_inv(graph))
    out = os.path.join(str(tmp_path), "ir.json")
    write_ir_json(ir, out)
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["workflow"]["entry_point"] == "a"
