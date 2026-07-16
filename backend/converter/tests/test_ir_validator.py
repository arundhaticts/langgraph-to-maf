"""Tests for Phase 2 -- the IR validation gate."""

from __future__ import annotations

from converter.contracts import (
    ComponentInventory,
    ConditionalEdge,
    ConfigSpec,
    FunctionSpec,
    GraphEdge,
    GraphNode,
    GraphSpec,
    StateField,
    ToolParam,
    ToolSpec,
)
from converter.ir import build_ir, validate_ir


def _inv(graph: GraphSpec, **kw) -> ComponentInventory:
    return ComponentInventory(graph=graph, **kw)


def test_clean_ir_has_no_issues():
    graph = GraphSpec(
        nodes=[GraphNode("gen", target_callable="gen"), GraphNode("gate", target_callable="route")],
        edges=[GraphEdge("gen", "gate")],
        conditional_edges=[ConditionalEdge("gate", "route", {"revise": "gen", "done": "END"})],
        entry_point="gen",
    )
    functions = {
        "gen": FunctionSpec(
            name="gen",
            params=[ToolParam("state")],
            source=(
                "def gen(state):\n"
                "    x = read(state['coverage'])\n"
                "    if state['coverage'] < MAX:\n        pass\n"
                "    return {'coverage': 1.0}"
            ),
        ),
        "route": FunctionSpec(
            name="route",
            params=[ToolParam("state")],
            source="def route(state):\n    return 'done' if state['coverage'] >= 1 else 'revise'",
        ),
    }
    inv = _inv(
        graph,
        state=[StateField("coverage", "float")],
        tools=[ToolSpec(name="read")],
        functions=functions,
        config=ConfigSpec(constants={"MAX": 3}),
    )
    ir = build_ir(inv)
    assert validate_ir(ir) == []


def test_flags_missing_router():
    graph = GraphSpec(
        nodes=[GraphNode("gate"), GraphNode("x")],
        conditional_edges=[ConditionalEdge("gate", "route", {"go": "x", "stop": "END"})],
        entry_point="gate",
    )
    ir = build_ir(_inv(graph))  # no functions -> router 'route' undefined
    issues = validate_ir(ir)
    assert any("Router 'route'" in i for i in issues)


def test_flags_orphan_tool():
    graph = GraphSpec(nodes=[GraphNode("a")], edges=[GraphEdge("a", "END")], entry_point="a")
    ir = build_ir(_inv(graph, tools=[ToolSpec(name="never_called")]))
    assert any("never_called" in i and "not referenced" in i for i in validate_ir(ir))


def test_flags_undeclared_state_field():
    graph = GraphSpec(
        nodes=[GraphNode("n", target_callable="n")],
        edges=[GraphEdge("n", "END")],
        entry_point="n",
    )
    functions = {
        "n": FunctionSpec(
            name="n",
            params=[ToolParam("state")],
            source="def n(state):\n    return {'undeclared': 1}",
        )
    }
    ir = build_ir(_inv(graph, state=[StateField("declared", "int")], functions=functions))
    assert any("undeclared" in i and "not declared" in i for i in validate_ir(ir))


def test_flags_hitl_without_payload():
    graph = GraphSpec(
        nodes=[GraphNode("approve", target_callable="approve")],
        edges=[GraphEdge("approve", "END")],
        entry_point="approve",
    )
    # HITL by name but body has no interrupt() payload.
    functions = {
        "approve": FunctionSpec(
            name="approve", params=[ToolParam("state")], source="def approve(state):\n    return state"
        )
    }
    ir = build_ir(_inv(graph, functions=functions))
    assert any("approve" in i and "payload" in i for i in validate_ir(ir))


def test_flags_edge_to_unknown_node():
    graph = GraphSpec(
        nodes=[GraphNode("a")],
        edges=[GraphEdge("a", "ghost")],
        entry_point="a",
    )
    ir = build_ir(_inv(graph))
    assert any("ghost" in i and "not a defined node" in i for i in validate_ir(ir))
