"""Tests for Module 5 -- IR builder."""

from __future__ import annotations

import json
import os

from converter.contracts import (
    ComponentInventory,
    ConditionalEdge,
    ConfigSpec,
    FunctionSpec,
    GraphEdge,
    GraphNode,
    GraphSpec,
    IRMetadata,
    NodeRole,
    OrchestrationMode,
    OrchestrationPattern,
    ReadmeSections,
    StateField,
    ToolParam,
    ToolSpec,
)
from converter.ir import build_ir, detect_target_version, write_ir_json


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


# ---------------------------------------------------------------------------
# Phase 0 -- orchestration mode, provider, checkpointer, version pinning
# ---------------------------------------------------------------------------

def test_single_agent_mode_for_trivial_graph():
    graph = GraphSpec(nodes=[GraphNode("act")], edges=[GraphEdge("act", "END")], entry_point="act")
    ir = build_ir(_inv(graph))
    assert ir.metadata.orchestration_mode is OrchestrationMode.SINGLE_AGENT


def test_graph_workflow_mode_when_branching():
    graph = GraphSpec(
        nodes=[GraphNode("gate"), GraphNode("x"), GraphNode("y")],
        conditional_edges=[ConditionalEdge("gate", "route", {"go": "x", "stop": "y"})],
        entry_point="gate",
    )
    ir = build_ir(_inv(graph))
    assert ir.metadata.orchestration_mode is OrchestrationMode.GRAPH_WORKFLOW


def test_llm_provider_and_checkpointer_recorded():
    graph = GraphSpec(nodes=[GraphNode("a")], edges=[GraphEdge("a", "END")], entry_point="a")
    inv = _inv(graph, config=ConfigSpec(llm_provider="ChatOpenAI"), checkpointer="MemorySaver")
    ir = build_ir(inv)
    assert ir.metadata.llm_provider == "ChatOpenAI"
    assert ir.metadata.checkpointer == "MemorySaver"


def test_target_version_pin_is_none_when_sdk_absent():
    # No such distribution -> best-effort pin returns None, never raises.
    assert detect_target_version("no-such-framework-xyz") is None


def test_target_version_recorded_in_metadata():
    graph = GraphSpec(nodes=[GraphNode("a")], edges=[GraphEdge("a", "END")], entry_point="a")
    ir = build_ir(_inv(graph), target_framework="no-such-framework-xyz")
    # Field is populated (value is whatever is installed; None here).
    assert ir.metadata.target_framework_version == detect_target_version("no-such-framework-xyz")


# ---------------------------------------------------------------------------
# Phase 1 -- data-flow, flat edges, loop guards, HITL payload
# ---------------------------------------------------------------------------

def _flow_inv():
    """A loop-with-exit graph whose node/router bodies drive the analysis."""
    graph = GraphSpec(
        nodes=[
            GraphNode("generate", target_callable="generate"),
            GraphNode("gate", target_callable="route"),
        ],
        edges=[GraphEdge("generate", "gate")],
        conditional_edges=[
            ConditionalEdge("gate", "route", {"revise": "generate", "done": "END"})
        ],
        entry_point="generate",
    )
    functions = {
        "generate": FunctionSpec(
            name="generate",
            params=[ToolParam("state")],
            source=(
                "def generate(state):\n"
                "    c = state['coverage']\n"
                "    hits = read_tests(c)\n"
                "    if c < MAX_GEN_RETRIES:\n"
                "        pass\n"
                "    return {'coverage': c + 0.1, 'audit_log': ['did']}"
            ),
            body="c = state['coverage']\n...",
        ),
        "route": FunctionSpec(
            name="route",
            params=[ToolParam("state")],
            source="def route(state):\n    return 'done' if state['coverage'] >= 1 else 'revise'",
            body="return 'done' if state['coverage'] >= 1 else 'revise'",
        ),
    }
    return _inv(
        graph,
        state=[
            StateField("coverage", "float"),
            StateField("audit_log", "list", is_append_only=True),
        ],
        tools=[ToolSpec(name="read_tests")],
        functions=functions,
        config=ConfigSpec(constants={"MAX_GEN_RETRIES": 3}),
    )


def test_node_data_flow_reads_writes_tools():
    ir = build_ir(_flow_inv())
    gen = next(n for n in ir.workflow.nodes if n.name == "generate")
    assert gen.reads == ["coverage"]
    assert gen.writes == ["audit_log", "coverage"]
    assert gen.calls_tools == ["read_tests"]


def test_flat_edges_are_explicit_triples_with_loop_flag():
    ir = build_ir(_flow_inv())
    flat = ir.workflow.flat_edges
    # Unconditional spine edge.
    assert any(e.source == "generate" and e.target == "gate" and e.condition_label is None
               for e in flat)
    # Conditional outcomes flattened, with the back-edge marked is_loop.
    revise = next(e for e in flat if e.condition_label == "revise")
    assert revise.source == "gate" and revise.target == "generate"
    assert revise.router == "route" and revise.is_loop is True
    done = next(e for e in flat if e.condition_label == "done")
    assert done.target == "END" and done.is_loop is False


def test_loop_guard_captures_router_and_counter():
    ir = build_ir(_flow_inv())
    guards = ir.workflow.loop_guards
    assert len(guards) == 1
    g = guards[0]
    assert g.loop_node == "generate"
    assert g.router == "route"
    assert g.counter_const == "MAX_GEN_RETRIES"
    assert "done" in g.exit_labels


def test_hitl_payload_and_resume_contract_extracted():
    graph = GraphSpec(
        nodes=[GraphNode("approve", target_callable="approve")],
        edges=[GraphEdge("approve", "END")],
        entry_point="approve",
    )
    functions = {
        "approve": FunctionSpec(
            name="approve",
            params=[ToolParam("state")],
            source=(
                "def approve(state):\n"
                "    decision = interrupt({'removals': state['removals']})\n"
                "    return {'ok': decision}"
            ),
            body="decision = interrupt({'removals': state['removals']})",
        )
    }
    ir = build_ir(_inv(graph, functions=functions))
    points = ir.workflow.hitl_points
    assert len(points) == 1
    assert points[0].node == "approve"
    assert "removals" in (points[0].payload or "")
    assert points[0].resume_contract and "interrupt()" in points[0].resume_contract
