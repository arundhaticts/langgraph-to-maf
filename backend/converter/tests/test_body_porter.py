"""Tests for the deterministic body porter (state -> ctx transform)."""

from __future__ import annotations

import ast

from converter.contracts import FunctionSpec, ToolParam, ToolSpec
from converter.generator.body_porter import (
    plugin_method_body,
    port_node_function,
    port_plain_function,
    transform_state_body,
)


def test_subscript_and_get_transform():
    src = 'a = state["x"]\nb = state.get("y")\nc = state.get("z", 5)'
    out = transform_state_body(src, "state", set())
    assert "a = ctx.x" in out
    assert "b = ctx.y" in out
    assert "c = ctx.z if ctx.z is not None else 5" in out


def test_dict_return_becomes_assignments():
    src = 'return {"count": n + 1, "name": "x"}'
    out = transform_state_body(src, "state", set())
    assert "ctx.count = n + 1" in out
    assert "ctx.name = 'x'" in out
    assert out.strip().endswith("return ctx")


def test_append_only_return_uses_append():
    src = 'return {"log": ["entry"], "value": 3}'
    out = transform_state_body(src, "state", {"log"})
    assert "ctx.log.append('entry')" in out
    assert "ctx.value = 3" in out


def test_port_node_function_full():
    f = FunctionSpec(
        name="generate",
        params=[ToolParam("state")],
        body='c = state["coverage"]\nreturn {"coverage": c + 0.1}',
    )
    out = port_node_function(f, "generate", "AgentContext", set())
    assert out.startswith("def generate(ctx: AgentContext) -> AgentContext:")
    assert "ctx.coverage = c + 0.1" in out
    ast.parse(out)  # valid Python


def test_port_node_function_missing_body_is_stub():
    out = port_node_function(None, "mystery", "AgentContext", set())
    assert "TODO: port logic from source node 'mystery'" in out
    assert out.strip().endswith("return ctx")
    ast.parse(out)


def test_port_router_renames_state_param():
    f = FunctionSpec(
        name="route",
        params=[ToolParam("state")],
        body='if state["done"]:\n    return "done"\nreturn "revise"',
        returns="str",
    )
    out = port_plain_function(f, "state", set())
    assert out.startswith("def route(ctx) -> str:")
    assert "if ctx.done:" in out
    ast.parse(out)


def test_port_helper_without_state_kept_verbatim():
    f = FunctionSpec(name="clamp", params=[ToolParam("v"), ToolParam("lo")], body="return max(v, lo)")
    out = port_plain_function(f, "state", set())
    assert out.startswith("def clamp(v, lo):")
    assert "return max(v, lo)" in out


def test_plugin_method_body_uses_real_logic():
    tool = ToolSpec(name="read", docstring="Reads.", body="return [1, 2, 3]")
    body = plugin_method_body(tool, indent=8)
    assert '"""Reads."""' in body
    assert "return [1, 2, 3]" in body
    assert "NotImplementedError" not in body


def test_plugin_method_body_stub_when_no_body():
    tool = ToolSpec(name="mystery")
    body = plugin_method_body(tool, indent=8)
    assert "NotImplementedError" in body
    assert "TODO: port logic from source tool 'mystery'" in body
