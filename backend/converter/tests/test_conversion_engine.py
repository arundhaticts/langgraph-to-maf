"""Tests for Module 6 -- conversion engine (Tier 1/2/3)."""

from __future__ import annotations

from converter.config import Config, ConversionMode
from converter.contracts import (
    IR,
    ConditionalEdge,
    ConfigSpec,
    GraphEdge,
    GraphNode,
    IRMetadata,
    NodeRole,
    OrchestrationPattern,
    StateField,
    Tier,
    Tier3Result,
    ToolSpec,
    WorkflowSpec,
)
from converter.engine import convert


def _ir(workflow: WorkflowSpec | None = None, **kw) -> IR:
    return IR(
        metadata=IRMetadata(target_framework="maf"),
        workflow=workflow,
        **kw,
    )


def _units_by_rule(result) -> dict:
    out: dict = {}
    for u in result.units:
        out.setdefault(u.rule_id, []).append(u)
    return out


# ---------------------------------------------------------------------------
# Tier 1 rules
# ---------------------------------------------------------------------------

def test_tool_rule_r01():
    ir = _ir(tools=[ToolSpec(name="read_tests"), ToolSpec(name="detect_flaky_tests")])
    units = _units_by_rule(convert(ir))
    r01 = units["R-01"]
    targets = {u.target_ref for u in r01}
    assert targets == {"ReadTestsPlugin", "DetectFlakyTestsPlugin"}
    assert all(u.tier is Tier.TIER1 for u in r01)


def test_state_rules_r02_and_reducers():
    ir = _ir(
        state=[
            StateField("project_id", "str"),
            StateField("audit_log", "list", is_append_only=True),
            StateField("tool_errors", "list", is_append_only=True),
        ]
    )
    units = _units_by_rule(convert(ir))
    assert units["R-02"][0].source_ref == "project_id"
    assert units["R-11"][0].source_ref == "audit_log"
    assert units["R-12"][0].source_ref == "tool_errors"


def test_config_rules_r14_r10():
    ir = _ir(
        config=ConfigSpec(constants={"MAX": 3}, llm_kwargs={"model": "x"})
    )
    units = _units_by_rule(convert(ir))
    assert units["R-14"][0].source_ref == "MAX"
    assert units["R-10"][0].source_ref == "llm"


def test_hitl_without_gemini_flags_manual_with_original_logic():
    from converter.contracts import FunctionSpec, ToolParam

    wf = WorkflowSpec(
        pattern=OrchestrationPattern.LINEAR,
        nodes=[GraphNode("start", role=NodeRole.ENTRY), GraphNode("hitl_x", role=NodeRole.HITL)],
        entry_point="start",
    )
    ir = _ir(
        wf,
        functions={
            "hitl_x": FunctionSpec(
                name="hitl_x",
                params=[ToolParam("state")],
                body="decision = interrupt({})\nreturn {}",
            )
        },
    )
    # No Gemini key/client -> deterministic stub + manual flag with original logic.
    r08 = _units_by_rule(convert(ir))["R-08"][0]
    assert r08.source_ref == "hitl_x"
    assert r08.tier is Tier.TIER1
    assert r08.manual_action is not None
    assert "interrupt(" in r08.reasoning  # original logic carried for the follow-up


def test_hitl_with_gemini_generates_flow_and_flags_review():
    wf = WorkflowSpec(
        pattern=OrchestrationPattern.LINEAR,
        nodes=[GraphNode("start", role=NodeRole.ENTRY), GraphNode("hitl_x", role=NodeRole.HITL)],
        entry_point="start",
    )

    def fake_hitl(ir_, node_name, source, cfg):
        return Tier3Result("hitl", "return ctx", "approval flow", 0.9)

    r08 = _units_by_rule(convert(_ir(wf), hitl_resolver=fake_hitl))["R-08"][0]
    assert r08.tier is Tier.TIER3
    assert r08.generated_code == "return ctx"
    assert r08.needs_review is True


def test_checkpointer_rule_r15():
    ir = _ir(config=ConfigSpec(constants={"SAVER": "MemorySaver"}))
    units = _units_by_rule(convert(ir))
    assert units["R-15"][0].manual_action is not None


# ---------------------------------------------------------------------------
# Workflow tier resolution
# ---------------------------------------------------------------------------

def test_linear_resolves_tier1():
    wf = WorkflowSpec(pattern=OrchestrationPattern.LINEAR, entry_point="a")
    unit = next(u for u in convert(_ir(wf)).units if u.source_ref == "workflow")
    assert unit.tier is Tier.TIER1
    assert unit.rule_id == "R-04"


def test_simple_branch_resolves_tier1():
    wf = WorkflowSpec(
        pattern=OrchestrationPattern.BRANCH,
        conditional_edges=[ConditionalEdge("gate", "route", {"a": "x", "b": "y"})],
    )
    unit = next(u for u in convert(_ir(wf)).units if u.source_ref == "workflow")
    assert unit.tier is Tier.TIER1
    assert unit.rule_id == "R-05"


def test_loop_resolves_tier1():
    wf = WorkflowSpec(pattern=OrchestrationPattern.LOOP_WITH_EXIT, entry_point="a")
    unit = next(u for u in convert(_ir(wf)).units if u.source_ref == "workflow")
    assert unit.tier is Tier.TIER1
    assert unit.rule_id == "R-06"


def test_tier2_reclassifies_from_readme():
    # 3-outcome branch => agent_driven structurally, but README prose says loop.
    wf = WorkflowSpec(
        pattern=OrchestrationPattern.AGENT_DRIVEN,
        conditional_edges=[
            ConditionalEdge("gate", "route", {"a": "x", "b": "y", "c": "z"})
        ],
        readme_description="It retries and loops back up to 3 times.",
    )
    unit = next(u for u in convert(_ir(wf)).units if u.source_ref == "workflow")
    assert unit.tier is Tier.TIER2
    assert "loop" in unit.target_ref


def test_tier3_used_when_structure_and_readme_fail():
    wf = WorkflowSpec(pattern=OrchestrationPattern.AGENT_DRIVEN, readme_description="")

    def fake_resolver(ir, workflow, config):
        return Tier3Result(
            pattern="agent_driven",
            generated_code="def run(ctx):\n    return ctx",
            reasoning="dynamic routing",
            confidence=0.9,
        )

    result = convert(_ir(wf), tier3_resolver=fake_resolver)
    unit = next(u for u in result.units if u.source_ref == "workflow")
    assert unit.tier is Tier.TIER3
    assert unit.needs_review is False
    assert "def run" in unit.generated_code


def test_tier3_below_threshold_flags_review():
    wf = WorkflowSpec(pattern=OrchestrationPattern.AGENT_DRIVEN, readme_description="")

    def low_conf(ir, workflow, config):
        return Tier3Result("agent_driven", "code", "unsure", confidence=0.4)

    result = convert(_ir(wf), tier3_resolver=low_conf)
    unit = next(u for u in result.units if u.source_ref == "workflow")
    assert unit.tier is Tier.TIER3
    assert unit.needs_review is True


def test_deterministic_mode_leaves_unresolved():
    # Approach 1: no LLM fallback. Default resolver returns None -> unresolved.
    wf = WorkflowSpec(pattern=OrchestrationPattern.AGENT_DRIVEN, readme_description="")
    config = Config(mode=ConversionMode.DETERMINISTIC)
    result = convert(_ir(wf), config=config)
    unit = next(u for u in result.units if u.source_ref == "workflow")
    assert unit.tier is Tier.UNRESOLVED
    assert unit.manual_action is not None
