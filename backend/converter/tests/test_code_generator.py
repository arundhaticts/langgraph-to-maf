"""Tests for Module 7 -- code generator."""

from __future__ import annotations

import ast
import os

from converter.contracts import (
    IR,
    ConditionalEdge,
    ConfigSpec,
    ConversionResult,
    ConversionUnit,
    FileAction,
    FileEntry,
    FileType,
    GraphEdge,
    GraphNode,
    IRMetadata,
    NodeRole,
    OrchestrationPattern,
    StateField,
    Tier,
    ToolParam,
    ToolSpec,
    WorkflowSpec,
)
from converter.engine import convert
from converter.generator import generate, generate_from_paths


def _read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        return fh.read()


def _sample_ir() -> IR:
    return IR(
        metadata=IRMetadata(target_framework="maf"),
        tools=[
            ToolSpec(
                name="read_tests",
                params=[ToolParam("path", "str", None)],
                docstring="Read.",
                returns="list",
                source_file="tools.py",
            )
        ],
        state=[
            StateField("project_id", "str"),
            StateField("audit_log", "list", is_append_only=True),
        ],
        config=ConfigSpec(constants={"MAX_GEN_RETRIES": 3}),
        workflow=WorkflowSpec(
            pattern=OrchestrationPattern.LINEAR,
            nodes=[GraphNode("read", role=NodeRole.ENTRY)],
            edges=[GraphEdge("read", "END")],
            entry_point="read",
        ),
        files=[FileEntry("tools.py", FileType.PYTHON, FileAction.ADAPT)],
    )


def test_generates_expected_files_all_parse(tmp_path):
    ir = _sample_ir()
    result = generate(ir, convert(ir), str(tmp_path))
    written = {p.replace("\\", "/") for p in result.written_files}
    assert {
        "plugins/tools.py",       # tools -> plugins/<source module>.py
        "agent_context.py",
        "orchestrator.py",
        "config.py",
        "main.py",
    } <= written
    assert result.syntax_errors == []
    for rel in result.written_files:
        if rel.endswith(".py"):
            ast.parse(_read(result.output_root, rel))  # every .py is valid Python


def test_plugin_class_naming(tmp_path):
    ir = _sample_ir()
    generate(ir, convert(ir), str(tmp_path))
    content = _read(str(tmp_path), os.path.join("plugins", "tools.py"))
    assert "class ReadTestsPlugin:" in content
    assert "def read_tests(path) -> list:" in content   # plain function
    assert "kernel_function" in content


def test_append_only_field_rendered_as_list(tmp_path):
    ir = _sample_ir()
    generate(ir, convert(ir), str(tmp_path))
    ctx = _read(str(tmp_path), "agent_context.py")
    assert "audit_log: list = field(default_factory=list)" in ctx
    assert "project_id: Optional[str] = None" in ctx


def test_hitl_auto_approves_with_commented_real_flow(tmp_path):
    ir = _sample_ir()
    ir.workflow.nodes.append(GraphNode("hitl_approve", role=NodeRole.HITL))
    result = generate(ir, convert(ir), str(tmp_path))
    orch = _read(result.output_root, "orchestrator.py")
    # The class is still defined (used when the real flow is uncommented).
    assert "class HumanApprovalRequired(Exception):" in orch
    # Auto-approve is ACTIVE so the agent runs end-to-end now.
    assert "AUTO-APPROVE (prototype)" in orch
    # The real approval flow is present but COMMENTED OUT.
    assert "# --- REAL HUMAN APPROVAL" in orch
    # The active body must be valid and return ctx (no live raise).
    ast.parse(orch)


def test_tier3_generated_code_is_stitched(tmp_path):
    ir = _sample_ir()
    ir.workflow.pattern = OrchestrationPattern.AGENT_DRIVEN

    def fake(ir_, wf, cfg):
        from converter.contracts import Tier3Result

        return Tier3Result(
            pattern="agent_driven",
            generated_code="def run(ctx):\n    # llm-authored\n    return ctx",
            reasoning="dynamic",
            confidence=0.95,
        )

    result = generate(ir, convert(ir, tier3_resolver=fake), str(tmp_path))
    orch = _read(result.output_root, "orchestrator.py")
    assert "Tier 3 (Gemini) generated orchestration" in orch
    assert "# llm-authored" in orch
    assert result.syntax_errors == []


def test_invalid_tier3_code_flagged_with_banner(tmp_path):
    ir = _sample_ir()
    ir.workflow.pattern = OrchestrationPattern.AGENT_DRIVEN

    def broken(ir_, wf, cfg):
        from converter.contracts import Tier3Result

        return Tier3Result("x", "def run(:::\n bad", "", 0.99)

    result = generate(ir, convert(ir, tier3_resolver=broken), str(tmp_path))
    assert "orchestrator.py" in result.syntax_errors
    orch = _read(result.output_root, "orchestrator.py")
    assert orch.startswith("# SYNTAX ERROR")


def test_config_py_carries_constants(tmp_path):
    ir = _sample_ir()
    generate(ir, convert(ir), str(tmp_path))
    cfg = _read(str(tmp_path), "config.py")
    assert "MAX_GEN_RETRIES = 3" in cfg


def test_generate_from_paths_copies_copy_through(tmp_path):
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    (input_root / "prompts").mkdir(parents=True)
    (input_root / "prompts" / "system.md").write_text("You are an agent.", encoding="utf-8")

    ir = _sample_ir()
    ir.files.append(
        FileEntry(os.path.join("prompts", "system.md"), FileType.PROMPT, FileAction.COPY_THROUGH)
    )

    result = generate_from_paths(ir, convert(ir), str(input_root), str(output_root))
    copied_norm = {c.replace("\\", "/") for c in result.copied_files}
    assert "prompts/system.md" in copied_norm
    assert os.path.exists(os.path.join(result.output_root, "prompts", "system.md"))


def test_no_validation_warnings_for_clean_output(tmp_path):
    ir = _sample_ir()
    result = generate(ir, convert(ir), str(tmp_path))
    assert result.validation_warnings == []


# ---------------------------------------------------------------------------
# Logic porting (deployable output)
# ---------------------------------------------------------------------------

def _ported_ir() -> IR:
    from converter.contracts import FunctionSpec

    return IR(
        metadata=IRMetadata(target_framework="maf"),
        tools=[
            ToolSpec(
                name="read_tests",
                params=[ToolParam("path", "str", None)],
                docstring="Reads.",
                returns="list",
                source_file="agent.py",
                body="return [1, 2, 3]",
            )
        ],
        state=[
            StateField("coverage", "float"),
            StateField("audit_log", "list", is_append_only=True),
        ],
        config=ConfigSpec(constants={"MAX_GEN_RETRIES": 3}),
        functions={
            "generate": FunctionSpec(
                name="generate",
                params=[ToolParam("state")],
                body='c = state["coverage"]\nreturn {"coverage": c + 0.1, "audit_log": ["did it"]}',
            ),
            "route": FunctionSpec(
                name="route",
                params=[ToolParam("state")],
                body='return "done" if state["coverage"] >= 1 else "revise"',
                returns="str",
            ),
        },
        imports=["import os"],
        preamble=["llm = None  # placeholder"],
        workflow=WorkflowSpec(
            pattern=OrchestrationPattern.LOOP_WITH_EXIT,
            nodes=[
                GraphNode("generate", target_callable="generate", role=NodeRole.ENTRY),
                GraphNode("gate", target_callable="gate", role=NodeRole.BRANCH),
            ],
            edges=[GraphEdge("generate", "gate")],
            conditional_edges=[
                ConditionalEdge("gate", "route", {"revise": "generate", "done": "END"})
            ],
            entry_point="generate",
        ),
    )


def test_tool_body_is_ported_not_stubbed(tmp_path):
    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    content = _read(str(tmp_path), os.path.join("plugins", "agent.py"))
    assert "return [1, 2, 3]" in content
    assert "NotImplementedError" not in content


def test_node_body_ported_state_to_ctx(tmp_path):
    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    orch = _read(str(tmp_path), "orchestrator.py")
    assert "def generate(ctx: AgentContext) -> AgentContext:" in orch
    assert "c = ctx.coverage" in orch
    assert "ctx.coverage = c + 0.1" in orch
    assert "ctx.audit_log.append('did it')" in orch  # append-only reducer


def test_router_and_imports_and_preamble_present(tmp_path):
    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    orch = _read(str(tmp_path), "orchestrator.py")
    assert "def route(ctx) -> str:" in orch
    assert "import os" in orch
    assert "llm = None" in orch
    assert "from config import MAX_GEN_RETRIES" in orch


def test_loop_uses_real_cap_and_router_exit(tmp_path):
    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    orch = _read(str(tmp_path), "orchestrator.py")
    assert "while guard < MAX_GEN_RETRIES:" in orch
    assert "outcome = route(ctx)" in orch
    assert 'if outcome == "done":' in orch


def test_hitl_stitched_when_gemini_available(tmp_path):
    ir = _ported_ir()
    ir.workflow.nodes.append(GraphNode("hitl_approve", role=NodeRole.HITL))

    def fake_hitl(ir_, node_name, source, cfg):
        from converter.contracts import Tier3Result

        return Tier3Result(
            pattern="hitl",
            generated_code="ctx.audit_log.append('approved')\nreturn ctx",
            reasoning="approval flow",
            confidence=0.9,
        )

    result = generate(ir, convert(ir, hitl_resolver=fake_hitl), str(tmp_path))
    orch = _read(result.output_root, "orchestrator.py")
    # Gemini's flow is carried in the commented real-approval block for review.
    assert "Gemini-generated; review then uncomment" in orch
    assert "# ctx.audit_log.append('approved')" in orch
    # Auto-approve is active; the agent still runs.
    assert "AUTO-APPROVE (prototype)" in orch
    assert result.syntax_errors == []
