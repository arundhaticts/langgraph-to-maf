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
    # True MAF: function-style @ai_function tools (no dead plugin classes).
    assert "@ai_function" in content
    assert "def read_tests(path) -> list:" in content
    assert "class ReadTestsTool" not in content


def test_state_is_pydantic_basemodel_with_advance(tmp_path):
    ir = _sample_ir()
    generate(ir, convert(ir), str(tmp_path))
    ctx = _read(str(tmp_path), "agent_context.py")
    assert "from pydantic import BaseModel, ConfigDict, Field" in ctx
    assert "class AgentContext(BaseModel):" in ctx
    assert "def advance(self, **updates)" in ctx
    assert "_REDUCER_FIELDS = {'audit_log'}" in ctx
    ast.parse(ctx)


def _import_module(path: str, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = mod  # pydantic resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_advance_returns_new_state_and_extends_reducer(tmp_path):
    ir = _sample_ir()
    generate(ir, convert(ir), str(tmp_path))
    mod = _import_module(
        os.path.join(str(tmp_path), "agent_context.py"), "gen_agent_context_advance"
    )
    AgentContext = mod.AgentContext
    a = AgentContext(project_id="p1", audit_log=["one"])
    b = a.advance(project_id="p2", audit_log=["two"])
    # advance() never mutates the original...
    assert a.project_id == "p1" and a.audit_log == ["one"]
    # ...replaces normal fields and EXTENDS reducer (append-only) fields.
    assert b.project_id == "p2"
    assert b.audit_log == ["one", "two"]
    assert b is not a


def test_type_erases_unresolvable_annotations(tmp_path):
    from converter.contracts import StateField

    ir = _sample_ir()
    ir.state.append(StateField("result", "CustomThing"))  # not importable in output
    generate(ir, convert(ir), str(tmp_path))
    ctx = _read(str(tmp_path), "agent_context.py")
    # Unresolvable custom type is erased to Any so the BaseModel still resolves.
    assert "result: Optional[Any] = None" in ctx
    ast.parse(ctx)


def test_append_only_field_rendered_as_list(tmp_path):
    ir = _sample_ir()
    generate(ir, convert(ir), str(tmp_path))
    ctx = _read(str(tmp_path), "agent_context.py")
    assert "audit_log: list = Field(default_factory=list)" in ctx
    assert "project_id: Optional[str] = None" in ctx


def test_hitl_auto_default_with_real_file_pause(tmp_path):
    ir = _sample_ir()
    ir.workflow.nodes.append(GraphNode("hitl_approve", role=NodeRole.HITL))
    result = generate(ir, convert(ir), str(tmp_path))
    orch = _read(result.output_root, "orchestrator.py")
    # Real, opt-in human pause (file-based) is ACTIVE code, not commented.
    assert 'os.environ.get("HITL_MODE", "auto")' in orch
    assert "hitl_approve.request.json" in orch
    assert "hitl_approve.response.json" in orch
    assert "HITL_TIMEOUT_SECONDS = 600" in orch
    # Auto-approve remains the default path (so the agent still runs end-to-end).
    assert 'ctx.audit_log.append({"node": "hitl_approve", "event": "auto_approved"})' in orch
    # The source/suggested approval logic is preserved as a reference comment.
    assert "HUMAN APPROVAL REFERENCE for 'hitl_approve'" in orch
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
                body='c = state["coverage"]\nnote = os.getenv("N", str(llm))\nreturn {"coverage": c + 0.1, "audit_log": ["did it"]}',
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


def test_maf_workflow_graph_emitted(tmp_path):
    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    orch = _read(str(tmp_path), "orchestrator.py")
    # Phase 4: one @executor per node, send_message + terminal yield_output.
    assert '@executor(id="generate")' in orch
    assert "async def generate_exec(state: AgentContext, ctx: WorkflowContext[AgentContext])" in orch
    assert "await ctx.send_message(" in orch
    assert "await ctx.yield_output(" in orch
    # Phase 5 + 8: WorkflowBuilder with a start executor and a guarded edge.
    assert "WorkflowBuilder().set_start_executor(generate_exec)" in orch
    assert "builder.add_edge(gate_exec, generate_exec, condition=lambda s" in orch
    assert "_r(s) == _l)  # loop back-edge" in orch
    # Offline fast-path still present alongside the true MAF graph.
    assert "def run(ctx):" in orch
    assert "_HAVE_AGENT_FRAMEWORK" in orch
    ast.parse(orch)


def test_maf_orchestrator_imports_with_stub(tmp_path):
    import sys

    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    # Put the output tree on sys.path so sibling modules (config, agent_context,
    # and the generated agent_framework/ stub) resolve as in the deployed package.
    sys.path.insert(0, str(tmp_path))
    for mod_name in ("agent_framework", "orchestrator", "agent_context", "config"):
        sys.modules.pop(mod_name, None)
    try:
        mod = _import_module(os.path.join(str(tmp_path), "orchestrator.py"), "orch_maf_stub")
        # The generated stub makes the MAF path active -> the real graph builds.
        assert mod._HAVE_AGENT_FRAMEWORK is True
        assert callable(mod.run)          # offline fast-path still present
        assert callable(mod.build_workflow)
        assert mod.build_workflow() is not None
    finally:
        sys.path.remove(str(tmp_path))
        for mod_name in ("agent_framework", "orchestrator", "agent_context", "config"):
            sys.modules.pop(mod_name, None)


def test_phase9_cli_entrypoint_run_stream(tmp_path):
    ir = _ported_ir()
    generate(ir, convert(ir), str(tmp_path))
    main = _read(str(tmp_path), "main.py")
    # True MAF driver: run_stream + request/resume loop until WorkflowOutputEvent.
    assert "async def run_workflow_stream(" in main
    assert "workflow.run_stream(state)" in main
    assert "send_responses_streaming(responses)" in main
    assert "WorkflowOutputEvent" in main and "RequestInfoEvent" in main
    # Offline fallback preserved.
    assert "return run(state)  # offline fast-path" in main
    assert "result = run_agent()" in main  # CLI mode
    assert "FastAPI" not in main
    ast.parse(main)


def test_phase9_api_entrypoint_fastapi_uvicorn(tmp_path):
    ir = _ported_ir()
    ir.metadata.entrypoint = "api"
    generate(ir, convert(ir), str(tmp_path))
    main = _read(str(tmp_path), "main.py")
    assert "from fastapi import FastAPI" in main
    assert "import uvicorn" in main
    assert "app = FastAPI()" in main
    assert '@app.post("/run")' in main
    assert "uvicorn.run(app" in main
    # Uses the same run_agent() (MAF workflow when SDK present, else offline).
    assert "run_agent(state)" in main
    ast.parse(main)
    # FastAPI + uvicorn become real runtime deps.
    reqs = _read(str(tmp_path), "requirements.txt")
    assert "fastapi" in reqs and "uvicorn" in reqs


def test_phase6_single_agent_wires_chatagent(tmp_path):
    from converter.contracts import OrchestrationMode

    ir = _sample_ir()
    ir.metadata.orchestration_mode = OrchestrationMode.SINGLE_AGENT
    # Mark the node as calling the tool (IR node->tool mapping).
    ir.workflow.nodes[0].calls_tools = ["read_tests"]
    generate(ir, convert(ir), str(tmp_path))
    orch = _read(str(tmp_path), "orchestrator.py")
    assert "AGENT_TOOLS = [read_tests]" in orch
    assert "def build_agent():" in orch
    assert "ChatAgent(chat_client=None, tools=AGENT_TOOLS)" in orch
    ast.parse(orch)


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
    # Gemini's flow is carried in the human-approval reference comment for review.
    assert "Gemini-suggested approval logic" in orch
    assert "# ctx.audit_log.append('approved')" in orch
    # The real file-based pause is active; auto-approve remains the default.
    assert 'os.environ.get("HITL_MODE", "auto")' in orch
    assert result.syntax_errors == []
