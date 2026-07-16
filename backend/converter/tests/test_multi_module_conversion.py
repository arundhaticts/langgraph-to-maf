"""Regression tests for converting a real multi-module LangGraph repo.

Runs the converter against the `multi_module_agent` fixture and asserts the
acceptance criteria. New assertions are enabled as each rework pass lands.
"""

from __future__ import annotations

import os

import pytest

from converter.config import Config, ConversionMode
from converter.extractor import extract_components
from converter.ir import build_ir
from converter.parser import parse_readme_file
from converter.pipeline.hybrid_pipeline import HybridPipeline
from converter.scanner import scan_repo

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "multi_module_agent")


def _build_ir():
    manifest = scan_repo(_FIXTURE)
    readme = parse_readme_file(os.path.join(manifest.input_root, manifest.readme_path))
    inv = extract_components(manifest, readme)
    return build_ir(inv, readme, manifest), inv


def _convert(tmp_path, mode=ConversionMode.DETERMINISTIC) -> dict[str, str]:
    """Convert the fixture; return {relative_path: file_contents} of the output."""
    out = str(tmp_path / "out")
    # write_ir_json writes ir.json to cwd; keep it out of the repo.
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        HybridPipeline(Config(mode=mode)).run(_FIXTURE, out)
    finally:
        os.chdir(cwd)
    files: dict[str, str] = {}
    for root, _dirs, names in os.walk(out):
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, out).replace("\\", "/")
            try:
                files[rel] = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                files[rel] = ""
    return files


# ---------------------------------------------------------------------------
# Pass 1 -- extraction scope
# ---------------------------------------------------------------------------

def test_orchestrator_excludes_tests_and_sample_data(tmp_path):
    files = _convert(tmp_path)
    orch = files.get("orchestrator.py", "")
    # Test functions and sample-data generators must never be dumped in.
    assert "test_login_success" not in orch
    assert "helper_used_only_in_tests" not in orch
    assert "write_ci_history" not in orch
    assert "make_fixtures" not in orch


def test_config_has_only_real_constants(tmp_path):
    files = _convert(tmp_path)
    cfg = files.get("config.py", "")
    # Real safety-blocker constants carried over...
    assert "MAX_GEN_RETRIES" in cfg
    assert "COVERAGE_FLOOR" in cfg
    # ...but sample-data / non-config constants must be gone.
    for junk in ("SEED", "GRAPH", "GOLDEN", "EXPECTED_FIELDS", "FILES"):
        assert junk not in cfg


# ---------------------------------------------------------------------------
# Pass 2 -- tool (by folder) / node / router detection
# ---------------------------------------------------------------------------

def test_tools_detected_by_folder_without_decorator():
    ir, _ = _build_ir()
    tool_names = {t.name for t in ir.tools}
    # Plain functions in src/tools/ are detected as tools (no @tool needed).
    assert {"read_tests", "detect_conventions"} <= tool_names
    # Their real bodies are carried for plugin conversion.
    read = next(t for t in ir.tools if t.name == "read_tests")
    assert read.body and "os.listdir" in read.body


def test_nodes_and_routers_detected_from_graph():
    ir, _ = _build_ir()
    node_names = {n.name for n in ir.workflow.nodes}
    assert {"intake", "prioritisation", "revise", "validation", "drop_failing"} <= node_names
    routers = {c.router for c in ir.workflow.conditional_edges}
    assert {"coverage_floor_gate", "route_after_validation"} <= routers


def test_tool_files_marked_for_conversion_not_copy_through():
    from converter.contracts import FileAction

    ir, _ = _build_ir()
    actions = {f.relative_path.replace("\\", "/"): f.file_action for f in ir.files}
    assert actions["src/tools/repo_reader.py"] is not FileAction.COPY_THROUGH


# ---------------------------------------------------------------------------
# Pass 3 -- codegen correctness (orchestrator + agent_context)
# ---------------------------------------------------------------------------

def test_orchestrator_is_langgraph_token_clean(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    for tok in ("StateGraph", "add_conditional_edges", "add_node", "MemorySaver",
                "langgraph", "interrupt(", "START", "END"):
        assert tok not in orch, f"orchestrator still contains {tok!r}"
    # graph-assembly functions must not be dumped in.
    assert "build_graph" not in orch
    assert "make_checkpointer" not in orch


def test_orchestrator_uses_ctx_not_state(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    assert "state.get(" not in orch
    assert "state[" not in orch
    # the state-consuming helper was rewritten to ctx uniformly (2nd-position param).
    assert "def is_protected(test_id: str, ctx)" in orch


def test_helper_signatures_preserved(tmp_path):
    # audit/configure_logging live in the carried support module util.py.
    util = _convert(tmp_path).get("util.py", "")
    # defaults and **kwargs must survive.
    assert "def audit(node, event, level='info', **details)" in util
    assert "def configure_logging(level='INFO')" in util


def test_orchestrator_and_context_parse(tmp_path):
    import ast as _ast

    files = _convert(tmp_path)
    _ast.parse(files["orchestrator.py"])
    _ast.parse(files["agent_context.py"])


def test_agent_context_imports_literal(tmp_path):
    ctx = _convert(tmp_path).get("agent_context.py", "")
    assert "Literal" in ctx and "from typing import" in ctx
    # it must actually compile (Literal defined).
    import ast as _ast
    _ast.parse(ctx)


def test_run_is_synthesized(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    assert "def run(ctx" in orch


# ---------------------------------------------------------------------------
# Pass 4 -- consolidated MAF tree (whole-output coherence)
# ---------------------------------------------------------------------------

def test_output_is_a_single_maf_tree(tmp_path):
    files = _convert(tmp_path)
    paths = set(files)
    # Expected consolidated artefacts.
    assert "agent_context.py" in paths
    assert "orchestrator.py" in paths
    assert "main.py" in paths
    assert "plugins/repo_reader.py" in paths
    # The original LangGraph tree must be GONE.
    assert not any(p.startswith("src/") for p in paths)
    assert not any(p.startswith("sample_data/") for p in paths)
    # The converter ships its OWN smoke test, but the source's tests are not copied.
    assert "tests/test_smoke.py" in paths
    assert not any(p.startswith("tests/") and p != "tests/test_smoke.py" for p in paths)


def test_no_langgraph_or_interrupt_anywhere(tmp_path):
    files = _convert(tmp_path)
    for rel, content in files.items():
        if not rel.endswith(".py"):
            continue
        # The vendored agent_framework/ stub is infrastructure, not converted code.
        if rel.replace("\\", "/").startswith("agent_framework/"):
            continue
        for tok in ("langgraph", "StateGraph", "add_conditional_edges", "interrupt("):
            assert tok not in content, f"{tok!r} still in {rel}"


def test_every_generated_py_parses(tmp_path):
    import ast as _ast

    for rel, content in _convert(tmp_path).items():
        if rel.endswith(".py"):
            _ast.parse(content)


def test_tools_are_plugin_classes_with_functions(tmp_path):
    plugin = _convert(tmp_path).get("plugins/repo_reader.py", "")
    # True MAF: function-style @ai_function tools, no dead plugin classes.
    assert "@ai_function" in plugin
    assert "def read_tests(path)" in plugin        # function present
    assert "os.listdir" in plugin                   # real body ported


def test_orchestrator_imports_tools_from_plugins(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    assert "from plugins.repo_reader import" in orch


def test_maf_workflow_graph_wired_from_ir(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    # Phase 4: executors wrapping the ported nodes.
    assert '@executor(id="intake")' in orch
    assert "await ctx.send_message(" in orch
    assert "await ctx.yield_output(" in orch
    # Phase 5 + 8: WorkflowBuilder with guarded + loop-back edges from the IR.
    assert "WorkflowBuilder().set_start_executor(intake_exec)" in orch
    assert "builder.add_edge(prioritisation_exec, revise_exec, condition=lambda s" in orch
    assert "# loop back-edge" in orch
    # Real routers drive the edge conditions.
    assert "_r=coverage_floor_gate" in orch
    assert "_r=route_after_validation" in orch


def test_phase6_tools_registered_and_mapped(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    # Tools registered (not orphaned) + node->tool mapping surfaced from the IR.
    assert "AGENT_TOOLS = [" in orch
    assert "read_tests" in orch and "detect_conventions" in orch
    assert "NODE_TOOLS = {" in orch
    assert '"intake":' in orch  # intake's tools are mapped


def test_phase7_hitl_request_info_and_checkpointing(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    # Fast-path switch + a real RequestInfoExecutor pause path, both wired.
    assert "AUTO_APPROVE_HITL = True" in orch
    assert "class HitlRemovalsRequest(RequestInfoMessage):" in orch
    assert 'RequestInfoExecutor(id="request_info")' in orch
    assert "async def hitl_removals_apply_exec(" in orch
    assert "builder.add_edge(hitl_removals_exec, _request_info_exec)" in orch
    assert "builder.add_edge(_request_info_exec, hitl_removals_apply_exec)" in orch
    # Fast-path (auto-approve) executor path is also present and real.
    assert "fast-path (auto-approve)" in orch
    # Source used MemorySaver -> file-based checkpointing enabled.
    assert "with_checkpointing(FileCheckpointStorage(" in orch


def test_main_imports_converted_modules(tmp_path):
    main = _convert(tmp_path).get("main.py", "")
    assert "from orchestrator import run" in main
    assert "from agent_context import AgentContext" in main
    assert "src.graph" not in main and "langgraph" not in main


def test_hitl_auto_approve_audit_is_consistent_dict(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    # HITL auto-approve must use the audit(...) helper (dict), never a bare string.
    assert 'ctx.audit_log.append(audit("hitl_removals", "auto_approved"))' in orch
    assert 'ctx.audit_log.append("auto-approved' not in orch


def test_all_tools_registered_no_orphans(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    # Every converted tool appears in AGENT_TOOLS (instantiated, not orphaned).
    assert "AGENT_TOOLS = [" in orch
    for tool in ("read_tests", "detect_conventions"):
        assert tool in orch.split("AGENT_TOOLS = [")[1].split("]")[0]


def test_smoke_test_is_generated_and_valid(tmp_path):
    import ast as _ast

    files = _convert(tmp_path)
    smoke = files.get("tests/test_smoke.py", "")
    assert "def test_state_constructs():" in smoke
    assert "def test_offline_entrypoint_present():" in smoke
    _ast.parse(smoke)


def test_phase9_entrypoint_drives_maf_workflow(tmp_path):
    main = _convert(tmp_path).get("main.py", "")
    # Phase 9: run_stream driver + request/resume loop, with offline fallback.
    assert "async def run_workflow_stream(" in main
    assert "workflow.run_stream(state)" in main
    assert "send_responses_streaming(responses)" in main
    assert "return run(state)  # offline fast-path" in main


# ---------------------------------------------------------------------------
# Pass 5 -- auto-detected requirements
# ---------------------------------------------------------------------------

def test_requirements_drop_langgraph_add_target(tmp_path):
    reqs = _convert(tmp_path).get("requirements.txt", "")
    assert "langgraph" not in reqs
    assert "langchain" not in reqs
    assert "agent-framework" in reqs           # target framework runtime dep


def test_requirements_only_lists_what_is_imported(tmp_path):
    # The fixture's converted code imports nothing third-party beyond the target
    # SDK and pydantic (the generated state is a pydantic BaseModel).
    reqs = _convert(tmp_path).get("requirements.txt", "")
    for spurious in ("sentence-transformers", "python-dotenv"):
        assert spurious not in reqs
    # pydantic IS required now -- the state model imports it.
    assert "pydantic" in reqs
    # stdlib is never listed.
    for stdlib in ("os", "operator", "typing", "dataclasses"):
        assert f"\n{stdlib}\n" not in reqs


# ---------------------------------------------------------------------------
# Pass 6 -- support modules carried with rewired imports
# ---------------------------------------------------------------------------

def test_support_module_is_carried(tmp_path):
    files = _convert(tmp_path)
    # util.py (audit/configure_logging) is a support module -> copied, rewritten.
    assert "util.py" in files
    assert "def audit(node, event, level='info', **details)" in files["util.py"]


def test_support_functions_imported_not_inlined(tmp_path):
    orch = _convert(tmp_path).get("orchestrator.py", "")
    # orchestrator imports the helper from its module...
    assert "from util import" in orch
    # ...and does NOT redefine it inline.
    assert not __import__("re").search(r"^def audit\b", orch, __import__("re").M)


def test_state_class_aliased(tmp_path):
    ctx = _convert(tmp_path).get("agent_context.py", "")
    assert "TestOptimiserState = AgentContext" in ctx


def test_no_internal_src_imports_anywhere(tmp_path):
    for rel, content in _convert(tmp_path).items():
        if rel.endswith(".py"):
            assert "from src." not in content and "import src." not in content, rel
