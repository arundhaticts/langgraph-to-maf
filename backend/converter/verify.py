"""Phase 11 -- acceptance gate.

Runs AFTER generation and diffs the emitted package against the IR: every IR
node, tool, and HITL point must have a corresponding artifact in the output;
every generated .py must compile; and the output must be free of source-framework
residue (langgraph / semantic-kernel / raw interrupt() / TypedDict state / ...).

`verify_output` returns an `AcceptanceReport`; the pipeline writes it to
ACCEPTANCE.md and routes any failures into the migration report's needs-review.
It never raises -- a failed check is data, not an exception.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field

from converter.contracts import IR, NodeRole

# Tokens that must NOT survive into the generated .py files.
_FORBIDDEN_CODE_TOKENS = (
    "semantic_kernel",
    "kernel_function",
    "from langgraph",
    "import langgraph",
    "StateGraph",
    "add_conditional_edges",
    "TypedDict",
    "from operator import add",
)
# Tokens that must NOT survive into requirements.txt.
_FORBIDDEN_REQ_TOKENS = ("langgraph", "langchain", "semantic-kernel", "zipfile")

_SENTINELS = frozenset({"START", "END", "__start__", "__end__"})


@dataclass
class AcceptanceReport:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _n, ok, _d in self.checks)

    def issues(self) -> list[str]:
        return [f"{name}: {detail}" for name, ok, detail in self.checks if not ok]


def _read(root: str, rel: str) -> str:
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return ""


def verify_output(ir: IR, generation) -> AcceptanceReport:
    """Diff the generated package against the IR and check for residue."""
    report = AcceptanceReport()
    root = generation.output_root
    py_files = [r for r in generation.written_files if r.endswith(".py")]

    # 1. Every generated .py compiles.
    non_compiling = list(generation.syntax_errors)
    for rel in py_files:
        if rel in non_compiling:
            continue
        try:
            ast.parse(_read(root, rel))
        except SyntaxError:
            non_compiling.append(rel)
    report.add(
        "all_python_compiles",
        not non_compiling,
        f"files failing to parse: {sorted(set(non_compiling))}" if non_compiling else "",
    )

    # 2. No source-framework residue in code. The vendored agent_framework/ stub
    #    is infrastructure (it legitimately names SDK constructs), not converted
    #    code, so it is excluded from the residue scan.
    code_hits: list[str] = []
    for rel in py_files:
        if rel.replace("\\", "/").startswith("agent_framework/"):
            continue
        text = _read(root, rel)
        for token in _FORBIDDEN_CODE_TOKENS:
            if token in text:
                code_hits.append(f"{rel}:{token}")
    report.add(
        "no_source_framework_residue",
        not code_hits,
        f"forbidden tokens: {code_hits}" if code_hits else "",
    )

    # 3. requirements.txt is clean.
    reqs = _read(root, "requirements.txt")
    req_hits = [t for t in _FORBIDDEN_REQ_TOKENS if t in reqs]
    report.add(
        "requirements_clean",
        not req_hits,
        f"forbidden requirements: {req_hits}" if req_hits else "",
    )

    # 4. IR coverage -- every node/tool/HITL point has an artifact.
    orch = _read(root, "orchestrator.py")
    plugins = "\n".join(
        _read(root, r) for r in py_files if r.replace("\\", "/").startswith("plugins/")
    )
    wf = ir.workflow

    missing_nodes: list[str] = []
    if wf:
        for node in wf.nodes:
            if node.role is NodeRole.AUX or node.name in _SENTINELS:
                continue
            if f"def {node.name}(" not in orch:
                missing_nodes.append(node.name)
    report.add(
        "all_nodes_generated",
        not missing_nodes,
        f"nodes with no function in orchestrator.py: {missing_nodes}" if missing_nodes else "",
    )

    missing_tools = [t.name for t in ir.tools if f"def {t.name}(" not in plugins]
    report.add(
        "all_tools_generated",
        not missing_tools,
        f"tools with no plugin function: {missing_tools}" if missing_tools else "",
    )

    missing_hitl: list[str] = []
    if wf:
        for point in wf.hitl_points:
            # Deterministic node fn OR a MAF apply-executor must exist.
            if f"def {point.node}(" not in orch and f"{point.node}_apply_exec" not in orch:
                missing_hitl.append(point.node)
    report.add(
        "all_hitl_points_generated",
        not missing_hitl,
        f"HITL points with no artifact: {missing_hitl}" if missing_hitl else "",
    )

    # 5. When the IR has a graph, the workflow was actually assembled.
    if wf and any(n.role is not NodeRole.AUX for n in wf.nodes):
        assembled = "WorkflowBuilder" in orch and "set_start_executor" in orch
        report.add(
            "workflow_assembled",
            assembled,
            "" if assembled else "orchestrator.py has no WorkflowBuilder/set_start_executor",
        )

    return report


_RUNNABLE_CHECKS = (
    (
        "stub_imports",
        "from agent_framework import (WorkflowBuilder, WorkflowContext, executor, "
        "RequestInfoExecutor, RequestInfoMessage, RequestInfoEvent, WorkflowOutputEvent, "
        "FileCheckpointStorage, ChatAgent, ai_function); print('stub ok')",
    ),
    (
        "graph_builds",
        "from orchestrator import build_workflow; wf = build_workflow(); print('graph ok')",
    ),
)


def verify_runnable(output_root: str, timeout: int = 90) -> list[tuple[str, bool, str]]:
    """Run the two acceptance subprocess checks in the OUTPUT dir.

    (1) import the generated agent_framework stub; (2) import orchestrator and
    build the workflow. Returns (name, ok, detail); on failure `detail` carries
    the exact stderr/stdout so the error is surfaced, never swallowed.
    """
    results: list[tuple[str, bool, str]] = []
    for name, code in _RUNNABLE_CHECKS:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=output_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            ok = proc.returncode == 0
            detail = "" if ok else (proc.stderr.strip() or proc.stdout.strip())[-800:]
        except Exception as exc:  # pragma: no cover - defensive (timeout, etc.)
            ok, detail = False, str(exc)
        results.append((name, ok, detail))
    return results


def render_acceptance(report: AcceptanceReport) -> str:
    """Render the acceptance report as Markdown for ACCEPTANCE.md."""
    status = "PASSED" if report.passed else "FAILED"
    lines = [f"# Acceptance Report - {status}", ""]
    for name, ok, detail in report.checks:
        mark = "PASS" if ok else "FAIL"
        lines.append(f"- [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    lines.append("")
    return "\n".join(lines)


def write_acceptance(report: AcceptanceReport, output_root: str) -> str:
    path = os.path.join(output_root, "ACCEPTANCE.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_acceptance(report))
    return path
