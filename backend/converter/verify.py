"""Phase 11 -- acceptance gate.

Runs AFTER generation and diffs the emitted package against the IR: every IR
node, tool, and HITL point must have a corresponding artifact in the output;
every generated .py must compile; and the output must be free of source-framework
residue.

`verify_output` returns an `AcceptanceReport`; the pipeline writes it to
ACCEPTANCE.md and routes any failures into the migration report's needs-review.
It never raises -- a failed check is data, not an exception.

Phase 10 checks (target-parameterized)
---------------------------------------
1. All generated .py files compile (ast.parse).
2. Per-target forbidden tokens absent from generated code (reject list).
3. requirements.txt is clean of forbidden packages.
4. IR coverage: every node / tool / HITL point has an artifact.
5. Per-target required symbols present in orchestrator.py (shape proof).
6. Loop reachability: every LoopGuard's loop_node appears in the orchestrator.
7. subprocess runnable checks (opt-in via validate_output flag).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from converter.contracts import IR, NodeRole

_SENTINELS = frozenset({"START", "END", "__start__", "__end__"})

# ---------------------------------------------------------------------------
# Per-target forbidden / required token tables (Phase 10)
# ---------------------------------------------------------------------------

# Tokens that must NOT appear in any generated .py (keyed by target framework).
# Each target bans the other three frameworks' source constructs.
_FORBIDDEN_CODE_TOKENS: dict[str, tuple[str, ...]] = {
    "maf": (
        "from langgraph",
        "import langgraph",
        "StateGraph(",
        "add_conditional_edges(",
        "from crewai",
        "import crewai",
        "from strands",
        "import strands",
        "semantic_kernel",
        "kernel_function",
    ),
    "langgraph": (
        "from agent_framework",
        "import agent_framework",
        "WorkflowBuilder(",
        "@executor",
        "@handler",
        "from crewai",
        "import crewai",
        "from strands",
        "import strands",
        "semantic_kernel",
    ),
    "crewai": (
        "from agent_framework",
        "import agent_framework",
        "WorkflowBuilder(",
        "StateGraph(",
        "from langgraph",
        "import langgraph",
        "from strands",
        "import strands",
        "semantic_kernel",
    ),
    "aws_strands": (
        "from agent_framework",
        "import agent_framework",
        "WorkflowBuilder(",
        "StateGraph(",
        "from langgraph",
        "import langgraph",
        "from crewai",
        "import crewai",
        "semantic_kernel",
    ),
}

# Default (unknown / dynamic target) — forbid all known source constructs.
_FORBIDDEN_CODE_TOKENS_DEFAULT: tuple[str, ...] = (
    "semantic_kernel",
    "kernel_function",
    "from langgraph",
    "import langgraph",
    "StateGraph(",
    "add_conditional_edges(",
    "TypedDict",
    "from operator import add",
)

# Tokens that MUST appear in orchestrator.py to prove the target shape landed.
# Only checked when the IR has a non-trivial workflow (more than zero real nodes).
_REQUIRED_ORCHESTRATOR_TOKENS: dict[str, tuple[str, ...]] = {
    "maf": ("WorkflowBuilder", "build_workflow"),
    "langgraph": ("StateGraph", "add_node", "set_entry_point", "build_graph"),
    "crewai": ("Crew", "Task", "Process"),
    "aws_strands": ("Agent", "build_agent"),
}

# Tokens that must NOT appear in requirements.txt (all targets share this list).
_FORBIDDEN_REQ_TOKENS: tuple[str, ...] = (
    "langgraph",
    "langchain",
    "semantic-kernel",
    "zipfile",
)

# Per-target runnable check code snippets (executed as subprocesses).
_RUNNABLE_CHECKS: dict[str, list[tuple[str, str]]] = {
    "maf": [
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
    ],
    "langgraph": [
        (
            "graph_builds",
            "from orchestrator import build_graph; g = build_graph(); print('graph ok')",
        ),
    ],
    "crewai": [
        (
            "crew_builds",
            "from orchestrator import build_crew; c = build_crew(); print('crew ok')",
        ),
    ],
    "aws_strands": [
        (
            "agent_builds",
            "from orchestrator import build_agent; a = build_agent(); print('agent ok')",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(root: str, rel: str) -> str:
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _target_forbidden_tokens(target: Optional[str]) -> tuple[str, ...]:
    if target and target in _FORBIDDEN_CODE_TOKENS:
        return _FORBIDDEN_CODE_TOKENS[target]
    return _FORBIDDEN_CODE_TOKENS_DEFAULT


def _target_required_tokens(target: Optional[str]) -> tuple[str, ...]:
    if target and target in _REQUIRED_ORCHESTRATOR_TOKENS:
        return _REQUIRED_ORCHESTRATOR_TOKENS[target]
    # Default: MAF (original behaviour)
    return _REQUIRED_ORCHESTRATOR_TOKENS["maf"]


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify_output(ir: IR, generation) -> AcceptanceReport:
    """Diff the generated package against the IR and check for residue."""
    report = AcceptanceReport()
    root = generation.output_root
    py_files = [r for r in generation.written_files if r.endswith(".py")]
    target = ir.metadata.target_framework or "maf"

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

    # 2. No source-framework residue in code.
    # The vendored agent_framework/ stub is infrastructure and excluded.
    forbidden_tokens = _target_forbidden_tokens(target)
    code_hits: list[str] = []
    for rel in py_files:
        if rel.replace("\\", "/").startswith("agent_framework/"):
            continue
        text = _read(root, rel)
        for token in forbidden_tokens:
            if token in text:
                code_hits.append(f"{rel}:{token!r}")
    report.add(
        "no_source_framework_residue",
        not code_hits,
        f"forbidden tokens found: {code_hits}" if code_hits else "",
    )

    # 3. requirements.txt is clean.
    reqs = _read(root, "requirements.txt")
    req_hits = [t for t in _FORBIDDEN_REQ_TOKENS if t in reqs]
    report.add(
        "requirements_clean",
        not req_hits,
        f"forbidden requirements: {req_hits}" if req_hits else "",
    )

    # 4. IR coverage -- every node / tool / HITL point has an artifact.
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

    # 5. Per-target required symbols prove the target shape landed.
    real_nodes = [n for n in (wf.nodes if wf else []) if n.name not in _SENTINELS and n.role is not NodeRole.AUX]
    if real_nodes:
        required = _target_required_tokens(target)
        missing_required = [tok for tok in required if tok not in orch]
        report.add(
            "target_shape_present",
            not missing_required,
            f"required tokens absent from orchestrator.py: {missing_required}" if missing_required else "",
        )

    # 6. Loop reachability: every LoopGuard's loop_node has a function.
    if wf and wf.loop_guards:
        missing_loop_fns = [
            g.loop_node for g in wf.loop_guards
            if f"def {g.loop_node}(" not in orch
        ]
        report.add(
            "loop_nodes_reachable",
            not missing_loop_fns,
            f"loop nodes with no function: {missing_loop_fns}" if missing_loop_fns else "",
        )

    return report


# ---------------------------------------------------------------------------
# Subprocess runnable checks (opt-in)
# ---------------------------------------------------------------------------

def verify_runnable(output_root: str, target: str = "maf", timeout: int = 90) -> list[tuple[str, bool, str]]:
    """Run per-target acceptance subprocess checks in the OUTPUT dir.

    Returns (name, ok, detail); on failure `detail` carries the exact
    stderr/stdout so the error is surfaced, never swallowed.
    """
    checks = _RUNNABLE_CHECKS.get(target, _RUNNABLE_CHECKS["maf"])
    results: list[tuple[str, bool, str]] = []
    for name, code in checks:
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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

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
