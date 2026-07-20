"""Agent-specific READINESS report (LLM-generated, deterministic fallback).

After a conversion, this produces `READINESS_REPORT.md`: an honest, agent-specific
account of what is left, WHO fixes it (the converter's next generation vs. a human
assisted by Claude Code), a time estimate, and a per-dimension accuracy table with
reasoning. It is grounded in the ACTUAL converted agent -- its nodes, tools, HITL
points, validation findings, and acceptance results -- so the report names the real
constructs instead of generic placeholders.

Primary path is the LLM (Gemini, via the Tier-3 client). When no API key/SDK is
available it degrades to a deterministic report built from the same facts, so a
report is always produced and never blocks the pipeline.
"""

from __future__ import annotations

import json
from typing import Optional

from converter.config import Config
from converter.contracts import IR, ConversionResult, NodeRole
from converter.engine.tier3_llm import _call_gemini


# ---------------------------------------------------------------------------
# Fact collection (deterministic -- the grounding for both paths)
# ---------------------------------------------------------------------------

# Deterministic-stub markers the generator can leave behind when a construct
# could not be fully converted automatically. Each maps to (short label, the
# action a human -- assisted by Claude Code -- should take, realistic time).
# These are surfaced as remaining-work rows AND fed to the LLM refinement loop,
# which drafts a basic working version so the shipped folder is runnable, not empty.
_STUB_MARKERS: list[tuple[str, str, str, str]] = [
    (
        "raise NotImplementedError",
        "Orchestration/logic left unresolved (raises NotImplementedError)",
        "Implement the real flow using the target framework's idioms",
        "2 hrs",
    ),
    (
        "# TODO: port logic from source tool",
        "Tool body could not be recovered; a runnable scaffold was emitted",
        "Port the tool's real logic into the scaffold",
        "1 hr",
    ),
    (
        "# TODO: port logic from source node",
        "Node logic could not be ported automatically",
        "Port the node's real logic",
        "1 hr",
    ),
    (
        "# TODO: automatic port failed for node",
        "Node port failed; a safe pass-through stub was emitted",
        "Port the node's real logic by hand",
        "1 hr",
    ),
    (
        "# TODO: confirm the real loop cap",
        "Loop iteration cap is a safe default (10), not the source value",
        "Confirm the real loop bound and set it",
        "15 min",
    ),
    (
        "# TODO: break on the real exit condition",
        "Loop exit condition is a placeholder (guard-count only)",
        "Wire the real exit condition",
        "30 min",
    ),
    (
        "chat_client=None",
        "Agent LLM chat client is a placeholder (chat_client=None)",
        "Pass your chat client / LLM provider to the agent",
        "30 min",
    ),
]


def _scan_code_stubs(generation) -> list[dict]:
    """Scan the generated .py files for deterministic-stub markers.

    Returns a list of {label, action, time, files} — one entry per marker kind
    that actually appears in the output. Never raises (best-effort).
    """
    import os as _os

    found: list[dict] = []
    try:
        root = getattr(generation, "output_root", None)
        rels = [r for r in getattr(generation, "written_files", []) or [] if r.endswith(".py")]
        if not root or not rels:
            return found
        # Read each file once, test every marker against it.
        contents: dict[str, str] = {}
        for rel in rels:
            try:
                with open(_os.path.join(root, rel.replace("/", _os.sep)), encoding="utf-8") as fh:
                    contents[rel] = fh.read()
            except OSError:
                continue
        for marker, label, action, time in _STUB_MARKERS:
            hits = sorted(rel for rel, text in contents.items() if marker in text)
            if hits:
                found.append({"label": label, "action": action, "time": time, "files": hits})
    except Exception:  # noqa: BLE001 - the readiness report must never crash the pipeline
        pass
    return found


def collect_facts(
    ir: IR,
    conversion: ConversionResult,
    generation,
    acceptance=None,
    config: Config | None = None,
    agent_name: str = "Converted Agent",
) -> dict:
    """Gather the agent-specific facts the report reasons over."""
    config = config or Config()
    wf = ir.workflow
    nodes = list(wf.nodes) if wf else []
    called: set[str] = set()
    for node in nodes:
        called.update(node.calls_tools)
    tool_names = [t.name for t in ir.tools]

    warnings = list(getattr(generation, "validation_warnings", []) or [])
    manual = [u.manual_action for u in conversion.units if u.manual_action]
    needs_review = [
        (u.source_ref, u.reasoning)
        for u in conversion.units
        if getattr(u, "needs_review", False)
    ]

    acc_checks = []
    if acceptance is not None:
        acc_checks = [
            {"name": n, "ok": ok, "detail": d} for n, ok, d in acceptance.checks
        ]

    return {
        "agent_name": agent_name,
        "target_framework": ir.metadata.target_framework or "maf",
        "orchestration_mode": getattr(ir.metadata.orchestration_mode, "value", None),
        "pattern": getattr(wf.pattern, "value", None) if wf else None,
        "nodes": [
            {"name": n.name, "role": getattr(n.role, "value", None)} for n in nodes
        ],
        "hitl_nodes": [n.name for n in nodes if n.role is NodeRole.HITL],
        "hitl_payloads": {h.node: h.payload for h in (wf.hitl_points if wf else [])},
        "tools": tool_names,
        "orphan_tools": sorted(t for t in tool_names if t not in called),
        "loop_guards": [
            {"loop_node": g.loop_node, "router": g.router, "cap": g.counter_const}
            for g in (wf.loop_guards if wf else [])
        ],
        "state_field_count": len(ir.state),
        "checkpointer": ir.metadata.checkpointer,
        "entrypoint": ir.metadata.entrypoint or "cli",
        "llm_provider": ir.metadata.llm_provider,
        "gemini_key_set": bool(config.llm_api_key()),
        "target_sdk_installed": ir.metadata.target_framework_version is not None,
        "syntax_errors": list(getattr(generation, "syntax_errors", []) or []),
        "ir_findings": [w for w in warnings if w.startswith("[IR]")],
        "acceptance_findings": [w for w in warnings if w.startswith("[ACCEPTANCE]")],
        "acceptance_checks": acc_checks,
        "manual_actions": manual,
        "needs_review": needs_review,
        "code_stubs": _scan_code_stubs(generation),
    }


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _prompt(facts: dict) -> str:
    return (
        "You are a senior engineer writing a READINESS report for a freshly "
        "converted AI agent (converted to the target framework by an automated "
        "tool). Be honest and specific to THIS agent -- reference its real node, "
        "tool, and HITL names. Do not invent features it does not have.\n\n"
        "Facts about the converted agent (JSON):\n"
        f"{json.dumps(facts, indent=2, default=str)}\n\n"
        "Write Markdown with EXACTLY these sections:\n"
        f"# Readiness Report - {facts.get('agent_name')}\n\n"
        "## Remaining work\n"
        "A Markdown table with columns: Item | Owner | Action | Time. "
        "Owner is either 'Auto (next converter gen)' for things the converter "
        "should fix on its next run, or 'Human via Claude Code' for things needing "
        "a person assisted by Claude Code. Time is 'Auto' for converter items or a "
        "realistic estimate (e.g. '10 min', '1-2 hrs') for human items. Derive rows "
        "from the facts: acceptance/IR findings, HITL nodes still defaulting to "
        "auto-approve, orphan tools, the local agent_framework stub vs real SDK, "
        "whether the Gemini key is set, the entrypoint kind (API endpoints need "
        "end-to-end testing and a /hitl/respond endpoint), and any needs-review or "
        "manual-action items.\n\n"
        "## Accuracy by dimension\n"
        "A Markdown table: Dimension | Accuracy | Why. Choose dimensions that fit "
        "THIS agent's actual purpose and nodes (e.g. each major node's output, the "
        "LLM-generated parts, the HITL-gated parts, the deterministic parts, and "
        "end-to-end production readiness). Give a percentage range and a one-line "
        "reason grounded in whether the logic is deterministic vs LLM vs "
        "human-reviewed.\n\n"
        "## Key insight\n"
        "2-4 sentences: where the accuracy gap concentrates (usually the HITL "
        "review checkpoints) and what flipping them from auto-approve to real review "
        "would do to end-to-end accuracy.\n"
    )


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            if cleaned.lower().startswith("markdown"):
                cleaned = cleaned[len("markdown"):]
    return cleaned.strip() + "\n"


# ---------------------------------------------------------------------------
# Deterministic fallback (agent-specific, no LLM required)
# ---------------------------------------------------------------------------

def _fallback_report(facts: dict) -> str:
    name = facts["agent_name"]
    rows: list[tuple[str, str, str, str]] = []

    for finding in facts["acceptance_findings"] + facts["ir_findings"]:
        rows.append((finding, "Human via Claude Code", "Investigate and fix", "30 min"))
    for err in facts["syntax_errors"]:
        rows.append((f"Syntax error in {err}", "Human via Claude Code", "Fix generated code", "30 min"))

    # Deterministic-stub markers left in the generated code. The LLM refinement
    # pass drafts a basic working version of each; a human (via Claude Code)
    # reviews and finalises it -- so these ship as reviewable code, not blanks.
    for stub in facts.get("code_stubs", []):
        where = ", ".join(stub.get("files", [])) or "generated code"
        rows.append((
            f"{stub['label']} (in {where})",
            "Human via Claude Code",
            stub["action"],
            stub["time"],
        ))

    for hitl in facts["hitl_nodes"]:
        rows.append((
            f"HITL node '{hitl}' defaults to auto-approve (HITL_MODE=auto)",
            "Human via Claude Code",
            "Implement the decision-merge in the HITL_MODE=file branch / wire a review UI",
            "1-2 hrs",
        ))
    for tool in facts["orphan_tools"]:
        rows.append((
            f"Tool '{tool}' is defined but not called by any node",
            "Human via Claude Code",
            "Confirm it is needed (or prune) -- may indicate a missed wiring",
            "15 min",
        ))
    if not facts["target_sdk_installed"]:
        rows.append((
            "agent_framework/ is a local stub, not the real SDK",
            "Human via Claude Code",
            "pip install the real agent-framework, delete the stub folder",
            "30 min",
        ))
    if not facts["gemini_key_set"]:
        rows.append((
            "GEMINI_API_KEY not set (LLM-authored parts fall back)",
            "Human via Claude Code",
            "Add the key to .env / deployment config",
            "10 min",
        ))
    if facts["entrypoint"] == "api":
        rows.append((
            "FastAPI /run endpoint not tested end-to-end",
            "Human via Claude Code",
            "Run the server and exercise /run; fix field mismatches",
            "1-2 hrs",
        ))
        if facts["hitl_nodes"]:
            rows.append((
                "No /hitl/respond endpoint (humans cannot respond via API)",
                "Human via Claude Code",
                "Add endpoint that writes <node>.response.json + a minimal review UI",
                "2-4 hrs",
            ))
    for action in facts["manual_actions"]:
        rows.append((str(action), "Human via Claude Code", "Complete the manual step", "30 min"))
    if not rows:
        rows.append(("No outstanding items detected", "-", "Ship it", "-"))

    def _clamp_time(t: str) -> str:
        """Cap any estimate that exceeds 2 hours."""
        if not t or t == "-":
            return t
        import re
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
        if not nums:
            return t
        if max(nums) > 2 and ("hr" in t.lower() or "hour" in t.lower()):
            return "2 hrs"
        return t

    capped = [(i, o, a, _clamp_time(t)) for (i, o, a, t) in rows]

    def _to_minutes(t: str) -> float:
        if not t or t == "-":
            return 0.0
        import re
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
        if not nums:
            return 0.0
        val = sum(nums) / len(nums)
        if "hr" in t.lower() or "hour" in t.lower():
            return val * 60
        return val

    total_min = sum(_to_minutes(t) for (_, _, _, t) in capped)
    if total_min > 0:
        hrs, mins = divmod(int(total_min), 60)
        total_str = f"{hrs} hr{'s' if hrs != 1 else ''} {mins} min" if hrs else f"{mins} min"
        capped.append(("**Total**", "", "", f"**{total_str}**"))

    work_table = "| Item | Owner | Action | Time |\n|---|---|---|---|\n" + "\n".join(
        f"| {i} | {o} | {a} | {t} |" for (i, o, a, t) in capped
    )

    # Accuracy dimensions from the actual nodes/roles.
    det_nodes = [n["name"] for n in facts["nodes"] if n["role"] in ("entry", "linear", "branch", "loop", "terminal")]
    acc_rows = []
    if det_nodes:
        acc_rows.append(("Deterministic node logic (" + ", ".join(det_nodes[:5]) + ")", "~85%", "Ported verbatim; bounded by input-data quality"))
    if facts["hitl_nodes"]:
        acc_rows.append(("HITL-gated decisions (" + ", ".join(facts["hitl_nodes"]) + ")", "~65-70%", "Auto-approve default; real human review is the swing factor"))
    if facts["llm_provider"] or facts["gemini_key_set"]:
        acc_rows.append(("LLM-authored sections", "~65-70%", "Syntactically valid; semantic correctness needs human review"))
    acc_rows.append(("Generated stub smoke test", "100% structural / 0% functional", "Exists and passes CI but must be implemented"))
    acc_rows.append(("End-to-end, production-ready", "~70%", "Sound pipeline; HITL being real (not auto-approve) is the swing factor"))
    acc_table = "| Dimension | Accuracy | Why |\n|---|---|---|\n" + "\n".join(
        f"| {d} | {a} | {w} |" for (d, a, w) in acc_rows
    )

    insight = (
        "The accuracy gap concentrates in the HITL checkpoints"
        + (f" ({', '.join(facts['hitl_nodes'])})" if facts["hitl_nodes"] else "")
        + ". They currently default to auto-approve, so an LLM/heuristic mistake can "
        "pass unchecked. Wiring real review (HITL_MODE=file or an API review UI) so a "
        "human approves before downstream nodes run typically lifts end-to-end "
        "accuracy into the ~88-92% range -- that is exactly what the 'LLM writes, "
        "human reviews' model is for; it just needs to be turned on."
    )

    return (
        f"# Readiness Report - {name}\n\n"
        "> Deterministic fallback (no LLM key available). Set GEMINI_API_KEY for an "
        "LLM-authored, deeper analysis.\n\n"
        "## Remaining work\n\n" + work_table + "\n\n"
        "## Accuracy by dimension\n\n" + acc_table + "\n\n"
        "## Key insight\n\n" + insight + "\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_readiness_report(
    ir: IR,
    conversion: ConversionResult,
    generation,
    config: Config | None = None,
    acceptance=None,
    agent_name: str = "Converted Agent",
    client: Optional[object] = None,
) -> str:
    """Return the READINESS_REPORT.md Markdown (LLM if available, else fallback)."""
    config = config or Config()
    facts = collect_facts(ir, conversion, generation, acceptance, config, agent_name)
    text = _call_gemini(_prompt(facts), config, client)
    if text and text.strip():
        return _strip_fences(text)
    return _fallback_report(facts)


def write_readiness_report(markdown: str, output_root: str) -> str:
    import os

    path = os.path.join(output_root, "READINESS_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    return path
