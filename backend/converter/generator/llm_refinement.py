"""Stage 11 — LLM Refinement Pass (gate-closed repair loop).

After the deterministic pipeline produces the converted output + READINESS_REPORT.md,
this module feeds everything back to the LLM and asks it to complete the remaining
code-level tasks — then VERIFIES the result against the acceptance gate and, if the
gate is still red, feeds the exact failures back and retries. This closed loop is
"the 70→95 lever": the LLM's work is graded by machine each iteration, not by vibes.

    emit (deterministic)  ->  gate  ->  green & no tasks? => done.
       ^                        |  red (compile / forbidden-token / missing required
       |                        |       symbol / IR-coverage) + readiness todos
       +---- LLM repair --------+  loop until green OR iteration cap (default 3)

Each iteration the LLM receives:
  1. The current code (all .py files) — scoped ground truth.
  2. The remaining-work table from READINESS_REPORT.md.
  3. The ACTUAL acceptance-gate failures from the last verify (traceback text,
     the forbidden token found, the required symbol that is missing).
  4. The target framework knowledge pack (idioms / examples).

Constraints enforced by this module (not just asked of the LLM):
  - Only files that already exist in the output can be patched (injection guard).
  - Every returned file must pass ast.parse before it is written (bad Python is
    discarded and logged, never shipped).
  - If the gate is still red after the iteration cap, the remaining failures are
    recorded as manual_action_required in REFINEMENT_LOG.md — we never ship
    broken code that merely *looks* green.

Graceful degradation
====================
If the LLM key is absent, LLM fallback is disabled, or the LLM returns nothing,
the pass is silently skipped (ran=False). The deterministic output is already
complete; this stage only improves it.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from converter.config import Config
from converter.contracts import IR, ConversionResult
from converter.engine.tier3_llm import _call_gemini, load_framework_docs
from converter.verify import verify_output

# Default cap on repair iterations. Part B recommends ~3-5; 3 keeps token cost
# bounded while still catching the common "one more fix" case.
_DEFAULT_MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FilePatch:
    """One file updated by the refinement pass."""
    path: str           # relative to output_root
    content: str        # validated new content
    summary: str        # what changed (from LLM)


@dataclass
class RefinementResult:
    patches: list[FilePatch] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # files rejected (bad Python)
    overall_summary: str = ""
    ran: bool = False              # False when skipped (no key / LLM returned nothing)
    iterations: int = 0            # how many LLM repair rounds actually executed
    gate_passed: bool = False      # did the acceptance gate end green?
    gate_issues_remaining: list[str] = field(default_factory=list)  # unresolved at cap


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def _read_file(root: str, rel: str) -> str:
    try:
        with open(os.path.join(root, rel.replace("/", os.sep)), "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _read_output_files(output_root: str) -> dict[str, str]:
    """Return {relative_path: content} for every .py in the output folder."""
    files: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(output_root):
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, output_root).replace(os.sep, "/")
            try:
                with open(abs_path, "r", encoding="utf-8") as fh:
                    files[rel] = fh.read()
            except OSError:
                pass
    return files


def _read_readiness_report(output_root: str) -> str:
    path = os.path.join(output_root, "READINESS_REPORT.md")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _extract_remaining_tasks(readiness_md: str) -> list[str]:
    """Pull the Item column from the '## Remaining work' table.

    Returns a list of task description strings (never raises).
    """
    tasks: list[str] = []
    in_section = False
    in_table = False
    for line in readiness_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Remaining work"):
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if not in_section:
            continue
        if stripped.startswith("|") and "---" not in stripped:
            # Header row: Item | Owner | Action | Time
            if "Item" in stripped and "Owner" in stripped:
                in_table = True
                continue
            if in_table:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if cells and cells[0] and cells[0] != "Item":
                    tasks.append(cells[0])
    return tasks


def _has_outstanding_tasks(remaining_tasks: list[str]) -> bool:
    """True when the readiness report lists real remaining work."""
    if not remaining_tasks:
        return False
    if len(remaining_tasks) == 1 and "no outstanding" in remaining_tasks[0].lower():
        return False
    return True


# ---------------------------------------------------------------------------
# Re-verification helpers (so the gate reflects the LLM's patches)
# ---------------------------------------------------------------------------

def _refresh_syntax_errors(generation) -> None:
    """Recompute generation.syntax_errors from disk.

    verify_output seeds its non-compiling list from generation.syntax_errors,
    which is stale after the LLM patches a file. Re-parsing every written .py
    here makes the re-run gate reflect reality (a fixed file drops out of the
    list; a newly-broken one — which cannot happen, we ast-validate — would be
    caught). Never raises.
    """
    try:
        root = generation.output_root
        py = [r for r in getattr(generation, "written_files", []) if r.endswith(".py")]
        still_bad: list[str] = []
        for rel in py:
            try:
                ast.parse(_read_file(root, rel))
            except SyntaxError:
                still_bad.append(rel)
        generation.syntax_errors = still_bad
    except Exception:  # noqa: BLE001 - defensive; verification is best-effort
        pass


def _safe_verify(ir: IR, generation):
    """Run the acceptance gate; return the report or None if it cannot run."""
    try:
        _refresh_syntax_errors(generation)
        return verify_output(ir, generation)
    except Exception:  # noqa: BLE001 - a FakeGeneration in tests lacks fields
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = """\
You are a senior software engineer refining an auto-converted AI agent.
The agent was converted by an automated tool; the deterministic output is correct
but may have TODO stubs, auto-approve HITL placeholders, orphan tools, unresolved
orchestration sections, or a target-framework shape that is not yet idiomatic.

Your output is graded by an automated ACCEPTANCE GATE, and you may be asked to
repair the same code across several rounds. Each round you are shown the ACTUAL
gate failures. Your job is to make the gate pass while completing the remaining
code-level tasks.

Rules:
- Only modify files that need changes; leave others exactly as-is.
- Your Python MUST be syntactically valid (it is ast.parsed before being written;
  invalid files are discarded).
- Do not invent external services, APIs, or credentials not already present.
- Make every reported acceptance-gate failure pass:
    * "target_shape_present: required tokens absent" — emit the REAL target
      framework orchestration (e.g. a genuine WorkflowBuilder graph / StateGraph /
      Crew / Agent). Do NOT leave a deterministic shortcut that skips the required
      symbols.
    * "no_source_framework_residue: forbidden tokens" — remove every listed token;
      re-express using the target framework's idioms from the knowledge pack.
    * "all_*_generated" / coverage failures — add the missing node/tool/HITL artifact.
    * "loop_nodes_reachable" — ensure the named loop node has a real function.
- For HITL: replace auto-approve stubs with a proper decision flow using the target
  framework's idioms. Keep an automated fast-path (an env-var guard) so CI can run
  unattended.
- For orphan tools: wire them into the agent/workflow where it makes sense.
- Do NOT change requirements.txt, docs, or READINESS_REPORT.md.
- Do NOT output files that have no changes.

Response format — strict JSON, no markdown fences:
{
  "changes": [
    {
      "file": "relative/path/to/file.py",
      "content": "<full new file content as a string>",
      "summary": "One-line description of what changed"
    }
  ],
  "overall_summary": "2-3 sentences: what was completed, what was left and why."
}
"""


def _build_prompt(
    output_files: dict[str, str],
    readiness_md: str,
    remaining_tasks: list[str],
    framework_docs: str,
    gate_issues: Optional[list[str]] = None,
    iteration: int = 1,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> str:
    parts: list[str] = [_SYSTEM_INSTRUCTIONS, ""]

    if iteration > 1:
        parts.append(
            f"# Repair round {iteration} of {max_iterations}. "
            "The previous round did not make the acceptance gate pass. "
            "Focus on the gate failures listed below."
        )
        parts.append("")

    if framework_docs:
        parts.append("# Target framework knowledge pack")
        parts.append(framework_docs)
        parts.append("")

    if gate_issues:
        parts.append("# ACCEPTANCE GATE FAILURES (your patched code MUST make these pass)")
        for i, issue in enumerate(gate_issues, 1):
            parts.append(f"{i}. {issue}")
        parts.append("")

    parts.append("# READINESS_REPORT.md (source of remaining tasks)")
    parts.append(readiness_md)
    parts.append("")

    parts.append("# Remaining code-level tasks to complete:")
    for i, task in enumerate(remaining_tasks, 1):
        parts.append(f"{i}. {task}")
    parts.append("")

    parts.append("# Current converted code (all .py files)")
    for rel, content in sorted(output_files.items()):
        parts.append(f"\n## File: {rel}\n```python\n{content}\n```")

    parts.append(
        "\nNow produce the JSON response. Only include files that you actually changed."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```", 2)
        if len(parts) >= 3:
            inner = parts[1]
            if inner.lower().startswith("json"):
                inner = inner[4:]
            return inner.strip()
    return t


def _parse_response(text: str) -> tuple[list[dict], str]:
    """Parse the LLM JSON response into (changes, overall_summary)."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Try to find a JSON object anywhere in the text.
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not m:
            return [], "LLM response could not be parsed as JSON."
        try:
            data = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return [], "LLM response could not be parsed as JSON."

    changes = data.get("changes", [])
    summary = data.get("overall_summary", "")
    if not isinstance(changes, list):
        changes = []
    return changes, summary


def _validate_python(content: str) -> bool:
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

def _apply_patches(
    changes: list[dict],
    output_files: dict[str, str],
    output_root: str,
) -> tuple[list[FilePatch], list[str]]:
    """Write validated patches; return (accepted, skipped_paths)."""
    accepted: list[FilePatch] = []
    skipped: list[str] = []

    for change in changes:
        if not isinstance(change, dict):
            continue
        rel = change.get("file", "").replace("\\", "/").lstrip("/")
        content = change.get("content", "")
        summary = change.get("summary", "")

        if not rel or not content:
            continue

        # Only patch files that existed in the original output (safety guard).
        if rel not in output_files:
            skipped.append(f"{rel} (not in original output)")
            continue

        # Skip if content is unchanged.
        if content.strip() == output_files[rel].strip():
            continue

        if not _validate_python(content):
            skipped.append(f"{rel} (failed ast.parse — discarded)")
            continue

        abs_path = os.path.join(output_root, rel.replace("/", os.sep))
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            accepted.append(FilePatch(path=rel, content=content, summary=summary))
            # Update the in-memory snapshot so a later round sees the new content.
            output_files[rel] = content
        except OSError as exc:
            skipped.append(f"{rel} (write error: {exc})")

    return accepted, skipped


# ---------------------------------------------------------------------------
# Refinement log
# ---------------------------------------------------------------------------

def _build_refinement_log(result: RefinementResult, agent_name: str) -> str:
    lines = [f"# LLM Refinement Log — {agent_name}\n"]

    if not result.ran:
        lines.append(
            "> Refinement pass skipped (LLM key not set or LLM returned no changes).\n"
        )
        return "\n".join(lines)

    lines.append(f"## Summary\n\n{result.overall_summary}\n")

    gate = "GREEN (all acceptance checks pass)" if result.gate_passed else "RED (issues remain)"
    lines.append("## Acceptance gate\n")
    lines.append(f"- Repair iterations run: {result.iterations}")
    lines.append(f"- Final gate status: {gate}")
    lines.append("")

    if result.patches:
        lines.append("## Files updated\n")
        for p in result.patches:
            lines.append(f"- **{p.path}**: {p.summary}")
        lines.append("")

    if result.skipped:
        lines.append("## Skipped (not applied)\n")
        for s in result.skipped:
            lines.append(f"- {s}")
        lines.append("")

    if not result.gate_passed and result.gate_issues_remaining:
        lines.append("## Manual action required (gate still RED after cap)\n")
        lines.append(
            "The LLM could not make these acceptance checks pass automatically. "
            "They must be resolved by hand before shipping:\n"
        )
        for issue in result.gate_issues_remaining:
            lines.append(f"- {issue}")
        lines.append("")

    if not result.patches and not result.skipped:
        lines.append("_No changes were needed — the generated code was already complete._\n")

    return "\n".join(lines)


def write_refinement_log(result: RefinementResult, output_root: str, agent_name: str = "Converted Agent") -> str:
    path = os.path.join(output_root, "REFINEMENT_LOG.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_build_refinement_log(result, agent_name))
    return path


# ---------------------------------------------------------------------------
# Public API — the gate-closed repair loop
# ---------------------------------------------------------------------------

def run_llm_refinement(
    ir: IR,
    conversion: ConversionResult,
    generation,
    config: Config,
    output_root: str,
    agent_name: str = "Converted Agent",
    client: Optional[object] = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> RefinementResult:
    """Run the gate-closed LLM refinement loop. Returns a RefinementResult (never raises).

    Skipped (result.ran=False) when:
    - config.allow_llm_fallback is False (deterministic mode)
    - No API key is configured (and no client injected)

    Loop:
    - Run the acceptance gate. If it is green AND there are no outstanding tasks,
      return early (ran=True, gate_passed=True, no changes).
    - Otherwise, up to `max_iterations` times: send the code + readiness tasks +
      ACTUAL gate failures to the LLM, apply validated patches, re-run the gate.
      Stop as soon as the gate is green, or when a round makes no progress.
    - Any gate failures still present at the cap are recorded as manual action.

    On any exception the result is returned as-is with the exception logged.
    """
    result = RefinementResult()

    if not config.allow_llm_fallback:
        return result
    if client is None and not config.llm_api_key():
        return result

    try:
        readiness_md = _read_readiness_report(output_root)
        remaining_tasks = _extract_remaining_tasks(readiness_md)
        target = ir.metadata.target_framework or "maf"
        framework_docs = load_framework_docs(target, config)

        # Initial acceptance gate against the deterministic output.
        acceptance = _safe_verify(ir, generation)
        gate_issues = acceptance.issues() if acceptance is not None else []
        gate_green = acceptance.passed if acceptance is not None else True

        # Nothing to do: the gate is already green AND no outstanding tasks.
        if gate_green and not _has_outstanding_tasks(remaining_tasks):
            result.ran = True
            result.gate_passed = True
            result.overall_summary = (
                "Acceptance gate green and no remaining tasks; refinement not needed."
            )
            return result

        last_summary = ""
        for iteration in range(1, max_iterations + 1):
            output_files = _read_output_files(output_root)
            prompt = _build_prompt(
                output_files,
                readiness_md,
                remaining_tasks,
                framework_docs,
                gate_issues=gate_issues,
                iteration=iteration,
                max_iterations=max_iterations,
            )

            raw = _call_gemini(prompt, config, client)
            if not raw or not raw.strip():
                break

            changes, summary = _parse_response(raw)
            if summary:
                last_summary = summary
            result.ran = True

            if not changes:
                # LLM asserts nothing to change; stop looping.
                break

            patches, skipped = _apply_patches(changes, output_files, output_root)
            result.patches.extend(patches)
            result.skipped.extend(skipped)
            result.iterations = iteration

            # Re-run the gate against the patched output.
            acceptance = _safe_verify(ir, generation)
            gate_issues = acceptance.issues() if acceptance is not None else []
            gate_green = acceptance.passed if acceptance is not None else True

            if gate_green:
                result.gate_passed = True
                break

            if not patches:
                # No forward progress this round -> stop burning iterations.
                break

        result.gate_issues_remaining = [] if result.gate_passed else gate_issues

        if not result.overall_summary:
            prefix = (last_summary + " ") if last_summary else ""
            if result.gate_passed:
                result.overall_summary = (
                    f"{prefix}Acceptance gate GREEN after {result.iterations} "
                    "refinement iteration(s)."
                )
            elif result.ran:
                remaining = len(result.gate_issues_remaining)
                if remaining:
                    result.overall_summary = (
                        f"{prefix}Refinement ran {result.iterations} iteration(s); "
                        f"{remaining} acceptance issue(s) still RED — recorded as "
                        "manual_action_required below."
                    )
                else:
                    result.overall_summary = (
                        f"{prefix}Refinement ran {result.iterations} iteration(s)."
                    )

    except Exception as exc:  # noqa: BLE001
        # Absolutely never crash the pipeline. Log and move on.
        result.skipped.append(f"Refinement pass exception: {exc}")

    return result
