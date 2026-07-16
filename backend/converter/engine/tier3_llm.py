"""Tier 3 -- Gemini LLM resolution for hard, framework-specific sections.

Used only when the deterministic (Tier 1) and README (Tier 2) tiers cannot
resolve something: complex/agent-driven orchestration and HITL flows. This is
the single seam the future local SLM swaps behind (Section 15).

Design rules:
- It NEVER raises for a missing API key or a disabled mode. It returns None, and
  the engine turns that into a "manual action required" entry + report note.
  This is what lets Approach 1 (Deterministic) reuse the engine with the LLM off.
- Framework knowledge is read at runtime from `frameworks/<target>/`, so we do
  not hard-code the target's hard idioms here.
- Provider is Gemini. It uses the `google-genai` SDK when installed and falls
  back to the stdlib REST API otherwise, so it works even where the SDK's wheels
  are blocked. Both paths are optional for the deterministic path / test suite.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from converter.config import Config
from converter.contracts import IR, FunctionSpec, Tier3Result, WorkflowSpec

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def load_framework_docs(target_framework: str, config: Config) -> str:
    """Read the target framework knowledge pack as the authoritative Tier-3 context.

    The pack is the folder uploaded through the UI and stored in
    `frameworks/<name>/`. Everything in it grounds the LLM:
      - docs.md          -> prose idioms / mapping rules
      - vocabulary.json  -> machine-readable term map + reject list
      - examples/*.py    -> few-shot SOURCE->TARGET code samples
    Attaching a pack for a brand-new framework is enough to target it -- no code
    change here. Absent files are skipped so a minimal pack still works.
    """
    chunks: list[str] = []
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(package_root, config.frameworks_dir, target_framework)

    for filename in ("docs.md", "vocabulary.json"):
        path = os.path.join(base, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    chunks.append(f"# {filename}\n{fh.read()}")
            except OSError:
                continue

    # Few-shot examples: every .py under examples/ (sorted for determinism).
    examples_dir = os.path.join(base, "examples")
    if os.path.isdir(examples_dir):
        for name in sorted(os.listdir(examples_dir)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(examples_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    chunks.append(f"# example: {name}\n```python\n{fh.read()}\n```")
            except OSError:
                continue

    if not chunks:
        return ""  # no pack on disk -> Tier 3 has no target grounding (never errors)

    header = (
        "# AUTHORITATIVE target-framework knowledge pack\n"
        f"Target framework: {target_framework}. Generate ONLY for this framework "
        "using the idioms below. When this conflicts with prior knowledge, THIS wins.\n"
    )
    return "\n\n".join([header, *chunks])


def _call_gemini_sdk(prompt: str, config: Config) -> Optional[str]:
    """Call Gemini via the `google-genai` SDK if it is installed.

    Returns the response text, None if the SDK is absent or the call fails. Note
    this is the NEWER `google-genai` package (`from google import genai`), which
    uses httpx -- not the older `google-generativeai` (which needs grpcio).
    """
    api_key = config.llm_api_key()
    if not api_key:
        return None
    try:
        from google import genai  # newer google-genai SDK (optional)
    except ImportError:
        return None
    # Behind a TLS-intercepting corporate proxy, httpx's bundled CA list rejects
    # the proxy cert. truststore makes ssl use the OS trust store (which has the
    # corporate CA), fixing the SDK's SSL. Best-effort; harmless if absent.
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.resolved_model(), contents=prompt
        )
        return response.text
    except Exception:
        return None


def _call_gemini_rest(prompt: str, config: Config) -> Optional[str]:
    """Call the Gemini REST API using only the stdlib (no SDK, no extra deps).

    This is the default path -- it needs nothing installed beyond Python, which
    matters in locked-down environments where the SDK's transitive wheels are
    blocked. Returns the response text, or None on any failure.
    """
    api_key = config.llm_api_key()
    if not api_key:
        return None
    url = _GEMINI_ENDPOINT.format(model=config.resolved_model())
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    except (urllib.error.URLError, KeyError, IndexError, ValueError, TimeoutError):
        return None


def _call_gemini(prompt: str, config: Config, client: Optional[object]) -> Optional[str]:
    """Low-level Gemini call. Returns raw text, or None if it cannot run.

    Resolution order:
      1. an injected `client` (tests, or a preconfigured model)
      2. the `google-genai` SDK, if installed
      3. the stdlib REST API (always available, zero dependencies)
    """
    if not config.allow_llm_fallback:
        return None
    if client is None and not config.llm_api_key():
        return None
    try:
        if client is not None:
            return client.generate_content(prompt).text
        text = _call_gemini_sdk(prompt, config)
        if text is not None:
            return text
        return _call_gemini_rest(prompt, config)
    except Exception:
        # Any failure (network, parse) degrades to manual review, never crashes.
        return None


def _parse_json(text: str, default_pattern: str) -> Optional[Tier3Result]:
    try:
        # Gemini sometimes fences JSON in ```json ... ```; strip fences.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)
        return Tier3Result(
            pattern=str(data.get("pattern", default_pattern)),
            generated_code=str(data.get("generated_code", "")),
            reasoning=str(data.get("reasoning", "")),
            confidence=float(data.get("confidence", 0.0)),
        )
    except Exception:
        return None


def _workflow_prompt(workflow: WorkflowSpec, framework_docs: str, target: str) -> str:
    return (
        f"You are converting an AI agent's orchestration to {target}.\n\n"
        f"Target framework knowledge:\n{framework_docs}\n\n"
        f"Orchestration pattern (neutral IR): {workflow.pattern.value}\n"
        f"Entry point: {workflow.entry_point}\n"
        f"Nodes: {[n.name for n in workflow.nodes]}\n"
        f"Conditional edges: "
        f"{[(c.source, c.outcomes) for c in workflow.conditional_edges]}\n\n"
        f"README workflow description:\n{workflow.readme_description or '(none)'}\n\n"
        "Generate ONLY the orchestration `run(ctx)` function for this framework. "
        "Respond as strict JSON with keys: pattern, generated_code, reasoning, "
        "confidence (0.0-1.0)."
    )


def _hitl_prompt(
    node_name: str, source: Optional[FunctionSpec], framework_docs: str, target: str
) -> str:
    original = source.body if source and source.body else "(source body unavailable)"
    return (
        f"You are converting a human-in-the-loop (HITL) step to {target}.\n\n"
        f"Target framework knowledge:\n{framework_docs}\n\n"
        f"The source node '{node_name}' used interrupt()/approval. Its logic:\n"
        f"```python\n{original}\n```\n\n"
        f"Generate ONLY the body of `def {node_name}(ctx):` implementing the "
        f"approval flow for {target} (return ctx). Respond as strict JSON with "
        "keys: pattern, generated_code, reasoning, confidence (0.0-1.0)."
    )


def resolve_with_llm(
    ir: IR,
    workflow: WorkflowSpec,
    config: Config,
    client: Optional[object] = None,
) -> Optional[Tier3Result]:
    """Resolve a hard orchestration section via Gemini. None if it cannot run."""
    target = ir.metadata.target_framework or "maf"
    docs = load_framework_docs(target, config)
    text = _call_gemini(_workflow_prompt(workflow, docs, target), config, client)
    if text is None:
        return None
    return _parse_json(text, workflow.pattern.value)


def resolve_hitl(
    ir: IR,
    node_name: str,
    source: Optional[FunctionSpec],
    config: Config,
    client: Optional[object] = None,
) -> Optional[Tier3Result]:
    """Generate a HITL approval flow via Gemini. None if it cannot run."""
    target = ir.metadata.target_framework or "maf"
    docs = load_framework_docs(target, config)
    text = _call_gemini(_hitl_prompt(node_name, source, docs, target), config, client)
    if text is None:
        return None
    return _parse_json(text, "hitl")
