"""Conversion service shared by the FastAPI app and the tests.

Turns an uploaded folder (list of {path, content}) into a zip of the converted
agent, targeting an uploaded framework pack. No web framework imported here --
pure logic, easy to test.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
import tempfile
import zipfile

# Make the converter package importable however the backend is launched
# (this file lives next to the `converter/` package inside backend/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converter.adapters import (  # noqa: E402
    frameworks_base,
    list_frameworks,
    list_source_frameworks,
)
from converter.config import Config, ConversionMode  # noqa: E402
from converter.pipeline.hybrid_pipeline import HybridPipeline  # noqa: E402

MODE_MAP = {
    "manual": ConversionMode.DETERMINISTIC,  # user implements the hard parts
    "llm": ConversionMode.HYBRID,            # Gemini writes them, user reviews
}

# Text file types kept from an uploaded framework pack.
_PACK_KEEP = re.compile(r"\.(md|markdown|txt|json|ya?ml|py)$", re.IGNORECASE)


def strip_common_top(files: list[dict]) -> list[dict]:
    """If every path shares one top-level folder, drop it so README.md is at root."""
    tops = set()
    for f in files:
        parts = f["path"].replace("\\", "/").split("/")
        tops.add(parts[0] if len(parts) > 1 else "")
    if len(tops) == 1 and "" not in tops:
        prefix = tops.pop() + "/"
        for f in files:
            p = f["path"].replace("\\", "/")
            f["path"] = p[len(prefix):] if p.startswith(prefix) else p
    return files


def _common_top(files: list[dict]) -> str | None:
    """The single shared top-level folder name of the uploaded files, or None."""
    tops = set()
    for f in files:
        parts = f["path"].replace("\\", "/").split("/")
        tops.add(parts[0] if len(parts) > 1 else "")
    return tops.pop() if len(tops) == 1 and "" not in tops else None


def _safe_name(name: str) -> str:
    """Sanitise a framework name for use as a folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip().lower()).strip("._-")
    return cleaned or "custom"


def save_framework_pack(files: list[dict], target_name: str | None = None) -> str:
    """Persist an uploaded framework pack to `frameworks/<name>/` and return the name.

    The pack must contain a `vocabulary.json` (the machine-readable term map that
    drives Tier 1/2). `docs.md` and `examples/*.py` are optional Tier-3 grounding.
    The framework name defaults to the uploaded folder's top-level directory.
    """
    if not files:
        raise ValueError("No framework files were uploaded.")

    # Derive the name from the uploaded folder unless one was given explicitly.
    name = _safe_name(target_name or _common_top(files) or "custom")

    # Strip the shared top folder so files land directly under frameworks/<name>/.
    staged = strip_common_top([dict(f) for f in files])
    kept = [f for f in staged if _PACK_KEEP.search(f["path"].replace("\\", "/"))]

    rels = {f["path"].replace("\\", "/").lstrip("/") for f in kept}
    if not any(r == "vocabulary.json" or r.endswith("/vocabulary.json") for r in rels):
        raise ValueError(
            "The framework folder must contain a 'vocabulary.json' at its root."
        )

    dest = os.path.join(frameworks_base(), name)
    # Replace any previous pack of the same name so re-uploads are clean.
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    for f in kept:
        rel = f["path"].replace("\\", "/").lstrip("/")
        if not rel or rel.endswith("/"):
            continue
        abs_path = os.path.join(dest, *rel.split("/"))
        os.makedirs(os.path.dirname(abs_path) or dest, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(f.get("content", ""))
    return name


def _extract_report_summary(out_dir: str) -> dict:
    """Load the computed readiness metrics written by the pipeline.

    Reads readiness_metrics.json (the machine-readable sidecar) so the UI shows
    the exact computed values -- no Markdown re-parsing, no placeholder text.
    """
    import json

    summary = {
        "total_time": "",
        "accuracy": "",
        "readiness_pct": "",
        "production_readiness": "",
        "confidence": "",
        "confidence_range": "",
        "low_end_effort": "",
        "high_end_effort": "",
        "avg_high_effort": "",
        "highest_accuracy": "",
        "lowest_accuracy": "",
    }
    metrics_path = os.path.join(out_dir, "readiness_metrics.json")
    try:
        with open(metrics_path, encoding="utf-8") as fh:
            m = json.load(fh)
    except Exception:
        return summary

    summary["total_time"] = str(m.get("recommended_effort", ""))
    summary["accuracy"] = str(m.get("accuracy_display", ""))
    summary["readiness_pct"] = str(m.get("readiness_pct", ""))
    summary["production_readiness"] = str(m.get("production_readiness", ""))
    summary["confidence"] = str(m.get("confidence", ""))
    summary["confidence_range"] = str(m.get("confidence_range", ""))
    summary["low_end_effort"] = str(m.get("low_end_effort", ""))
    summary["high_end_effort"] = str(m.get("high_end_effort", ""))
    summary["avg_high_effort"] = str(m.get("average_high_end_effort", ""))
    summary["highest_accuracy"] = str(m.get("highest_accuracy", ""))
    summary["lowest_accuracy"] = str(m.get("lowest_accuracy", ""))
    return summary


def _run_and_zip(
    input_path: str, mode: str, target: str, source: str | None = None
) -> tuple[bytes, dict]:
    """Run the pipeline on `input_path` for `target` and zip the output folder.

    Returns (zip_bytes, summary_dict) where summary_dict has total_time and accuracy.
    The scanner ignores dependency dirs (.venv, node_modules, __pycache__, ...),
    so a huge folder is scanned in terms of its source only. `source` is an
    optional explicit source-framework override (else it is auto-detected).
    """
    if target not in list_frameworks():
        known = ", ".join(list_frameworks()) or "(none)"
        raise ValueError(
            f"Unknown target framework '{target}'. Known targets: {known}."
        )
    if source and source not in list_source_frameworks():
        known = ", ".join(list_source_frameworks()) or "(none)"
        raise ValueError(
            f"Unknown source framework '{source}'. Known sources: {known}."
        )
    import time

    conversion_mode = MODE_MAP.get(mode, ConversionMode.HYBRID)
    config = Config(mode=conversion_mode, target_framework=target, source_framework=source)
    out_dir = tempfile.mkdtemp(prefix="fcu_out_")
    try:
        started = time.monotonic()
        HybridPipeline(config).run(input_path, out_dir)
        elapsed = time.monotonic() - started

        summary = _extract_report_summary(out_dir)
        buffer = io.BytesIO()
        file_count = 0
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, filenames in os.walk(out_dir):
                for name in filenames:
                    abs_path = os.path.join(root, name)
                    zf.write(abs_path, os.path.relpath(abs_path, out_dir))
                    file_count += 1
        summary["files_converted"] = str(file_count)
        summary["conversion_time"] = _fmt_elapsed(elapsed)
        return buffer.getvalue(), summary
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _fmt_elapsed(seconds: float) -> str:
    """Human 'Nm Ss' / 'Ns' for a wall-clock duration."""
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def convert_local_path(
    input_path: str,
    mode: str,
    target: str = "maf",
    framework_files: list[dict] | None = None,
    source: str | None = None,
) -> tuple[bytes, dict]:
    """Convert a folder already on disk (no source upload) to `target`.

    The target is normally chosen from the frameworks/ folder on disk. The legacy
    `framework_files` upload is still honoured (saved as the target pack first).
    """
    if framework_files:
        target = save_framework_pack(framework_files, target)
    expanded = os.path.abspath(os.path.expanduser(input_path.strip().strip('"')))
    if not os.path.isdir(expanded):
        raise NotADirectoryError(f"Not a folder: {input_path}")
    return _run_and_zip(expanded, mode, target, source=source)


def convert_folder(
    files: list[dict],
    mode: str,
    target: str = "maf",
    framework_files: list[dict] | None = None,
    source: str | None = None,
) -> tuple[bytes, dict]:
    """Write uploaded source files to a temp dir, convert to `target`, return (zip, summary)."""
    if framework_files:
        target = save_framework_pack(framework_files, target)
    in_dir = tempfile.mkdtemp(prefix="fcu_in_")
    try:
        for f in strip_common_top(list(files)):
            rel = f["path"].replace("\\", "/").lstrip("/")
            if not rel or rel.endswith("/"):
                continue
            abs_path = os.path.join(in_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(abs_path) or in_dir, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(f.get("content", ""))
        return _run_and_zip(in_dir, mode, target, source=source)
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
