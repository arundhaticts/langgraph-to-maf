"""Conversion service shared by the FastAPI app and the tests.

Turns an uploaded folder (list of {path, content}) into a zip of the converted
MAF agent. No web framework imported here -- pure logic, easy to test.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import zipfile

# Make the converter package importable however the backend is launched
# (this file lives next to the `converter/` package inside backend/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converter.config import Config, ConversionMode  # noqa: E402
from converter.pipeline.hybrid_pipeline import HybridPipeline  # noqa: E402

MODE_MAP = {
    "manual": ConversionMode.DETERMINISTIC,  # user implements the hard parts
    "llm": ConversionMode.HYBRID,            # Gemini writes them, user reviews
}


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


def _run_and_zip(input_path: str, mode: str) -> bytes:
    """Run the pipeline on `input_path` and return a zip of the output folder.

    The scanner ignores dependency dirs (.venv, node_modules, __pycache__, ...),
    so a huge folder is scanned in terms of its source only.
    """
    conversion_mode = MODE_MAP.get(mode, ConversionMode.HYBRID)
    out_dir = tempfile.mkdtemp(prefix="fcu_out_")
    try:
        HybridPipeline(Config(mode=conversion_mode)).run(input_path, out_dir)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, filenames in os.walk(out_dir):
                for name in filenames:
                    abs_path = os.path.join(root, name)
                    zf.write(abs_path, os.path.relpath(abs_path, out_dir))
        return buffer.getvalue()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def convert_local_path(input_path: str, mode: str) -> bytes:
    """Convert a folder that already exists on disk (no upload).

    Ideal when the backend runs on the same machine: nothing is uploaded, and the
    scanner skips .venv / node_modules / __pycache__, so a 60k-file folder is
    handled instantly.
    """
    expanded = os.path.abspath(os.path.expanduser(input_path.strip().strip('"')))
    if not os.path.isdir(expanded):
        raise NotADirectoryError(f"Not a folder: {input_path}")
    return _run_and_zip(expanded, mode)


def convert_folder(files: list[dict], mode: str) -> bytes:
    """Write uploaded files to a temp dir, convert, and return a zip of output."""
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
        return _run_and_zip(in_dir, mode)
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
