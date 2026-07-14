"""Tests for Module 1 -- repo scanner."""

from __future__ import annotations

import os

import pytest

from converter.contracts import FileType
from converter.scanner import ScannerError, scan_repo


def _write(root: str, rel_path: str, content: str = "") -> None:
    abs_path = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(abs_path) or root, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _make_valid_repo(root: str) -> None:
    _write(root, "README.md", "# Agent\n## Purpose\nDoes things.\n")
    _write(root, "agent.py", "from langgraph.graph import StateGraph\n")
    _write(root, "tools/repo_reader.py", "import os\n")
    _write(root, "prompts/system.md", "You are an agent.\n")
    _write(root, "requirements.txt", "langgraph\n")


def test_valid_repo_builds_manifest(tmp_path):
    root = str(tmp_path)
    _make_valid_repo(root)

    manifest = scan_repo(root)

    assert manifest.input_root == os.path.abspath(root)
    assert manifest.readme_path == "README.md"
    assert manifest.detected_framework == "langgraph"
    # Two python files present.
    assert len(manifest.python_files()) == 2


def test_file_classification(tmp_path):
    root = str(tmp_path)
    _make_valid_repo(root)

    manifest = scan_repo(root)
    by_path = {f.relative_path: f.file_type for f in manifest.files}

    assert by_path["README.md"] is FileType.README
    assert by_path["agent.py"] is FileType.PYTHON
    assert by_path[os.path.join("prompts", "system.md")] is FileType.PROMPT
    assert by_path["requirements.txt"] is FileType.REQUIREMENTS


def test_missing_readme_hard_stops(tmp_path):
    root = str(tmp_path)
    _write(root, "agent.py", "import langgraph\n")

    with pytest.raises(ScannerError, match="README.md"):
        scan_repo(root)


def test_lowercase_readme_is_not_accepted(tmp_path):
    root = str(tmp_path)
    _write(root, "readme.md", "# nope\n")  # wrong case
    _write(root, "agent.py", "import langgraph\n")

    with pytest.raises(ScannerError, match="README.md"):
        scan_repo(root)


def test_no_python_files_hard_stops(tmp_path):
    root = str(tmp_path)
    _write(root, "README.md", "# Agent\n")

    with pytest.raises(ScannerError, match="No Python"):
        scan_repo(root)


def test_missing_path_hard_stops(tmp_path):
    missing = os.path.join(str(tmp_path), "does-not-exist")

    with pytest.raises(ScannerError, match="does not exist"):
        scan_repo(missing)


def test_unknown_framework_is_not_a_hard_stop(tmp_path):
    root = str(tmp_path)
    _write(root, "README.md", "# Agent\n")
    _write(root, "agent.py", "import flask\n")

    manifest = scan_repo(root)
    assert manifest.detected_framework is None


def test_detection_via_from_import(tmp_path):
    root = str(tmp_path)
    _write(root, "README.md", "# Agent\n")
    _write(root, "agent.py", "from langgraph.graph import StateGraph\n")

    manifest = scan_repo(root)
    assert manifest.detected_framework == "langgraph"


def test_ignored_dirs_are_skipped(tmp_path):
    root = str(tmp_path)
    _make_valid_repo(root)
    _write(root, "__pycache__/agent.cpython-311.pyc", "junk")
    _write(root, ".venv/lib/site.py", "import langgraph\n")

    manifest = scan_repo(root)
    rel_paths = {f.relative_path for f in manifest.files}
    assert not any(".venv" in p or "__pycache__" in p for p in rel_paths)


def test_unparseable_python_does_not_crash_detection(tmp_path):
    root = str(tmp_path)
    _write(root, "README.md", "# Agent\n")
    _write(root, "broken.py", "def (((\n")  # syntax error
    _write(root, "good.py", "import langgraph\n")

    manifest = scan_repo(root)
    assert manifest.detected_framework == "langgraph"
