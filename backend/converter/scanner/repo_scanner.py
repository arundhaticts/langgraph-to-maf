"""Module 1 -- Repo scanner: validation and manifest building.

Responsibilities (Section 11, Module 1 of the build plan):

- Validate the input folder: `README.md` at root (case-sensitive) and at least
  one `.py` file. Hard-stop with a clear message otherwise -- it must NOT let a
  half-valid repo through to the parser.
- Tag every file by type into a `RepoManifest`.
- Auto-detect the source framework by scanning imports and asking each
  registered `SourceAdapter.detect()` for its confidence (Section 16).

This module is framework-agnostic by construction: it produces the frozen
`RepoManifest` contract and knows nothing about the target framework or the
conversion approach. All three approaches (Deterministic / Full LLM / Hybrid)
start here.

Framework detection now lives behind the source-adapter registry: `scan_repo`
collects the imported module roots across the repo and delegates to
`detect_source_framework`, which scores every registered adapter. Adding a
source adapter is enough to make its framework detectable -- no changes here.
"""

from __future__ import annotations

import ast
import os

from converter.adapters import detect_source_framework
from converter.config import Config
from converter.contracts import (
    FileEntry,
    FileType,
    RepoManifest,
)


class ScannerError(Exception):
    """Raised when the input folder fails validation.

    The pipeline treats this as a hard stop: the message is shown to the user
    and the converter exits before the parser ever runs.
    """


# Directories we never descend into or count as source.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "build",
        "dist",
        ".egg-info",
    }
)

def _classify_file(relative_path: str, config: Config) -> FileType:
    """Map a file to its `FileType`. `relative_path` uses OS separators."""
    name = os.path.basename(relative_path)
    lower = name.lower()

    # Root README is special: exact, case-sensitive name at repo root.
    if relative_path == config.required_readme_name:
        return FileType.README

    if lower.endswith(".py"):
        return FileType.PYTHON

    if lower in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"):
        return FileType.REQUIREMENTS

    if lower.endswith(".md"):
        # Non-root markdown is treated as prompt/doc content -> copy through.
        return FileType.PROMPT

    return FileType.OTHER


def _iter_repo_files(input_root: str) -> list[str]:
    """Return every non-ignored file path, relative to `input_root`.

    Paths use the OS separator; comparisons against contract fields normalise.
    """
    collected: list[str] = []
    for dirpath, dirnames, filenames in os.walk(input_root):
        # Prune ignored directories in place so os.walk skips them.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _IGNORED_DIRS and not d.endswith(".egg-info")
        ]
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, input_root)
            collected.append(rel_path)
    return sorted(collected)


def _imported_roots(python_source: str) -> set[str]:
    """Top-level module names imported by a Python source string.

    Uses AST, never regex (build-plan rule: no regex on Python code). Returns an
    empty set if the source does not parse -- detection tolerates unparseable
    files rather than crashing the scan.
    """
    try:
        tree = ast.parse(python_source)
    except SyntaxError:
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (node.module is None or level > 0 with no
            # framework meaning); only absolute module roots identify a framework.
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


def _detect_framework(input_root: str, python_files: list[FileEntry]) -> str | None:
    """Scan imports across all Python files and match a known framework.

    Collects the imported module roots repo-wide and delegates scoring to the
    source-adapter registry. Returns the framework name (e.g. "langgraph") or
    None if no adapter matched.
    """
    seen_roots: set[str] = set()
    for entry in python_files:
        abs_path = os.path.join(input_root, entry.relative_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        seen_roots |= _imported_roots(source)

    return detect_source_framework(seen_roots)


def scan_repo(input_root: str, config: Config | None = None) -> RepoManifest:
    """Validate `input_root` and build a `RepoManifest`.

    Validation (hard-stop via `ScannerError`):
      1. path exists and is a directory
      2. `README.md` present at root (exact, case-sensitive name)
      3. at least one `.py` file anywhere in the tree

    Framework detection is best-effort: an unrecognised framework leaves
    `detected_framework=None` and is NOT a hard stop here -- the pipeline decides
    whether an unsupported source framework should halt (keeps this module
    reusable across all three approaches and future source frameworks).
    """
    config = config or Config()

    if not os.path.exists(input_root):
        raise ScannerError(f"Input path does not exist: {input_root}")
    if not os.path.isdir(input_root):
        raise ScannerError(f"Input path is not a folder: {input_root}")

    relative_paths = _iter_repo_files(input_root)

    files: list[FileEntry] = [
        FileEntry(relative_path=rel, file_type=_classify_file(rel, config))
        for rel in relative_paths
    ]

    readme_entries = [f for f in files if f.file_type is FileType.README]
    python_entries = [f for f in files if f.file_type is FileType.PYTHON]

    if not readme_entries:
        raise ScannerError(
            f"Missing required '{config.required_readme_name}' at the root of "
            f"{input_root} (exact, case-sensitive name)."
        )
    if not python_entries:
        raise ScannerError(
            f"No Python (.py) files found under {input_root}; nothing to convert."
        )

    detected = _detect_framework(input_root, python_entries)

    return RepoManifest(
        input_root=os.path.abspath(input_root),
        files=files,
        detected_framework=detected,
        readme_path=readme_entries[0].relative_path,
    )
