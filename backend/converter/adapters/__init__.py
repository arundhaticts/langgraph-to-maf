"""Framework adapters.

`base.py` defines the `SourceAdapter` / `TargetAdapter` interfaces. Source
adapters own framework detection (Section 16); target adapters own Tier 1/2
deterministic mappings and vocabulary.

Targets can be added two ways:
  1. a built-in adapter class registered in `TARGET_ADAPTERS`, or
  2. an uploaded framework pack folder (`frameworks/<name>/vocabulary.json`),
     loaded at runtime via `DynamicTargetAdapter` -- no code change needed.

Built-in names win over on-disk packs so bundled behaviour is stable; any other
name resolves from its uploaded pack. This is the seam that makes the tool
"any framework -> any framework" (Section 16).
"""

from __future__ import annotations

import json
import os

from converter.adapters.base import SourceAdapter, TargetAdapter, to_pascal_case
from converter.adapters.source.langgraph_adapter import LangGraphSourceAdapter
from converter.adapters.target.dynamic_adapter import DynamicTargetAdapter
from converter.adapters.target.maf_adapter import MAFTargetAdapter

# Registry: target framework name -> adapter class. Add a row to onboard a
# built-in target; uploaded packs need no row here.
TARGET_ADAPTERS: dict[str, type[TargetAdapter]] = {
    "maf": MAFTargetAdapter,
}

SOURCE_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "langgraph": LangGraphSourceAdapter,
}

# Default framework-pack store, relative to the converter package. Mirrors
# Config.frameworks_dir; kept here so adapter resolution has no Config dependency.
_FRAMEWORKS_DIRNAME = "frameworks"


def frameworks_base() -> str:
    """Absolute path to the framework-pack store (`converter/frameworks`)."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, _FRAMEWORKS_DIRNAME)


def _load_vocabulary(name: str) -> dict | None:
    """Read `frameworks/<name>/vocabulary.json`, or None if absent/invalid."""
    path = os.path.join(frameworks_base(), name, "vocabulary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def list_frameworks() -> list[str]:
    """All targetable framework names: built-ins plus uploaded packs on disk."""
    names = set(TARGET_ADAPTERS)
    base = frameworks_base()
    if os.path.isdir(base):
        for entry in os.listdir(base):
            if os.path.isfile(os.path.join(base, entry, "vocabulary.json")):
                names.add(entry)
    return sorted(names)


def get_target_adapter(name: str) -> TargetAdapter:
    """Resolve a target adapter by name.

    Built-in classes take precedence; otherwise the name must have an uploaded
    pack (`frameworks/<name>/vocabulary.json`) which drives a DynamicTargetAdapter.
    """
    cls = TARGET_ADAPTERS.get(name)
    if cls is not None:
        return cls()

    vocab = _load_vocabulary(name)
    if vocab is not None:
        return DynamicTargetAdapter(name, vocab)

    known = ", ".join(list_frameworks()) or "(none)"
    raise ValueError(
        f"No target adapter or uploaded pack for '{name}'. "
        f"Upload a framework folder with a vocabulary.json. Known: {known}."
    )


def get_source_adapter(name: str | None) -> SourceAdapter | None:
    """Return the source adapter for `name`, or None if unknown/unset."""
    cls = SOURCE_ADAPTERS.get(name or "")
    return cls() if cls else None


__all__ = [
    "SourceAdapter",
    "TargetAdapter",
    "to_pascal_case",
    "LangGraphSourceAdapter",
    "MAFTargetAdapter",
    "DynamicTargetAdapter",
    "TARGET_ADAPTERS",
    "SOURCE_ADAPTERS",
    "frameworks_base",
    "list_frameworks",
    "get_target_adapter",
    "get_source_adapter",
]
