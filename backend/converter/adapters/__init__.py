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
from converter.adapters.source.aws_strands_adapter import AWSStrandsSourceAdapter
from converter.adapters.source.crewai_adapter import CrewAISourceAdapter
from converter.adapters.source.langgraph_adapter import LangGraphSourceAdapter
from converter.adapters.source.maf_adapter import MAFSourceAdapter
from converter.adapters.target.aws_strands_adapter import AWSStrandsTargetAdapter
from converter.adapters.target.crewai_adapter import CrewAITargetAdapter
from converter.adapters.target.dynamic_adapter import DynamicTargetAdapter
from converter.adapters.target.langgraph_adapter import LangGraphTargetAdapter
from converter.adapters.target.maf_adapter import MAFTargetAdapter

# Registry: target framework name -> adapter class. Add a row to onboard a
# built-in target; uploaded packs need no row here.
TARGET_ADAPTERS: dict[str, type[TargetAdapter]] = {
    "maf": MAFTargetAdapter,
    "langgraph": LangGraphTargetAdapter,
    "crewai": CrewAITargetAdapter,
    "aws_strands": AWSStrandsTargetAdapter,
}

SOURCE_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "langgraph": LangGraphSourceAdapter,
    "crewai": CrewAISourceAdapter,
    "aws_strands": AWSStrandsSourceAdapter,
    "maf": MAFSourceAdapter,
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


def _prettify(name: str) -> str:
    """`aws_strands` -> `Aws Strands` (fallback display name)."""
    return " ".join(p.capitalize() for p in name.replace("-", "_").split("_") if p) or name


def list_frameworks_detailed() -> list[dict]:
    """Every known framework with its capabilities, for the UI dropdowns.

    Each entry: `{name, display_name, source: bool, target: bool}`. A framework
    is known if it has a registered source/target adapter OR a `vocabulary.json`
    pack on disk. Source/target support defaults from the adapter registries and
    on-disk presence, and a pack may override either via `supports_source` /
    `supports_target` in its vocabulary.json (e.g. LangGraph is source-only until
    a LangGraph target generator lands). This is what makes "drop a folder in
    frameworks/ and it appears in the UI" true.
    """
    base = frameworks_base()
    on_disk: set[str] = set()
    if os.path.isdir(base):
        for entry in os.listdir(base):
            if os.path.isfile(os.path.join(base, entry, "vocabulary.json")):
                on_disk.add(entry)

    names = set(SOURCE_ADAPTERS) | set(TARGET_ADAPTERS) | on_disk
    out: list[dict] = []
    for name in sorted(names):
        vocab = _load_vocabulary(name)
        supports_source = name in SOURCE_ADAPTERS
        supports_target = name in TARGET_ADAPTERS or name in on_disk
        display = _prettify(name)
        if vocab:
            meta = vocab.get("_meta") if isinstance(vocab.get("_meta"), dict) else {}
            display = vocab.get("display_name") or (meta or {}).get("display_name") or display
            if "supports_source" in vocab:
                supports_source = bool(vocab["supports_source"])
            if "supports_target" in vocab:
                supports_target = bool(vocab["supports_target"])
        out.append(
            {
                "name": name,
                "display_name": display,
                "source": supports_source,
                "target": supports_target,
            }
        )
    return out


def list_frameworks() -> list[str]:
    """All targetable framework names (built-ins + packs that allow targeting)."""
    return [f["name"] for f in list_frameworks_detailed() if f["target"]]


def list_source_frameworks() -> list[str]:
    """All framework names usable as a conversion source."""
    return [f["name"] for f in list_frameworks_detailed() if f["source"]]


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


def detect_source_framework(imported_roots: set[str]) -> str | None:
    """Pick the registered source framework that best matches `imported_roots`.

    Each adapter scores its confidence via `SourceAdapter.detect()`; the highest
    positive score wins. Returns the framework name, or None if nothing matched.
    This is the registry-driven replacement for the scanner's old hardcoded
    signature table -- adding a source adapter is enough to make it detectable.
    """
    best_name: str | None = None
    best_score = 0.0
    for name, cls in SOURCE_ADAPTERS.items():
        score = cls().detect(imported_roots)
        if score > best_score:
            best_name, best_score = name, score
    return best_name


__all__ = [
    "SourceAdapter",
    "TargetAdapter",
    "to_pascal_case",
    "LangGraphSourceAdapter",
    "CrewAISourceAdapter",
    "AWSStrandsSourceAdapter",
    "MAFSourceAdapter",
    "MAFTargetAdapter",
    "LangGraphTargetAdapter",
    "CrewAITargetAdapter",
    "AWSStrandsTargetAdapter",
    "DynamicTargetAdapter",
    "TARGET_ADAPTERS",
    "SOURCE_ADAPTERS",
    "frameworks_base",
    "list_frameworks",
    "list_frameworks_detailed",
    "list_source_frameworks",
    "get_target_adapter",
    "get_source_adapter",
    "detect_source_framework",
]
