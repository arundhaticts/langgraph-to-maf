"""Framework adapters.

`base.py` defines the `SourceAdapter` / `TargetAdapter` interfaces. Source
adapters own framework detection (Section 16); target adapters own Tier 1/2
deterministic mappings and vocabulary. New frameworks are added here, not in the
core pipeline.
"""

from converter.adapters.base import SourceAdapter, TargetAdapter, to_pascal_case
from converter.adapters.source.langgraph_adapter import LangGraphSourceAdapter
from converter.adapters.target.maf_adapter import MAFTargetAdapter

# Registry: target framework name -> adapter class. Add a row to onboard a target.
TARGET_ADAPTERS: dict[str, type[TargetAdapter]] = {
    "maf": MAFTargetAdapter,
}

SOURCE_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "langgraph": LangGraphSourceAdapter,
}


def get_target_adapter(name: str) -> TargetAdapter:
    try:
        return TARGET_ADAPTERS[name]()
    except KeyError:
        raise ValueError(
            f"No target adapter for '{name}'. Known: {', '.join(TARGET_ADAPTERS)}."
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
    "TARGET_ADAPTERS",
    "SOURCE_ADAPTERS",
    "get_target_adapter",
    "get_source_adapter",
]
