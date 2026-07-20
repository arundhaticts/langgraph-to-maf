"""Tests for the DynamicTargetAdapter — the 'any target framework' seam.

A dynamic adapter is driven entirely by an uploaded vocabulary.json. It must
fulfil the same TargetAdapter contract as a hand-written adapter (e.g. MAF),
including a capability_matrix() so Phase-5b negotiation works for custom packs.
"""

from __future__ import annotations

from converter.adapters.target.dynamic_adapter import DynamicTargetAdapter
from converter.contracts import ConstructSupport, ConstructType


def test_capability_matrix_reads_pack_declaration():
    vocab = {
        "capabilities": {
            "tools": "direct",
            "hitl": "lossy",
            "checkpointing": "unsupported",
        }
    }
    adapter = DynamicTargetAdapter("custom", vocab)
    matrix = adapter.capability_matrix()
    assert matrix[ConstructType.TOOLS] is ConstructSupport.DIRECT
    assert matrix[ConstructType.HITL] is ConstructSupport.LOSSY
    assert matrix[ConstructType.CHECKPOINTING] is ConstructSupport.UNSUPPORTED


def test_capability_matrix_covers_every_construct():
    """Every ConstructType is present so negotiation never KeyErrors."""
    adapter = DynamicTargetAdapter("custom", {"capabilities": {"tools": "direct"}})
    matrix = adapter.capability_matrix()
    for construct in ConstructType:
        assert construct in matrix


def test_capability_matrix_defaults_to_direct_when_absent():
    """A minimal pack with no capabilities block still works (optimistic default)."""
    adapter = DynamicTargetAdapter("custom", {})
    matrix = adapter.capability_matrix()
    assert all(s is ConstructSupport.DIRECT for s in matrix.values())


def test_capability_matrix_ignores_invalid_support_value():
    adapter = DynamicTargetAdapter("custom", {"capabilities": {"tools": "bogus"}})
    assert adapter.capability_matrix()[ConstructType.TOOLS] is ConstructSupport.DIRECT
