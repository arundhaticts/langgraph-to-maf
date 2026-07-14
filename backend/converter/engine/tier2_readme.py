"""Tier 2 -- README-assisted classification.

When Tier 1's structural rules cannot pin down an orchestration shape, we fall
back to keyword matching on the verbatim README workflow prose (Section 11,
Module 6). This is deterministic and cheap -- it runs before any LLM call.
"""

from __future__ import annotations

from typing import Optional

from converter.contracts import OrchestrationPattern

# Ordered most-specific first: loop cues beat branch cues beat linear cues.
_KEYWORDS: list[tuple[OrchestrationPattern, tuple[str, ...]]] = [
    (
        OrchestrationPattern.LOOP,
        ("retries", "retry", "loops back", "loop back", "up to", "times", "until"),
    ),
    (
        OrchestrationPattern.BRANCH,
        ("if ", "depending on", "when ", "otherwise", "branch", "else"),
    ),
    (
        OrchestrationPattern.LINEAR,
        ("then", "next", "followed by", "after that", "finally"),
    ),
]


def classify_from_readme(workflow_text: Optional[str]) -> Optional[OrchestrationPattern]:
    """Best-effort pattern from workflow prose, or None if nothing matches."""
    if not workflow_text:
        return None
    lowered = workflow_text.lower()
    for pattern, keywords in _KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return pattern
    return None
