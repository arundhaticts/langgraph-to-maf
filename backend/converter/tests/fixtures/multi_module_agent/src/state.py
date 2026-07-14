"""Shared state (TypedDict) with append-only reducers and Literal fields."""

from operator import add
from typing import Annotated, Literal, TypedDict


class TestOptimiserState(TypedDict, total=False):
    project_id: str
    coverage: float
    gen_retry_count: int
    optimization_goal: Literal["speed", "coverage", "reliability", "cost"]
    audit_log: Annotated[list, add]
    tool_errors: Annotated[list, add]
