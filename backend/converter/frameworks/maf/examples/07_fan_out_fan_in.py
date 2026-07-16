"""
MAF Example 07 — Fan-out / fan-in (parallel branches).

SOURCE PATTERN (LangGraph):
    graph.add_node("dispatch", dispatch_fn)
    graph.add_node("worker_a", worker_a_fn)
    graph.add_node("worker_b", worker_b_fn)
    graph.add_node("worker_c", worker_c_fn)
    graph.add_node("aggregate", aggregate_fn)
    graph.add_conditional_edges("dispatch", lambda s: ["worker_a","worker_b","worker_c"])
    graph.add_edge("worker_a", "aggregate")
    graph.add_edge("worker_b", "aggregate")
    graph.add_edge("worker_c", "aggregate")

TARGET PATTERN (MAF):
    .add_fan_out_edges(dispatch, [worker_a, worker_b, worker_c])   # parallel
    .add_fan_in_edges([worker_a, worker_b, worker_c], aggregate)   # join
"""
from __future__ import annotations

from pydantic import BaseModel
from agent_framework import WorkflowBuilder, Executor, handler, WorkflowContext


# ---------------------------------------------------------------------------
# Message models
# ---------------------------------------------------------------------------
class Suite(BaseModel):
    tests: list[dict]


class AnalysisShard(BaseModel):
    dimension: str     # "coverage" | "redundancy" | "flakiness"
    flags: list[dict]


class AggregatedReport(BaseModel):
    coverage_flags: list[dict]
    redundancy_flags: list[dict]
    flakiness_flags: list[dict]


# ---------------------------------------------------------------------------
# Dispatch executor
# ---------------------------------------------------------------------------
class Dispatch(Executor):
    @handler
    async def run(self, msg: Suite, ctx: WorkflowContext[Suite]) -> None:
        # Fan-out is declared in the workflow builder, not in the executor.
        # The executor just forwards the same message; the builder fans it out.
        await ctx.send_message(msg)


# ---------------------------------------------------------------------------
# Worker executors (run in parallel after fan-out)
# ---------------------------------------------------------------------------
class CoverageWorker(Executor):
    @handler
    async def run(self, msg: Suite, ctx: WorkflowContext[AnalysisShard]) -> None:
        flags = _analyse_coverage(msg.tests)
        await ctx.send_message(AnalysisShard(dimension="coverage", flags=flags))


class RedundancyWorker(Executor):
    @handler
    async def run(self, msg: Suite, ctx: WorkflowContext[AnalysisShard]) -> None:
        flags = _analyse_redundancy(msg.tests)
        await ctx.send_message(AnalysisShard(dimension="redundancy", flags=flags))


class FlakinessWorker(Executor):
    @handler
    async def run(self, msg: Suite, ctx: WorkflowContext[AnalysisShard]) -> None:
        flags = _analyse_flakiness(msg.tests)
        await ctx.send_message(AnalysisShard(dimension="flakiness", flags=flags))


# ---------------------------------------------------------------------------
# Aggregate executor (runs after all workers complete — fan-in)
# ---------------------------------------------------------------------------
class Aggregate(Executor):
    @handler
    async def run(self, shards: list[AnalysisShard], ctx: WorkflowContext[None, AggregatedReport]) -> None:
        # The fan-in executor receives a list of all upstream results.
        result = AggregatedReport(
            coverage_flags=[],
            redundancy_flags=[],
            flakiness_flags=[],
        )
        for shard in shards:
            if shard.dimension == "coverage":
                result.coverage_flags = shard.flags
            elif shard.dimension == "redundancy":
                result.redundancy_flags = shard.flags
            elif shard.dimension == "flakiness":
                result.flakiness_flags = shard.flags
        await ctx.yield_output(result)


# ---------------------------------------------------------------------------
# Instantiate executors
# ---------------------------------------------------------------------------
dispatch        = Dispatch()
coverage_worker = CoverageWorker()
redundancy_worker = RedundancyWorker()
flakiness_worker  = FlakinessWorker()
aggregate       = Aggregate()


# ---------------------------------------------------------------------------
# Workflow — fan-out then fan-in
# ---------------------------------------------------------------------------
workflow = (
    WorkflowBuilder()
    .set_start_executor(dispatch)
    # Fan-out: dispatch sends the same message to all three workers in parallel.
    .add_fan_out_edges(dispatch, [coverage_worker, redundancy_worker, flakiness_worker])
    # Fan-in: aggregate waits for all three workers before proceeding.
    .add_fan_in_edges([coverage_worker, redundancy_worker, flakiness_worker], aggregate)
    .build()
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    result = await workflow.run(Suite(tests=[{"id": "test_login"}, {"id": "test_logout"}]))
    print("Coverage flags:   ", result.coverage_flags)
    print("Redundancy flags: ", result.redundancy_flags)
    print("Flakiness flags:  ", result.flakiness_flags)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _analyse_coverage(tests):   return []
def _analyse_redundancy(tests): return []
def _analyse_flakiness(tests):  return []
