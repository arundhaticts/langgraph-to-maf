"""
AWS Strands Example 08 — Hooks, callbacks, observability, and HITL.

Covers:
  1. Hooks       -> typed lifecycle events (strands.hooks) via a HookProvider.
  2. callback_handler -> simple streaming callback for text/tool events.
  3. Streaming   -> agent.stream_async yields incremental events.
  4. OpenTelemetry -> Strands emits OTel traces/metrics.
  5. HITL (honest note) -> Strands has NO native interrupt; use a human-input
     tool or break the loop.

IR MAPPING:
    Hooks, callbacks, and telemetry map to the node's observability config —
    they are not structural nodes/edges. A human-input @tool parses as just
    another tool on the node, flagged as HITL.
"""
from __future__ import annotations

import logging
import time

from strands import Agent, tool
from strands.models import BedrockModel
from strands.hooks import (
    HookProvider,
    HookRegistry,
    BeforeInvocationEvent,
    AfterInvocationEvent,
)

log = logging.getLogger(__name__)
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


# ---------------------------------------------------------------------------
# 1) Hooks — register callbacks for lifecycle events.
#    A HookProvider wires its callbacks into the registry.
# ---------------------------------------------------------------------------
class TimingHooks(HookProvider):
    """Logs how long each agent invocation takes."""

    def __init__(self) -> None:
        self._start = 0.0

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before)
        registry.add_callback(AfterInvocationEvent, self._after)

    def _before(self, event: BeforeInvocationEvent) -> None:
        self._start = time.monotonic()
        log.info("[agent] invocation started")

    def _after(self, event: AfterInvocationEvent) -> None:
        elapsed = time.monotonic() - self._start
        log.info("[agent] invocation finished in %.3fs", elapsed)


# ---------------------------------------------------------------------------
# 2) callback_handler — simple streaming callback.
#    Receives kwargs for text deltas / tool events as they occur. Pass None to
#    silence the default stdout handler.
# ---------------------------------------------------------------------------
def my_callback_handler(**kwargs) -> None:
    if "data" in kwargs:                       # incremental text
        print(kwargs["data"], end="", flush=True)
    elif "current_tool_use" in kwargs:         # a tool is being invoked
        tool_use = kwargs["current_tool_use"]
        log.info("[tool] %s", tool_use.get("name"))


# ---------------------------------------------------------------------------
# 5) HITL — Strands has NO native interrupt. Implement it as a tool that
#    solicits human input; the model calls it when it needs a decision and the
#    return value flows back into the loop.
# ---------------------------------------------------------------------------
@tool
def request_human_approval(summary: str) -> str:
    """Ask a human to approve a proposed action; returns the decision.

    Args:
        summary: A short description of the action needing approval.
    """
    # Block on real human input (CLI prompt, UI callback, queue, webhook, ...).
    return _get_human_decision(summary)


# ---------------------------------------------------------------------------
# Assemble the agent with hooks + callback + the HITL tool.
# ---------------------------------------------------------------------------
def build_agent() -> Agent:
    return Agent(
        model=BedrockModel(model_id=MODEL_ID),
        system_prompt=(
            "You optimise test suites. Before deleting any test, call "
            "request_human_approval and respect the decision."
        ),
        tools=[request_human_approval],
        hooks=[TimingHooks()],
        callback_handler=my_callback_handler,
    )


# ---------------------------------------------------------------------------
# 3) Streaming invocation.
# ---------------------------------------------------------------------------
async def streaming_example() -> None:
    agent = build_agent()
    async for event in agent.stream_async("Which tests should we remove?"):
        if "data" in event:
            print(event["data"], end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 4) OpenTelemetry — enable OTel traces/metrics (configure exporters via the
#    standard OTEL_* environment variables). Do this once at startup.
# ---------------------------------------------------------------------------
def setup_telemetry() -> None:
    from strands.telemetry import StrandsTelemetry

    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter()   # OTLP endpoint via OTEL_EXPORTER_OTLP_ENDPOINT
    # telemetry.setup_console_exporter()  # for local debugging


def main() -> None:
    agent = build_agent()
    print(agent("Recommend and apply safe test removals.").message)


# ---------------------------------------------------------------------------
# Stub — replace with a real human-input mechanism.
# ---------------------------------------------------------------------------
def _get_human_decision(summary: str) -> str:
    return "approved"


if __name__ == "__main__":
    main()
