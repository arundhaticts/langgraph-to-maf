"""
MAF Example 08 — Middleware and observability.

Covers:
- Wrapping agent runs with middleware (logging, retry, guardrails)
- OpenTelemetry tracing (MAF's built-in telemetry)
- Audit logging inside executors (replacing LangGraph's append-reducer audit_log)

SOURCE PATTERN (LangGraph):
    # Logging typically done inside node functions or via LangSmith callbacks.
    # No built-in middleware concept.

TARGET PATTERN (MAF):
    - Middleware: registered on the agent or run config; wraps every tool call / agent run.
    - Telemetry: enable MAF's OTel integration; traces flow automatically.
    - Per-executor audit: append to shared state or carry in the message model.
"""
from __future__ import annotations

import time
import logging
from pydantic import BaseModel, Field
from agent_framework import (
    ChatAgent, WorkflowBuilder,
    executor, WorkflowContext,
    Middleware, MiddlewareContext,
)
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.telemetry import configure_telemetry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Telemetry — enable OTel tracing (call once at startup)
# ---------------------------------------------------------------------------
def setup_telemetry():
    """
    Enable MAF's built-in OpenTelemetry integration.
    Traces every agent run, tool call, and executor hop automatically.
    Configure the OTel exporter via standard OTEL_* environment variables.
    """
    configure_telemetry(
        service_name="test-optimiser-agent",
        # exporter="otlp",          # default; configure via OTEL_EXPORTER_OTLP_ENDPOINT
        # trace_content=True,       # include prompt/response content in spans (disable in prod)
    )


# ---------------------------------------------------------------------------
# 2. Middleware — wraps agent/tool calls
# ---------------------------------------------------------------------------
class LoggingMiddleware(Middleware):
    """Logs every tool call with duration and outcome."""

    async def on_tool_call(self, ctx: MiddlewareContext, call_next):
        tool_name = ctx.tool_name
        start = time.monotonic()
        log.info("[tool] %s started", tool_name)
        try:
            result = await call_next()
            elapsed = time.monotonic() - start
            log.info("[tool] %s ok (%.3fs)", tool_name, elapsed)
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            log.error("[tool] %s failed after %.3fs: %s", tool_name, elapsed, exc)
            raise


class RetryMiddleware(Middleware):
    """Retries transient tool failures up to max_retries times."""

    def __init__(self, max_retries: int = 3, backoff: float = 2.0):
        self.max_retries = max_retries
        self.backoff = backoff

    async def on_tool_call(self, ctx: MiddlewareContext, call_next):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await call_next()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.backoff ** (attempt - 1)
                    log.warning("[retry] attempt %d/%d failed; retrying in %.1fs: %s",
                                attempt, self.max_retries, wait, exc)
                    await _async_sleep(wait)
        raise last_exc


class GuardrailMiddleware(Middleware):
    """Blocks tool calls that exceed a token budget (illustrative guardrail)."""

    def __init__(self, max_input_tokens: int = 8000):
        self.max_input_tokens = max_input_tokens

    async def on_tool_call(self, ctx: MiddlewareContext, call_next):
        tokens = _estimate_tokens(ctx.input)
        if tokens > self.max_input_tokens:
            raise ValueError(
                f"Tool '{ctx.tool_name}' input too large: {tokens} tokens "
                f"(limit {self.max_input_tokens})"
            )
        return await call_next()


# ---------------------------------------------------------------------------
# 3. Register middleware on the agent
# ---------------------------------------------------------------------------
def build_agent_with_middleware() -> ChatAgent:
    return ChatAgent(
        chat_client=AzureOpenAIChatClient(),
        instructions="You are a test-optimisation assistant.",
        tools=[],
        middleware=[
            LoggingMiddleware(),
            RetryMiddleware(max_retries=3),
            GuardrailMiddleware(max_input_tokens=8000),
        ],
    )


# ---------------------------------------------------------------------------
# 4. Per-executor audit logging (inside a workflow)
# ---------------------------------------------------------------------------
class PipelineState(BaseModel):
    tests: list[dict] = Field(default_factory=list)
    audit_log: list[dict] = Field(default_factory=list)    # was Annotated[list, add]
    tool_errors: list[dict] = Field(default_factory=list)  # was Annotated[list, add]


@executor(id="analyse")
async def analyse(msg: PipelineState, ctx: WorkflowContext[PipelineState]) -> None:
    start = time.monotonic()
    try:
        result = _run_analysis(msg.tests)
        elapsed = time.monotonic() - start
        msg.audit_log.append({
            "node": "analyse",
            "event": "ok",
            "elapsed_s": round(elapsed, 3),
            "count": len(result),
        })
    except Exception as exc:
        elapsed = time.monotonic() - start
        msg.tool_errors.append({
            "source": "analyse",
            "error": str(exc),
            "action": "skipped — deterministic fallback used",
        })
        msg.audit_log.append({
            "node": "analyse",
            "event": "error",
            "elapsed_s": round(elapsed, 3),
            "error": str(exc),
        })

    await ctx.send_message(msg)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _run_analysis(tests): return []

def _estimate_tokens(obj) -> int:
    return len(str(obj)) // 4

async def _async_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)
