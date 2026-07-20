"""
AWS Strands Example 01 — Basic agent (the core primitive).

NATIVE PATTERN (Strands):
    An Agent is a model-driven tool-calling loop. You give it a model, a
    system prompt, and (optionally) tools, then call it like a function.
    The MODEL decides control flow — there is no author-declared graph.

IR MAPPING (Strands as a conversion SOURCE):
    Agent(model=, tools=, system_prompt=)  -> ONE SINGLE_AGENT node
    the agentic loop                        -> the node's execution semantics
    model constructor                       -> node model/config
    system_prompt=                          -> node instruction

This whole file parses to a single SINGLE_AGENT node with a Bedrock model
config and the given system prompt (no tools here).
"""
from __future__ import annotations

from strands import Agent
from strands.models import BedrockModel


# ---------------------------------------------------------------------------
# Build the agent.
# `model=` is a provider instance. If omitted entirely, Strands defaults to a
# Bedrock Claude model (which requires AWS credentials + region).
# ---------------------------------------------------------------------------
agent = Agent(
    model=BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-east-1",
        temperature=0.3,
    ),
    system_prompt=(
        "You are a test-optimisation assistant. Explain your reasoning "
        "concisely and prefer actionable recommendations."
    ),
)


def main() -> None:
    # Agent instances are callable. The synchronous call runs the full agentic
    # loop and returns an AgentResult.
    result = agent("Give me three ways to reduce flaky tests in a CI suite.")

    # AgentResult carries the final assistant message plus metadata.
    print("Final message:", result.message)
    print("Stop reason:", result.stop_reason)   # e.g. "end_turn"
    # result.metrics -> token usage / latency metrics


if __name__ == "__main__":
    main()
