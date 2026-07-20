"""
AWS Strands Example 04 — Conversation managers, agent state, and sessions.

NATIVE PATTERN (Strands) — three distinct concerns:
    1. agent.state          -> key/value scratch NOT part of the model conversation.
    2. conversation_manager -> how history is trimmed as it grows (context-window).
    3. session_manager      -> durable persistence of the whole session across restarts.

IR MAPPING:
    All three map to the node's state/memory config. Strands has NO global
    mutable graph-state dict (unlike LangGraph) — represent this as per-agent
    memory, never as reducer fields on a shared state model.
"""
from __future__ import annotations

from strands import Agent
from strands.models import BedrockModel
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from strands.session import FileSessionManager


# ---------------------------------------------------------------------------
# 1) agent.state — scratch key/value store (not sent to the model as chat).
# ---------------------------------------------------------------------------
def agent_state_example() -> None:
    agent = Agent(
        model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
        system_prompt="You are a test-optimisation assistant.",
    )
    agent.state.set("run_mode", "automated")
    agent.state.set("max_removals", 5)

    print("run_mode:", agent.state.get("run_mode"))
    agent("Proceed according to the configured run_mode.")


# ---------------------------------------------------------------------------
# 2) Conversation managers — control context-window growth.
# ---------------------------------------------------------------------------
def sliding_window_example() -> Agent:
    # Keep only the most recent 20 messages in context.
    return Agent(
        model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
        system_prompt="You are a test-optimisation assistant.",
        conversation_manager=SlidingWindowConversationManager(window_size=20),
    )


def summarizing_example() -> Agent:
    # Summarise older turns instead of dropping them, to preserve context.
    return Agent(
        model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
        system_prompt="You are a test-optimisation assistant.",
        conversation_manager=SummarizingConversationManager(
            summary_ratio=0.3,             # fraction of history to compress
            preserve_recent_messages=10,   # always keep this many verbatim
        ),
    )


# ---------------------------------------------------------------------------
# 3) Session persistence — survive process restarts.
#    Multi-turn memory: the agent naturally remembers earlier turns because it
#    carries the conversation; a session_manager makes that durable.
# ---------------------------------------------------------------------------
def session_example() -> None:
    agent = Agent(
        model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
        system_prompt="You are a test-optimisation assistant.",
        session_manager=FileSessionManager(session_id="user-123"),
        # For durable cloud storage instead:
        # from strands.session import S3SessionManager
        # session_manager=S3SessionManager(session_id="user-123", bucket="my-bucket"),
    )

    # Turn 1 and turn 2 share memory; if the process restarts, FileSessionManager
    # reloads the conversation for session_id="user-123".
    agent("Load the suite at ./uploads/suite/")
    result = agent("Which of those tests fail more than 10% of the time?")
    print(result.message)


def main() -> None:
    session_example()


if __name__ == "__main__":
    main()
