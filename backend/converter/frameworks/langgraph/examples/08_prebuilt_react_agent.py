"""
LangGraph Example 08 — Prebuilt ReAct agent (create_react_agent).

When the source is genuinely ONE tool-calling agent with no bespoke control
flow, use the prebuilt create_react_agent instead of hand-wiring the
model/tools cycle (Example 04). It returns a compiled graph with the same
.invoke / .stream interface and accepts a checkpointer for memory.

Do NOT use this to collapse a multi-node source graph (named steps, branching,
loops) into a single agent — that silently drops the explicit control flow.
Use StateGraph for those.

Requires: pip install langgraph langchain-openai
"""
from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_history(test_id: str) -> dict:
    """Return CI run stats for one test id."""
    return {"test_id": test_id, "failures": 2, "runs": 40}


@tool
def quarantine(test_id: str) -> str:
    """Quarantine a flaky test by id. Reversible."""
    return f"{test_id} quarantined"


# ---------------------------------------------------------------------------
# Model — any LangChain tool-calling chat model. Swap the constructor to change
# provider (e.g. ChatAnthropic, or init_chat_model("anthropic:...")).
# ---------------------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Prebuilt agent = compiled graph. `prompt` sets the system instructions.
# A checkpointer gives multi-turn memory keyed by thread_id.
# ---------------------------------------------------------------------------
agent = create_react_agent(
    model,
    tools=[get_history, quarantine],
    prompt="You are a test-optimisation assistant. Use tools before deciding.",
    checkpointer=MemorySaver(),
)


def main():
    config = {"configurable": {"thread_id": "session-1"}}
    result = agent.invoke(
        {"messages": [("user", "Is test_login flaky? If so, quarantine it.")]},
        config,
    )
    print(result["messages"][-1].content)

    # Streaming per-node updates:
    for chunk in agent.stream(
        {"messages": [("user", "What did you just do?")]}, config, stream_mode="updates"
    ):
        print(chunk)


if __name__ == "__main__":
    main()
