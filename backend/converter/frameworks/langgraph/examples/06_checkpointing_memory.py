"""
LangGraph Example 06 — Checkpointing (durable state) and multi-turn memory.

A checkpointer saves state after every superstep, keyed by thread_id. This gives:
    - durable resume (survive process restarts with a persistent saver),
    - HITL pauses that outlive the process,
    - multi-turn conversation memory (same thread_id = same accumulated state).

RULE: When a checkpointer is attached, EVERY invoke MUST pass
      config={"configurable": {"thread_id": ...}} or LangGraph raises an error.

This example uses MemorySaver (in-process). For persistence, swap in SqliteSaver
or PostgresSaver (they are context managers / need .setup()):
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        graph = builder.compile(checkpointer=saver)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage


class State(MessagesState):
    pass


# ---------------------------------------------------------------------------
# A trivial "assistant" node. `messages` accumulates across turns because the
# checkpointer persists state under the thread_id between invokes.
# ---------------------------------------------------------------------------
def respond(state: State) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"You said: {last}")]}


builder = StateGraph(State)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

# checkpointer = per-thread durable state; store = optional cross-thread memory.
graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())


def main():
    config = {"configurable": {"thread_id": "user-42"}}

    graph.invoke({"messages": [("user", "hello")]}, config)
    result = graph.invoke({"messages": [("user", "what did I say first?")]}, config)

    # The whole conversation is present because state was checkpointed per turn.
    print("turns in thread:", len(result["messages"]))

    # Inspect / time-travel utilities:
    snapshot = graph.get_state(config)
    print("checkpoint has", len(snapshot.values["messages"]), "messages")

    # A different thread_id starts with a fresh, empty conversation.
    fresh = graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "other"}})
    print("fresh thread turns:", len(fresh["messages"]))


if __name__ == "__main__":
    main()
