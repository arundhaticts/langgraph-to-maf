"""
LangGraph Example 04 — Tools, bind_tools, ToolNode, and the ReAct loop.

Building the tool-calling loop by hand (what create_react_agent does for you):
    - @tool wraps a function; its signature + docstring become the schema.
    - llm.bind_tools([...]) lets the model emit tool calls on the AIMessage.
    - ToolNode executes those calls and appends ToolMessages to `messages`.
    - tools_condition routes to "tools" if the last message has tool calls,
      else to END. The "model -> tools -> model" cycle IS the ReAct loop.

Requires: pip install langgraph langchain-openai
"""
from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# Tools — plain functions decorated with @tool.
# ---------------------------------------------------------------------------
@tool
def get_history(test_id: str) -> dict:
    """Return CI run stats for one test id."""
    return {"test_id": test_id, "failures": 2, "runs": 40}


@tool
def read_tests(path: str) -> list[str]:
    """List test names found at the given path."""
    return ["test_login", "test_logout"]


TOOLS = [get_history, read_tests]

# Bind the tools to the model so it can request them.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# State — MessagesState gives a `messages` key with the add_messages reducer.
# ---------------------------------------------------------------------------
class State(MessagesState):
    pass


# ---------------------------------------------------------------------------
# Model node — returns the AIMessage as a partial update (reducer appends it).
# ---------------------------------------------------------------------------
def call_model(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph assembly — the ReAct cycle.
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))   # prebuilt tool executor node

builder.add_edge(START, "model")
# tools_condition -> "tools" if the AIMessage has tool calls, else END.
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")           # loop back after tools run

graph = builder.compile()


def main():
    result = graph.invoke(
        {"messages": [("user", "How flaky is test_login? Use the history tool.")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
