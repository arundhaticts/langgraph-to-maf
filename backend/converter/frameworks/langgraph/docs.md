# LangGraph — Framework Knowledge Pack

> **Purpose of this file.** This is the Tier‑3 documentation pack for the framework
> converter. It is injected into the LLM prompt as context when a "hard" section
> (HITL, checkpointing, conditional/loop graph wiring, tool loops, etc.) cannot be
> resolved deterministically by Tier‑1 adapters or Tier‑2 templates. LangGraph can be
> either the conversion **TARGET** (generate code from the idioms in this file) or the
> **SOURCE** (read a graph and map its constructs into the neutral IR). When this file
> conflicts with prior training memory of "LangGraph", trust this file.
>
> **API stability note.** LangGraph moves fast but the graph model is stable. The
> *concepts and mappings* here are stable; if an exact class/argument name has drifted
> in the installed `langgraph` version, prefer the installed signature but keep the same
> structure (`StateGraph`, reducers, `interrupt`/`Command`, checkpointers, `thread_id`).

---

## 0. Identity — what LangGraph is and is NOT

**LangGraph = a low-level graph runtime for stateful, multi-actor LLM applications.**
- PyPI package: `langgraph`  →  `pip install langgraph`
- Import root: `langgraph`  →  `from langgraph.graph import StateGraph, START, END`
- Built on top of LangChain Core primitives. It uses LangChain **chat models**
  (`langchain_openai.ChatOpenAI`, `langchain_anthropic.ChatAnthropic`, …),
  **messages** (`HumanMessage`, `AIMessage`, `ToolMessage`), and **tools** (`@tool`),
  but the orchestration layer is LangGraph's own.

**LangGraph is NOT just an agent SDK.** It is a **graph execution runtime** (a
Pregel-style superstep engine). The core object you build is a `StateGraph`: nodes are
plain functions that read state and return partial updates, edges (static or
conditional) define control flow, and a checkpointer gives you durable, resumable state.
A single ReAct agent (`create_react_agent`) is just a *prebuilt* graph — do not assume
LangGraph means "one agent with tools".

**LangGraph is NOT LangChain's `AgentExecutor`.** Do not emit legacy LangChain agent
constructs (`initialize_agent`, `AgentExecutor`, `LLMChain`, `RunnableAgent` glue).
Those are the deprecated predecessor. In LangGraph, agent behavior is a compiled graph.

**When LangGraph is the TARGET, never carry the source framework's constructs into the
output** (full list in §10). Examples of un-converted source that are conversion errors:
- MAF: `WorkflowBuilder`, `@executor`, `Executor`, `WorkflowContext`, `ctx.send_message`,
  `ctx.yield_output`, `RequestInfoExecutor`, `agent_framework` imports.
- CrewAI: `Crew(...)`, `Task(...)`, `Agent(role=..., goal=...)`, `Process.sequential`.
- Autogen: `AssistantAgent`, `GroupChat`, `register_reply`.

---

## 1. Choosing the orchestration primitive

LangGraph has two levels. Pick based on the **source graph shape**:

| Source shape | LangGraph primitive |
|---|---|
| One LLM agent that decides which tools to call in a loop (ReAct-style) | **`create_react_agent`** (prebuilt) |
| A multi-step graph with explicit nodes, conditional edges, cycles, HITL, or custom state | **`StateGraph`** (build it yourself) |

**Default to `StateGraph`** whenever the source has named steps, branching, loops, or a
custom state object. Only use `create_react_agent` when the source is genuinely a single
tool-calling agent with no bespoke control flow. Do NOT collapse a multi-node graph into
a single `create_react_agent` call — that silently drops the explicit edges and loops.

Both compile to the same runnable interface (`.invoke`, `.stream`, `.ainvoke`,
`.astream`) and both accept a `checkpointer`.

---

## 2. `StateGraph` — the graph you build

### 2.1 State schema

State is a **schema** (a `TypedDict`, a `dataclass`, or a Pydantic `BaseModel`). Nodes
receive the current state and return a **partial dict** of the keys they changed; the
runtime merges those updates into the state using each key's **reducer**.

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langgraph.graph.message import add_messages  # same symbol, canonical import

class State(TypedDict):
    # No reducer -> last write wins (the update replaces the value).
    topic: str
    score: float
    # With a reducer -> updates are combined. add_messages appends & dedupes messages.
    messages: Annotated[list, add_messages]
```

Reducers:
```python
import operator
from typing import Annotated

class State(TypedDict):
    audit_log: Annotated[list[dict], operator.add]   # concatenate lists
    messages:  Annotated[list, add_messages]          # message-aware append/merge
    counter:   int                                    # no reducer: replace
```

`MessagesState` is a prebuilt convenience: a `TypedDict` with a single
`messages: Annotated[list, add_messages]` field. Subclass it to add your own keys.
```python
from langgraph.graph import MessagesState

class State(MessagesState):   # inherits `messages`
    remaining_steps: int
```

Pydantic state is supported and gives runtime validation:
```python
from pydantic import BaseModel

class State(BaseModel):
    topic: str
    score: float = 0.0
```

### 2.2 Nodes

A node is `def node(state) -> dict`. It **reads** the whole state and **returns only the
keys it changed** (a partial update). The runtime applies reducers to merge the return.

```python
def analyse(state: State) -> dict:
    gaps = find_gaps(state["suite"])
    # Return ONLY changed keys. Do NOT return the whole state object.
    return {"coverage_gaps": gaps, "audit_log": [{"node": "analyse"}]}
```

Rules:
- Return a **partial dict** of changed keys, not the full mutated state and not a bare
  value. Keys not returned are left unchanged.
- For a key with a reducer (`operator.add`, `add_messages`), the returned value is
  **combined** with the existing value, not replaced — so return the *delta*
  (`{"audit_log": [one_new_entry]}`), not the whole accumulated list.
- Nodes may be `async def` — mix freely; use `.ainvoke`/`.astream` to run them.
- A node can accept a second `config` arg (`def node(state, config)`) to read
  `config["configurable"]` (e.g. `thread_id`, model params).

### 2.3 Building, compiling, running

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("analyse", analyse)
builder.add_node("report", report)

builder.add_edge(START, "intake")     # entry point
builder.add_edge("intake", "analyse")
builder.add_edge("analyse", "report")
builder.add_edge("report", END)       # terminal

graph = builder.compile()             # optionally compile(checkpointer=...)

result = graph.invoke({"suite": [...]})   # returns the final state dict
```

`START` and `END` are sentinel node names (import from `langgraph.graph`).
`add_node` may take just the function (name inferred from `__name__`) or an explicit
name + function.

### 2.4 Conditional edges (branching / routing)

`add_conditional_edges` runs a **router function** over the state and picks the next
node. The router returns a node name (or a key that a mapping dict translates to a name):

```python
def route(state: State) -> str:
    if state["validation_passed"]:
        return "approve"
    if state["retries"] >= MAX_RETRIES:
        return "drop"
    return "gap_gen"

builder.add_conditional_edges(
    "validate",
    route,
    {"approve": "approve", "drop": "drop", "gap_gen": "gap_gen"},  # optional mapping
)
```

The mapping dict is optional; if omitted, the router's returned string is used directly
as the next node name. A router may also return `END`, or a **list** of node names to
fan out to several nodes in parallel.

### 2.5 Loops / cycles (bounded retry)

A loop is just an edge that goes **back** to an earlier node, with a conditional edge
providing the exit. The exit guard is the source's iteration cap:

```python
builder.add_edge("gap_gen", "validate")
builder.add_conditional_edges(
    "validate",
    lambda s: "gap_gen" if (not s["passed"] and s["retries"] < MAX_RETRIES) else "done",
    {"gap_gen": "gap_gen", "done": END},
)
```

⚠️ Always include a termination guard (iteration cap) so the cycle cannot run forever.
Set `graph.invoke(..., {"recursion_limit": N})` as a backstop; hitting it raises
`GraphRecursionError`.

### 2.6 `Command` — update state AND route in one node

A node can return a `Command` to combine a state update with an explicit `goto`. This is
the idiomatic replacement for a node that both writes state and decides the next hop
(and the primary tool for multi-agent handoffs):

```python
from langgraph.types import Command
from typing import Literal

def supervisor(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    decision = pick_next(state)
    return Command(
        update={"audit_log": [{"node": "supervisor", "decision": decision}]},
        goto=decision,   # a node name, or END
    )
```

When a node returns a `Command` with `goto`, you do **not** need static edges out of it.

### 2.7 Fan-out / fan-in (parallel branches)

Add multiple edges out of one node to run branches in parallel; their state updates merge
via reducers at the join. `Send` dispatches dynamic parallel work (map-reduce):

```python
from langgraph.types import Send

def dispatch(state: State):
    return [Send("worker", {"item": x}) for x in state["items"]]

builder.add_conditional_edges("dispatch", dispatch, ["worker"])
```

---

## 3. Tools

A tool is a function wrapped by `@tool`; its signature + docstring become the schema.

```python
from langchain_core.tools import tool

@tool
def get_history(test_id: str, path: str | None = None) -> dict | None:
    """Look up CI evidence for a single test."""
    ...
```

**Binding tools to a model** (so the LLM can request tool calls):
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([get_history, read_tests])
```

**`ToolNode`** is a prebuilt node that executes the tool calls found on the last
`AIMessage` and appends `ToolMessage`s to `messages`:
```python
from langgraph.prebuilt import ToolNode, tools_condition

builder.add_node("tools", ToolNode([get_history, read_tests]))
builder.add_node("model", call_model)          # your node calling llm_with_tools
builder.add_conditional_edges("model", tools_condition)  # -> "tools" or END
builder.add_edge("tools", "model")             # loop back after tools run
```

`tools_condition` is a prebuilt router: routes to `"tools"` if the last message has tool
calls, else to `END`. This "model → tools → model" cycle IS the ReAct loop.

---

## 4. State reducers (the append pattern)

For any key that should **accumulate** rather than be overwritten, annotate it with a
reducer. This is the LangGraph-native way; do NOT hand-roll list concatenation in nodes.

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:   Annotated[list, add_messages]      # chat history
    audit_log:  Annotated[list[dict], operator.add]  # append entries
    errors:     Annotated[list[dict], operator.add]

def node(state: State) -> dict:
    # Return only the DELTA. The reducer concatenates it onto the existing list.
    return {"audit_log": [{"node": "node", "event": "done"}]}
```

- `add_messages`: appends messages, and updates existing messages by `id` (used for the
  `messages` key almost always).
- `operator.add`: list/number concatenation/addition.
- No reducer: last-write-wins (the returned value replaces the stored value).

---

## 5. Human-in-the-loop (HITL) — HARD SECTION

Modern LangGraph HITL uses **`interrupt()`** inside a node to pause the graph and surface
a payload to the caller, then **resume with `Command(resume=...)`**. This requires a
**checkpointer** and a `thread_id` (the interrupt is persisted).

```python
from langgraph.types import interrupt, Command

def approve_removals(state: State) -> dict:
    if state["run_mode"] == "automated":
        # Automated fast-path: accept the recommendation, no human, no pause.
        return {"approved": state["recommended"]}

    # Interactive path: pause and surface the payload to the caller.
    human = interrupt({
        "question": "Approve these test removals?",
        "candidates": state["candidates"],
        "recommended": state["recommended"],
    })
    # Execution resumes HERE when the caller sends Command(resume=...).
    approved = human if human is not None else state["recommended"]
    return {"approved": approved}
```

Driving the pause/resume from the caller (checkpointer + thread_id required):
```python
config = {"configurable": {"thread_id": "run-123"}}

result = graph.invoke({"run_mode": "interactive", ...}, config)

# If the graph interrupted, result contains an "__interrupt__" payload.
if "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    decision = present_to_human(payload)          # your UI/API
    # Resume: the resume value becomes the return value of interrupt(...).
    result = graph.invoke(Command(resume=decision), config)
```

**Static interrupts** are the coarser alternative — pause *before* or *after* a node
without an `interrupt()` call in code:
```python
graph = builder.compile(checkpointer=saver, interrupt_before=["approve"])
# ... invoke pauses before "approve"; resume with graph.invoke(None, config)
# optionally edit state first: graph.update_state(config, {"approved": [...]})
```

**Conversion guidance:**
- Source `interrupt(payload)` → LangGraph is native `interrupt(payload)`; resume with
  `Command(resume=value)`. Keep the checkpointer + `thread_id`.
- MAF `RequestInfoExecutor`/`RequestInfoMessage` → a node that calls `interrupt(payload)`.
- Always keep an `automated` fast-path that returns the recommendation without pausing.
- **Never** implement HITL as a hard-coded auto-approve with the real logic in comments.
  Auto-approve is only acceptable as the explicit `automated` branch.

---

## 6. Checkpointing & persistence (durability / memory) — HARD SECTION

A **checkpointer** saves state after every superstep, keyed by `thread_id`. This gives
you durable resume, HITL pauses that outlive the process, and multi-turn memory.

```python
from langgraph.checkpoint.memory import MemorySaver      # in-process, ephemeral
graph = builder.compile(checkpointer=MemorySaver())

# Persistent options:
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.checkpoint.postgres import PostgresSaver

config = {"configurable": {"thread_id": "user-42"}}
graph.invoke({"messages": [("user", "hi")]}, config)
graph.invoke({"messages": [("user", "what did I just say?")]}, config)  # remembers
```

- Every stateful invocation MUST pass `config={"configurable": {"thread_id": ...}}` when
  a checkpointer is attached — without it, LangGraph raises an error.
- Inspect/rewind: `graph.get_state(config)`, `graph.get_state_history(config)`, and
  `graph.update_state(config, values)` for time-travel / manual edits.
- `SqliteSaver`/`PostgresSaver` are context managers (`with SqliteSaver.from_conn_string(
  ...) as saver:`) or use their `.setup()`; async variants `AsyncSqliteSaver`,
  `AsyncPostgresSaver` exist.

**Long-term memory (across threads)** uses a `Store`, not the checkpointer:
```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
graph = builder.compile(checkpointer=MemorySaver(), store=store)
# In a node: store.put((namespace,), key, value); store.get((namespace,), key)
```

---

## 7. Invocation

```python
graph.invoke(state, config)                 # run to completion, return final state
async_result = await graph.ainvoke(state, config)

for chunk in graph.stream(state, config, stream_mode="updates"):
    ...                                     # per-node state deltas
async for chunk in graph.astream(state, config, stream_mode="messages"):
    ...                                     # token/message stream
```

`stream_mode` options:
- `"values"` — full state after each step.
- `"updates"` — only the delta each node produced (most common for debugging flow).
- `"messages"` — LLM tokens + messages (for chat UIs).
- `"custom"` — data emitted via `get_stream_writer()`.
- Pass a list to combine modes.

---

## 8. Providers (LangChain chat models)

LangGraph is model-agnostic; you inject a LangChain chat model.

```python
from langchain_openai import ChatOpenAI          # pip install langchain-openai
llm = ChatOpenAI(model="gpt-4o", temperature=0)

from langchain_anthropic import ChatAnthropic     # pip install langchain-anthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Provider-agnostic factory (picks the integration from the string):
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4o")            # or "anthropic:claude-3-5-sonnet-latest"
```

API keys come from env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). Swap providers by
swapping the constructor; the graph itself is unchanged. `.bind_tools([...])` works on
any tool-calling chat model.

---

## 9. Concept ↔ LangGraph mapping table (bidirectional)

| Neutral IR / typical source concept | LangGraph |
|---|---|
| graph / workflow object | `StateGraph(State)` compiled to a runnable |
| node / step / executor | `def node(state) -> dict` added via `add_node` |
| typed graph state | `TypedDict` / Pydantic `BaseModel` / `dataclass` state schema |
| accumulate/append field | `Annotated[list, operator.add]` or `Annotated[list, add_messages]` |
| unconditional edge | `builder.add_edge("a", "b")` |
| conditional edge / router | `builder.add_conditional_edges("a", router_fn, mapping)` |
| entry / terminal | `add_edge(START, ...)` / `add_edge(..., END)` |
| update-state-and-route | node returns `Command(update=..., goto=...)` |
| cycle / bounded retry | back-edge + conditional edge with iteration-cap guard |
| dynamic parallel / map-reduce | `Send(node, substate)` from a conditional edge |
| tool function | `@tool` (`from langchain_core.tools import tool`) |
| tool execution node | `ToolNode([...])` + `tools_condition` |
| single ReAct agent + tools | `create_react_agent(model, tools)` (prebuilt) |
| human-in-the-loop pause | `interrupt(payload)` + `Command(resume=value)` |
| durable state / memory | checkpointer (`MemorySaver`/`SqliteSaver`) + `thread_id` |
| long-term cross-thread memory | `Store` (`InMemoryStore`, `PostgresStore`) |
| run graph | `graph.invoke(state, config)` / `.stream` / `.ainvoke` / `.astream` |
| chat model / provider | LangChain chat model (`ChatOpenAI`, `ChatAnthropic`, `init_chat_model`) |

---

## 10. Reject-list — output containing any of these is WRONG (when LangGraph is TARGET)

- MAF carry-over: `from agent_framework`, `WorkflowBuilder`, `@executor`, `Executor`,
  `WorkflowContext`, `ctx.send_message`, `ctx.yield_output`, `RequestInfoExecutor`,
  `RequestInfoMessage`, `.build()` on a workflow.
- CrewAI carry-over: `from crewai`, `Crew(`, `Task(`, `Agent(role=`, `Process.sequential`.
- Autogen carry-over: `AssistantAgent`, `GroupChat`, `register_reply`.
- Legacy LangChain: `initialize_agent`, `AgentExecutor`, `LLMChain`, `ConversationChain`.
- A node returning the **whole mutated state object** or a bare value instead of a
  **partial dict** of changed keys (breaks the reducer/merge model). Exception: nodes
  legitimately returning a `Command`.
- Returning the **full accumulated list** for a reducer key instead of just the delta
  (double-appends every step).
- A router function defined but never wired into `add_conditional_edges` (dead code).
- Collapsing a multi-node source graph into a single `create_react_agent` call, dropping
  its conditional edges / loops.
- HITL as `interrupt(...)` **without** a checkpointer + `thread_id` (it will not resume),
  or HITL as hard-coded auto-approve with the real logic commented out.
- `graph.compile(checkpointer=...)` then invoking **without**
  `config={"configurable": {"thread_id": ...}}`.
- A cycle with **no** termination guard / iteration cap.
- `requirements.txt` listing stdlib modules (`json`, `os`, `io`, `operator`, `typing`, …)
  as packages, or listing `langchain` when only `langgraph` + one provider integration is
  needed.

---

## 11. Minimal end-to-end skeleton (reference the converter can pattern-match)

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import operator


class State(TypedDict):
    suite: list[dict]
    coverage_gaps: list[dict]
    audit_log: Annotated[list[dict], operator.add]   # append reducer
    messages: Annotated[list, add_messages]


def intake(state: State) -> dict:
    return {"suite": [normalise(t) for t in state["suite"]],
            "audit_log": [{"node": "intake"}]}

def analyse(state: State) -> dict:
    gaps = find_gaps(state["suite"])
    return {"coverage_gaps": gaps, "audit_log": [{"node": "analyse"}]}

def report(state: State) -> dict:
    return {"audit_log": [{"node": "report", "gaps": len(state["coverage_gaps"])}]}


builder = StateGraph(State)
builder.add_node("intake", intake)
builder.add_node("analyse", analyse)
builder.add_node("report", report)
builder.add_edge(START, "intake")
builder.add_edge("intake", "analyse")
builder.add_edge("analyse", "report")
builder.add_edge("report", END)

graph = builder.compile()


def main():
    final = graph.invoke({"suite": [{"id": "test_login"}], "coverage_gaps": [],
                          "audit_log": [], "messages": []})
    print(final["audit_log"])


def normalise(t): return {**t, "entities": []}
def find_gaps(suite): return []
```
