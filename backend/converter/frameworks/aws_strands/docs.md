# AWS Strands (strands-agents SDK) — Source Framework Knowledge Pack

> **Purpose of this file.** This is the Tier-3 documentation pack for the framework
> converter. AWS Strands is a conversion **SOURCE only** in this converter — there is no
> Strands *target* generator yet. This file therefore serves two jobs: (1) an
> authoritative **native reference** for the real `strands-agents` SDK, and (2) a
> description of **how Strands constructs map to the neutral IR** when a Strands program
> is parsed as a source. When this file conflicts with prior training memory of
> "Strands", trust this file.
>
> **API stability note.** The `strands-agents` SDK is young and evolving. The *concepts
> and mappings* here are stable; if an exact class/argument name has drifted in the
> installed version, prefer the installed signature but keep the same structure. Where a
> precise signature is uncertain it is flagged inline rather than invented.

---

## 0. Identity — what Strands is and is NOT

**Strands = AWS Strands Agents SDK.**
- Core PyPI package: `strands-agents`  →  `pip install strands-agents`
- Prebuilt tools package: `strands-agents-tools`  →  `pip install strands-agents-tools`
- Import root: `strands`  →  `from strands import Agent, tool`
- Model providers live under `strands.models` (`from strands.models import BedrockModel`).
- Prebuilt tools import from `strands_tools` (note the underscore) after installing the
  `strands-agents-tools` distribution — e.g. `from strands_tools import calculator`.

**What Strands IS.** A **model-driven, single-agent tool-calling loop**. You give one
`Agent` a model, a system prompt, and a list of tools. On each invocation the agent runs
an *agentic loop*: it calls the model, the model may request tool calls, Strands executes
those tools, feeds the results back to the model, and repeats until the model returns a
final answer. The **model decides** control flow — there is no author-defined graph of
steps.

**What Strands is NOT.**
- **NOT an explicit graph runtime.** There is no `StateGraph`, no author-declared nodes
  and edges, no `add_conditional_edges`, no compiled DAG for the core agent. (Strands
  *does* offer opt-in multi-agent orchestration — `Swarm`, `Graph`, `workflow` — see §8,
  but the default unit of work is one agent.)
- **NOT LangGraph and NOT Microsoft Agent Framework.** Do not confuse `strands.Agent`
  with `agent_framework.ChatAgent` or a LangGraph node.
- The default model provider is **Amazon Bedrock**, not OpenAI. A bare `Agent()` with no
  `model=` uses a Bedrock Claude model and requires AWS credentials.

---

## 1. The core primitive — `Agent`

Strands has **one** central primitive: `Agent`. Pick the shape from the source:

| Source shape | Strands construct | Neutral IR |
|---|---|---|
| One LLM that loops over tool calls (ReAct-style) | `Agent(model=, tools=, system_prompt=)` | a single `SINGLE_AGENT` node |
| Several agents cooperating (handoff / parallel / routed) | `Swarm`, `Graph`, or `workflow` | multiple nodes + orchestration notes |
| A sub-agent invoked by another agent | agent wrapped as a `@tool` | tool edge / note |

```python
from strands import Agent
from strands.models import BedrockModel

agent = Agent(
    model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
    system_prompt="You are a test-optimisation assistant.",
    tools=[get_history, read_tests],   # @tool functions — see §4
)

result = agent("Analyse this suite.")   # synchronous call -> AgentResult
print(result.message)                    # final assistant message
```

`Agent(...)` is callable. The most important arguments:
- `model=` — a model-provider instance (default: a `BedrockModel`). See §3.
- `system_prompt=` — the system instruction string.
- `tools=` — a list of `@tool` functions, Python modules, prebuilt `strands_tools`, or
  MCP tool objects. See §4.
- `messages=` — optional seed conversation history.
- `conversation_manager=` — controls context-window trimming/summarisation. See §5.
- `session_manager=` — durable session persistence. See §5 and §7.
- `hooks=` / `callback_handler=` — observability and lifecycle hooks. See §9.

---

## 2. The agentic loop

The behaviour that makes Strands "model-driven": each call runs an internal loop.

1. Strands sends `system_prompt` + conversation history + the new user input +
   tool schemas to the model.
2. The model responds either with a **final answer** (loop ends) or with one or more
   **tool-use requests**.
3. Strands executes each requested tool and appends the tool results to the conversation.
4. Go to step 1. The loop repeats until the model produces a final answer (or a stop
   condition / max-iteration guard fires).

There is **no author-written routing** between steps — the model chooses which tool to
call next and when to stop. This is the key difference from a LangGraph/MAF workflow,
where the author declares the edges. In the neutral IR this whole loop collapses to a
**single `SINGLE_AGENT` node**; the tools become that node's tool set.

---

## 3. Model providers (`strands.models`)

The model is injected — swap the provider by swapping the constructor. **Default is
Bedrock.**

```python
from strands.models import BedrockModel
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    temperature=0.3,
)
```

```python
# Anthropic API directly (pip install 'strands-agents[anthropic]')
from strands.models.anthropic import AnthropicModel
model = AnthropicModel(model_id="claude-sonnet-4-20250514", max_tokens=1024)
```

```python
# OpenAI (pip install 'strands-agents[openai]')
from strands.models.openai import OpenAIModel
model = OpenAIModel(model_id="gpt-4o", client_args={"api_key": "..."})
```

```python
# LiteLLM — one interface to 100+ providers (pip install 'strands-agents[litellm]')
from strands.models.litellm import LiteLLMModel
model = LiteLLMModel(model_id="gemini/gemini-1.5-pro")
```

```python
# Ollama — local models (pip install 'strands-agents[ollama]')
from strands.models.ollama import OllamaModel
model = OllamaModel(host="http://localhost:11434", model_id="llama3")
```

| Provider | Import | Notes |
|---|---|---|
| Amazon Bedrock (default) | `from strands.models import BedrockModel` | uses AWS credentials + region |
| Anthropic | `from strands.models.anthropic import AnthropicModel` | needs `ANTHROPIC_API_KEY` |
| OpenAI | `from strands.models.openai import OpenAIModel` | needs `OPENAI_API_KEY` |
| LiteLLM | `from strands.models.litellm import LiteLLMModel` | proxy to many providers |
| Ollama | `from strands.models.ollama import OllamaModel` | local inference |

In the neutral IR the model constructor maps to the node's **model/config** (provider,
model_id, temperature, etc.), not to a node of its own.

---

## 4. Tools

A Strands tool is what the agent may call inside the loop. Four ways to supply tools:

### 4.1 The `@tool` decorator (Python functions)

```python
from strands import tool

@tool
def get_history(test_id: str, path: str | None = None) -> dict:
    """Return CI run stats for one test.

    Args:
        test_id: The unique test identifier.
        path: Optional path to a CI-history JSON file.
    """
    ...
```

Strands introspects the **type hints and docstring** to build the tool schema — the
docstring description and `Args:` map to the tool/parameter descriptions. Register a tool
by passing the function object in `tools=[...]`.

### 4.2 Prebuilt tools (`strands_tools`)

Install `strands-agents-tools`, then import ready-made tools:

```python
from strands_tools import calculator, file_read, http_request

agent = Agent(tools=[calculator, file_read, http_request])
```

Common prebuilt tools: `calculator`, `file_read`, `file_write`, `http_request`,
`python_repl`, `shell`, `current_time`, `use_aws`, and more. (Some, like `shell` and
`python_repl`, execute arbitrary code — treat as sensitive.)

### 4.3 Python-module tools

A module that exposes `TOOL_SPEC` and a matching function can be passed directly (by
module object or dotted path), letting you author tools without the decorator.

### 4.4 MCP tools

Strands can consume tools from a Model Context Protocol server via `MCPClient`. The MCP
client is used as a context manager; `client.list_tools_sync()` returns tool objects to
pass into `tools=[...]`.

```python
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient

mcp = MCPClient(lambda: stdio_client(StdioServerParameters(command="uvx", args=["some-mcp-server"])))
with mcp:
    agent = Agent(tools=mcp.list_tools_sync())
    agent("Use the MCP tools to ...")
```

**IR mapping.** Every `@tool` function, every prebuilt tool, and every MCP tool becomes a
**tool on the single agent node**. Do not model tools as separate graph nodes.

---

## 5. State & memory

Strands separates three concerns: per-agent scratch state, conversation-window
management, and durable session persistence.

### 5.1 Agent state (`agent.state`)

`agent.state` is a key/value store carried with the agent for data that is *not* part of
the model conversation (counters, flags, cached artefacts):

```python
agent.state.set("run_mode", "automated")
mode = agent.state.get("run_mode")
```

There is also **request state** available inside tools/hooks for values scoped to a single
invocation.

### 5.2 Conversation managers (context-window control)

The `conversation_manager=` argument controls how history is trimmed as it grows:

```python
from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

agent = Agent(
    conversation_manager=SlidingWindowConversationManager(window_size=20),
)
```

- `SlidingWindowConversationManager(window_size=N)` — keeps the last N messages.
- `SummarizingConversationManager(...)` — summarises older turns to preserve context
  while staying under the window.
- `NullConversationManager` — no trimming (default behaviour is a sliding window).

### 5.3 Session persistence

Session managers persist the whole agent session (messages + state) so a conversation
survives process restarts:

```python
from strands.session import FileSessionManager, S3SessionManager

agent = Agent(session_manager=FileSessionManager(session_id="user-123"))
# or
agent = Agent(session_manager=S3SessionManager(session_id="user-123", bucket="my-bucket"))
```

**IR mapping.** `agent.state` and conversation/session managers map to the node's
**state/memory config**. There is no global mutable graph-state dict as in LangGraph —
represent it as per-agent memory, not as reducer fields on a shared state model.

---

## 6. Structured output

Strands can coerce the final answer into a Pydantic model instead of free text:

```python
from pydantic import BaseModel
from strands import Agent

class SuiteReport(BaseModel):
    score: float
    flaky_tests: list[str]

agent = Agent(system_prompt="Analyse the suite.")
report: SuiteReport = agent.structured_output(
    SuiteReport,
    "Summarise the quality of the suite at ./uploads/suite/",
)
print(report.score, report.flaky_tests)
```

`agent.structured_output(Model, prompt)` runs the loop and returns a validated instance of
`Model`. An async variant `structured_output_async(...)` also exists. In the IR this is a
property of the node (its output is a typed schema), not a separate step.

---

## 7. Invocation

```python
# 1) Synchronous, callable form — most common
result = agent("Analyse this suite.")
print(result.message)          # AgentResult: final message, .stop_reason, .metrics, etc.

# 2) Async
result = await agent.invoke_async("Analyse this suite.")

# 3) Streaming (async generator of events: text deltas, tool calls, lifecycle)
async for event in agent.stream_async("Analyse this suite."):
    if "data" in event:
        print(event["data"], end="", flush=True)
```

- `agent("...")` and `agent.invoke_async("...")` return an `AgentResult` (final message
  plus metadata such as `stop_reason` and token `metrics`).
- `agent.stream_async("...")` yields incremental events (text chunks, tool-use events,
  lifecycle markers).

**IR mapping.** All three are the *same* single-agent invocation of the one node; the
sync/async/streaming distinction is an invocation-style note, not a structural difference.

---

## 8. Multi-agent patterns

Strands supports composing multiple agents. These are the only cases that map to
**more than one node** in the IR.

### 8.1 Agents-as-tools (hierarchical)

Wrap a specialist agent in a `@tool` and give it to an orchestrator agent. The
orchestrator's model decides when to delegate.

```python
from strands import Agent, tool

@tool
def research_test_quality(query: str) -> str:
    """Delegate CI-history research to a specialist agent."""
    specialist = Agent(system_prompt="You retrieve CI history and summarise test quality.",
                       tools=[get_history])
    return str(specialist(query).message)

orchestrator = Agent(system_prompt="You optimise test suites.",
                     tools=[research_test_quality])
```

### 8.2 `Swarm` — self-organising handoff

Agents hand off to each other via a shared `handoff_to_agent` tool; the swarm manages
turns until completion.

```python
from strands import Agent
from strands.multiagent import Swarm

researcher = Agent(name="researcher", system_prompt="...")
writer = Agent(name="writer", system_prompt="...")

swarm = Swarm([researcher, writer], max_handoffs=10, max_iterations=20)
result = swarm("Investigate and report on the test suite.")
```

### 8.3 `Graph` / `GraphBuilder` — explicit multi-agent DAG

The one place Strands has an author-declared graph — of **agents**, not of low-level
steps:

```python
from strands import Agent
from strands.multiagent import GraphBuilder

builder = GraphBuilder()
builder.add_node(researcher, "research")
builder.add_node(writer, "write")
builder.add_edge("research", "write")
builder.set_entry_point("research")
graph = builder.build()
result = graph("Investigate and report on the test suite.")
```

Edges may carry conditions for conditional routing between agents.

### 8.4 `workflow`

A `workflow` tool/utility sequences agent tasks with dependencies. Treat as another
multi-agent orchestration form.

**IR mapping.** Each participating `Agent` becomes its own node. Swarm handoffs and Graph
edges become **orchestration notes / edges between agent nodes**. A single agent that only
uses `@tool` functions is still one node — do not inflate its tools into extra nodes.

---

## 9. Human-in-the-loop (HITL) — HARD SECTION (honest note)

**Strands has NO native interrupt/resume primitive** like LangGraph's `interrupt()` or
MAF's `RequestInfoExecutor`. The agentic loop does not pause itself for a human. There is
no built-in checkpoint-and-await-human mechanism in the core agent.

Two honest, idiomatic ways to get HITL:

1. **Human-input tool.** Define a `@tool` that solicits human input (prompt the user,
   read from a queue/UI, wait for a webhook). The model calls this tool when it needs a
   human decision, and its return value flows back into the loop like any other tool
   result. `strands_tools` also ships a `handoff_to_user` tool for this purpose.

   ```python
   from strands import tool

   @tool
   def request_human_approval(summary: str) -> str:
       """Ask a human to approve an action. Returns the human's decision."""
       # Block on real human input (CLI prompt, UI callback, queue, webhook...).
       return get_human_decision(summary)   # e.g. "approved" / "rejected: ..."
   ```

2. **Break the loop.** Stop the agent at a point where a human must act (e.g. after a
   `structured_output` that flags "needs approval"), do the human step in your own host
   code, then start a new invocation (optionally with a `session_manager` so the
   conversation persists across the pause).

**Conversion guidance.** When a source used a real interrupt primitive and the target were
Strands, you would represent HITL as a human-input tool **or** an explicit loop break —
and you must say so honestly. Do **not** claim Strands natively pauses/resumes. Do **not**
fabricate a `strands.interrupt(...)` call. When Strands is the *source*, a human-input
tool parses as just another tool on the agent node, with a note that it implements HITL.

---

## 10. Observability & hooks

- **Hooks.** Strands exposes a typed hook system (`strands.hooks`) for lifecycle events —
  e.g. `BeforeInvocationEvent`, `AfterInvocationEvent`, and tool-invocation events. You
  register a `HookProvider` (or add callbacks) to run logging, guardrails, metrics, or
  retries around agent/tool execution. Pass providers via `Agent(hooks=[...])`.
- **`callback_handler`.** A simpler streaming-callback mechanism: pass
  `callback_handler=` to receive text/tool events as they occur (there is a default
  handler that prints to stdout; pass `None` to silence it).
- **OpenTelemetry.** Strands emits OTel traces/metrics for agent runs, model calls, and
  tool calls; configure via `StrandsTelemetry` / standard `OTEL_*` environment variables
  rather than hand-rolling logging.

**IR mapping.** Hooks, callbacks, and telemetry map to the node's **observability config**;
they are not structural nodes/edges.

---

## 11. Strands → neutral IR mapping table (apply mechanically)

| Strands construct (source) | Neutral IR |
|---|---|
| `Agent(model=, tools=, system_prompt=)` | one `SINGLE_AGENT` node |
| the agentic loop | the node's execution semantics (model-driven, no explicit edges) |
| `@tool` function | a tool on the agent node |
| prebuilt `strands_tools.*` / module tool / MCP tool | a tool on the agent node |
| model constructor (`BedrockModel(...)`, etc.) | node model/config (provider, model_id, params) |
| `system_prompt=` | node instruction |
| `agent.state`, conversation/session managers | node state/memory config |
| `agent.structured_output(Model, ...)` | node typed-output schema |
| `agent("...")` / `invoke_async` / `stream_async` | node invocation style (sync/async/stream) |
| agent-as-`@tool` | tool edge / delegation note between agent nodes |
| `Swarm([...])` | multiple agent nodes + handoff orchestration notes |
| `GraphBuilder` / `Graph` | multiple agent nodes + edges (multi-agent DAG) |
| `workflow` | multiple agent nodes + task-dependency notes |
| human-input `@tool` / loop break | tool on the node + HITL note (no native interrupt) |
| `hooks=`, `callback_handler=`, OTel | node observability config |

---

## 12. Reject-list — Strands parsing / output containing any of these is WRONG

- `strands.interrupt(`, `Agent.interrupt(`, or any claim that Strands natively
  pauses/resumes — **no such primitive exists** (§9).
- `from strands import StateGraph`, `add_conditional_edges`, `TypedDict` graph state,
  `interrupt()` — LangGraph constructs; not Strands.
- `from strands import ChatAgent`, `WorkflowBuilder`, `@ai_function`, `Executor` — MAF
  constructs; not Strands.
- Importing prebuilt tools from `strands` instead of `strands_tools`
  (correct: `from strands_tools import calculator`).
- Listing `strands_tools` as a dependency — the *distribution* is `strands-agents-tools`
  (import name is `strands_tools`).
- Modelling each `@tool` on a single agent as its own IR node (tools are node tools, not
  nodes).
- Inventing an OpenAI default — the default provider is **Bedrock**.
- `@kernel_function`, `Kernel()`, `import semantic_kernel` — wrong framework entirely.
- Treating a lone `Agent(tools=[...])` as a multi-node graph — it is one `SINGLE_AGENT`
  node.
- stdlib modules (`json`, `os`, `io`, `re`, `math`, ...) listed in `requirements.txt`.

---

## 13. Minimal end-to-end skeleton (reference the parser can pattern-match)

```python
from strands import Agent, tool
from strands.models import BedrockModel


@tool
def get_history(test_id: str) -> dict:
    """Return CI run stats for one test."""
    return {"test_id": test_id, "runs": 100, "fails": 3}


@tool
def validate_test(code: str) -> dict:
    """Check that a Python test snippet compiles. Returns {valid, error}."""
    try:
        compile(code, "<string>", "exec")
        return {"valid": True, "error": ""}
    except SyntaxError as exc:
        return {"valid": False, "error": str(exc)}


agent = Agent(
    model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
    system_prompt="You are a test-optimisation assistant.",
    tools=[get_history, validate_test],
)


def main():
    result = agent("Analyse the suite and flag flaky tests.")
    print(result.message)


if __name__ == "__main__":
    main()
```

This whole program parses to a **single `SINGLE_AGENT` node** with two tools
(`get_history`, `validate_test`), a Bedrock model config, and the given system prompt.
