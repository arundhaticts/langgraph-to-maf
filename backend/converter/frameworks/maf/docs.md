# Microsoft Agent Framework (MAF) — Target Framework Knowledge Pack

> **Purpose of this file.** This is the Tier‑3 documentation pack for the framework
> converter. It is injected into the LLM prompt as context when a "hard" section
> (HITL, checkpointing, conditional/loop workflow wiring, agent-as-tool, etc.) cannot
> be resolved deterministically by Tier‑1 adapters or Tier‑2 templates. Generate code
> for the target framework **only** from the idioms in this file. When this file
> conflicts with prior training memory of "MAF", trust this file.
>
> **API stability note.** The `agent_framework` API is young and evolving. The
> *concepts and mappings* here are stable; if an exact class/argument name has drifted
> in the installed `agent-framework` version, prefer the installed signature but keep
> the same structure.

---

## 0. Identity — what MAF is and is NOT

**MAF = Microsoft Agent Framework.**
- PyPI package: `agent-framework`  →  `pip install agent-framework`
- Import root: `agent_framework`  →  `from agent_framework import ChatAgent, WorkflowBuilder`
- Provider clients live in subpackages: `agent_framework.azure`, `agent_framework.openai`.

**MAF is NOT Semantic Kernel.** Do not emit any of the following — they belong to a
different (predecessor) framework and are a conversion error:
- dependency `semantic-kernel`
- `from semantic_kernel ...`, `@kernel_function`, `Kernel()`, `kernel.add_plugin(...)`
- "skill" / "plugin class" vocabulary

**MAF is the conversion TARGET, not the source.** Never carry LangGraph constructs into
the output: `StateGraph`, `add_node`, `add_conditional_edges`, `TypedDict` graph state,
`Annotated[list, operator.add]` reducers, `interrupt()`, `from operator import add`.

---

## 1. Choosing the orchestration primitive

MAF has two orchestration primitives. Pick based on the **source graph shape**:

| Source shape | MAF primitive |
|---|---|
| One LLM agent that decides which tools to call in a loop (ReAct-style) | **`ChatAgent`** |
| A multi-step graph with explicit nodes, conditional edges, cycles, or HITL | **`Workflow`** (`WorkflowBuilder` + `Executor`s) |

A **LangGraph `StateGraph` maps to a MAF `Workflow`.** Do not collapse a graph into a
linear `run()` that calls step functions in sequence — that silently drops the
conditional edges and loops.

---

## 2. `ChatAgent` — the LLM agent

```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
# or: from agent_framework.openai import OpenAIChatClient

agent = ChatAgent(
    chat_client=AzureOpenAIChatClient(),   # picks up endpoint / key / deployment from env
    instructions="You are a test-optimisation assistant.",
    tools=[get_history, read_tests],       # plain typed functions — see §4
)

result = await agent.run("Analyse this suite.")   # async
print(result.text)

# Streaming:
async for update in agent.run_stream("..."):
    print(update.text, end="")
```

**Conversation memory (threads):**
```python
thread = agent.get_new_thread()
await agent.run("first turn", thread=thread)
await agent.run("second turn, remembers the first", thread=thread)
```

**Creating an agent from a client (equivalent shorthand):**
```python
agent = AzureOpenAIChatClient().create_agent(
    instructions="...",
    tools=[...],
)
```

---

## 3. `Workflow` — graph orchestration (the StateGraph target)

### 3.1 Executors

An **executor** is a node. It receives a typed message and emits typed messages to
downstream executors via a `WorkflowContext`.

Class form:
```python
from agent_framework import Executor, handler, WorkflowContext

class Intake(Executor):
    @handler
    async def run(self, message: RawSuite, ctx: WorkflowContext[NormalisedSuite]) -> None:
        normalised = normalise(message)
        await ctx.send_message(normalised)     # -> flows to the next executor(s)
```

Function form (use `@executor` for simple nodes):
```python
from agent_framework import executor, WorkflowContext

@executor(id="coverage")
async def coverage(message: NormalisedSuite, ctx: WorkflowContext[CoverageResult]) -> None:
    await ctx.send_message(analyse_coverage(message))
```

Rules:
- `WorkflowContext[T]` — `T` is the type this executor **sends downstream**.
- `WorkflowContext[T, U]` — `T` sent downstream, `U` is the workflow **output** type.
- Emit results with `await ctx.send_message(...)`; emit a final workflow output with
  `await ctx.yield_output(...)`.
- Every executor must send/yield a **typed object** (pydantic `BaseModel` or dataclass),
  never a bare partial `dict`.

### 3.2 Building and running the workflow

```python
from agent_framework import WorkflowBuilder

workflow = (
    WorkflowBuilder()
    .set_start_executor(intake)
    .add_edge(intake, coverage)
    .add_edge(coverage, redundancy)
    .add_edge(redundancy, report)
    .build()
)

# Run to completion:
result = await workflow.run(RawSuite(...))

# Or stream events (executor completions, requests, outputs):
async for event in workflow.run_stream(RawSuite(...)):
    ...
```

### 3.3 Conditional edges (source `add_conditional_edges`)

A router that returns the next node name becomes **one guarded edge per branch**:

```python
.add_edge(validation, approve_tests,
          condition=lambda m: m.validation_passed)
.add_edge(validation, drop_failing,
          condition=lambda m: not m.validation_passed and m.gen_retry_count >= MAX_GEN_RETRIES)
.add_edge(validation, gap_gen,
          condition=lambda m: not m.validation_passed and m.gen_retry_count < MAX_GEN_RETRIES)
```

The `condition` is a predicate over the message the source executor emitted. Exactly the
branch whose condition is true fires.

### 3.4 Loops (source cycles / bounded retry)

A loop is just a **back-edge guarded by a condition** that eventually goes false. The
guard is the source's iteration cap (e.g. `MAX_GEN_RETRIES`, `MAX_REVISE_ITERS`):

```python
# gap_gen -> validation -> (loop back to gap_gen while failing and under the retry cap)
.add_edge(gap_gen, validation)
.add_edge(validation, gap_gen,
          condition=lambda m: not m.validation_passed and m.gen_retry_count < MAX_GEN_RETRIES)
.add_edge(validation, drop_failing,
          condition=lambda m: not m.validation_passed and m.gen_retry_count >= MAX_GEN_RETRIES)
```

⚠️ A LangGraph router function that is *defined but never referenced* is a conversion
bug. Every source router must become live guarded edges.

### 3.5 Fan-out / fan-in

```python
.add_fan_out_edges(dispatch, [worker_a, worker_b, worker_c])   # parallel branches
.add_fan_in_edges([worker_a, worker_b, worker_c], aggregate)   # join
```

---

## 4. Tools (source `@tool` functions)

A MAF tool is a **typed Python function**. The framework introspects the signature and
docstring to build the tool schema. Use `@ai_function` only to override name/description.

```python
from typing import Annotated
from pydantic import Field
from agent_framework import ai_function

@ai_function(description="Return CI run stats for one test, or None if no history.")
def get_history(
    test_id: Annotated[str, Field(description="The test identifier")],
    path: Annotated[str | None, Field(description="Optional CI-history file path")] = None,
) -> dict | None:
    """Look up CI evidence for a single test."""
    ...
```

Register tools by passing them to the agent (`tools=[...]`) or by calling them from
inside an executor. **Do not** wrap tools in decorator classes that are never
instantiated or registered — that is dead code and a conversion error.

An agent can itself be used as a tool by another agent:
```python
research_tool = research_agent.as_tool(
    name="research", description="Research a topic and return findings."
)
supervisor = ChatAgent(chat_client=..., tools=[research_tool])
```

---

## 5. State (source TypedDict + reducers)

MAF has **no global mutable graph-state dict**. State flows as typed messages, plus an
optional shared store for cross-cutting values.

**Typed messages** (preferred — pydantic model or dataclass):
```python
from pydantic import BaseModel

class CoverageResult(BaseModel):
    coverage_map: dict
    coverage_gaps: list[dict]
    projected_coverage: float
```

**Shared state** for values many executors read/write:
```python
await ctx.set_shared_state("audit_log", log)
log = await ctx.get_shared_state("audit_log")
```

**Append-reducer fields.** LangGraph `Annotated[list, add]` becomes a plain list on the
state model that you `.append()` to (in shared state or the carried message). Drop the
`operator.add` reducer and the `Annotated[..., add]` wrapper entirely.

**Consistency rule.** Every executor emits the same state contract the next executor
expects. Never return a raw partial `dict` (a LangGraph node idiom) — it will break
attribute access downstream.

---

## 6. Human-in-the-loop (source `interrupt()`)  — HARD SECTION

MAF implements HITL with a **request/response** pattern: an executor sends a request,
the workflow **pauses and emits a request event**, external code supplies a response,
and the workflow **resumes**. Combine with checkpointing (§7) to survive a real pause.

```python
from dataclasses import dataclass
from agent_framework import (
    Executor, handler, WorkflowContext,
    RequestInfoExecutor, RequestInfoMessage, RequestResponse,
)

# 1) Define the request payload (what the human is asked to approve).
@dataclass
class ApprovalRequest(RequestInfoMessage):
    checkpoint: str = ""
    candidates: list | None = None
    recommended: list | None = None

# 2) The node asks for approval instead of calling interrupt().
class HitlRemovals(Executor):
    @handler
    async def run(self, message: RemovalPayload, ctx: WorkflowContext) -> None:
        if message.run_mode == "automated":
            # automated fast-path: accept the safe recommendation, no human needed
            await ctx.send_message(ApprovedRemovals(ids=message.recommended))
            return
        # interactive path: emit a request and pause
        await ctx.send_message(
            ApprovalRequest(
                checkpoint="approve_removals",
                candidates=message.candidates,
                recommended=message.recommended,
            ),
            target_id=request_info.id,   # route to the RequestInfoExecutor
        )

# 3) A RequestInfoExecutor bridges to the outside world.
request_info = RequestInfoExecutor(id="request_info")

# 4) A node handles the human's reply on resume.
class ApplyRemovalDecision(Executor):
    @handler
    async def run(self, reply: RequestResponse[ApprovalRequest, list], ctx: WorkflowContext) -> None:
        approved = reply.data if reply.data is not None else reply.original_request.recommended
        await ctx.send_message(ApprovedRemovals(ids=approved))
```

**Driving the pause/resume loop from the host:**
```python
# First run — collect any human requests the workflow emits.
responses = None
while True:
    events = (await workflow.run(input_data)
              if responses is None
              else await workflow.send_responses_streaming(responses))

    pending = RequestInfoExecutor.pending_requests(events)   # requests needing a human
    if not pending:
        break

    # Present each request to the human, gather answers.
    responses = {req.request_id: get_human_decision(req) for req in pending}
```

**Conversion guidance:**
- `interrupt(payload)` → send a `RequestInfoMessage` subclass carrying `payload`.
- The resume value → arrives as `RequestResponse.data`; `None` means "accept the
  recommended default".
- Always keep an `automated` fast-path that approves the recommendation without pausing.
- **Never** implement HITL as a hard-coded auto-approve with the real logic left in a
  comment block. Auto-approve is only acceptable as the explicit `automated` branch.

---

## 7. Checkpointing (pause / resume / durability) — HARD SECTION

Checkpointing lets a workflow persist between steps and resume later (essential for HITL
pauses that outlive the process).

```python
from agent_framework import WorkflowBuilder, FileCheckpointStorage

storage = FileCheckpointStorage(dir_path="./.checkpoints")

workflow = (
    WorkflowBuilder()
    .set_start_executor(intake)
    .add_edge(intake, coverage)
    # ... edges ...
    .with_checkpointing(storage)
    .build()
)

# Resume from a saved checkpoint:
await workflow.run_from_checkpoint(checkpoint_id, checkpoint_storage=storage)
```

- Checkpoints are written at superstep boundaries; each captures executor + shared state.
- List/select checkpoints via the storage object to resume the right one.
- Pair checkpointing with the HITL request/response loop so an approval can arrive long
  after the workflow first paused.

---

## 8. Middleware & observability (optional)

- **Middleware** wraps agent runs or function/tool calls for logging, retries,
  guardrails, or termination. Register via the agent/run configuration.
- MAF emits **OpenTelemetry** traces/metrics; enable the framework's telemetry setup
  rather than hand-rolling logging. Keep source audit-logging as middleware or as plain
  calls inside executors.

---

## 9. LangGraph → MAF mapping table (apply mechanically)

| LangGraph (source) | Microsoft Agent Framework (target) |
|---|---|
| `StateGraph` / compiled graph | `Workflow` built via `WorkflowBuilder` |
| node `def n(state) -> dict` | `Executor` subclass w/ `@handler`, or `@executor` function |
| `TypedDict` graph state | typed messages (pydantic/dataclass) + shared state |
| `Annotated[list, add]` reducer | plain list field + `.append()`; drop `operator.add` |
| `builder.add_edge(a, b)` | `.add_edge(a, b)` |
| `add_conditional_edges(a, router)` | one `.add_edge(a, target, condition=...)` per branch |
| cycle / bounded retry loop | back-edge guarded by a `condition` (iteration cap) |
| `@tool` function | typed function tool / `@ai_function` |
| `interrupt(payload)` | `RequestInfoExecutor` + `RequestInfoMessage`/`RequestResponse` |
| checkpointer (`MemorySaver`, etc.) | `with_checkpointing(FileCheckpointStorage(...))` |
| `graph.invoke()` / `graph.stream()` | `await workflow.run(...)` / `workflow.run_stream(...)` |
| single ReAct agent + tools | `ChatAgent(chat_client, instructions, tools=[...])` |
| supervisor / sub-agent | agent used as a tool via `agent.as_tool(...)` |

---

## 10. Reject-list — output containing any of these is WRONG

- `import semantic_kernel`, `@kernel_function`, `Kernel()` → wrong framework
- `semantic-kernel` in `requirements.txt` → wrong dependency (must be `agent-framework`)
- `from langgraph ...`, `TypedDict` graph state, `from operator import add`, `interrupt(` → un-converted source
- A linear `run(ctx)` calling nodes in sequence when the source graph had conditional edges or loops
- Router functions that are defined but never wired into `.add_edge(..., condition=...)`
- Tool wrapper classes that are never instantiated or registered
- Executors returning bare partial `dict`s instead of typed state/messages
- HITL implemented as hard-coded auto-approve with real logic commented out
- `requirements.txt` listing stdlib modules (`zipfile`, `json`, `io`, `os`, ...) as packages

---

## 11. Minimal end-to-end skeleton (reference the converter can pattern-match)

```python
from pydantic import BaseModel
from agent_framework import WorkflowBuilder, executor, WorkflowContext

class Suite(BaseModel):
    tests: list[dict]

class Report(BaseModel):
    summary: dict

@executor(id="intake")
async def intake(msg: Suite, ctx: WorkflowContext[Suite]) -> None:
    await ctx.send_message(Suite(tests=[normalise(t) for t in msg.tests]))

@executor(id="analyse")
async def analyse(msg: Suite, ctx: WorkflowContext[Report]) -> None:
    await ctx.send_message(Report(summary=score(msg.tests)))

@executor(id="report")
async def report(msg: Report, ctx: WorkflowContext[None, Report]) -> None:
    await ctx.yield_output(msg)   # final workflow output

workflow = (
    WorkflowBuilder()
    .set_start_executor(intake)
    .add_edge(intake, analyse)
    .add_edge(analyse, report)
    .build()
)

async def main():
    result = await workflow.run(Suite(tests=[...]))
    print(result)
```
