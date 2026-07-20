# New Framework — Target Framework Knowledge Pack (Template)

> **Purpose of this file.** This is the Tier-3 documentation pack template.
> It is injected into the LLM prompt as context when a "hard" section (HITL,
> checkpointing, conditional/loop workflow wiring, agent-as-tool, etc.) cannot
> be resolved deterministically by Tier-1 adapters or Tier-2 templates.
>
> **Replace every `YOUR_*` / `your_*` placeholder** with the real framework
> API before using this pack. See `FRAMEWORK_AUTHORING.md` for the full process.
>
> When this file conflicts with prior training memory of the framework,
> trust this file.

---

## 0. Identity — what this framework is and is NOT

**Framework name:** New Framework (replace with real name)  
- PyPI package: `your-framework-package` → `pip install your-framework-package`
- Import root: `your_framework` → `from your_framework import YOUR_ORCHESTRATOR_CLASS`
- Provider clients live in: `your_framework.openai`, `your_framework.azure` (update as needed)

**This framework is the conversion TARGET, not the source.** Never carry
source-framework constructs into the output:
- No `StateGraph`, `add_conditional_edges`, `TypedDict` reducers, `interrupt()` (LangGraph)
- No `WorkflowBuilder`, `@executor`, `@handler`, `RequestInfoExecutor` (MAF)
- No `from crewai import`, `Task(`, `Crew(`, `@listen(` (CrewAI)
- No `from strands import`, `Agent(tools=[` (AWS Strands)

---

## 1. Choosing the orchestration primitive

> Replace this section with the framework's real primitives.

| Source shape | Your framework primitive |
|---|---|
| Single LLM agent with tools (ReAct-style) | `YOUR_AGENT_CLASS` |
| Multi-step workflow with explicit nodes and edges | `YOUR_BUILDER_CLASS` + steps |
| Conditional routing / loops | `YOUR_BUILDER_CLASS` + conditional step wiring |

---

## 2. `YOUR_AGENT_CLASS` — the single LLM agent

```python
from your_framework import YOUR_AGENT_CLASS
from your_framework.openai import OpenAIChatClient  # or your provider

agent = YOUR_AGENT_CLASS(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    tools=[fetch, summarise],         # @your_tool_decorator-decorated functions
)
result = agent.run("Do the task.")
```

---

## 3. `YOUR_BUILDER_CLASS` — the multi-step workflow

```python
from your_framework import YOUR_BUILDER_CLASS, YOUR_STEP_BASE, YOUR_CONTEXT_TYPE

class GatherStep(YOUR_STEP_BASE):
    def run(self, input: InputType, ctx: YOUR_CONTEXT_TYPE) -> OutputType:
        data = fetch(input.url)
        return OutputType(result=data)

class SummariseStep(YOUR_STEP_BASE):
    def run(self, input: InputType, ctx: YOUR_CONTEXT_TYPE) -> FinalOutput:
        return FinalOutput(summary=summarise(input.result))

workflow = (
    YOUR_BUILDER_CLASS()
    .add_step(GatherStep())
    .add_step(SummariseStep())
    .set_entry_point(GatherStep)
    .build()
)
result = workflow.run(InitialInput(url="..."))
```

---

## 4. Conditional edges and routing

```python
# Every branch needs its own wiring call:
workflow = (
    YOUR_BUILDER_CLASS()
    .add_step(ValidateStep())
    .add_step(ApproveStep(),  condition=lambda ctx: ctx.passed)
    .add_step(RetryStep(),    condition=lambda ctx: not ctx.passed and ctx.retries < 3)
    .add_step(DropStep(),     condition=lambda ctx: not ctx.passed and ctx.retries >= 3)
    .build()
)
```

**Anti-pattern:** Do not collapse a conditional graph into a linear `run()`
that calls steps in sequence — that silently drops the branching logic.

---

## 5. Tools (`@your_tool_decorator`)

```python
from your_framework import your_tool_decorator

@your_tool_decorator
def fetch(url: str) -> str:
    """Fetch a resource from the given URL."""
    return requests.get(url).text

@your_tool_decorator
def summarise(text: str) -> str:
    """Return a one-paragraph summary of the text."""
    ...
```

Registration: pass tool functions directly in the agent/step constructor:
```python
agent = YOUR_AGENT_CLASS(tools=[fetch, summarise], ...)
```

---

## 6. Human-in-the-loop (HITL)

> Replace this section with the framework's real HITL story.

If the framework has **no native pause/resume primitive**, model HITL as a
human-input tool that the agent calls when a decision is needed:

```python
@your_tool_decorator
def request_human_approval(summary: str, options: list[str]) -> str:
    """Ask a human to approve an action. Returns the approved option."""
    # In production: send to a UI / ticketing system and await response.
    # During testing: return options[0] (auto-approve fast-path).
    return options[0]
```

Always keep an automated fast-path so the agent can run unattended in CI.

---

## 7. State and checkpointing

```python
from your_framework import YOUR_CHECKPOINT_CLASS

storage = YOUR_CHECKPOINT_CLASS(path="./.checkpoints")
workflow = (
    YOUR_BUILDER_CLASS()
    .with_checkpointing(storage)
    ...
    .build()
)
# Resume from a checkpoint:
result = workflow.run_from_checkpoint(checkpoint_id, storage=storage)
```

---

## 8. Invocation patterns

```python
# Synchronous:
result = workflow.run(InitialInput(...))

# Asynchronous:
result = await workflow.run_async(InitialInput(...))

# Streaming:
async for chunk in workflow.stream(InitialInput(...)):
    print(chunk)
```

---

## 9. LLM providers

The framework is provider-pluggable; swap the client import to change the provider:

```python
# Default (replace with framework default):
from your_framework import DefaultModelClient
client = DefaultModelClient()  # reads YOUR_API_KEY from env

# OpenAI:
from your_framework.openai import OpenAIChatClient
client = OpenAIChatClient()    # reads OPENAI_API_KEY from env

# Azure OpenAI:
from your_framework.azure import AzureOpenAIChatClient
client = AzureOpenAIChatClient()  # reads AZURE_OPENAI_* from env
```

---

## 10. Dependencies

```
your-framework-package          # core
your-framework-package[openai]  # OpenAI provider
your-framework-package[azure]   # Azure OpenAI provider
python-dotenv                   # optional: load .env
```

**Never** include stdlib modules in requirements.txt:
`json`, `os`, `sys`, `io`, `re`, `math`, `hashlib`, `uuid`, `typing`,
`dataclasses`, `functools`, `subprocess`, `logging`, `datetime`, `pathlib`.

---

## 11. Reject list

Generated output containing any of the following is a conversion error:

- `from langgraph import` / `StateGraph(` / `add_conditional_edges(`
- `from agent_framework import` / `WorkflowBuilder(` / `@executor` / `@handler`
- `from crewai import` / `Task(` / `Crew(` / `@listen(`
- `from strands import` / `Agent(tools=[`
- stdlib packages in `requirements.txt`
