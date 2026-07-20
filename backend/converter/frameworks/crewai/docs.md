# CrewAI — Target / Source Framework Knowledge Pack

> **Purpose of this file.** This is the Tier‑3 documentation pack for the framework
> converter. It is injected into the LLM prompt as context when a "hard" section
> (HITL, task dependencies, hierarchical delegation, conditional/event-driven control
> flow, custom tools, memory/knowledge, etc.) cannot be resolved deterministically by
> Tier‑1 adapters or Tier‑2 templates. When CrewAI is the conversion **TARGET**, generate
> code **only** from the idioms in this file. When CrewAI is the **SOURCE**, use this file
> to recognise its constructs. When this file conflicts with prior training memory of
> "CrewAI", trust this file.
>
> **API stability note.** CrewAI evolves quickly. The *concepts and mappings* here are
> stable; if an exact class/argument name has drifted in the installed `crewai` version,
> prefer the installed signature but keep the same structure.

---

## 0. Identity — what CrewAI is and is NOT

**CrewAI = an open-source Python framework for orchestrating role-playing, autonomous AI agents.**
- PyPI package: `crewai`  →  `pip install crewai`
- Tools add-on: `crewai-tools`  →  `pip install crewai-tools` (prebuilt tools + `BaseTool`/`@tool` helpers)
- Import root: `crewai`  →  `from crewai import Agent, Task, Crew, Process, Flow, LLM`
- Tool helpers: `from crewai.tools import tool, BaseTool`; prebuilt tools: `from crewai_tools import SerperDevTool, FileReadTool, ...`

**CrewAI is a high-level role/task orchestration framework. It is NOT a low-level graph
runtime.** It has no `StateGraph`, no `add_node`/`add_edge` graph API, and no per-step
checkpointer. Its two orchestration models are:
- **Crew** — a team of `Agent`s executing a list of `Task`s under a `Process`
  (`sequential` or `hierarchical`).
- **Flow** — an event-driven, decorator-based controller (`@start`, `@listen`,
  `@router`) with typed/dict state, used when you need explicit branching, loops, or to
  chain multiple crews.

**When CrewAI is the TARGET, never carry source-framework constructs into the output.**
These are a conversion error:
- MAF: `WorkflowBuilder`, `Executor`, `@executor`, `@handler`, `WorkflowContext`,
  `RequestInfoExecutor`, `agent-framework` in requirements.
- LangGraph: `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`,
  `TypedDict` graph state, `Annotated[list, operator.add]` reducers, `interrupt()`,
  `from operator import add`.
- Semantic Kernel: `@kernel_function`, `Kernel()`, `kernel.add_plugin`.

---

## 1. Choosing the orchestration primitive

CrewAI has two primitives. Pick based on the **source graph shape**:

| Source shape | CrewAI primitive |
|---|---|
| A team of specialised roles each doing a step, run top-to-bottom or delegated by a manager | **`Crew`** of `Agent`s + `Task`s (`Process.sequential` / `Process.hierarchical`) |
| A single LLM agent that loops over tools (ReAct-style) | **one `Agent` + one `Task`** in a small `Crew` |
| Explicit event-driven control flow: branching, loops, routers, chaining multiple crews | **`Flow`** (`@start`, `@listen`, `@router`, `or_`/`and_`) |
| A multi-step graph with conditional edges / cycles | **`Flow`** driving one or more `Crew`s |

A **LangGraph `StateGraph` with conditional edges/loops maps to a CrewAI `Flow`**, not to
a bare `Crew`. A simple linear pipeline of steps maps to a **sequential `Crew`**. Do not
collapse a branching graph into a linear sequential crew — that silently drops the
conditional routing.

---

## 2. `Agent` — the role-playing LLM worker

```python
from crewai import Agent, LLM

researcher = Agent(
    role="Senior Test Analyst",
    goal="Identify flaky and redundant tests in the suite",
    backstory=(
        "You are a meticulous QA engineer with years of experience "
        "reading CI history and spotting unreliable tests."
    ),
    tools=[get_history_tool, read_tests_tool],   # BaseTool / @tool instances — see §6
    llm=LLM(model="openai/gpt-4o"),               # optional; defaults from env otherwise
    allow_delegation=False,                        # True lets this agent delegate to peers
    verbose=True,
    max_iter=25,                                   # optional tool-loop cap
)
```

Key arguments:
- `role`, `goal`, `backstory` — the three **required** identity fields. They form the
  agent's system prompt. Always provide all three; they replace an MAF `instructions`
  string or a LangGraph node's prompt.
- `tools` — list of tool instances the agent may call (see §6).
- `llm` — an `LLM(...)` instance or a model string; if omitted, uses the default model
  from environment (`OPENAI_API_KEY` / `OPENAI_MODEL_NAME`, etc.).
- `allow_delegation` — when `True`, the agent can hand work to other agents in the crew.
  In `Process.hierarchical`, the manager delegates; workers usually keep this `False`.
- `verbose`, `max_iter`, `max_rpm`, `memory` — optional behaviour controls.

An `Agent` alone does nothing — it must be assigned to a `Task` that is added to a `Crew`.

---

## 3. `Task` — one unit of work (the "node")

```python
from crewai import Task
from pydantic import BaseModel

class CoverageReport(BaseModel):
    score: float
    gaps: list[str]

analyse_task = Task(
    description=(
        "Analyse the test suite at {suite_path}. "   # {placeholders} filled from kickoff inputs
        "Find coverage gaps and score the suite."
    ),
    expected_output="A JSON object with a coverage score and a list of gaps.",
    agent=researcher,                # which agent executes this task
    context=[intake_task],           # outputs of these tasks are injected as context — see §4
    output_pydantic=CoverageReport,  # parse the result into a typed model
    human_input=False,               # True → pause for human review (see §7)
    # output_file="report.md",       # optional: also write the result to a file
)
```

Key arguments:
- `description` — what to do. May contain `{placeholders}` that are interpolated from
  `crew.kickoff(inputs={...})`.
- `expected_output` — **required.** A description of what a good result looks like. This
  is not optional flavour text; CrewAI uses it to shape the output.
- `agent` — the `Agent` that runs this task. In `Process.hierarchical` you may omit it and
  let the manager assign.
- `context=[...]` — a list of **other Task objects** whose outputs are fed in as context.
  This is how you express data dependencies / edges (see §4).
- `output_pydantic` / `output_json` — parse the raw result into a typed object; access via
  `task_output.pydantic`. `output_file` also writes it to disk.
- `human_input=True` — after the task runs, pause and ask a human to review/approve the
  output before continuing (see §7).

**Every Task you define MUST be added to a Crew's `tasks=[...]` list** (or produced inside
a Flow method). A Task that is constructed but never given to a Crew is dead code and a
conversion error.

---

## 4. Task ordering, `context`, and dependencies (the "edges")

CrewAI does not have an explicit edge API. Ordering and data flow come from three places:

1. **`Process.sequential` order.** Tasks run **in the order they appear** in the crew's
   `tasks=[...]` list. Each task implicitly receives the previous task's output.

2. **Explicit `context=[...]` dependencies.** To feed the output of specific earlier
   tasks into a later task (a DAG edge, not just the immediate predecessor), list those
   Task objects in `context`:
   ```python
   report_task = Task(
       description="Write a report combining the coverage and redundancy findings.",
       expected_output="A markdown report.",
       agent=writer,
       context=[coverage_task, redundancy_task],   # both outputs injected
   )
   ```

3. **`Process.hierarchical` manager delegation.** A manager agent (via `manager_llm` or
   `manager_agent`) decides which worker agent handles each task and in what order,
   delegating dynamically instead of following the static list order (see §5).

Conditional branching, loops, and routing are **NOT** expressible inside a plain Crew —
they require a **Flow** (see §8). Do not fake a conditional edge with prompt text inside a
sequential crew.

---

## 5. `Crew` — the team that runs the tasks

```python
from crewai import Crew, Process, LLM

crew = Crew(
    agents=[researcher, redundancy_analyst, writer],
    tasks=[intake_task, coverage_task, redundancy_task, report_task],
    process=Process.sequential,     # or Process.hierarchical
    verbose=True,
    memory=True,                    # enable short/long/entity memory (see §7 memory)
    # Hierarchical only:
    # manager_llm=LLM(model="openai/gpt-4o"),   # auto-creates a manager agent
    # manager_agent=custom_manager_agent,        # or supply your own manager Agent
)

result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
print(result.raw)              # final task's raw text
print(result.pydantic)         # typed output if the final task set output_pydantic
```

- `process=Process.sequential` — tasks run top-to-bottom; each sees the prior output.
- `process=Process.hierarchical` — a **manager** coordinates. You must supply either
  `manager_llm` (CrewAI builds a manager agent) or `manager_agent` (your own). Worker
  tasks may omit `agent=`; the manager assigns and delegates. Requires `allow_delegation`
  behaviour on the manager side.
- `memory=True` — turns on the memory subsystem (short-term, long-term, entity).
- `knowledge_sources=[...]` — attach documents/knowledge the crew can retrieve from.

---

## 6. Tools (source `@tool` functions / tool classes)

CrewAI tools are **instances** passed to `Agent(tools=[...])` (or `Task(tools=[...])`).
Three idiomatic ways to define them:

**(a) `@tool` decorator** — quickest for a single function:
```python
from crewai.tools import tool

@tool("CI History Lookup")
def get_history(test_id: str) -> str:
    """Return CI run stats (runs, fails, avg_seconds) for a single test id."""
    return str(_load_ci_history().get(test_id, {}))
```
The decorated object is a ready-to-use tool **instance** — pass it directly in `tools=[...]`.

**(b) Subclass `BaseTool`** — when you need typed args / reusable config:
```python
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class HistoryInput(BaseModel):
    test_id: str = Field(..., description="The unique test identifier")

class CIHistoryTool(BaseTool):
    name: str = "CI History Lookup"
    description: str = "Return CI run stats for a single test id."
    args_schema: Type[BaseModel] = HistoryInput

    def _run(self, test_id: str) -> str:
        return str(_load_ci_history().get(test_id, {}))

history_tool = CIHistoryTool()   # instantiate before passing to an agent
```

**(c) Prebuilt tools from `crewai_tools`:**
```python
from crewai_tools import SerperDevTool, FileReadTool, ScrapeWebsiteTool
search_tool = SerperDevTool()
```

Rules:
- Pass **instances**, not classes: `tools=[history_tool]`, not `tools=[CIHistoryTool]`.
- `BaseTool` subclasses implement `_run(self, ...)`; the arg names must match
  `args_schema`.
- **Do not** emit tool classes that are never instantiated or never assigned to an agent —
  that is dead code and a conversion error.

---

## 7. HITL, memory & knowledge

### 7.1 Human-in-the-loop (source `interrupt()`)  — HARD SECTION

CrewAI's built-in HITL is the `human_input=True` flag on a **Task**. After the task's
agent produces its output, execution **pauses** and the human is prompted to review,
correct, or approve it; their feedback is fed back to the agent, which may revise before
the crew continues.

```python
approve_task = Task(
    description="Propose which flaky tests to quarantine and justify each choice.",
    expected_output="A list of test ids to quarantine, with reasons.",
    agent=analyst,
    human_input=True,    # ← pauses after the agent responds; human reviews/approves
)
```

- The pause is at the **task boundary**, not mid-tool-call — this is the CrewAI-native
  equivalent of a LangGraph `interrupt()` or an MAF `RequestInfoExecutor` request.
- For programmatic / non-interactive HITL, drive review outside the crew: run one crew,
  inspect `result`, then kick off the next crew — orchestrate this with a **Flow** and a
  `@router` that branches on human approval.
- **Never** implement HITL as a hard-coded auto-approve with the real logic left in a
  comment block. Either set `human_input=True`, or route through a Flow that captures a
  real decision.

### 7.2 Memory (there is no per-step checkpointer)

CrewAI has **no MAF-style step checkpoint / resume-from-checkpoint**. Instead it offers a
**memory** subsystem, enabled with `Crew(memory=True)`:
- **Short-term memory** — context within the current run (recent interactions).
- **Long-term memory** — persisted across runs (stored on disk / SQLite by default),
  lets the crew learn from past executions.
- **Entity memory** — tracks entities (people, concepts) encountered during the run.

```python
crew = Crew(agents=[...], tasks=[...], process=Process.sequential, memory=True)
```

Map a source checkpointer (`MemorySaver`, `FileCheckpointStorage`) to `memory=True` —
noting it is **not** a pause/resume checkpoint, only cross-run recall. True durable
pause/resume must be modelled as separate `crew.kickoff()` calls orchestrated by a Flow.

### 7.3 Knowledge sources

Attach reference material the crew can retrieve from:
```python
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

policy = StringKnowledgeSource(content="Pinned tests are never removed. ...")
crew = Crew(agents=[...], tasks=[...], knowledge_sources=[policy])
```

---

## 8. `Flow` — event-driven control flow (the StateGraph target)

Use a **Flow** when the source has explicit branching, loops, routers, or chains multiple
crews. A Flow is a class subclassing `Flow`, with methods decorated to wire the control
flow. State is a Pydantic model (typed) or a dict.

```python
from crewai.flow.flow import Flow, start, listen, router, or_, and_
from pydantic import BaseModel

class SuiteState(BaseModel):
    suite_path: str = ""
    score: float = 0.0
    passed: bool = False
    retries: int = 0

class OptimiseFlow(Flow[SuiteState]):

    @start()
    def intake(self):
        # runs first; may return a value passed to listeners
        self.state.suite_path = "./uploads/suite/"
        return "loaded"

    @listen(intake)
    def analyse(self, _):
        result = AnalysisCrew().crew().kickoff(inputs={"path": self.state.suite_path})
        self.state.score = result.pydantic.score
        self.state.passed = self.state.score >= 0.8

    @router(analyse)
    def route_on_score(self):
        # returns a label; the matching @listen(<label>) fires next
        if self.state.passed:
            return "approve"
        if self.state.retries < 3:
            self.state.retries += 1
            return "retry"
        return "escalate"

    @listen("approve")
    def approve(self):
        return f"approved score={self.state.score}"

    @listen("retry")
    def retry(self):
        return self.analyse(None)   # loop back

    @listen("escalate")
    def escalate(self):
        return "escalated to human"

flow = OptimiseFlow()
final = flow.kickoff()
```

- `@start()` — entry method(s); run when the flow kicks off.
- `@listen(trigger)` — runs when `trigger` completes. `trigger` can be another method or a
  string label emitted by a `@router`.
- `@router(method)` — runs after `method`, **returns a string label**; the `@listen("label")`
  with that label fires. This is how conditional edges and loops are expressed.
- `or_(a, b)` / `and_(a, b)` — a listener fires when **any** / **all** of the triggers
  complete: `@listen(or_(retry, approve))`.
- `self.state` — the typed Pydantic model (or dict) shared across the flow; replaces a
  LangGraph graph-state `TypedDict`.
- A `@router` that is defined but whose labels are never listened to is a conversion bug —
  every branch label must have a matching `@listen`.

---

## 9. Invocation

```python
# Single run (fills {placeholders} in task descriptions):
result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})

# Fan out over many inputs (one crew run per dict):
results = crew.kickoff_for_each(inputs=[{"suite_path": p} for p in paths])

# Async:
result = await crew.kickoff_async(inputs={"suite_path": "..."})
results = await crew.kickoff_for_each_async(inputs=[...])

# Flow:
final = flow.kickoff()             # sync
final = await flow.kickoff_async() # async

# Reading results:
result.raw          # final task raw text
result.pydantic     # typed model if final task set output_pydantic
result.json_dict    # dict if output_json was set
result.tasks_output # per-task outputs
```

---

## 10. Providers

CrewAI uses **LiteLLM** under the hood, so the provider is chosen by the model string.

```python
from crewai import LLM

llm = LLM(model="openai/gpt-4o", temperature=0.2)
llm = LLM(model="azure/<deployment>")            # + AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
llm = LLM(model="anthropic/claude-3-5-sonnet")   # + ANTHROPIC_API_KEY
llm = LLM(model="gemini/gemini-1.5-pro")         # + GEMINI_API_KEY
```

- OpenAI: `OPENAI_API_KEY` (and optional `OPENAI_MODEL_NAME`).
- Azure OpenAI: `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`, model
  `azure/<deployment-name>`.
- Anthropic: `ANTHROPIC_API_KEY`, model `anthropic/<model>`.
- Gemini: `GEMINI_API_KEY`, model `gemini/<model>`.
- If no `llm=` is passed to an `Agent`, CrewAI uses the default model from environment.
  Prefer passing an explicit `LLM(model=...)` in generated code.

---

## 11. Source → CrewAI mapping table (apply mechanically)

| Neutral IR / typical source concept | CrewAI (target) |
|---|---|
| Graph / compiled pipeline | `Crew` (linear) or `Flow` (branching/loops) |
| Node `def n(state) -> dict` | a `Task` executed by an `Agent` |
| Sequential edge `add_edge(a, b)` | order of tasks in `Crew(tasks=[a, b, ...])` (`Process.sequential`) |
| DAG dependency (specific upstream outputs) | `Task(context=[upstream_task, ...])` |
| Conditional edge / router | `Flow` `@router` returning a label → `@listen("label")` |
| Cycle / bounded retry loop | `Flow` `@router` that loops back, guarded by a state counter |
| Fan-out / parallel | `Flow` `@listen` on multiple triggers + `and_(...)` join, or `kickoff_for_each` |
| Supervisor / dynamic assignment | `Process.hierarchical` with `manager_llm` / `manager_agent` |
| Single ReAct agent + tools | one `Agent` + one `Task` in a small `Crew` |
| Graph-state `TypedDict` | `Flow` state Pydantic model (`Flow[StateModel]`) |
| `@tool` function | `@tool` from `crewai.tools`, or a `BaseTool` subclass |
| Prebuilt tool | `crewai_tools` (e.g. `SerperDevTool`, `FileReadTool`) |
| `interrupt(payload)` / HITL | `Task(human_input=True)`, or a `Flow` `@router` on a human decision |
| Checkpointer (`MemorySaver`, etc.) | `Crew(memory=True)` (cross-run recall — **not** a step checkpoint) |
| `graph.invoke(inputs)` | `crew.kickoff(inputs=...)` / `flow.kickoff()` |
| Batched invoke | `crew.kickoff_for_each(inputs=[...])` |
| Async invoke | `crew.kickoff_async(...)` / `flow.kickoff_async()` |

---

## 12. Reject-list — output containing any of these is WRONG

- `WorkflowBuilder`, `Executor`, `@executor`, `@handler`, `WorkflowContext`,
  `RequestInfoExecutor` → un-converted MAF
- `agent-framework` in `requirements.txt` → wrong dependency (must be `crewai`)
- `from langgraph`, `StateGraph(`, `add_node(`, `add_edge(` (graph API),
  `add_conditional_edges(`, `from operator import add`, `interrupt(` → un-converted LangGraph
- `@kernel_function`, `Kernel()`, `kernel.add_plugin`, `semantic-kernel` in requirements → wrong framework
- A `Task` constructed but never added to a `Crew(tasks=[...])` (or produced by a Flow) → dead task
- A tool class (`BaseTool` subclass) never instantiated or never assigned to an agent → dead tool
- Passing a tool **class** instead of an **instance** in `tools=[...]`
- Using a plain sequential `Crew` when the source had conditional branching or loops (needs a `Flow`)
- A `Flow` `@router` label that no `@listen` responds to
- Omitting `expected_output` on a `Task` (it is required)
- Omitting `role`/`goal`/`backstory` on an `Agent`
- HITL implemented as hard-coded auto-approve with real logic commented out
- `requirements.txt` listing stdlib modules (`json`, `os`, `io`, `re`, `pathlib`, ...) as packages

---

## 13. Minimal end-to-end skeleton (reference the converter can pattern-match)

```python
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel

class Report(BaseModel):
    score: float
    gaps: list[str]

llm = LLM(model="openai/gpt-4o")

analyst = Agent(
    role="Test Analyst",
    goal="Find coverage gaps in the suite at {suite_path}",
    backstory="A meticulous QA engineer who reads CI history.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Report Writer",
    goal="Summarise findings into a clear report",
    backstory="A technical writer who turns analysis into actionable reports.",
    llm=llm,
)

analyse_task = Task(
    description="Analyse the suite at {suite_path} and list coverage gaps.",
    expected_output="A list of coverage gaps.",
    agent=analyst,
)

report_task = Task(
    description="Write a report with a coverage score and the gaps.",
    expected_output="A JSON object: {score: float, gaps: [str]}.",
    agent=writer,
    context=[analyse_task],
    output_pydantic=Report,
)

crew = Crew(
    agents=[analyst, writer],
    tasks=[analyse_task, report_task],
    process=Process.sequential,
    verbose=True,
)

def main():
    result = crew.kickoff(inputs={"suite_path": "./uploads/suite/"})
    print(result.pydantic)
```
