# Framework Reference

A consolidated reference for the four frameworks the converter understands, plus
the integration model (registration, discovery, `vocabulary.json` schema,
mapping/translation rules, templates). For the engine that uses this knowledge,
see [ENGINEERING_GUIDE.md](ENGINEERING_GUIDE.md); for the architecture, see
[ARCHITECTURE.md](ARCHITECTURE.md).

How to read this document: each framework section describes the framework's own
model *and* how the converter reads it (its `SourceAdapter`) and writes it (its
`TargetAdapter` + `TargetGenerator`). Real class/field names are cited from
`backend/converter/`.

---

## Supported frameworks at a glance

| | MAF | LangGraph | CrewAI | AWS Strands |
|---|---|---|---|---|
| Registry name | `maf` | `langgraph` | `crewai` | `aws_strands` |
| Source adapter | `MAFSourceAdapter` | `LangGraphSourceAdapter` | `CrewAISourceAdapter` | `AWSStrandsSourceAdapter` |
| Target adapter | `MAFTargetAdapter` | `LangGraphTargetAdapter` | `CrewAITargetAdapter` | `AWSStrandsTargetAdapter` |
| Target generator | `MAFTargetGenerator` | `LangGraphTargetGenerator` | `CrewAITargetGenerator` | `AWSStrandsTargetGenerator` |
| Import signature | `agent_framework` | `langgraph` | `crewai` | `strands` |
| Tool decorator (target) | `@ai_function` | `@tool` | `@tool` | `@tool` |
| Tool style | function | function | function | function |
| Context class | `AgentContext` | `AgentState` | `AgentContext` | `AgentContext` |
| Orchestration idiom | `WorkflowBuilder` graph | `StateGraph` | `Crew` / `Flow` | `Agent` + tools |
| Ships offline SDK stub | **yes** (`agent_framework/`) | no | no | no |

---

## Microsoft Agent Framework (MAF)

- **Purpose:** Microsoft's framework for building agent workflows as a graph of
  executors. It is the converter's **reference target** (the richest generator).
- **Core concepts:** a `WorkflowBuilder` wires `@executor` nodes with edges;
  `ChatAgent` is a single ReAct-style agent; `RequestInfoExecutor` implements
  HITL; `FileCheckpointStorage` persists runs.
- **Execution model:** graph workflow (`run_stream` yields events, HITL surfaces
  as `RequestInfoEvent`, results as `WorkflowOutputEvent`) or a single
  `ChatAgent` loop for the SINGLE_AGENT mode.
- **Key components / terminology:** `WorkflowBuilder`, `@executor`,
  `WorkflowContext`, `set_start_executor`, `add_edge(..., condition=...)`,
  `RequestInfoExecutor`, `ai_function`, `ChatAgent`.
- **Agent model:** implicit — agents are graph nodes/executors (SINGLE_AGENT
  collapses to `ChatAgent(tools=[...])`).
- **Tool model:** `@ai_function(description=...)` decorated functions
  (`tool_style="function"`).
- **Workflow model:** `WorkflowBuilder` + executors + guarded edges; loops via
  back-edges.
- **Memory/state:** pydantic `BaseModel` context (`AgentContext`); the source
  `Annotated[list, add]` reducer becomes a plain list + `.append()`.
- **HITL:** native — `RequestInfoExecutor`; the generated entrypoint drives a
  `run_stream` + `send_responses_streaming` loop, with an `AUTO_APPROVE_HITL`
  switch for CI.
- **Converter specifics:** `MAFTargetGenerator` is the only generator that ships
  a **pure-Python offline SDK stub** (`_AGENT_FRAMEWORK_STUB`, ~200 lines:
  `WorkflowBuilder`, `executor`, `WorkflowContext`, `RequestInfoExecutor`,
  `FileCheckpointStorage`, `ChatAgent`, `ai_function`), so the output imports and
  runs with no real SDK. `orchestrator_must_tokens` (+workflow) require
  `WorkflowBuilder`, `@executor`, `set_start_executor`. `MAFSourceAdapter`
  recognises `@ai_function`, `TypedDict`/`BaseModel` state, and
  `WorkflowBuilder`/`@executor`/`@handler` graph tokens.

## LangGraph

- **Purpose:** LangChain's graph-based agent framework — the converter's
  **reference source** (its vocabulary is the `SourceVocabulary` default).
- **Core concepts:** `StateGraph` over a typed state (`TypedDict`), `add_node` /
  `add_edge` / `add_conditional_edges`, `MemorySaver` checkpointing, `interrupt()`
  HITL, `@tool` tools.
- **Execution model:** compiled graph invoked with a state dict; conditional
  edges route by a router function's return; loops are cycles in the graph.
- **Key components / terminology:** `StateGraph`, `START`/`END`, `add_node`,
  `add_conditional_edges`, `MemorySaver`/`SqliteSaver`, `interrupt`,
  `Annotated[list, add]` reducers, `MessagesState`.
- **Agent model:** implicit in graph nodes.
- **Tool model:** `@tool` (docstring-driven schema); `tool_style="function"`,
  bare decorator.
- **Workflow model:** explicit graph — `StateGraph` with entry point, edges, and
  `{label: target}` conditional maps.
- **Memory/state:** `TypedDict` state; append-only reducer fields
  (`Annotated[list, add]`) detected by the parser (`is_append_only`).
- **HITL:** `interrupt(payload)` inside a node; captured as a `HitlPoint`
  (verbatim payload).
- **Converter specifics:** `LangGraphSourceAdapter.vocabulary()` states the
  defaults explicitly (StateGraph builder methods, `@tool`, `TypedDict`,
  `MemorySaver`/`SqliteSaver`/…, `Chat` LLM prefix, drop root `langgraph`). As a
  target, `LangGraphTargetGenerator` emits a real `StateGraph`
  (`add_node`/`set_entry_point`/`add_conditional_edges`), `MemorySaver` when a
  checkpointer or HITL is present, import-guarded by `_HAVE_LANGGRAPH`;
  `orchestrator_must_tokens` (+workflow) require `StateGraph`, `add_node`,
  `set_entry_point`. Dropped source packages: `langgraph`,
  `langgraph-checkpoint-sqlite`, `langchain-core`, `langchain`.

## CrewAI

- **Purpose:** a role-based multi-agent framework — agents with
  role/goal/backstory collaborate on tasks.
- **Core concepts:** `Agent(role, goal, backstory, tools, llm)`,
  `Task(description, expected_output, agent, context)`,
  `Crew(agents, tasks, process)`, and `Flow` (`@start`/`@listen`/`@router`) for
  branching/looping control.
- **Execution model:** `Crew.kickoff()` runs tasks (sequential or DAG by
  `context=` dependencies); `Flow` runs event-driven steps.
- **Key components / terminology:** `Agent`, `Task`, `Crew`, `Process.sequential`,
  `Flow`, `@start`, `@listen`, `@router`, `human_input=True`.
- **Agent model:** **explicit roles** — `AgentSpec` (`role`, `goal`, `backstory`,
  `allowed_tools`, `llm`); `TaskSpec` (`description`, `expected_output`,
  `assigned_agent`, `depends_on`).
- **Tool model:** `@tool` / `BaseTool`; `tool_style="function"`, bare decorator.
- **Workflow model:** sequential crew *or* dependency DAG (from `Task.context`)
  *or* `Flow` for branches/loops.
- **Memory/state:** memory/knowledge, not per-step shared TypedDict — state is
  emulated (a `_state_dict` / `self.ctx` in generated code).
- **HITL:** `Task(human_input=True)`.
- **Converter specifics:** `CrewAISourceAdapter` overrides `extract_agents`,
  `extract_tasks`, and `extract_graph` (each `Task` var is a node; `Crew(tasks=[…])`
  order wins; `context=[…]` becomes dependency edges; otherwise a linear chain).
  `CrewAITargetGenerator` picks **sequential** (`Crew` + `Process.sequential`) or
  **branch/loop** (`Flow` subclass with `@start`/`@listen`/`@router`, running the
  ported node function on `self.ctx`), maps HITL to `Task(human_input=True)`, and
  is the only generator emitting non-Python `extra_files` — editable
  `prompts/agent_*.md`, `prompts/task_*.md`. `capability_matrix`: AGENT_ROLES and
  MULTI_AGENT DIRECT; STATE/CONDITIONAL/LOOPS/HITL LOSSY; CHECKPOINTING
  UNSUPPORTED.

## AWS Strands

- **Purpose:** AWS's model-driven agent SDK — an `Agent` with a model and tools
  decides how to act; multi-agent via agents-as-tools / graphs / swarms.
- **Core concepts:** `@tool` functions, `Agent(model=..., tools=[...],
  system_prompt=...)`, `BedrockModel` (and Anthropic/OpenAI/LiteLLM/Ollama model
  classes), session managers for state.
- **Execution model:** the model drives the agentic loop; there is no
  deterministic branching API and no native checkpointing.
- **Key components / terminology:** `@tool`, `Agent`, `BedrockModel`,
  `system_prompt`, agents-as-tools, `Graph`, `Swarm`.
- **Agent model:** explicit single agent (an `AgentSpec` with `model`,
  `system_prompt`, `tools`); the source graph is synthesized to **one** node
  (SINGLE_AGENT) by `AWSStrandsSourceAdapter.extract_graph`.
- **Tool model:** `@tool` (type-hint + docstring schema); `tool_style="function"`,
  bare decorator, `from strands import tool`.
- **Workflow model:** model-driven; the converter reconstructs deterministic
  order/routing when the source had a graph (see below).
- **Memory/state:** conversation/session state, not an explicit shared dict
  (LOSSY).
- **HITL:** no native pause — emulated via a blocking `request_human_approval`
  `@tool` (with a `HITL_MODE=auto` CI fast-path).
- **Converter specifics (recently hardened for genuine nativeness):**
  `AWSStrandsTargetGenerator` emits, for a source with real nodes, one
  `@tool call_<node>` per node whose body runs the **real ported node logic** on
  a shared module-level `_CTX`; `build_agent()` wires the real callables into an
  `Agent(model=BedrockModel("claude-sonnet-4"))`; and `run_agent()` is the
  **primary** entrypoint that drives the tools in the source-derived order,
  honoring the original routing and loop bounds (`_orchestration_lines` handles
  LINEAR / BRANCH / LOOP with loop-cap-constant detection). HITL adds
  `request_human_approval` to the tool list. When the source has no graph, it
  falls back to a model-driven `Agent(tools=...)`. Import-guarded by
  `_HAVE_STRANDS` with a no-op `tool` shim so it also runs offline.
  `orchestrator_must_tokens`: `def run`, `def build_agent`, `def run_agent`
  (+ `Agent`, `_HAVE_STRANDS`). `capability_matrix`: TOOLS DIRECT;
  CONDITIONAL_EDGES and CHECKPOINTING UNSUPPORTED; most others LOSSY.
  Dropped source packages: `strands-agents`, `strands-agents-tools`.
  Known semantic limit: a source `llm.py` using a non-Bedrock provider (e.g.
  Gemini) is not auto-rewritten to Bedrock — the `Agent` uses `BedrockModel`, but
  a carried client module needs a manual port.

---

## Framework integration

### Framework registration process

Two paths (see [FRAMEWORK_AUTHORING.md](FRAMEWORK_AUTHORING.md) for the full
lifecycle):

1. **Built-in class** — add a `SourceAdapter` and/or `TargetAdapter`
   (+ `TargetGenerator`) and register in the dicts in `adapters/__init__.py`
   (`SOURCE_ADAPTERS`, `TARGET_ADAPTERS`) and
   `generator/targets/__init__.py` (`TARGET_GENERATORS`).
2. **Uploaded pack** — a `frameworks/<name>/vocabulary.json` loaded at runtime as
   a `DynamicTargetAdapter`. Built-in names win over on-disk packs of the same
   name.

### Framework discovery mechanism

`adapters/__init__.py::list_frameworks_detailed()` unions the two registries with
on-disk folders containing a `vocabulary.json`, returning
`{name, display_name, source, target}` per framework. `list_frameworks()` /
`list_source_frameworks()` filter by capability. `GET /api/frameworks` returns
this list; the SPA splits it into the Source and Target dropdowns. So a new pack
folder appears in the UI with **no code change**.

### Framework configuration & metadata structure

A pack folder is:

```
frameworks/<name>/
├── vocabulary.json     # machine-readable construct map + UI metadata (required)
├── docs.md             # human-readable overview (Tier 3 grounding)
└── examples/*.py       # few-shot working agents (Tier 3 grounding)
```

Only Tier 3 reads `docs.md` / `examples/` (`tier3_llm.load_framework_docs`);
Tier 1/2 use adapters + templates.

### `vocabulary.json` schema

Observed across the four packs (all keys optional unless noted; unknown keys are
ignored, so packs may be minimal or rich):

| Key | Purpose |
|---|---|
| `_meta` | `framework`, `pypi_package`, `import_root`, `version_tested`, `purpose`, optional `source_framework`. |
| `display_name` | **UI label** (discovery). |
| `supports_source` / `supports_target` | **Booleans** that place the framework in the Source/Target dropdowns. |
| `conventions` | Drives `DynamicTargetAdapter`: `tool_decorator`, `context_class`, `plugin_class_suffix`, tool decorator import. |
| `primitives` | source-concept → target equivalent (`import`/`pattern`/`notes`), e.g. `StateGraph → WorkflowBuilder`. |
| `state` | State mapping (`TypedDict → BaseModel`, reducer rules, checkpoint storage, shared-state API). |
| `nodes` | `node_function`, `final_node`, role-specific node shapes (crewai `agent`, strands `single_agent`). |
| `edges` | `add_edge`, `add_conditional_edges`, `cycle_loop`, fan-out/in, DAG/sequential/hierarchical variants. |
| `tools` | Tool decorator + registration mapping. |
| `hitl` | HITL mechanism (or `native_support: false` + workarounds). |
| `checkpointing` | Persistence mechanism (or `native_*: false`). |
| `invocation`, `providers` | How to run; per-provider `import` + `env_vars`. |
| `capabilities` | Per-`ConstructType` support (DIRECT/LOSSY/UNSUPPORTED) → `DynamicTargetAdapter.capability_matrix`. |
| `reject_list` | `description` + `items[]`: tokens that make output a conversion error (anti-patterns, wrong-framework imports, stdlib in requirements) → reject tokens the acceptance gate enforces. |
| `dependencies` | `required[]`, `optional_by_feature{}`, `never_include_in_requirements[]`. |

Minimal viable pack: `display_name`, `supports_target: true`, and
`conventions.tool_decorator` + `conventions.context_class` +
`dependencies.required`.

### Mapping strategy & translation rules

Mapping happens in the Tier-1 rule engine (`engine/tier1_rules.py`, `R-01…R-15`)
and records *decisions*; code is emitted later by the generator. Rule map:

| Rule | Meaning |
|---|---|
| R-01 | `@tool` → target plugin method. |
| R-02 | TypedDict field → context dataclass field. |
| R-04 | LINEAR spine → sequential calls. |
| R-05 | small BRANCH (≤2 outcomes) → if/elif (3+ escalates to Tier 2/3). |
| R-06 | LOOP → `while` + counter guard; LOOP_WITH_EXIT → `while` + exit check. |
| R-08 | HITL → Tier 3 approval flow, else deterministic flagged stub (`manual_action`). |
| R-09 | `call_tool` wrapper module handling (`*_wrapper.py` excluded from tools). |
| R-10 | LLM instantiation → target invocation wrapper (fires if `llm_kwargs`). |
| R-11 / R-12 | append-only reducer fields (`audit_log`, `tool_errors`) → direct `.append()`. |
| R-13 | README heading vocabulary (`## Tools` → `## Skills`) via `TargetAdapter.README_VOCAB`. |
| R-14 | config constants carried over unchanged. |
| R-15 | `MemorySaver`/`SqliteSaver` → persistence TODO stub (`manual_action`). |

Escalation IDs: `R-TIER2` (README keyword classification), `R-TIER3` (LLM),
`None`/`Tier.UNRESOLVED` (nothing resolved). R-03/R-07 are reserved/documented
but not fired. The report buckets units accordingly.

### Template organization & variables

Deterministic output is rendered from `converter/templates/`:

| Template | Renders | Key variables |
|---|---|---|
| `agent_context.py.jinja` | state model (`BaseModel` + `advance`) | `typing_imports`, `reducer_fields`, `context_class`, `fields_block` |
| `orchestrator.py.jinja` | orchestrator skeleton | `context_class`, `config_import`, `imports`, `extra_defs`, `preamble`, `helper_functions`, `node_functions`, `run_block`, `maf_workflow` (any target's workflow block) |
| `plugin_class.py.jinja` | plugin classes (plugin_class style only) | `imports`, `tools[]` (`class_name`, `method_name`, `signature`, `body`, …) |
| `readme_maf.md.jinja` | output README | `title`, `purpose`, `framework`, `skills[]`, `workflow_pattern`, `has_hitl`, `context_fields[]`, `config_constants`, `temperature` |

Framework-specific code (the `workflow_block`) is emitted in Python by each
`TargetGenerator`, not by a template; Tier-3 `generated_code` is stitched
directly into `orchestrator.py`.

### Framework-specific generation rules (summary)

- **MAF** — real `WorkflowBuilder` graph + bundled offline SDK stub; `ChatAgent`
  for SINGLE_AGENT; `RequestInfoExecutor` HITL.
- **LangGraph** — real `StateGraph`; `MemorySaver` when checkpointer/HITL;
  public pip package (no stub).
- **CrewAI** — sequential `Crew` or branching `Flow`; emits editable prompt
  templates in `prompts/`.
- **AWS Strands** — node-tools running real ported logic on a shared `_CTX`,
  driven by a primary `run_agent()` honoring routing/loops; `BedrockModel`
  default; model-driven fallback when the source has no graph.

All four import-guard the target SDK and keep a deterministic offline `run()` so
the generated package imports and (for graph sources) runs without the SDK
installed — the converter itself never executes the output.
