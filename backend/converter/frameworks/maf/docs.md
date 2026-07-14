# MAF (Microsoft Agent Framework) — Tier 3 Conversion Knowledge Pack

This document is injected into the Tier 3 LLM (Gemini) prompt as context when the
deterministic (Tier 1) and README (Tier 2) tiers cannot resolve a section —
namely **HITL flows** and **complex / agent-driven orchestration**.

Generated code MUST fit the conventions below so it stitches into the files the
converter already produces (`agent_context.py`, `orchestrator.py`, the plugin
modules). Do NOT invent a different module layout.

---

## 1. Target conventions the converter emits (match these exactly)

The converted agent is plain Python using these shapes:

- **Context object** — a dataclass `AgentContext` in `agent_context.py`. All agent
  state lives on it as attributes (`ctx.coverage`, `ctx.audit_log`, ...).
  Append-only list fields use `ctx.field.append(x)` (NOT reducer returns).

- **Node functions** — a node is a function that takes and returns the context:

  ```python
  def generate(ctx: AgentContext) -> AgentContext:
      ...
      return ctx
  ```

- **Router functions** — return a string label used for branching:

  ```python
  def route(ctx: AgentContext) -> str:
      return "done" if ctx.coverage >= COVERAGE_FLOOR else "revise"
  ```

- **Skills / tools** — plugin classes with `@kernel_function` methods:

  ```python
  from semantic_kernel.functions import kernel_function

  class ReadTestsPlugin:
      @kernel_function(description="Reads test files from the repo.")
      def read_tests(self, path: str) -> list:
          ...
  ```

- **Orchestration entry point** — a single `run(ctx)` function that drives the
  node/router functions and returns the final context:

  ```python
  def run(ctx: AgentContext) -> AgentContext:
      ...
      return ctx
  ```

Constants (e.g. `MAX_GEN_RETRIES`, `COVERAGE_FLOOR`) live in `config.py` and are
imported into `orchestrator.py`. Reference them by name; do not hard-code values.

---

## 2. LangGraph → MAF mapping reference

| LangGraph (source)                         | MAF target (this converter)                    |
|--------------------------------------------|------------------------------------------------|
| `StateGraph(State)`                        | a `run(ctx)` orchestration function            |
| `TypedDict` state                          | `@dataclass AgentContext`                       |
| `Annotated[list, add]` reducer field       | plain `list`, updated via `ctx.field.append()`  |
| node `def n(state) -> dict`                | `def n(ctx: AgentContext) -> AgentContext`      |
| `state["x"]` / `state.get("x")`            | `ctx.x`                                         |
| `return {"x": v}`                          | `ctx.x = v; return ctx`                          |
| `add_edge(a, b)`                           | sequential call `ctx = b(ctx)` after `a`        |
| `add_conditional_edges(src, router, map)`  | `outcome = router(ctx)` + `if/elif` dispatch    |
| generation/validation loop                 | `while guard < CAP:` with a router break        |
| `@tool` function                           | `@kernel_function` plugin method                |
| `interrupt(...)` (HITL)                    | approval flow raising `HumanApprovalRequired`   |
| `MemorySaver` / checkpointer               | persistence layer (out of scope; TODO stub)     |

---

## 3. Orchestration patterns

### Linear
```python
def run(ctx: AgentContext) -> AgentContext:
    ctx = read(ctx)
    ctx = analyze(ctx)
    return ctx
```

### Branch (2 outcomes)
```python
def run(ctx: AgentContext) -> AgentContext:
    ctx = gate(ctx)
    outcome = route(ctx)
    if outcome == "revise":
        ctx = generate(ctx)
    else:
        ctx = finish(ctx)
    return ctx
```

### Loop with exit (generation/validation)
```python
def run(ctx: AgentContext) -> AgentContext:
    guard = 0
    while guard < MAX_GEN_RETRIES:
        ctx = generate(ctx)
        ctx = gate(ctx)
        if route(ctx) == "done":
            break
        guard += 1
    return ctx
```

### Complex / agent-driven (3+ outcomes or dynamic routing)
Dispatch on the router label; keep every branch a call that returns `ctx`. If the
router can pick tools dynamically, dispatch by name into the plugin methods. Keep
a bounded guard so the loop always terminates.

---

## 4. HITL (human-in-the-loop) flows

The source used `interrupt(payload)` to pause for human input. In the converted
agent, model this as an **approval boundary**. Two acceptable shapes:

**(a) Raise for an external approver to catch** (default; matches the stub):
```python
def hitl_approve(ctx: AgentContext) -> AgentContext:
    if not ctx.approved:
        raise HumanApprovalRequired({"coverage": ctx.coverage})
    ctx.audit_log.append("human approved")
    return ctx
```

**(b) Callback/hook** if the target app supplies an approval function:
```python
def hitl_approve(ctx: AgentContext, approver=None) -> AgentContext:
    decision = approver(ctx) if approver else False
    ctx.audit_log.append(f"human decision: {decision}")
    return ctx
```

`HumanApprovalRequired` is already defined in `orchestrator.py`. Preserve the
original decision-handling logic from the source node; only translate the
`interrupt()` call itself into the approval boundary.

---

## 5. Tier 3 output contract (STRICT)

Respond with strict JSON, no prose outside it:

```json
{
  "pattern": "loop_with_exit",
  "generated_code": "def run(ctx: AgentContext) -> AgentContext:\n    ...\n    return ctx",
  "reasoning": "why this structure; note anything the human must verify",
  "confidence": 0.0
}
```

Rules for `generated_code`:
- For orchestration requests: return ONLY the `run(ctx)` function.
- For HITL requests: return ONLY the **body** of the node function (statements,
  indented at 4 spaces is fine — the converter re-indents), ending in `return ctx`.
- Use the context attribute idiom (`ctx.x`), append-only `.append()`, and config
  constants by name.
- Do not include imports or class definitions unless strictly required.
- `confidence` in [0,1]; below the configured threshold flags the section for
  human review in the migration report.
