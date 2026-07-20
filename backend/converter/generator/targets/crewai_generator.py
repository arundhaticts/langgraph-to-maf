"""CrewAI target generator.

Emits genuine CrewAI orchestration -- the primary output, not a thin layer on
top of a plain-Python run(). Two strategies based on the IR:

  SEQUENTIAL (no conditional edges, no loops)
    → `Crew(agents=[agent], tasks=[...], process=Process.sequential)`.
      Each IR node becomes one `Task`. `build_crew()` is the primary runner;
      the offline `run(ctx)` synthesised by the core generator is the fallback
      when crewai is not installed.

  BRANCHING / LOOP (has conditional edges or loops)
    → A `Flow` subclass with proper `@start` / `@listen` / `@router` decorators.
      Entry node → `@start()`.  Plain successor nodes → `@listen("source")`.
      The source of a conditional edge gets a paired `@router("source")` method
      that calls the ported router function and returns the label string.
      Outcome targets → `@listen("label")` so CrewAI's event bus connects them.
      `build_flow()` is the primary runner; `run(ctx)` is the offline fallback.

HITL nodes (NodeRole.HITL) become `Task(human_input=True)` -- CrewAI pauses
execution after the agent responds and lets the human review/approve before
continuing.  An automated fast-path is preserved when `HITL_MODE=auto` via the
ported node function.

CrewAI is a public pip package, so `sdk_stub_files()` is empty.
"""

from __future__ import annotations

from converter.adapters.base import TargetAdapter
from converter.contracts import IR, NodeRole, OrchestrationPattern
from converter.generator.targets.base import TargetGenerator

_SENTINEL_NODES = frozenset({"START", "END", "__start__", "__end__"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_nodes(ir: IR) -> list:
    """Non-sentinel, non-AUX nodes."""
    wf = ir.workflow
    if not wf:
        return []
    return [n for n in wf.nodes if n.name not in _SENTINEL_NODES and n.role is not NodeRole.AUX]


def _flow_order(ir: IR, real_names: set[str]) -> list[str]:
    """Follow single-successor edges from the entry point; append any remaining."""
    wf = ir.workflow
    single_out: dict[str, str] = {}
    for e in wf.edges:
        if e.source in real_names and e.target in real_names:
            single_out.setdefault(e.source, e.target)
    order: list[str] = []
    seen: set[str] = set()
    cur = wf.entry_point if wf.entry_point in real_names else None
    while cur and cur in real_names and cur not in seen:
        order.append(cur)
        seen.add(cur)
        cur = single_out.get(cur)
    for n in wf.nodes:
        if n.name in real_names and n.name not in seen:
            order.append(n.name)
            seen.add(n.name)
    return order


def _task_var(name: str) -> str:
    return f"{name}_task"


def _agent_var(name: str) -> str:
    return f"{name}_agent"


def _is_hitl(ir: IR, node_name: str) -> bool:
    wf = ir.workflow
    if not wf:
        return False
    for n in wf.nodes:
        if n.name == node_name and n.role is NodeRole.HITL:
            return True
    return False


def _docstring_first_line(ir: IR, node_name: str) -> str:
    func = ir.functions.get(node_name)
    if func and func.docstring:
        first = func.docstring.strip().splitlines()[0]
        if first:
            return first
    return f"Run the '{node_name}' step of the converted workflow."


def _full_docstring(ir: IR, node_name: str) -> str:
    """The node's complete docstring (multi-line), or a sensible default."""
    func = ir.functions.get(node_name)
    if func and func.docstring:
        text = func.docstring.strip()
        if text:
            return text
    return f"Run the '{node_name}' step of the converted workflow."


def _node_tool_names(ir: IR, node_name: str) -> list[str]:
    """Tools the node calls in the source (so the task can instruct the agent)."""
    wf = ir.workflow
    if not wf:
        return []
    for n in wf.nodes:
        if n.name == node_name and n.calls_tools:
            return list(n.calls_tools)
    return []


def _task_description(ir: IR, node_name: str) -> str:
    """A rich CrewAI Task description: full intent + which tools to use.

    This is what makes the Crew the PRIMARY driver -- the agent is given the real
    production instruction (the ported node's docstring) plus the tools it should
    call, rather than a one-line label with the real work hidden in offline code.
    """
    parts = [_full_docstring(ir, node_name)]
    tools = _node_tool_names(ir, node_name)
    if tools:
        parts.append("Use these tools where appropriate: " + ", ".join(tools) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sequential crew (no conditional edges, no loops)
# ---------------------------------------------------------------------------

def _agent_vars_from_specs(ir: IR, all_tools: list[str]) -> list[str]:
    """Emit Agent definitions using ir.agent_specs (role/goal/backstory preserved)."""
    out: list[str] = []
    for spec in ir.agent_specs:
        role = (spec.role or f"Agent for {spec.name}").replace('"', '\\"')
        goal = (spec.goal or "Execute the assigned workflow steps.").replace('"', '\\"')
        backstory = (spec.backstory or "An agent converted from the source framework.").replace('"', '\\"')
        spec_tools = [t for t in (spec.allowed_tools or []) if t in all_tools] or all_tools
        out.append(f"        {_agent_var(spec.name)} = Agent(")
        out.append(f'            role="{role}",')
        out.append(f'            goal="{goal}",')
        out.append(f'            backstory="{backstory}",')
        tools_expr = repr(spec_tools) if spec_tools else "[]"
        out.append(f"            tools={tools_expr},")
        out.append("            allow_delegation=False,")
        out.append("        )")
    return out


def _task_from_spec(ir: IR, node_name: str, task_specs_by_name: dict, agent_var: str, is_hitl: bool) -> str:
    """Build a Task(...) line using TaskSpec when available, else synthesize."""
    spec = task_specs_by_name.get(node_name)
    if spec:
        desc = (spec.description or _task_description(ir, node_name)).replace('"', '\\"')
        expected = (spec.expected_output or f"The result of the '{node_name}' step.").replace('"', '\\"')
    else:
        desc = _task_description(ir, node_name).replace('"', '\\"')
        expected = f"Human-reviewed decision for the '{node_name}' step." if is_hitl else f"The result of the '{node_name}' step."
    parts = [
        f"description={desc!r}",
        f"expected_output={expected!r}",
        f"agent={agent_var}",
    ]
    if is_hitl:
        parts.append("human_input=True")
    return f"        {_task_var(node_name)} = Task({', '.join(parts)})"


def _sequential_crew_block(ir: IR, adapter: TargetAdapter) -> str:
    ctx_class = adapter.context_class_name
    nodes = _real_nodes(ir)
    if not nodes:
        return ""
    real = {n.name for n in nodes}
    order = _flow_order(ir, real)
    all_tools = sorted({t.name for t in ir.tools})
    # task_spec lookup by node var name
    task_specs_by_name = {s.name: s for s in ir.task_specs}
    # agent_spec lookup by name; agent for each task is the assigned agent or the first/only
    agent_specs = ir.agent_specs
    task_to_agent: dict[str, str] = {}
    for ts in ir.task_specs:
        if ts.assigned_agent:
            task_to_agent[ts.name] = _agent_var(ts.assigned_agent)
    use_specs = bool(agent_specs)

    out: list[str] = [
        "# --- CrewAI crew (primary orchestration) ---",
        "# Each converted node becomes one CrewAI Task executed by an Agent that",
        "# carries the converted @tool functions. build_crew() is the primary",
        "# entry point. run_crew() first executes the converted node logic to build",
        "# real state, then hands that state to the Crew's agent to reason over --",
        "# so the deterministic converted logic AND the CrewAI agents both run.",
        "# run(ctx) above is the offline fallback used only when crewai is absent.",
        "try:",
        "    from crewai import Agent, Task, Crew, Process",
        "    _HAVE_CREWAI = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_CREWAI = False",
        "",
        "",
        "if _HAVE_CREWAI:",
        "",
    ]
    if all_tools:
        out.append(f"    AGENT_TOOLS = [{', '.join(all_tools)}]")
        out.append("")

    out.append("    def _state_dict(ctx) -> dict:")
    out.append('        """Flatten the converted context into kickoff() inputs."""')
    out.append('        fields = getattr(type(ctx), "model_fields", {})')
    out.append("        return {f: getattr(ctx, f, None) for f in fields}")
    out.append("")
    out.append("    def build_crew() -> Crew:")
    out.append('        """Build and return the CrewAI Crew (agent + one Task per node)."""')

    if use_specs:
        # Emit one Agent per AgentSpec using real role/goal/backstory.
        out += _agent_vars_from_specs(ir, all_tools)
        # Fallback agent for nodes not covered by any spec.
        fallback_agent_var = _agent_var(agent_specs[0].name)
    else:
        # Synthesize a generic agent.
        out.append("        agent = Agent(")
        out.append('            role="Workflow Agent",')
        out.append('            goal="Execute the converted workflow steps in order.",')
        out.append('            backstory="An agent converted from the source framework.",')
        out.append(f"            tools={'AGENT_TOOLS' if all_tools else '[]'},")
        out.append("            allow_delegation=False,")
        out.append("        )")
        fallback_agent_var = "agent"

    for name in order:
        av = task_to_agent.get(name, fallback_agent_var)
        out.append(_task_from_spec(ir, name, task_specs_by_name, av, _is_hitl(ir, name)))

    task_list = ", ".join(_task_var(n) for n in order)
    if use_specs:
        agent_list = ", ".join(_agent_var(s.name) for s in agent_specs)
    else:
        agent_list = "agent"
    out.append(
        f"        return Crew(agents=[{agent_list}], tasks=[{task_list}], "
        "process=Process.sequential)"
    )
    out.append("")
    out.append("    def run_crew(inputs: dict | None = None):")
    out.append('        """Primary runner: the CrewAI Crew drives execution.')
    out.append("")
    out.append("        The Crew (agents + tasks + tools) is the production path. The")
    out.append("        deterministic offline run() is only the fallback for when no LLM")
    out.append("        is configured (e.g. CI) or the Crew raises.")
    out.append('        """')
    out.append(f"        seed = inputs or _state_dict({ctx_class}())")
    out.append("        try:")
    out.append("            return build_crew().kickoff(inputs=seed)")
    out.append("        except Exception:  # no LLM configured / offline -> deterministic fallback")
    out.append(f"            return run({ctx_class}())")
    out.append("# --- end CrewAI crew ---")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Flow-based orchestration (conditional edges / loops)
# ---------------------------------------------------------------------------

def _flow_block(ir: IR, adapter: TargetAdapter) -> str:
    """Emit a CrewAI Flow with @start / @listen / @router decorators.

    Strategy:
    - Entry node        → @start()  method
    - Plain successors  → @listen("predecessor_name") method
    - Conditional source→ additional @router("node_name") method on the same step
    - Outcome targets   → @listen("label") method (the label the router returns)
    - Each method runs the ported node function, wraps it in a per-step Task+Agent
      so the LLM can improve the output, and returns a state snapshot.
    """
    wf = ir.workflow
    ctx_class = adapter.context_class_name
    nodes = _real_nodes(ir)
    if not nodes:
        return ""
    real = {n.name for n in nodes}
    all_tools = sorted({t.name for t in ir.tools})

    # Build listener map: node_name -> list[str] events it listens to
    # (either a predecessor node name or a router label string)
    router_sources: dict[str, str] = {}  # node_name -> router function name
    router_outcomes: dict[str, dict[str, str]] = {}  # node_name -> {label: target}
    for ce in wf.conditional_edges:
        if ce.source in real:
            router_sources[ce.source] = ce.router or f"route_from_{ce.source}"
            router_outcomes[ce.source] = ce.outcomes

    # For each node compute what event(s) it listens on.
    plain_predecessors: dict[str, list[str]] = {}  # node_name -> [predecessor]
    label_listeners: dict[str, list[str]] = {}  # node_name -> [label]
    for e in wf.edges:
        if e.source in _SENTINEL_NODES or e.target in _SENTINEL_NODES:
            continue
        if e.source not in real or e.target not in real:
            continue
        plain_predecessors.setdefault(e.target, []).append(e.source)
    for src, outcomes in router_outcomes.items():
        for label, tgt in outcomes.items():
            if tgt in real:
                label_listeners.setdefault(tgt, []).append(label)

    entry = wf.entry_point if wf.entry_point in real else (nodes[0].name if nodes else None)

    out: list[str] = [
        "# --- CrewAI Flow (primary orchestration with conditional routing) ---",
        "# A Flow subclass IS the driver -- it replaces the if/else routing in run().",
        "# @start -> entry node, @listen -> successor nodes, @router -> branching.",
        "# Every method runs the converted node function against a shared context",
        "# (self.ctx) so the real converted logic executes, then runs a CrewAI",
        "# Agent+Task so an LLM reasons over the result. Routers call the converted",
        "# router function on the live context. HITL nodes use Task(human_input=True)",
        "# so CrewAI pauses for a human before continuing.",
        "try:",
        "    from crewai import Agent, Task, Crew, Process",
        "    from crewai.flow.flow import Flow, start, listen, router",
        "    _HAVE_CREWAI = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_CREWAI = False",
        "",
        "",
        "if _HAVE_CREWAI:",
        "",
    ]
    if all_tools:
        out.append(f"    AGENT_TOOLS = [{', '.join(all_tools)}]")
        out.append("")

    # Module-level helpers that bridge the CrewAI Flow to the converted node
    # functions (which live at module scope in this same orchestrator).
    out += [
        "    def _run_converted(node_name, ctx):",
        '        """Execute one converted node function against the shared context."""',
        "        fn = globals().get(node_name)",
        "        return fn(ctx) if callable(fn) else ctx",
        "",
        "    def _run_router(router_name, ctx, fallback):",
        '        """Call the converted router function; return its label (or fallback)."""',
        "        fn = globals().get(router_name)",
        "        if callable(fn):",
        "            outcome = fn(ctx)",
        "            if outcome:",
        "                return outcome",
        "        return fallback",
        "",
        "    def _record(ctx, node_name, result):",
        '        """Fold the CrewAI agent output back into the converted context."""',
        '        log = getattr(ctx, "audit_log", None)',
        "        if isinstance(log, list):",
        '            log.append({"node": node_name, "crew_output": getattr(result, "raw", str(result))})',
        "",
        "    def _apply_human(ctx, node_name, result):",
        '        """Write the human decision from a human_input Task into the context.',
        "",
        "        Sets the first matching state field so the decision actually drives",
        "        downstream routing -- not just an audit-log entry.",
        '        """',
        '        text = getattr(result, "raw", str(result))',
        '        fields = getattr(type(ctx), "model_fields", {})',
        f"        for _cand in (node_name, node_name + '_decision', 'human_decision', 'approval', 'approved'):",
        "            if _cand in fields:",
        "                setattr(ctx, _cand, text)",
        "                break",
        "        _record(ctx, node_name, result)",
        "",
    ]

    out.append("    class ConvertedFlow(Flow):")
    out.append('        """CrewAI Flow converted from the source framework (the driver)."""')
    out.append("")
    out.append(f"        def __init__(self, ctx=None):")
    out.append("            super().__init__()")
    out.append(f"            self.ctx = ctx if ctx is not None else {ctx_class}()")
    out.append("")

    # Emit agent helper — uses real AgentSpec role/goal/backstory when available.
    agent_specs = ir.agent_specs
    agent_spec_lookup = {s.name: s for s in agent_specs}
    out.append("        def _make_agent(self, role: str) -> Agent:")
    out.append('            """Create a per-step CrewAI Agent using spec data when available."""')
    if agent_specs:
        out.append("            # Use real role/goal/backstory from the source AgentSpec if found.")
        out.append("            _spec_map = {")
        for s in agent_specs:
            role_s = (s.role or s.name).replace('"', '\\"')
            goal_s = (s.goal or "Carry out the assigned step.").replace('"', '\\"')
            back_s = (s.backstory or "Converted from the source agent.").replace('"', '\\"')
            out.append(f'                {s.name!r}: ({role_s!r}, {goal_s!r}, {back_s!r}),')
        out.append("            }")
        out.append("            _r, _g, _b = _spec_map.get(role, (role, 'Carry out the assigned step.', 'Converted from the source agent.'))")
    else:
        out.append("            _r, _g, _b = role, 'Carry out the assigned step.', 'Converted from the source agent.'")
    out.append("            return Agent(")
    out.append("                role=_r,")
    out.append("                goal=_g,")
    out.append("                backstory=_b,")
    out.append(f"                tools={'AGENT_TOOLS' if all_tools else '[]'},")
    out.append("                allow_delegation=False,")
    out.append("            )")
    out.append("")

    def _dedupe(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        keep: list[str] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                keep.append(x)
        return keep

    # Every node emits its own name on completion; every router emits its outcome
    # labels. These are the ONLY events anything can listen for.
    produced_events: set[str] = set(real)
    for outcomes in router_outcomes.values():
        produced_events.update(outcomes.keys())

    # A node's triggers = the UNION of its plain-edge predecessors' emit-names AND
    # the router labels that target it, deduped. (Using union not if/elif is what
    # keeps a forward edge from being dropped when the node is also a loop target.)
    node_triggers: dict[str, list[str]] = {}
    for node in nodes:
        node_triggers[node.name] = _dedupe(
            plain_predecessors.get(node.name, []) + label_listeners.get(node.name, [])
        )

    # Emit each node as a Flow method
    for node in nodes:
        name = node.name
        hitl = node.role is NodeRole.HITL
        desc = _docstring_first_line(ir, name)
        triggers = node_triggers[name]

        if name == entry:
            out.append("        @start()")
            # Loop-back edges (a router outcome targeting the entry) are wired by
            # ALSO listening for those events -- re-entering the entry step
            # implements the source's loop natively.
            for trig in triggers:
                out.append(f"        @listen({trig!r})")
            sig = ", event=None" if triggers else ""
            out.append(f"        def {name}(self{sig}):")
        elif triggers:
            for trig in triggers:
                out.append(f"        @listen({trig!r})")
            out.append(f"        def {name}(self, event=None):")
        else:
            # No incoming edge in the source graph -- genuinely unreachable.
            out.append(f"        # NOTE: '{name}' has no incoming edge in the source graph.")
            out.append(f"        def {name}(self, event=None):")

        out.append(f'            """{desc}"""')
        # 1) Run the converted node logic on the live context (the real work,
        #    including the file-mode human pause for HITL nodes).
        out.append(f"            self.ctx = _run_converted({name!r}, self.ctx)")
        # 2) CrewAI reasoning over the step; its output is folded back into ctx.
        out.append(f"            _agent = self._make_agent({name!r})")
        out.append("            _task = Task(")
        out.append(f"                description={desc!r},")
        if hitl:
            out.append(f'                expected_output="Human-reviewed decision for {name!r}.",')
            out.append("                agent=_agent,")
            out.append("                human_input=True,")
        else:
            out.append(f'                expected_output="Result of the {name!r} step.",')
            out.append("                agent=_agent,")
        out.append("            )")
        out.append("            _crew = Crew(agents=[_agent], tasks=[_task], process=Process.sequential)")
        out.append("            try:")
        if hitl:
            # The human's decision MUST mutate the context, not just get logged.
            out.append(f"                _apply_human(self.ctx, {name!r}, _crew.kickoff())")
            out.append("            except Exception as _exc:  # surfaced, not silently swallowed")
            out.append(f"                _record(self.ctx, {name!r}, f'[human review unavailable: {{_exc}}]')")
        else:
            out.append(f"                _record(self.ctx, {name!r}, _crew.kickoff())")
            out.append("            except Exception as _exc:  # surfaced, not silently swallowed")
            out.append(f"                _record(self.ctx, {name!r}, f'[reasoning unavailable: {{_exc}}]')")
        out.append(f"            return {name!r}")
        out.append("")

        # Emit @router if this node is a conditional edge source
        if name in router_sources:
            router_fn = router_sources[name]
            outcomes = router_outcomes[name]
            labels = list(outcomes.keys())
            fallback = labels[0] if labels else "END"
            out.append(f"        @router({name!r})")
            out.append(f"        def route_from_{name}(self, event=None):")
            out.append(f'            """Route after {name}: calls the converted router on the live context."""')
            out.append(f"            return _run_router({router_fn!r}, self.ctx, {fallback!r})")
            out.append("")

    # --- Graph validation (fail-loud on an unrunnable Flow) ---
    # Every @listen trigger must be produced by some node/router, and every router
    # label must have a listener; otherwise the emitted Flow would silently stall.
    unresolved_triggers = sorted(
        {t for trigs in node_triggers.values() for t in trigs} - produced_events
    )
    listened_events = {t for trigs in node_triggers.values() for t in trigs}
    orphan_labels = sorted(
        lbl
        for outcomes in router_outcomes.values()
        for lbl, tgt in outcomes.items()
        if tgt in real and lbl not in listened_events
    )
    if unresolved_triggers or orphan_labels:
        out.append("    # !!! GRAPH VALIDATION WARNINGS (review MIGRATION_REPORT.md) !!!")
        for t in unresolved_triggers:
            out.append(f"    #   listener for {t!r} has no matching producer.")
        for lbl in orphan_labels:
            out.append(f"    #   router label {lbl!r} has no listener node.")
        out.append("")

    # Flow runner
    out.append(f"    def build_flow(ctx=None) -> ConvertedFlow:")
    out.append('        """Return a ready-to-run ConvertedFlow (the primary orchestrator)."""')
    out.append("        return ConvertedFlow(ctx)")
    out.append("")
    out.append("    def run_flow(inputs: dict | None = None):")
    out.append('        """Run the flow to completion; returns its final result."""')
    out.append("        return build_flow().kickoff()")
    out.append("# --- end CrewAI Flow ---")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Smoke test, entrypoint
# ---------------------------------------------------------------------------

def _smoke_test(adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    return (
        f'"""Smoke test for the converted {target} agent (offline-safe)."""\n\n'
        f"from agent_context import {ctx}\n"
        "import orchestrator\n\n\n"
        "def test_state_constructs():\n"
        f"    assert {ctx}() is not None\n\n\n"
        "def test_offline_entrypoint_present():\n"
        '    """The offline fast-path run() must exist even without crewai."""\n'
        '    assert callable(getattr(orchestrator, "run", None))\n\n\n'
        "def test_state_advance_returns_new_instance():\n"
        f"    a = {ctx}()\n"
        "    b = a.advance()\n"
        f"    assert isinstance(b, {ctx}) and b is not a\n"
    )


def _entrypoint(ir: IR, adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    is_api = (ir.metadata.entrypoint or "cli") == "api"
    has_flow = bool(ir.workflow and ir.workflow.conditional_edges)
    runner_fn = "build_flow" if has_flow else "build_crew"
    # Thread the request-built state into the runner (Flow: seed self.ctx; Crew:
    # pass as kickoff inputs) -- never run on an empty context.
    if has_flow:
        runner_call = "build_flow(state).kickoff()"
    else:
        runner_call = (
            "build_crew().kickoff("
            "inputs={f: getattr(state, f, None) for f in type(state).model_fields})"
        )

    lines: list[str] = [
        f'"""Generated entrypoint for the converted {target} agent."""',
        "from __future__ import annotations",
        "",
        f"from agent_context import {ctx}",
        "from orchestrator import run",
        "",
        "try:",
        f"    from orchestrator import {runner_fn}",
        "    _HAVE_CREWAI = True",
        "except ImportError:  # pragma: no cover - SDK provided at deploy time",
        "    _HAVE_CREWAI = False",
        "",
        "",
        f"def run_agent(state: {ctx} | None = None):",
        f'    """Run via the CrewAI {"Flow" if has_flow else "Crew"} when available, else offline run()."""',
        f"    state = state or {ctx}()",
        "    if _HAVE_CREWAI:",
        f"        return {runner_call}",
        "    return run(state)  # offline fast-path",
        "",
        "",
    ]

    if is_api:
        lines += [
            "from fastapi import FastAPI",
            "import uvicorn",
            "",
            "app = FastAPI()",
            "",
            "",
            '@app.post("/run")',
            "async def run_endpoint(payload: dict | None = None):",
            '    """Run the agent; converted from the source API entrypoint."""',
            f"    state = {ctx}(**(payload or {{}}))",
            "    return run_agent(state)",
            "",
            "",
            "def main():",
            '    uvicorn.run(app, host="0.0.0.0", port=8000)',
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    else:
        lines += [
            "def main():",
            "    result = run_agent()",
            "    print(result)",
            "    return result",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates (prompts/)
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_") or "agent"


def _prompt_files(ir: IR, adapter: TargetAdapter) -> dict[str, str]:
    """Editable CrewAI prompt templates: one per agent, one per task, + a README.

    CrewAI agents ARE prompts (role/goal/backstory) and tasks ARE prompts
    (description/expected_output). Emitting them as files under prompts/ makes the
    production prompts first-class, editable artifacts instead of string literals
    buried in orchestrator.py. The orchestrator still carries working defaults;
    these files document and let a human iterate on them.
    """
    files: dict[str, str] = {}
    nodes = _real_nodes(ir)
    agent_specs = ir.agent_specs

    # --- Agent prompt templates ---
    if agent_specs:
        for spec in agent_specs:
            lines = [
                f"# Agent: {spec.name}",
                "",
                f"**Role:** {spec.role or '(set the agent role)'}",
                "",
                f"**Goal:** {spec.goal or '(set the agent goal)'}",
                "",
                "**Backstory:**",
                "",
                (spec.backstory or "(set the agent backstory / persona)"),
                "",
                "**Tools available:** "
                + (", ".join(spec.allowed_tools) if spec.allowed_tools else "(none)"),
                "",
            ]
            files[f"prompts/agent_{_slug(spec.name)}.md"] = "\n".join(lines)
    else:
        files["prompts/agent_workflow.md"] = "\n".join([
            "# Agent: Workflow Agent",
            "",
            "**Role:** Workflow Agent",
            "",
            "**Goal:** Execute the converted workflow steps in order.",
            "",
            "**Backstory:**",
            "",
            (ir.metadata.description or "An agent converted from the source framework."),
            "",
            "**Tools available:** "
            + (", ".join(sorted({t.name for t in ir.tools})) or "(none)"),
            "",
        ])

    # --- Task prompt templates (one per node) ---
    task_specs = {s.name: s for s in ir.task_specs}
    for node in nodes:
        spec = task_specs.get(node.name)
        description = (spec.description if spec and spec.description else _task_description(ir, node.name))
        expected = (spec.expected_output if spec and spec.expected_output
                    else f"The result of the '{node.name}' step.")
        tools = _node_tool_names(ir, node.name)
        lines = [
            f"# Task: {node.name}",
            "",
            "## Description",
            "",
            description,
            "",
            "## Expected output",
            "",
            expected,
            "",
            "## Tools",
            "",
            (", ".join(tools) if tools else "(the agent may use any of its tools)"),
            "",
        ]
        files[f"prompts/task_{_slug(node.name)}.md"] = "\n".join(lines)

    # --- README explaining the folder ---
    listing = "\n".join(f"- `{p}`" for p in sorted(files))
    files["prompts/README.md"] = "\n".join([
        "# Prompt templates",
        "",
        "These files hold the editable prompts for the converted CrewAI agent:",
        "",
        "- `agent_*.md` — an agent's role / goal / backstory (its system prompt).",
        "- `task_*.md` — a task's description and expected output.",
        "",
        "`orchestrator.py` ships with working defaults derived from the source; edit",
        "these files (and copy the text into the corresponding `Agent(...)` / `Task(...)`",
        "call) to iterate on the prompts without touching the orchestration logic.",
        "",
        "## Files",
        "",
        listing,
        "",
    ])
    return files


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class CrewAITargetGenerator(TargetGenerator):
    """Emits CrewAI output (any source -> CrewAI Crew or Flow)."""

    name = "crewai"

    def workflow_block(self, ir: IR, adapter: TargetAdapter) -> str:
        wf = ir.workflow
        if not wf or not wf.nodes:
            return ""
        # Branching or loops → emit a Flow with proper @router / @listen
        if wf.conditional_edges or (wf.pattern in (
            OrchestrationPattern.LOOP, OrchestrationPattern.LOOP_WITH_EXIT
        )):
            return _flow_block(ir, adapter)
        # Sequential → emit a simple Crew
        return _sequential_crew_block(ir, adapter)

    def sdk_stub_files(self) -> dict[str, str]:
        return {}

    def extra_files(self, ir: IR, adapter: TargetAdapter) -> dict[str, str]:
        # Editable prompt templates under prompts/ (agents + tasks + README).
        return _prompt_files(ir, adapter)

    def smoke_test(self, adapter: TargetAdapter, target: str) -> str:
        return _smoke_test(adapter, target)

    def entrypoint(self, ir: IR, adapter: TargetAdapter, target: str) -> str:
        return _entrypoint(ir, adapter, target)

    def orchestrator_must_tokens(self, has_workflow_block: bool) -> list[str]:
        must = ["def run"]
        if has_workflow_block:
            must += ["Crew", "Task", "Process"]
        return must
