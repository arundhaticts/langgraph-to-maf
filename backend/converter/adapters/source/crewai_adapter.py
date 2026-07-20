"""CrewAI source adapter.

CrewAI models work differently from LangGraph: there is no `StateGraph` with
`add_node`/`add_edge` calls. Instead a crew is assembled from constructor calls --
`Agent(role=..., tools=[...])`, `Task(description=..., agent=..., context=[...])`,
and `Crew(agents=[...], tasks=[...], process=...)`. So this adapter overrides
`extract_graph` to parse that constructor shape into the neutral `GraphSpec`:

- each `Task(...)` bound to a variable becomes a node,
- a sequential crew (or no explicit deps) becomes linear edges in crew order,
- `context=[taskA, taskB]` on a task becomes dependency edges taskA/taskB -> task.

Tools (`@tool` functions), config (LLM constructors), imports, and preamble are
still handled by the default vocabulary-driven parser via `vocabulary()` below --
those constructs look the same as anywhere else. State is left empty: CrewAI has
no shared TypedDict state (task outputs flow implicitly).
"""

from __future__ import annotations

import ast
from typing import Optional

from converter.adapters.base import SourceAdapter
from converter.contracts import AgentSpec, GraphEdge, GraphNode, GraphSpec, SourceVocabulary, TaskSpec


def _call_name(node: ast.AST) -> Optional[str]:
    """Trailing name of a call/attribute: `Task(...)` -> 'Task', `x.Crew` -> 'Crew'."""
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _name_list(node: Optional[ast.AST]) -> list[str]:
    """The bare names inside a list literal (`[t1, t2]` -> ['t1', 't2'])."""
    if not isinstance(node, ast.List):
        return []
    return [e.id for e in node.elts if isinstance(e, ast.Name)]


class CrewAISourceAdapter(SourceAdapter):
    name = "crewai"

    def import_signatures(self) -> tuple[str, ...]:
        return ("crewai",)

    def source_packages(self) -> tuple[str, ...]:
        return ("crewai", "crewai-tools")

    def vocabulary(self) -> SourceVocabulary:
        # Graph is parsed by extract_graph() below, so graph_methods is empty.
        # Tools are still `@tool`; LLMs are `LLM(...)` / Chat*; crew wiring tokens
        # keep Crew/Task/Agent construction out of the carried preamble.
        return SourceVocabulary(
            tool_decorators=frozenset({"tool"}),
            graph_methods=frozenset(),
            llm_constructors=frozenset(
                {"LLM", "OpenAI", "AzureOpenAI", "Anthropic", "AnthropicLLM"}
            ),
            llm_constructor_prefixes=("Chat",),
            checkpointer_constructors=frozenset(),
            sentinels={"END": "END", "__end__": "END"},
            state_base_classes=frozenset(),
            dropped_import_roots=frozenset({"crewai"}),
            graph_tokens=("Crew(", "Task(", "Agent(", ".kickoff("),
        )

    def extract_agents(self, source: str) -> list[AgentSpec]:
        """Parse `agent_var = Agent(role=..., goal=..., backstory=..., tools=[...])` calls."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        specs: list[AgentSpec] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "Agent"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            var = node.targets[0].id
            kw_map: dict[str, ast.expr] = {kw.arg: kw.value for kw in node.value.keywords if kw.arg}
            def _str(k: str) -> Optional[str]:
                v = kw_map.get(k)
                if v is None:
                    return None
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    return v.value
                try:
                    return ast.unparse(v)
                except Exception:
                    return None
            tools = _name_list(kw_map.get("tools")) if "tools" in kw_map else []
            llm_node = kw_map.get("llm")
            llm_str = None
            if llm_node is not None:
                try:
                    llm_str = ast.unparse(llm_node)
                except Exception:
                    pass
            specs.append(AgentSpec(
                name=var,
                role=_str("role"),
                goal=_str("goal"),
                backstory=_str("backstory"),
                allowed_tools=tools,
                llm=llm_str,
            ))
        return specs

    def extract_tasks(self, source: str) -> list[TaskSpec]:
        """Parse `task_var = Task(description=..., expected_output=..., agent=..., context=[...])` calls."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        specs: list[TaskSpec] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "Task"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            var = node.targets[0].id
            kw_map: dict[str, ast.expr] = {kw.arg: kw.value for kw in node.value.keywords if kw.arg}
            def _str(k: str) -> Optional[str]:
                v = kw_map.get(k)
                if v is None:
                    return None
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    return v.value
                try:
                    return ast.unparse(v)
                except Exception:
                    return None
            agent_node = kw_map.get("agent")
            agent_var = None
            if isinstance(agent_node, ast.Name):
                agent_var = agent_node.id
            depends_on = _name_list(kw_map.get("context")) if "context" in kw_map else []
            specs.append(TaskSpec(
                name=var,
                description=_str("description"),
                expected_output=_str("expected_output"),
                assigned_agent=agent_var,
                depends_on=depends_on,
            ))
        return specs

    def extract_graph(self, source: str) -> GraphSpec | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        tasks: dict[str, list[str]] = {}  # task var -> its context (dependency) vars
        order: list[str] = []             # task definition order
        crew_tasks: list[str] | None = None
        has_crew = False

        for node in ast.walk(tree):
            # `t = Task(..., context=[...])`
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "Task"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                var = node.targets[0].id
                ctx: list[str] = []
                for kw in node.value.keywords:
                    if kw.arg == "context":
                        ctx = _name_list(kw.value)
                tasks[var] = ctx
                order.append(var)
            # `Crew(tasks=[...])`
            elif isinstance(node, ast.Call) and _call_name(node) == "Crew":
                has_crew = True
                for kw in node.keywords:
                    if kw.arg == "tasks":
                        crew_tasks = _name_list(kw.value)

        # Not a crew-defining file -> let the default parser have a go (finds none).
        if not tasks and not has_crew:
            return None

        # Node order: crew's task list wins; append any tasks defined but not listed.
        node_order = list(crew_tasks) if crew_tasks else list(order)
        for var in order:
            if var not in node_order:
                node_order.append(var)

        graph = GraphSpec()
        for var in node_order:
            graph.nodes.append(GraphNode(name=var, target_callable=None))
        if not node_order:
            return graph

        has_deps = any(tasks.get(var) for var in node_order)
        if has_deps:
            # Dependency graph: edges from each context task into the dependent task.
            depended_on: set[str] = set()
            for var in node_order:
                for dep in tasks.get(var, []):
                    graph.edges.append(GraphEdge(source=dep, target=var))
                    depended_on.add(dep)
            entries = [v for v in node_order if not tasks.get(v)] or node_order[:1]
            graph.entry_point = entries[0]
            for var in node_order:
                if var not in depended_on:  # nothing depends on it -> terminal
                    graph.edges.append(GraphEdge(source=var, target="END"))
        else:
            # Sequential crew: linear chain in crew order.
            for src, tgt in zip(node_order, node_order[1:]):
                graph.edges.append(GraphEdge(source=src, target=tgt))
            graph.entry_point = node_order[0]
            graph.edges.append(GraphEdge(source=node_order[-1], target="END"))

        return graph
