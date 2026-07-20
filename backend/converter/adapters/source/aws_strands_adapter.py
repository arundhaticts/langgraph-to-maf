"""AWS Strands source adapter.

Strands agents are a single tool-calling loop, not a multi-node graph: you build
`agent = Agent(model=..., tools=[...], system_prompt=...)` and invoke `agent(...)`.
There is no `StateGraph`/`add_edge` wiring and no shared TypedDict state. So this
adapter overrides `extract_graph` to synthesize ONE node for the agent (which the
IR builder classifies as SINGLE_AGENT mode, so the target wires all tools onto a
single agent). Tools (`@tool` functions), config (model constructors), imports,
and preamble are handled by the default vocabulary-driven parser via `vocabulary()`.
"""

from __future__ import annotations

import ast
from typing import Optional

from converter.adapters.base import SourceAdapter
from converter.contracts import AgentSpec, GraphEdge, GraphNode, GraphSpec, SourceVocabulary


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class AWSStrandsSourceAdapter(SourceAdapter):
    name = "aws_strands"

    def import_signatures(self) -> tuple[str, ...]:
        return ("strands",)

    def source_packages(self) -> tuple[str, ...]:
        return ("strands-agents", "strands-agents-tools")

    def vocabulary(self) -> SourceVocabulary:
        # No graph-builder methods (the agent loop is implicit); the single node
        # is synthesized in extract_graph. Tools are `@tool`; models are the
        # Strands model constructors; agent construction is kept out of preamble.
        return SourceVocabulary(
            tool_decorators=frozenset({"tool"}),
            graph_methods=frozenset(),
            llm_constructors=frozenset(
                {
                    "BedrockModel",
                    "AnthropicModel",
                    "OpenAIModel",
                    "LiteLLMModel",
                    "OllamaModel",
                    "Anthropic",
                    "OpenAI",
                }
            ),
            llm_constructor_prefixes=("Chat",),
            checkpointer_constructors=frozenset(),
            sentinels={"END": "END", "__end__": "END"},
            state_base_classes=frozenset(),
            dropped_import_roots=frozenset({"strands", "strands_tools"}),
            graph_tokens=("Agent(", ".run("),
        )

    def extract_agents(self, source: str) -> list[AgentSpec]:
        """Parse `agent_var = Agent(model=..., system_prompt=..., tools=[...])` calls."""
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
            # tools kwarg: list literal of tool names or objects
            tools_node = kw_map.get("tools")
            tool_names: list[str] = []
            if isinstance(tools_node, ast.List):
                for elt in tools_node.elts:
                    if isinstance(elt, ast.Name):
                        tool_names.append(elt.id)
            # model kwarg: may be a constructor call like BedrockModel(...)
            model_node = kw_map.get("model")
            model_str = None
            if model_node is not None:
                try:
                    model_str = ast.unparse(model_node)
                except Exception:
                    pass
            specs.append(AgentSpec(
                name=var,
                system_prompt=_str("system_prompt"),
                allowed_tools=tool_names,
                model=model_str,
            ))
        return specs

    def extract_graph(self, source: str) -> GraphSpec | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        agent_vars: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "Agent"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                agent_vars.append(node.targets[0].id)

        if not agent_vars:
            return None  # not an agent-defining file -> default parser (finds none)

        # Primary agent: the one actually invoked (`agent(...)`), else the first.
        invoked: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in agent_vars
            ):
                invoked.add(node.func.id)
        primary = next((v for v in agent_vars if v in invoked), agent_vars[0])

        # One node = the agent's tool-calling loop; edge to END. SINGLE_AGENT mode.
        graph = GraphSpec()
        graph.nodes.append(GraphNode(name=primary, target_callable=None))
        graph.entry_point = primary
        graph.edges.append(GraphEdge(source=primary, target="END"))
        return graph
