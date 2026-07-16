"""Module 2 -- Code parser: AST extraction of tools, graph, state, config.

Four extractors, one per construct, each operating on a single file's source
string via `ast.parse` (build-plan rule: no regex on Python code):

- `extract_tools`   -- `@tool` functions -> `ToolSpec`
- `extract_graph`   -- `add_node` / `add_edge` / `add_conditional_edges` /
                       `set_entry_point` calls -> `GraphSpec`
- `extract_state`   -- `TypedDict` fields, detecting `Annotated[list, add]`
- `extract_config`  -- LLM constructor kwargs + `os.getenv` env vars

These produce the frozen contracts and stay source-framework aware but
target-agnostic. Module 4 runs them across every file and consolidates. LangGraph
is the source today; the call-name tables below are the only framework-specific
knowledge and are easy to extend.
"""

from __future__ import annotations

import ast
from typing import Any, Optional

from converter.contracts import (
    ConditionalEdge,
    ConfigSpec,
    FunctionSpec,
    GraphEdge,
    GraphNode,
    GraphSpec,
    StateField,
    ToolParam,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Framework-specific vocabulary (the only source-framework knowledge here)
# ---------------------------------------------------------------------------

# Decorator names that mark a tool function.
_TOOL_DECORATORS = frozenset({"tool"})

# Graph-builder method names we care about.
_GRAPH_METHODS = frozenset(
    {"add_node", "add_edge", "add_conditional_edges", "set_entry_point", "set_finish_point"}
)

# LLM constructor call names (exact matches). Anything starting with "Chat" is
# also treated as an LLM constructor (ChatOpenAI, ChatAnthropic, ...).
_LLM_CONSTRUCTORS = frozenset(
    {
        "init_chat_model",
        "OpenAI",
        "AzureOpenAI",
        "Anthropic",
        "AnthropicLLM",
    }
)

# Persistence / checkpointer constructors (LangGraph savers).
_CHECKPOINTER_CONSTRUCTORS = frozenset(
    {"MemorySaver", "SqliteSaver", "AsyncSqliteSaver", "PostgresSaver", "AsyncPostgresSaver"}
)

# LangGraph sentinel node names, normalised for the IR.
_SENTINELS = {"START": "START", "__start__": "START", "END": "END", "__end__": "END"}


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------

def _safe_unparse(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return None


def _callable_name(node: ast.AST) -> Optional[str]:
    """The trailing name of a callable/decorator expression.

    `tool` -> "tool", `tool(...)` -> "tool", `lc.tools.tool` -> "tool".
    """
    if isinstance(node, ast.Call):
        return _callable_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal(node: Optional[ast.AST]) -> Any:
    """Python value of a literal node, else its unparsed source string."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return _safe_unparse(node)


def _node_ref(node: ast.AST) -> Optional[str]:
    """Resolve a node reference (edge endpoint / mapping target) to a name.

    Handles string literals, START/END sentinels, and bare names/attributes.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _SENTINELS.get(node.value, node.value)
    if isinstance(node, ast.Name):
        return _SENTINELS.get(node.id, node.id)
    return _safe_unparse(node)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _is_tool(func: ast.AST) -> bool:
    node = func  # FunctionDef / AsyncFunctionDef
    for dec in getattr(node, "decorator_list", []):
        if _callable_name(dec) in _TOOL_DECORATORS:
            return True
    return False


def _function_body(node: ast.AST) -> Optional[str]:
    """Source of a function's body statements (docstring stripped), or None.

    Uses `ast.unparse`, so it returns clean, valid Python at column 0.
    """
    stmts = list(getattr(node, "body", []))
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]  # drop the docstring
    if not stmts:
        return None
    try:
        return "\n".join(ast.unparse(s) for s in stmts)
    except Exception:  # pragma: no cover - defensive
        return None


def _decorator_names(node: ast.AST) -> list[str]:
    return [
        name
        for name in (_callable_name(d) for d in getattr(node, "decorator_list", []))
        if name
    ]


def _extract_params(args: ast.arguments) -> list[ToolParam]:
    params: list[ToolParam] = []
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    num_without_default = len(positional) - len(args.defaults)

    for i, arg in enumerate(positional):
        if arg.arg in ("self", "cls"):
            continue
        default = None
        default_index = i - num_without_default
        if default_index >= 0:
            default = _safe_unparse(args.defaults[default_index])
        params.append(
            ToolParam(
                name=arg.arg,
                annotation=_safe_unparse(arg.annotation),
                default=default,
            )
        )

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg in ("self", "cls"):
            continue
        params.append(
            ToolParam(
                name=arg.arg,
                annotation=_safe_unparse(arg.annotation),
                default=_safe_unparse(default_node) if default_node else None,
            )
        )

    return params


def extract_tools(source: str, source_file: Optional[str] = None) -> list[ToolSpec]:
    """Find `@tool`-decorated functions and describe them."""
    tree = ast.parse(source)
    tools: list[ToolSpec] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_tool(node):
            tools.append(
                ToolSpec(
                    name=node.name,
                    params=_extract_params(node.args),
                    docstring=ast.get_docstring(node),
                    returns=_safe_unparse(node.returns),
                    source_file=source_file,
                    body=_function_body(node),
                    signature=_safe_unparse(node.args),
                )
            )
    return tools


def extract_functions(
    source: str, source_file: Optional[str] = None
) -> dict[str, FunctionSpec]:
    """Every top-level function definition, keyed by name (for logic porting)."""
    tree = ast.parse(source)
    functions: dict[str, FunctionSpec] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = FunctionSpec(
                name=node.name,
                params=_extract_params(node.args),
                body=_function_body(node),
                returns=_safe_unparse(node.returns),
                docstring=ast.get_docstring(node),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                decorators=_decorator_names(node),
                source_file=source_file,
                source=_safe_unparse(node),
                signature=_safe_unparse(node.args),
            )
    return functions


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _graph_method(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Attribute) and call.func.attr in _GRAPH_METHODS:
        return call.func.attr
    return None


def _handle_add_node(call: ast.Call, graph: GraphSpec) -> None:
    args = call.args
    if not args:
        return
    first = args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        name = first.value
        target = _callable_name(args[1]) if len(args) > 1 else None
    else:
        # add_node(action) -- node name is the callable's name.
        target = _callable_name(first)
        name = target
    if name:
        graph.nodes.append(GraphNode(name=name, target_callable=target))


def _handle_add_edge(call: ast.Call, graph: GraphSpec) -> None:
    if len(call.args) < 2:
        return
    source = _node_ref(call.args[0])
    target = _node_ref(call.args[1])
    if source and target:
        graph.edges.append(GraphEdge(source=source, target=target))
        if source == "START" and graph.entry_point is None:
            graph.entry_point = target


def _handle_conditional_edges(call: ast.Call, graph: GraphSpec) -> None:
    if not call.args:
        return
    source = _node_ref(call.args[0])
    router = _callable_name(call.args[1]) if len(call.args) > 1 else None

    outcomes: dict[str, str] = {}
    mapping = call.args[2] if len(call.args) > 2 else None
    if isinstance(mapping, ast.Dict):
        for key_node, value_node in zip(mapping.keys, mapping.values):
            key = _literal(key_node)
            value = _node_ref(value_node) if value_node is not None else None
            if key is not None and value is not None:
                outcomes[str(key)] = value

    if source:
        graph.conditional_edges.append(
            ConditionalEdge(source=source, router=router, outcomes=outcomes)
        )


def extract_graph(source: str) -> GraphSpec:
    """Find graph-builder calls and assemble a `GraphSpec`."""
    tree = ast.parse(source)
    graph = GraphSpec()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method = _graph_method(node)
        if method == "add_node":
            _handle_add_node(node, graph)
        elif method == "add_edge":
            _handle_add_edge(node, graph)
        elif method == "add_conditional_edges":
            _handle_conditional_edges(node, graph)
        elif method == "set_entry_point" and node.args:
            graph.entry_point = _node_ref(node.args[0])
        elif method == "set_finish_point" and node.args:
            finish = _node_ref(node.args[0])
            if finish:
                graph.edges.append(GraphEdge(source=finish, target="END"))

    return graph


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _is_typeddict_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if _callable_name(base) == "TypedDict":
            return True
    return False


def _is_append_only(annotation: Optional[ast.AST]) -> bool:
    """True for `Annotated[list, add]` / `Annotated[list[...], operator.add]`."""
    if not isinstance(annotation, ast.Subscript):
        return False
    if _callable_name(annotation.value) != "Annotated":
        return False
    # Collect metadata elements after the first (the type).
    slc = annotation.slice
    elements = slc.elts if isinstance(slc, ast.Tuple) else [slc]
    for meta in elements[1:]:
        name = _callable_name(meta)
        if name in ("add", "operator.add"):
            return True
        # `operator.add` shows as Attribute(attr="add"); _callable_name gets "add".
    return False


def extract_state_class_names(source: str) -> list[str]:
    """Names of TypedDict state classes defined at module level."""
    tree = ast.parse(source)
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _is_typeddict_class(node)
    ]


def extract_state(source: str) -> list[StateField]:
    """Find TypedDict state schemas and extract their fields."""
    tree = ast.parse(source)
    fields: list[StateField] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_typeddict_class(node):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(
                        StateField(
                            name=stmt.target.id,
                            type=_safe_unparse(stmt.annotation) or "Any",
                            is_append_only=_is_append_only(stmt.annotation),
                            default=_safe_unparse(stmt.value) if stmt.value else None,
                        )
                    )

    return fields


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _is_llm_constructor(call: ast.Call) -> bool:
    name = _callable_name(call)
    if name is None:
        return False
    return name in _LLM_CONSTRUCTORS or name.startswith("Chat")


def _env_var_from_call(call: ast.Call) -> Optional[str]:
    """`os.getenv("X")` / `os.environ.get("X")` -> "X"."""
    if not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr in ("getenv", "get") and call.args:
        value = _literal(call.args[0])
        if isinstance(value, str):
            return value
    return None


def extract_config(source: str) -> ConfigSpec:
    """Extract LLM kwargs, env vars, and module-level constants."""
    tree = ast.parse(source)
    config = ConfigSpec()

    for node in ast.walk(tree):
        # os.getenv / os.environ.get / os.environ["X"]
        if isinstance(node, ast.Call):
            env = _env_var_from_call(node)
            if env and env not in config.env_vars:
                config.env_vars.append(env)
            if _is_llm_constructor(node):
                if config.llm_provider is None:
                    config.llm_provider = _callable_name(node)
                for kw in node.keywords:
                    if kw.arg is None:  # **kwargs spread
                        continue
                    config.llm_kwargs[kw.arg] = _literal(kw.value)
            elif config.checkpointer is None:
                saver = _callable_name(node)
                if saver in _CHECKPOINTER_CONSTRUCTORS:
                    config.checkpointer = saver
        elif isinstance(node, ast.Subscript):
            # os.environ["X"]
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                if node.slice.value not in config.env_vars:
                    config.env_vars.append(node.slice.value)

    # Module-level ALL_CAPS constants (safety blockers, caps, etc.).
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    config.constants[target.id] = _literal(stmt.value)

    # Surface temperature explicitly (never invented -- None if absent).
    temp = config.llm_kwargs.get("temperature")
    if isinstance(temp, (int, float)):
        config.temperature = float(temp)

    return config


# ---------------------------------------------------------------------------
# Imports and module preamble (carried into the output so ported bodies resolve)
# ---------------------------------------------------------------------------

# Import roots dropped from the output (source-framework wiring, not logic).
_DROPPED_IMPORT_ROOTS = frozenset({"langgraph"})

# Tokens that mark a statement as graph wiring rather than portable setup.
_GRAPH_TOKENS = (
    "StateGraph",
    "MessageGraph",
    ".add_node",
    ".add_edge",
    ".add_conditional_edges",
    ".set_entry_point",
    ".set_finish_point",
    ".compile(",
)


def extract_imports(source: str) -> list[str]:
    """Top-level import lines to carry over (source-framework imports dropped)."""
    tree = ast.parse(source)
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            kept = [a for a in node.names if a.name.split(".")[0] not in _DROPPED_IMPORT_ROOTS]
            if kept:
                new = ast.Import(names=kept)
                lines.append(ast.unparse(new))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in _DROPPED_IMPORT_ROOTS:
                continue
            lines.append(ast.unparse(node))
    return lines


def extract_preamble(source: str) -> list[str]:
    """Module-level setup assignments (e.g. `llm = ChatOpenAI(...)`).

    Excludes ALL_CAPS constants (they go to config.py) and graph-wiring
    statements (StateGraph construction, add_node/edge, compile, ...).
    """
    tree = ast.parse(source)
    lines: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        # Skip ALL_CAPS constants (handled as config).
        if all(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            continue
        rendered = ast.unparse(node)
        if any(token in rendered for token in _GRAPH_TOKENS):
            continue
        lines.append(rendered)
    return lines
