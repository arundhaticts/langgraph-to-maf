"""Deterministic logic porting: source function bodies -> target function bodies.

The hardest deterministic part of a real conversion is translating a LangGraph
node (`def node(state) -> dict`) into a target context function
(`def node(ctx) -> ctx`). The logic is the same; only the state access idiom
differs. This module does that translation with an AST transform:

    state["x"]              -> ctx.x
    state.get("x")          -> ctx.x
    state.get("x", default) -> (ctx.x if ctx.x is not None else default)
    return {"a": v}         -> ctx.a = v; return ctx
    return {"log": [e]}     -> ctx.log.append(e)          (append-only fields)
    state                   -> ctx

Tool bodies are framework-agnostic, so they are carried over verbatim.
Anything the transform cannot handle falls back to a clearly-marked stub -- it
never emits invalid Python.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Optional

from converter.contracts import FunctionSpec, ToolSpec


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _StateToCtx(ast.NodeTransformer):
    """Rewrite state-dict access into attribute access on `ctx`.

    `transform_returns` controls whether `return {dict}` is rewritten into
    `ctx.x = ...; return ctx` -- correct for NODE functions (which return state
    updates) but WRONG for helpers/routers (whose dict returns are real values,
    e.g. a payload). Read-access rewrites (state["x"], state.get, bare state)
    always apply.
    """

    def __init__(self, state_param: str, append_only: set[str], transform_returns: bool = True):
        self.state_param = state_param
        self.append_only = append_only
        self.transform_returns = transform_returns

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == self.state_param:
            key = _const_str(node.slice)
            if key is not None:
                return ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id="ctx", ctx=ast.Load()),
                        attr=key,
                        ctx=node.ctx,
                    ),
                    node,
                )
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id == self.state_param
            and node.args
        ):
            key = _const_str(node.args[0])
            if key is not None:
                node.args = [self.visit(a) for a in node.args]
                attr_src = f"ctx.{key}"
                if len(node.args) >= 2:
                    default_src = ast.unparse(node.args[1])
                    expr = f"({attr_src} if {attr_src} is not None else {default_src})"
                else:
                    expr = attr_src
                return ast.copy_location(ast.parse(expr, mode="eval").body, node)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id == self.state_param:
            return ast.copy_location(ast.Name(id="ctx", ctx=node.ctx), node)
        return node

    def visit_Return(self, node: ast.Return):
        if self.transform_returns and isinstance(node.value, ast.Dict):
            stmts: list[ast.stmt] = []
            for key_node, val_node in zip(node.value.keys, node.value.values):
                key = _const_str(key_node) if key_node is not None else None
                if key is None:
                    continue
                val = self.visit(val_node)
                val_src = ast.unparse(val)
                if key in self.append_only:
                    if isinstance(val, ast.List):
                        for elt in val.elts:
                            stmts.append(ast.parse(f"ctx.{key}.append({ast.unparse(elt)})").body[0])
                    else:
                        stmts.append(ast.parse(f"ctx.{key}.extend({val_src})").body[0])
                else:
                    stmts.append(ast.parse(f"ctx.{key} = {val_src}").body[0])
            stmts.append(ast.parse("return ctx").body[0])
            for s in stmts:
                ast.copy_location(s, node)
            return stmts
        # Node returns a COMPUTED value (e.g. `out` where out is a dict built
        # above, or `{**state, ...}`). Apply it to ctx at runtime via the injected
        # `_apply_updates` helper. `return ctx` / `return None` are left as-is.
        if self.transform_returns and node.value is not None:
            val = self.visit(node.value)
            if isinstance(val, ast.Name) and val.id == "ctx":
                node.value = val
                return node
            new = ast.parse(f"return _apply_updates(ctx, {ast.unparse(val)})").body[0]
            return ast.copy_location(new, node)
        self.generic_visit(node)
        return node


def _ends_with_return(body_src: str) -> bool:
    try:
        tree = ast.parse(body_src)
    except SyntaxError:
        return False
    return bool(tree.body) and isinstance(tree.body[-1], ast.Return)


def transform_state_body(body_src: str, state_param: str, append_only: set[str]) -> str:
    """Apply the state->ctx transform to a body string; return new body source."""
    module = ast.parse(body_src)
    new_module = _StateToCtx(state_param, append_only).visit(module)
    ast.fix_missing_locations(new_module)
    return ast.unparse(new_module)


def _indent(src: str, spaces: int) -> str:
    return textwrap.indent(src, " " * spaces)


def _default_return_for(returns: Optional[str]) -> str:
    """A type-appropriate default return expression for a bodiless tool.

    Keeps the tool CALLABLE (a complete, runnable scaffold) instead of raising,
    so the converted agent can be exercised end-to-end while the human fills in
    the real logic. The annotation is matched leniently (Optional[...], list[...]
    etc. all resolve to their base container).
    """
    if not returns:
        return "None"
    r = returns.strip().lower()
    if r.startswith(("list", "sequence", "tuple", "iterable")):
        return "[]"
    if r.startswith(("dict", "mapping")):
        return "{}"
    if r.startswith("set"):
        return "set()"
    if r.startswith("str"):
        return '""'
    if r.startswith("bool"):
        return "False"
    if r.startswith(("int", "float")):
        return "0"
    if r.startswith("none"):
        return "None"
    return "None"


def plugin_method_body(tool: ToolSpec, indent: int = 8) -> str:
    """Body for a plugin method: real tool logic if available, else a runnable scaffold.

    A bodiless tool (documented in the README but with no captured source, or a
    tool whose body could not be recovered) is emitted as a COMPLETE, runnable
    scaffold that returns a type-appropriate default and logs a warning -- not a
    `raise NotImplementedError` that breaks the package the moment the tool runs.
    The `# TODO` marker still flags it for a human to complete.
    """
    lines: list[str] = []
    if tool.docstring:
        lines.append(f'"""{tool.docstring}"""')
    if tool.body:
        lines.append(tool.body)
    else:
        default = _default_return_for(tool.returns)
        lines.append(f"# TODO: port logic from source tool '{tool.name}' (scaffold below is runnable).")
        lines.append("import logging")
        lines.append(
            f"logging.getLogger(__name__).warning("
            f"\"tool '{tool.name}' is a generated scaffold; implement its real logic\")"
        )
        lines.append(f"return {default}")
    return _indent("\n".join(lines), indent)


def port_node_function(
    func: Optional[FunctionSpec],
    node_name: str,
    context_class: str,
    append_only: set[str],
) -> str:
    """A full `def node(ctx) -> ctx:` with ported body, or a stub if unavailable."""
    header = f"def {node_name}(ctx: {context_class}) -> {context_class}:"
    if func is None or not func.body:
        body = (
            f"    # TODO: port logic from source node '{node_name}'\n"
            "    return ctx"
        )
        return f"{header}\n{body}"

    state_param = func.first_param or "state"
    try:
        ported = transform_state_body(func.body, state_param, append_only)
        if not _ends_with_return(ported):
            ported += "\nreturn ctx"
        body = _indent(ported, 4)
        # Validate the whole function compiles; fall back to stub otherwise.
        ast.parse(f"{header}\n{body}")
        return f"{header}\n{body}"
    except Exception:
        body = (
            f"    # TODO: automatic port failed for node '{node_name}'; port by hand\n"
            "    return ctx"
        )
        return f"{header}\n{body}"


def _iter_arg_nodes(args: ast.arguments):
    yield from getattr(args, "posonlyargs", [])
    yield from args.args
    if args.vararg:
        yield args.vararg
    yield from args.kwonlyargs
    if args.kwarg:
        yield args.kwarg


def _references_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def port_plain_function(
    func: FunctionSpec, state_param: Optional[str], append_only: set[str]
) -> str:
    """Port a router/helper, preserving its EXACT signature (defaults/*args/**kwargs).

    If it consumes the state param (in any position), that param is renamed to
    `ctx` and read-accesses are rewritten; dict returns are NOT rewritten (a
    helper's dict return is a real value, not a state update). Non-state helpers
    are emitted unchanged.
    """
    # Preferred path: transform the full original AST so nothing in the signature
    # is lost (this is what fixes dropped **kwargs / defaults).
    if func.source:
        try:
            fn = ast.parse(func.source).body[0]
            consumes = False
            if state_param:
                for arg in _iter_arg_nodes(fn.args):
                    if arg.arg == state_param:
                        arg.arg = "ctx"
                        consumes = True
                if not consumes and _references_name(fn, state_param):
                    consumes = True
            if consumes:
                _StateToCtx(state_param, append_only, transform_returns=False).visit(fn)
                ast.fix_missing_locations(fn)
            out = ast.unparse(fn)
            ast.parse(out)  # validate
            return out
        except Exception:
            pass

    # Fallback (body-only FunctionSpec, e.g. hand-built in tests).
    params = ", ".join(p.name for p in func.params)
    ret = f" -> {func.returns}" if func.returns else ""
    prefix = "async def" if func.is_async else "def"
    if state_param is not None and func.first_param == state_param:
        renamed = ", ".join(("ctx" if i == 0 else p.name) for i, p in enumerate(func.params))
        header = f"{prefix} {func.name}({renamed}){ret}:"
        try:
            module = ast.parse(func.body or "pass")
            _StateToCtx(state_param, append_only, transform_returns=False).visit(module)
            ast.fix_missing_locations(module)
            body = _indent(ast.unparse(module), 4)
            ast.parse(f"{header}\n{body}")
            return f"{header}\n{body}"
        except Exception:
            pass
    header = f"{prefix} {func.name}({params}){ret}:"
    body = _indent(func.body or "pass", 4)
    try:
        ast.parse(f"{header}\n{body}")
        return f"{header}\n{body}"
    except Exception:
        return f"{header}\n    pass  # TODO: port helper '{func.name}' by hand"
