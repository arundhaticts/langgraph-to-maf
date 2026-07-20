"""Module 7 -- Output repo generator.

Writes the converted agent to the output folder:

- `copy_through` files are copied verbatim at the same relative path.
- Tools become plugin modules (rendered from `plugin_class.py.jinja`) at the same
  relative path as their source file.
- The state schema becomes `agent_context.py` and the graph becomes
  `orchestrator.py`, with any Tier 3 `generated_code` stitched into the
  orchestrator.
- Config constants are carried into `config.py`.

Every generated `.py` is checked with `ast.parse`. On failure the file is still
written but prefixed with a `# SYNTAX ERROR` banner and recorded so the report
flags it. A light "target framework validation" pass confirms the expected
framework constructs are present.

Framework-specific naming/idioms come from the `TargetAdapter`; the pipeline
core stays framework-agnostic.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from converter.adapters import get_source_adapter, get_target_adapter
from converter.adapters.base import TargetAdapter
from converter.config import Config
from converter.contracts import (
    IR,
    ConditionalEdge,
    ConversionResult,
    ConversionUnit,
    FileAction,
    FunctionSpec,
    NodeRole,
    OrchestrationPattern,
    StateField,
    Tier,
    ToolSpec,
)
from converter.generator.body_porter import (
    plugin_method_body,
    port_node_function,
    port_plain_function,
)
from converter.generator.targets import TargetGenerator, get_target_generator

_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
)
_SYNTAX_BANNER = "# SYNTAX ERROR - review required\n"

# Tokens marking a source function as graph wiring (excluded from the emitted
# orchestrator -- the run() is synthesized from the IR instead).
_GRAPH_ASSEMBLY_TOKENS = (
    "StateGraph", "MessageGraph", ".add_node", ".add_edge",
    ".add_conditional_edges", ".set_entry_point", ".set_finish_point",
    ".compile(", "MemorySaver", "SqliteSaver",
)

# Entrypoint modules: their functions (run/initial_state/main/...) are NOT node
# helpers and must not be dumped into the orchestrator. Pass 4 generates a fresh
# entrypoint instead.
_ENTRYPOINT_MODULES = frozenset(
    {"main.py", "api.py", "app.py", "cli.py", "server.py", "run.py", "__main__.py"}
)


@dataclass
class GenerationResult:
    """Module 7 output: what was written and what needs attention."""

    output_root: str
    written_files: list[str] = field(default_factory=list)
    copied_files: list[str] = field(default_factory=list)
    syntax_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=True
    )


# ---------------------------------------------------------------------------
# Cooked-block builders (kept in Python; templates stay thin and robust)
# ---------------------------------------------------------------------------

def _tool_signature(tool: ToolSpec) -> str:
    # Prefer the exact original signature (preserves defaults / *args / **kwargs).
    if tool.signature:
        return f", {tool.signature}"
    parts = []
    for p in tool.params:
        piece = p.name
        if p.annotation:
            piece += f": {p.annotation}"
        if p.default is not None:
            piece += f" = {p.default}"
        parts.append(piece)
    return (", " + ", ".join(parts)) if parts else ""


def _tool_context(tool: ToolSpec, adapter: TargetAdapter) -> dict:
    return {
        "class_name": adapter.plugin_class_name(tool.name),
        "method_name": adapter.method_name(tool.name),
        "source_name": tool.name,
        "description_repr": repr(tool.docstring or tool.name),
        "signature": _tool_signature(tool),
        "return_annotation": f" -> {tool.returns}" if tool.returns else "",
        "body": plugin_method_body(tool, indent=8),
    }


# typing symbols we import into agent_context.py when a field type uses them.
_TYPING_SYMBOLS = (
    "Optional", "Any", "Literal", "Union", "List", "Dict", "Tuple", "Set",
    "Callable", "Annotated", "Sequence", "Mapping",
)


def _typing_imports(ir: IR) -> str:
    """Import exactly the typing symbols the emitted field types reference."""
    used = {"Any", "Optional"}  # the template always emits these
    for f in ir.state:
        text = f.type or ""
        for sym in _TYPING_SYMBOLS:
            if sym in text:
                used.add(sym)
    return "from typing import " + ", ".join(sorted(used))


# Names allowed to appear in a state field annotation emitted into a pydantic
# model. Anything referencing a name outside this set is type-erased to `Any`
# so the generated BaseModel always resolves its annotations (pydantic evaluates
# them at class-definition time, unlike a plain dataclass).
_PYDANTIC_SAFE_NAMES = frozenset(
    {
        "str", "int", "float", "bool", "bytes", "complex", "list", "dict",
        "tuple", "set", "frozenset", "bytearray", "None", "NoneType", "object",
        "Any", "Optional", "Union", "List", "Dict", "Tuple", "Set", "Callable",
        "Annotated", "Sequence", "Mapping", "Literal", "Iterable", "Iterator",
    }
)

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _pydantic_safe_type(type_str: str) -> str:
    """Return `type_str` if every identifier it names is resolvable, else "Any".

    Literal string/number members are not identifiers, so `Literal['a', 'b']`
    stays intact; a custom class the output would not import becomes `Any`.
    """
    text = (type_str or "").strip()
    if not text:
        return "Any"
    # Identifiers that are attribute members (x.y -> y) are always safe to ignore
    # for resolvability of the leading name; check every bare identifier.
    for match in _IDENTIFIER_RE.finditer(text):
        name = match.group(0)
        # Skip attribute members: if preceded by '.', it's an attribute access.
        start = match.start()
        if start > 0 and text[start - 1] == ".":
            continue
        if name not in _PYDANTIC_SAFE_NAMES:
            return "Any"
    return text


def _state_field_decl(f: StateField) -> str:
    if f.is_append_only:
        return (
            f"    {f.name}: list = Field(default_factory=list)"
            f"  # was Annotated[{f.type}, add]; .advance() extends it"
        )
    base = _pydantic_safe_type(f.type or "Any")
    return f"    {f.name}: Optional[{base}] = None"


def _append_only(ir: IR) -> set[str]:
    return {f.name for f in ir.state if f.is_append_only}


def _local_roots(ir: IR) -> set[str]:
    """Top-level module/package names internal to the source repo."""
    roots: set[str] = set()
    for entry in ir.files:
        p = entry.relative_path.replace("\\", "/")
        if not p.endswith(".py"):
            continue
        segs = p.split("/")
        roots.add(segs[0] if len(segs) > 1 else segs[0][:-3])
    return roots


def _import_root(line: str) -> str | None:
    tokens = line.replace(",", " ").split()
    if tokens[:1] == ["import"] and len(tokens) > 1:
        return tokens[1].split(".")[0]
    if tokens[:1] == ["from"] and len(tokens) > 1:
        return tokens[1].split(".")[0]
    return None


def _external_imports(ir: IR) -> list[str]:
    """Source imports minus internal cross-module ones (legacy; superseded by
    import rewriting for the consolidated tree)."""
    local = _local_roots(ir)
    return [line for line in ir.imports if _import_root(line) not in local]


# ---------------------------------------------------------------------------
# Pass 6: import rewriting so the consolidated tree resolves across modules
# ---------------------------------------------------------------------------

def _source_root(ir: IR) -> str | None:
    """The source's top-level package (e.g. 'src'), or None for a flat repo."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for entry in ir.files:
        p = entry.relative_path.replace("\\", "/")
        if p.endswith(".py"):
            segs = p.split("/")
            if len(segs) > 1:
                counts[segs[0]] += 1
    return counts.most_common(1)[0][0] if counts else None


def _rewrite_module(mod: str, source_root: str | None) -> str | None:
    """Map a source module path to its consolidated-tree path.

    src.state -> agent_context ; src.config -> config ; src.tools.X -> plugins.X ;
    src.<rest> -> <rest>. Returns None to signal "drop this import" (bare `import
    src`). Modules outside the source root are returned unchanged.
    """
    parts = mod.split(".")
    if not source_root or parts[0] != source_root:
        return mod
    rest = parts[1:]
    if not rest:
        return None
    if rest[0] == "state":
        return "agent_context"
    if rest[0] == "config":
        return "config"
    if rest[0] == "tools":
        return ".".join(["plugins"] + rest[1:]) if len(rest) > 1 else "plugins"
    return ".".join(rest)


_IMPORT_RE = re.compile(r"^(?P<indent>\s*)from\s+(?P<mod>[\w.]+)\s+import\s+(?P<names>.+)$")
_IMPORT_PLAIN_RE = re.compile(r"^(?P<indent>\s*)import\s+(?P<mod>[\w.]+)(?P<tail>.*)$")


def _rewrite_import_line(line: str, source_root: str, inlined: set[str]) -> str | None:
    """Rewrite one import line for the consolidated tree, or None to drop it.

    Drops langgraph imports and any imported name that is defined inline in the
    orchestrator (avoids clashing with the inlined node/helper defs).
    """
    m = _IMPORT_RE.match(line)
    if m:
        mod = m.group("mod")
        if mod.split(".")[0] == "langgraph":
            return None
        new_mod = _rewrite_module(mod, source_root)
        if new_mod is None:
            return None
        raw = m.group("names").replace("(", "").replace(")", "").strip().rstrip(",")
        names = [n.strip() for n in raw.split(",") if n.strip()]
        kept = [n for n in names if n.split(" as ")[0].strip() not in inlined]
        if not kept:
            return None
        return f"{m.group('indent')}from {new_mod} import {', '.join(kept)}"
    m = _IMPORT_PLAIN_RE.match(line)
    if m:
        mod = m.group("mod")
        if mod.split(".")[0] == "langgraph":
            return None
        new_mod = _rewrite_module(mod, source_root)
        if new_mod is None:
            return None
        return f"{m.group('indent')}import {new_mod}{m.group('tail')}"
    return line


def _rewrite_path_anchors(content: str, source_root: str | None) -> str:
    """Re-base source-relative path string literals onto the flattened output tree.

    The output drops the source's top package dir (e.g. `src/`), so a literal like
    "src/prompts/system.md" must become "prompts/system.md" to still resolve at
    runtime. Only string literals whose FIRST path segment is the source root are
    rewritten; everything else is left untouched.
    """
    if not source_root:
        return content
    pattern = re.compile(r"([\"'])" + re.escape(source_root) + r"([/\\])")
    return pattern.sub(r"\1", content)


def _rewrite_imports_in_text(content: str, source_root: str, inlined: set[str]) -> str:
    """Rewrite/drop every import line in a block of generated code."""
    out: list[str] = []
    for line in content.splitlines():
        if line.lstrip().startswith(("from ", "import ")):
            new = _rewrite_import_line(line, source_root, inlined)
            if new is None:
                continue
            out.append(new)
        else:
            out.append(line)
    return "\n".join(out)


import builtins as _builtins

_BUILTIN_NAMES = frozenset(dir(_builtins)) | {"__name__", "__file__", "self"}


def _names_used(text: str) -> set[str]:
    """Every identifier referenced in a block of code (Name + attribute roots)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Best-effort: fall back to a word scan so curation still trims obvious junk.
        return set(re.findall(r"[A-Za-z_]\w*", text))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
    return used


def _import_bound_names(line: str) -> list[str]:
    """Names an import statement binds into scope (asname wins)."""
    try:
        node = ast.parse(line.strip()).body[0]
    except (SyntaxError, IndexError):
        return []
    out: list[str] = []
    if isinstance(node, ast.Import):
        for a in node.names:
            out.append((a.asname or a.name.split(".")[0]))
    elif isinstance(node, ast.ImportFrom):
        for a in node.names:
            out.append(a.asname or a.name)
    return out


def _curate_imports(lines: list[str], used: set[str], config_consts) -> str:
    """Keep only import lines whose bound names are actually used in the body.

    - Drops `from __future__` (rendered once at the top of the template).
    - For `from config import ...`, keeps only names config.py actually defines
      AND that are used -- this kills the ImportError from blind-merged constants.
    - Rebuilds `from X import a, b, c` with just the used subset (trims residue
      like `TypedDict, Annotated`). De-duplicates.
    """
    config_consts = set(config_consts or ())
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        try:
            node = ast.parse(line.strip()).body[0]
        except (SyntaxError, IndexError):
            continue
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "__future__":
                continue
            if mod.split(".")[0] == "langgraph":
                continue
            kept = []
            for a in node.names:
                bound = a.asname or a.name
                if bound not in used:
                    continue
                if mod == "config" and a.name not in config_consts:
                    continue  # config.py never defines this -> would ImportError
                kept.append(a.name if not a.asname else f"{a.name} as {a.asname}")
            if not kept:
                continue
            rebuilt = f"from {mod} import {', '.join(kept)}"
        elif isinstance(node, ast.Import):
            kept_aliases = []
            for a in node.names:
                if a.name.split(".")[0] == "langgraph":
                    continue
                bound = a.asname or a.name.split(".")[0]
                if bound in used:
                    kept_aliases.append(a.name if not a.asname else f"{a.name} as {a.asname}")
            if not kept_aliases:
                continue
            rebuilt = f"import {', '.join(kept_aliases)}"
        else:
            continue
        if rebuilt not in seen:
            seen.add(rebuilt)
            out.append(rebuilt)
    return "\n".join(out)


def _curate_preamble(preamble: list[str], used: set[str], available: set[str]) -> str:
    """Keep module-level preamble lines that are safe and actually used.

    A line is kept only if it assigns a name used in the body AND every name it
    references is resolvable (imported, a builtin, or defined earlier). This drops
    blindly-merged junk like `_default = VectorStore()` (VectorStore not imported)
    and a stray `app = FastAPI(...)` (unused), which caused NameErrors at import.
    """
    resolvable = set(available) | set(_BUILTIN_NAMES)
    out: list[str] = []
    block = "\n".join(preamble)
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return ""  # can't reason about it safely -> drop rather than break import
    for stmt in tree.body:
        targets: list[str] = []
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            targets.append(stmt.target.id)
        else:
            continue
        if not targets or not any(t in used for t in targets):
            continue
        refs = {
            n.id for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and n.id not in targets
        }
        if not refs <= resolvable:
            continue
        out.append(ast.unparse(stmt))
        resolvable.update(targets)
    return "\n".join(out)


def _node_or_router_callables(ir: IR) -> set[str]:
    names: set[str] = set()
    wf = ir.workflow
    if wf:
        for n in wf.nodes:
            names.add(n.name)
            if n.target_callable:
                names.add(n.target_callable)
        for c in wf.conditional_edges:
            if c.router:
                names.add(c.router)
    return names


def _node_files(ir: IR) -> set[str]:
    """Source files that define a node or router (their non-node functions are
    co-located helpers, inlined; the file itself is NOT copied as support)."""
    callables = _node_or_router_callables(ir)
    return {
        f.source_file
        for name, f in ir.functions.items()
        if name in callables and f.source_file
    }


class _ImportRewriter(ast.NodeTransformer):
    """Rewrite/drop imports throughout a copied support module."""

    def __init__(self, source_root: str | None):
        self.root = source_root

    def visit_Import(self, node: ast.Import):
        kept = []
        for alias in node.names:
            if alias.name.split(".")[0] == "langgraph":
                continue
            new = _rewrite_module(alias.name, self.root)
            if new:
                kept.append(ast.alias(name=new, asname=alias.asname))
        if not kept:
            return None
        node.names = kept
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level:
            return node
        if (node.module or "").split(".")[0] == "langgraph":
            return None
        new = _rewrite_module(node.module or "", self.root)
        if new is None:
            return None
        node.module = new
        return node


def _rewrite_support_module(source: str, source_root: str | None) -> str:
    tree = ast.parse(source)
    _ImportRewriter(source_root).visit(tree)
    ast.fix_missing_locations(tree)
    return _rewrite_path_anchors(ast.unparse(tree), source_root)


def _state_param(ir: IR) -> str:
    """The parameter name source node functions use for the state dict."""
    workflow = ir.workflow
    if workflow:
        for node in workflow.nodes:
            func = _func_for_node(ir, node)
            if func and func.first_param:
                return func.first_param
    return "state"


def _func_for_node(ir: IR, node) -> FunctionSpec | None:
    return ir.functions.get(node.target_callable or "") or ir.functions.get(node.name)


def _hitl_unit(conversion: ConversionResult | None, node_name: str) -> ConversionUnit | None:
    if conversion is None:
        return None
    for u in conversion.units:
        if u.rule_id == "R-08" and u.source_ref == node_name:
            return u
    return None


def _node_functions(
    ir: IR, adapter: TargetAdapter, conversion: ConversionResult | None
) -> str:
    blocks: list[str] = []
    workflow = ir.workflow
    nodes = workflow.nodes if workflow else []
    ctx_class = adapter.context_class_name
    append_only = _append_only(ir)

    has_audit_log = any(f.name == "audit_log" for f in ir.state)
    for node in nodes:
        if node.role is NodeRole.HITL:
            unit = _hitl_unit(conversion, node.name)
            blocks.append(_hitl_block(ir, node, unit, ctx_class, has_audit_log))
        else:
            func = _func_for_node(ir, node)
            blocks.append(
                port_node_function(func, node.name, ctx_class, append_only)
            )

    return "\n\n\n".join(blocks) if blocks else "# (no graph nodes detected)"


def _comment_lines(src: str, indent: int = 4) -> str:
    """Comment out every line of `src` at the given indent."""
    pad = " " * indent
    out = []
    for line in src.splitlines():
        out.append(f"{pad}# {line}" if line.strip() else f"{pad}#")
    return "\n".join(out)


def _audit_append(node_name: str, event: str, has_audit_helper: bool, indent: str) -> str:
    """One audit_log.append(...) line -- via the source audit() helper if present."""
    if has_audit_helper:
        return f'{indent}ctx.audit_log.append(audit("{node_name}", "{event}"))'
    return f'{indent}ctx.audit_log.append({{"node": "{node_name}", "event": "{event}"}})'


def _hitl_block(
    ir: IR,
    node,
    unit: ConversionUnit | None,
    ctx_class: str,
    has_audit_log: bool,
) -> str:
    """HITL node with a REAL, opt-in human pause + an auto-approve default.

    Default (`HITL_MODE=auto`): auto-approves so the agent runs end-to-end.
    `HITL_MODE=file`: writes `<node>.request.json`, waits for `<node>.response.json`
    (a dict of state updates), applies it, and continues -- a genuine "LLM writes,
    human reviews" pause with no extra infrastructure. Times out to auto-approve
    after HITL_TIMEOUT_SECONDS. The source/Gemini-suggested approval logic is kept
    as a reference comment to fold into the file-mode branch.
    """
    name = node.name
    has_audit_helper = "audit" in ir.functions
    appr = _audit_append(name, "human_approved", has_audit_helper, "                ") if has_audit_log else "                pass"
    tmo = _audit_append(name, "timeout_auto_approved", has_audit_helper, "        ") if has_audit_log else "        pass"
    auto = _audit_append(name, "auto_approved", has_audit_helper, "    ") if has_audit_log else "    pass"

    body = [
        f"def {name}(ctx: {ctx_class}) -> {ctx_class}:",
        '    """Human-in-the-loop node (converted).',
        "",
        "    Default auto-approves (HITL_MODE=auto) so the agent runs end-to-end.",
        f"    Set HITL_MODE=file for a real review: writes {name}.request.json, waits",
        f"    for {name}.response.json (a dict of state updates), applies it, continues.",
        '    """',
        '    import os',
        '    if os.environ.get("HITL_MODE", "auto").lower() == "file":',
        "        import json, pathlib, time",
        f'        _req = pathlib.Path("{name}.request.json")',
        f'        _resp = pathlib.Path("{name}.response.json")',
        "        _snap = {f: getattr(ctx, f, None) for f in type(ctx).model_fields}",
        "        _req.write_text(json.dumps(_snap, indent=2, default=str))",
        "        for _ in range(HITL_TIMEOUT_SECONDS):",
        "            if _resp.exists():",
        "                _decision = json.loads(_resp.read_text())",
        "                _resp.unlink()",
        "                for _k, _v in _decision.items():",
        "                    setattr(ctx, _k, _v)",
        appr,
        "                return ctx",
        "            time.sleep(1)",
        tmo,
        "        return ctx",
        auto,
        "    return ctx",
    ]

    if unit and unit.generated_code:
        real_src = unit.generated_code.rstrip()
        origin = "Gemini-suggested approval logic; fold into the HITL_MODE=file branch"
    else:
        real_src = (
            "# The source's real approval logic is recorded in MIGRATION_REPORT.md.\n"
            "# Apply the human decision to ctx (the file-mode branch above shows how),\n"
            "# e.g. set the approved fields from _decision, then return ctx."
        )
        origin = "reference: implement in the HITL_MODE=file branch"

    return "\n".join(
        body
        + [
            "",
            f"# --- HUMAN APPROVAL REFERENCE for '{name}' ({origin}) ---",
            _comment_lines(real_src, indent=0),
            "# --- end human approval reference ---",
        ]
    )


def _helper_functions(ir: IR, adapter: TargetAdapter) -> str:
    """Emit routers and other top-level helpers (ported if they consume state)."""
    workflow = ir.workflow
    tool_names = {t.name for t in ir.tools}
    node_callables = set()
    if workflow:
        for node in workflow.nodes:
            node_callables.add(node.name)
            if node.target_callable:
                node_callables.add(node.target_callable)

    state_param = _state_param(ir)
    append_only = _append_only(ir)
    node_files = _node_files(ir)
    blocks: list[str] = []
    for name, func in ir.functions.items():
        if name in tool_names or name in node_callables:
            continue
        # Skip entrypoint-module functions (run/initial_state/main/...).
        base = os.path.basename(func.source_file or "")
        if base in _ENTRYPOINT_MODULES:
            continue
        # Skip graph-assembly functions (build_graph, make_checkpointer, ...):
        # the orchestration is synthesized from the IR, not copied verbatim.
        if func.source and any(tok in func.source for tok in _GRAPH_ASSEMBLY_TOKENS):
            continue
        # Only inline routers + helpers CO-LOCATED with nodes. Functions from
        # pure-support modules (nlp/memory/llm/util/...) are copied as modules
        # (Pass 6) and imported, not inlined -- so their module-level classes and
        # state stay intact. (source_file None => flat/hand-built -> inline.)
        if func.source_file and func.source_file not in node_files:
            continue
        blocks.append(port_plain_function(func, state_param, append_only))
    return "\n\n\n".join(blocks)


def _ordered_nodes(ir: IR) -> list[str]:
    workflow = ir.workflow
    if not workflow:
        return []
    names = {n.name for n in workflow.nodes}
    single_out = {}
    for e in workflow.edges:
        single_out.setdefault(e.source, e.target)
    order: list[str] = []
    seen: set[str] = set()
    cur = workflow.entry_point
    while cur and cur in names and cur not in seen:
        order.append(cur)
        seen.add(cur)
        cur = single_out.get(cur)
    for n in workflow.nodes:
        if n.name not in seen:
            order.append(n.name)
    return order


def _loop_cap_constant(ir: IR) -> str | None:
    """A real cap constant name if one looks like a loop bound, else None."""
    for name in ir.config.constants:
        upper = name.upper()
        if any(tok in upper for tok in ("RETRIES", "RETRY", "MAX", "ITER", "LIMIT")):
            return name
    return None


def _flow_nodes(ir: IR) -> list[str]:
    """Ordered nodes that participate in the main flow (HITL/AUX excluded)."""
    roles = {n.name: n.role for n in (ir.workflow.nodes if ir.workflow else [])}
    return [
        n
        for n in _ordered_nodes(ir)
        if roles.get(n) not in (NodeRole.HITL, NodeRole.AUX)
    ]


def _exit_label(cond: ConditionalEdge) -> Optional[str]:
    for label, target in cond.outcomes.items():
        if target in ("END", "__end__"):
            return label
    return None


def _run_block(ir: IR, workflow_unit: Optional[ConversionUnit]) -> str:
    """Build the run() function, or stitch Tier 3 generated code."""
    ctx = "ctx"
    # Tier 3 (Gemini): stitch the generated orchestration verbatim.
    if workflow_unit and workflow_unit.generated_code:
        return (
            "# --- Tier 3 (Gemini) generated orchestration (stitched) ---\n"
            f"{workflow_unit.generated_code.rstrip()}\n"
            "# --- end Tier 3 section ---"
        )

    # Unresolved: leave an explicit manual stub.
    if workflow_unit and workflow_unit.tier is Tier.UNRESOLVED:
        return (
            "def run(ctx):\n"
            "    # TODO: orchestration could not be converted automatically\n"
            "    #       (see MIGRATION_REPORT.md).\n"
            "    raise NotImplementedError('Convert the orchestration by hand.')"
        )

    workflow = ir.workflow
    pattern = workflow.pattern if workflow else OrchestrationPattern.LINEAR
    flow = _flow_nodes(ir)

    if pattern in (OrchestrationPattern.LOOP, OrchestrationPattern.LOOP_WITH_EXIT):
        cap = _loop_cap_constant(ir)
        calls = "\n".join(f"        {ctx} = {n}({ctx})" for n in flow) or "        pass"
        lines = ["def run(ctx):", "    guard = 0"]
        if cap is None:
            lines.append("    _cap = 10  # TODO: confirm the real loop cap")
            cap = "_cap"
        lines += [
            f"    while guard < {cap}:",
            calls,
        ]
        cond = workflow.conditional_edges[0] if (workflow and workflow.conditional_edges) else None
        exit_label = _exit_label(cond) if cond else None
        if cond and cond.router and exit_label:
            lines.append(f"        outcome = {cond.router}(ctx)")
            lines.append(f'        if outcome == "{exit_label}":')
            lines.append("            break")
        else:
            lines.append("        # TODO: break on the real exit condition")
        lines.append("        guard += 1")
        lines.append("    return ctx")
        return "\n".join(lines)

    if pattern is OrchestrationPattern.BRANCH and workflow and workflow.conditional_edges:
        cond = workflow.conditional_edges[0]
        lines = ["def run(ctx):"]
        for n in flow:
            lines.append(f"    {ctx} = {n}({ctx})")
        router = cond.router or "route"
        lines.append(f"    outcome = {router}(ctx)")
        for i, (label, target) in enumerate(cond.outcomes.items()):
            kw = "if" if i == 0 else "elif"
            lines.append(f'    {kw} outcome == "{label}":')
            if target in ("END", "__end__"):
                lines.append("        return ctx")
            else:
                lines.append(f"        {ctx} = {target}({ctx})")
        lines.append("    return ctx")
        return "\n".join(lines)

    # Linear (default).
    calls = "\n".join(f"    {ctx} = {n}({ctx})" for n in flow) or "    pass"
    return f"def run(ctx):\n{calls}\n    return ctx"


# ---------------------------------------------------------------------------
# Writing + validation
# ---------------------------------------------------------------------------

def _write_python(
    result: GenerationResult, rel_path: str, content: str
) -> None:
    abs_path = os.path.join(result.output_root, rel_path)
    os.makedirs(os.path.dirname(abs_path) or result.output_root, exist_ok=True)
    try:
        ast.parse(content)
    except SyntaxError:
        content = _SYNTAX_BANNER + content
        result.syntax_errors.append(rel_path)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    result.written_files.append(rel_path)


def _validate_target(result: GenerationResult, rel_path: str, content: str, must: list[str]) -> None:
    for token in must:
        if token not in content:
            result.validation_warnings.append(
                f"{rel_path}: expected target construct '{token}' not found."
            )


# ---------------------------------------------------------------------------
# requirements.txt -- auto-detected from the CONVERTED output's actual imports
# ---------------------------------------------------------------------------

_STDLIB_ROOTS = frozenset({
    "os", "sys", "re", "json", "math", "time", "typing", "dataclasses",
    "collections", "itertools", "functools", "pathlib", "operator", "datetime",
    "enum", "abc", "logging", "random", "asyncio", "io", "contextlib", "types",
    "textwrap", "uuid", "hashlib", "subprocess", "dataclasses", "string",
    "warnings", "copy", "inspect", "traceback", "argparse", "sqlite3", "csv",
    "__future__", "annotations", "ast", "importlib", "pkgutil", "glob",
    "shutil", "tempfile", "base64", "secrets", "threading", "queue", "socket",
    "struct", "decimal", "statistics", "pprint", "difflib", "unittest", "signal",
    "concurrent", "urllib", "http", "xml", "html", "shlex", "getpass", "platform",
    "zipfile", "gzip", "tarfile", "mimetypes", "fnmatch", "bisect", "heapq",
    "weakref", "gc", "select", "errno", "stat", "posixpath", "ntpath", "unicodedata",
    "zlib", "binascii", "codecs", "locale", "calendar", "numbers", "fractions",
})
# Modules the consolidated package defines itself (never a pip dependency).
_INTERNAL_MODULES = frozenset(
    {"agent_context", "orchestrator", "config", "plugins", "main", "skills"}
)
# Best-effort import-root -> pip package name (only where they differ).
_PIP_NAME = {
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
    "google": "google-genai",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "semantic_kernel": "semantic-kernel",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "pillow",
}


def _import_roots_in(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _write_text(result: GenerationResult, rel_path: str, content: str) -> None:
    abs_path = os.path.join(result.output_root, rel_path)
    os.makedirs(os.path.dirname(abs_path) or result.output_root, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    result.written_files.append(rel_path)


def _write_env_example(result: GenerationResult, ir: IR) -> None:
    """Emit .env.example with one placeholder line per env var found in the source."""
    vars_found: list[str] = list(ir.config.env_vars) if ir.config.env_vars else []
    if not vars_found:
        return
    lines = ["# Copy this file to .env and fill in your values.", ""]
    for var in vars_found:
        lines.append(f"{var}=<your-value-here>")
    lines.append("")
    _write_text(result, ".env.example", "\n".join(lines))


def _write_requirements(result: GenerationResult, ir: IR, adapter: TargetAdapter) -> None:
    """Write requirements.txt from what the converted code ACTUALLY imports.

    Scans the generated .py files' import roots (ground truth of runtime deps),
    maps them to pip names, drops stdlib / internal / source-framework packages,
    and adds the target framework's own runtime packages.
    """
    source_adapter = get_source_adapter(ir.metadata.source_framework)
    drop_roots = set(source_adapter.import_signatures()) if source_adapter else set()
    drop_pkgs = set(source_adapter.source_packages()) if source_adapter else set()

    # Local modules = every top-level module/package present in the output tree
    # (agent_context, config, plugins, nlp, memory, ...) -- never pip deps.
    local_modules = set(_INTERNAL_MODULES)
    for rel in result.written_files:
        if rel.endswith(".py"):
            segs = rel.replace("\\", "/").split("/")
            local_modules.add(segs[0] if len(segs) > 1 else segs[0][:-3])

    packages: set[str] = set()
    for rel in result.written_files:
        if not rel.endswith(".py"):
            continue
        source = open(os.path.join(result.output_root, rel), encoding="utf-8").read()
        for root in _import_roots_in(source):
            if root in _STDLIB_ROOTS or root in local_modules or root in drop_roots:
                continue
            packages.add(_PIP_NAME.get(root, root.replace("_", "-")))

    packages -= drop_pkgs
    packages |= set(adapter.runtime_requirements())

    header = [
        "# Auto-generated for the converted "
        f"{(ir.metadata.target_framework or 'target').upper()} agent.",
        "# Detected from the converted code's imports; the source framework's",
        "# packages were dropped and the target framework's added.",
        "",
    ]
    _write_text(result, "requirements.txt", "\n".join(header + sorted(packages)) + "\n")


# ---------------------------------------------------------------------------
# Plugins (tools -> plugin module: ported plain functions + @kernel_function classes)
# ---------------------------------------------------------------------------

def _decorator_shim(adapter: TargetAdapter) -> str:
    """Import the target's tool decorator, with a no-op fallback for offline runs."""
    return (
        "try:\n"
        f"    {adapter.tool_decorator_import()}\n"
        "except ImportError:  # pragma: no cover - target SDK provided at deploy time\n"
        f"    def {adapter.tool_decorator()}(*args, **kwargs):\n"
        "        # Supports both bare `@dec` and called `@dec(...)` usage offline.\n"
        "        if len(args) == 1 and callable(args[0]) and not kwargs:\n"
        "            return args[0]\n"
        "        def _wrap(fn):\n"
        "            return fn\n"
        "        return _wrap\n"
    )


def _plugin_module_name(source_file: str | None) -> str:
    base = os.path.basename(source_file or "skills.py")
    return base[:-3] if base.endswith(".py") else "skills"


def _tool_plain_function(tool: ToolSpec, adapter: TargetAdapter | None = None) -> str:
    sig = tool.signature or ", ".join(p.name for p in tool.params)
    ret = f" -> {tool.returns}" if tool.returns else ""
    lines: list[str] = []
    # Function-style targets (e.g. MAF @ai_function) decorate the function itself,
    # so it IS the registered tool -- no separate wrapper class needed.
    if adapter is not None and adapter.tool_style() == "function":
        lines.append(f"@{adapter.tool_decorator_call(repr(tool.docstring or tool.name))}")
    lines.append(f"def {tool.name}({sig}){ret}:")
    if tool.docstring:
        lines.append(f'    """{tool.docstring}"""')
    if tool.body:
        lines.append(textwrap.indent(tool.body, "    "))
    else:
        lines.append(f"    # TODO: port logic from source tool '{tool.name}'")
        lines.append("    raise NotImplementedError")
    return "\n".join(lines)


def _tool_plugin_class(tool: ToolSpec, adapter: TargetAdapter) -> str:
    sig = tool.signature or ", ".join(p.name for p in tool.params)
    method_params = f"self, {sig}" if sig else "self"
    ret = f" -> {tool.returns}" if tool.returns else ""
    call_args = ", ".join(p.name for p in tool.params)
    decorator = adapter.tool_decorator()
    return (
        f"class {adapter.plugin_class_name(tool.name)}:\n"
        f'    """Skill converted from source tool \'{tool.name}\'."""\n\n'
        f"    @{decorator}(description={tool.docstring or tool.name!r})\n"
        f"    def {adapter.method_name(tool.name)}({method_params}){ret}:\n"
        f"        return {tool.name}({call_args})\n"
    )


def _plugin_module(tools: list[ToolSpec], adapter: TargetAdapter, imports_block: str) -> str:
    parts = [
        '"""Generated skills (converted from source tool functions)."""',
        "from __future__ import annotations",
    ]
    if imports_block.strip():
        parts.append(imports_block)
    parts.append("\n" + _decorator_shim(adapter))
    function_style = adapter.tool_style() == "function"
    for tool in tools:
        parts.append("\n" + _tool_plain_function(tool, adapter))
    # Plugin-class-style targets also emit a wrapper class; function-style targets
    # do NOT (the decorated function above is the registered tool -- no dead class).
    if not function_style:
        for tool in tools:
            parts.append("\n" + _tool_plugin_class(tool, adapter))
    return "\n".join(parts) + "\n"


_SECRET_BASENAMES = frozenset({".env", ".env.local", ".env.production", "credentials", "credentials.json"})
_SECRET_SUFFIXES = (".pem", ".key", ".pfx", ".p12")


def _is_secret_file(rel: str) -> bool:
    """True for files that must never be copied into the output (secrets)."""
    base = os.path.basename(rel.replace("\\", "/")).lower()
    if base in _SECRET_BASENAMES or base.startswith(".env"):
        return True
    if base.startswith("id_rsa") or base.startswith("secrets"):
        return True
    return base.endswith(_SECRET_SUFFIXES)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(
    ir: IR,
    conversion: ConversionResult,
    output_path: str,
    config: Config | None = None,
    adapter: TargetAdapter | None = None,
    generator: TargetGenerator | None = None,
) -> GenerationResult:
    """Render and write ONE coherent target package. Returns a `GenerationResult`.

    Layout: agent_context.py, config.py, plugins/<mod>.py (+__init__), orchestrator.py,
    main.py. The original source tree is NOT reproduced -- the output is a
    self-contained target package. All framework-specific code (the orchestration
    graph, the SDK stub, the entrypoint, the smoke test) is emitted by the
    resolved `TargetGenerator`; everything else here is framework-agnostic.
    """
    config = config or Config()
    target_name = ir.metadata.target_framework or "maf"
    adapter = adapter or get_target_adapter(target_name)
    generator = generator or get_target_generator(target_name)
    target = target_name.upper()
    env = _env()
    result = GenerationResult(output_root=os.path.abspath(output_path))
    os.makedirs(result.output_root, exist_ok=True)
    imports_block = "\n".join(_external_imports(ir))

    # 1. Plugins: one module per source tools file (plain functions + plugin classes).
    by_file: dict[str, list[ToolSpec]] = {}
    for tool in ir.tools:
        by_file.setdefault(tool.source_file or "skills.py", []).append(tool)
    tool_import_lines: list[str] = []
    if by_file:
        _write_python(result, os.path.join("plugins", "__init__.py"), "")
        for source_file, tools in by_file.items():
            mod = _plugin_module_name(source_file)
            rel = os.path.join("plugins", f"{mod}.py")
            content = _plugin_module(tools, adapter, imports_block)
            _write_python(result, rel, content)
            must = [adapter.tool_decorator()]
            if adapter.tool_style() != "function":
                must.append("class ")
            _validate_target(result, rel, content, must)
            names = ", ".join(sorted({t.name for t in tools}))
            tool_import_lines.append(f"from plugins.{mod} import {names}")

    # 2. Context dataclass (+ backward-compat aliases for the source state names).
    fields_block = (
        "\n".join(_state_field_decl(f) for f in ir.state) if ir.state else "    pass"
    )
    append_only = sorted(_append_only(ir))
    ctx_rendered = env.get_template("agent_context.py.jinja").render(
        context_class=adapter.context_class_name,
        fields_block=fields_block,
        typing_imports=_typing_imports(ir),
        reducer_fields=(repr(set(append_only)) if append_only else "set()"),
    )
    aliases = [
        f"{name} = {adapter.context_class_name}"
        for name in ir.state_class_names
        if name != adapter.context_class_name
    ]
    if aliases:
        ctx_rendered += (
            "\n\n# Backward-compatible aliases for the original state class name(s).\n"
            + "\n".join(aliases)
            + "\n"
        )
    _write_python(result, "agent_context.py", ctx_rendered)
    _validate_target(result, "agent_context.py", ctx_rendered, ["BaseModel", "def advance"])

    # 3. config.py from carried constants.
    if ir.config.constants:
        lines = ['"""Generated config (carried from the source)."""', ""]
        for name, value in ir.config.constants.items():
            lines.append(f"{name} = {value!r}")
        _write_python(result, "config.py", "\n".join(lines) + "\n")

    # 4. Orchestrator (ported nodes/routers/helpers + synthesized run()).
    workflow_unit = next(
        (u for u in conversion.units if u.source_ref == "workflow"), None
    )
    has_hitl = any(u.rule_id == "R-08" for u in conversion.units) or any(
        n.role is NodeRole.HITL for n in (ir.workflow.nodes if ir.workflow else [])
    )
    has_checkpointer = any(u.rule_id == "R-15" for u in conversion.units)
    # Runtime helper: applies a node's returned dict of state updates onto ctx.
    # Makes nodes that `return out` / `return {**state, ...}` work as ctx-returning
    # functions without per-node data-flow analysis.
    append_only_names = sorted(_append_only(ir))
    extra_defs_parts = [
        "\n\n_REDUCER_FIELDS = " + (repr(set(append_only_names)) if append_only_names else "set()"),
        "\n\n\ndef _apply_updates(ctx, updates):\n"
        '    """Apply a node\'s returned {field: value} dict onto the context object."""\n'
        "    if updates is ctx or not isinstance(updates, dict):\n"
        "        return ctx\n"
        "    for _k, _v in updates.items():\n"
        "        _cur = getattr(ctx, _k, None)\n"
        "        if _k in _REDUCER_FIELDS and isinstance(_cur, list):\n"
        "            _cur.extend(_v if isinstance(_v, list) else [_v])\n"
        "        else:\n"
        "            setattr(ctx, _k, _v)\n"
        "    return ctx",
    ]
    if has_hitl:
        extra_defs_parts.append(
            "\n\n# Human-in-the-loop pause budget (seconds) for HITL_MODE=file nodes.\n"
            "HITL_TIMEOUT_SECONDS = 600\n\n\n"
            "class HumanApprovalRequired(Exception):\n"
            '    """Raised where the source paused for human-in-the-loop approval."""'
        )
    if has_checkpointer:
        extra_defs_parts.append(
            "\n\n# TODO: wire persistence layer (source used a checkpointer/saver)"
        )

    helper_block = _helper_functions(ir, adapter)
    node_block = _node_functions(ir, adapter, conversion)
    run_block = _run_block(ir, workflow_unit)
    # The target framework's orchestration graph (empty for Tier 3 stitched
    # orchestration, which brings its own control flow). Emitted by the resolved
    # TargetGenerator -- the variable name is kept for template compatibility.
    maf_block = "" if (workflow_unit and workflow_unit.generated_code) else generator.workflow_block(ir, adapter)
    # Names to DROP from carried imports: functions defined inline here, all node/
    # router callables (inlined under node names, imported under target_callables),
    # and graph-assembly functions (their modules are dropped, not copied).
    inlined = set(re.findall(r"^def (\w+)", helper_block + "\n" + node_block, re.M))
    inlined |= _node_or_router_callables(ir)
    inlined |= {
        name
        for name, f in ir.functions.items()
        if f.source and any(tok in f.source for tok in _GRAPH_ASSEMBLY_TOKENS)
    }

    # Carry the source modules' imports, rewritten for the consolidated tree
    # (src.* -> flat / agent_context / config / plugins; langgraph dropped;
    # inlined names dropped) + explicit tool imports from plugins.
    source_root = _source_root(ir)
    rewritten = []
    for line in ir.imports:
        new = _rewrite_import_line(line, source_root, inlined) if source_root else line
        if new:
            rewritten.append(new)

    # Curate the header against what the body ACTUALLY uses. This is what stops the
    # blind-merge failures: undefined `config` constants, duplicate `__future__`,
    # a stray FastAPI app, and `_default = VectorStore()` no longer leak in.
    body_text = "\n\n".join([helper_block, node_block, run_block, maf_block])
    used = _names_used(body_text)
    used |= {"AgentContext", adapter.context_class_name, "HumanApprovalRequired"}

    config_import = ""
    if ir.config.constants:
        needed = [n for n in ir.config.constants if n in used]
        if needed:
            config_import = "from config import " + ", ".join(needed)

    orch_imports = _curate_imports([*rewritten, *tool_import_lines], used, ir.config.constants)
    import_bound = {b for ln in orch_imports.splitlines() for b in _import_bound_names(ln)}
    import_bound |= set(ir.config.constants) | {"AgentContext", adapter.context_class_name}
    preamble = _curate_preamble(ir.preamble, used, import_bound)

    orch_rendered = env.get_template("orchestrator.py.jinja").render(
        context_class=adapter.context_class_name,
        config_import=config_import,
        imports=orch_imports,
        preamble=preamble,
        extra_defs="".join(extra_defs_parts),
        helper_functions=helper_block,
        node_functions=node_block,
        run_block=run_block,
        maf_workflow=maf_block,
    )
    # Final pass: rewrite any remaining function-level source imports in bodies.
    if source_root:
        orch_rendered = _rewrite_imports_in_text(orch_rendered, source_root, inlined)
        orch_rendered = _rewrite_path_anchors(orch_rendered, source_root)
    _write_python(result, "orchestrator.py", orch_rendered)
    _validate_target(
        result, "orchestrator.py", orch_rendered,
        generator.orchestrator_must_tokens(bool(maf_block)),
    )

    # 5. Fresh entrypoint importing the converted modules.
    _write_python(result, "main.py", generator.entrypoint(ir, adapter, target))

    # 4b. A minimal smoke test so the converted package ships with a test.
    _write_python(
        result, os.path.join("tests", "test_smoke.py"), generator.smoke_test(adapter, target)
    )

    # 4c. Any offline SDK stub files the target needs so the emitted graph BUILDS
    #     AND RUNS without the real SDK. Python resolves these local packages
    #     first; delete them once the real SDK is installed (drop-in replacement).
    for stub_rel, stub_content in generator.sdk_stub_files().items():
        _write_python(result, stub_rel, stub_content)

    # 4d. Extra non-Python artifacts (e.g. CrewAI prompt templates in prompts/).
    #     These are editable production prompts, written verbatim (not ast-checked).
    for extra_rel, extra_content in generator.extra_files(ir, adapter).items():
        _write_text(result, extra_rel, extra_content)

    # 6. requirements.txt. (generate_from_paths re-writes it after support modules
    #    are carried, so their local modules are excluded from deps.)
    _write_requirements(result, ir, adapter)

    # 7. .env.example — one KEY=<your-value-here> line per env var detected in the
    #    source. Always emitted so the team knows which secrets to provision.
    _write_env_example(result, ir)

    return result


def _plugin_module_from_source(
    source: str, tools: list[ToolSpec], adapter: TargetAdapter, source_root: str | None
) -> str:
    """Build a plugin module from the tool file's FULL source.

    Unlike the stripped version, this keeps the module's classes, module-level
    singletons (e.g. `_default = VectorStore()`), and private helpers -- so the
    tool functions' dependencies still resolve -- while rewriting `src.*` imports
    and decorating the public tool functions (function-style targets).
    """
    tree = ast.parse(source)
    _ImportRewriter(source_root).visit(tree)
    ast.fix_missing_locations(tree)

    tool_by_name = {t.name: t for t in tools}
    function_style = adapter.tool_style() == "function"
    if function_style:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in tool_by_name:
                desc = tool_by_name[node.name].docstring or node.name
                dec = ast.parse(adapter.tool_decorator_call(repr(desc))).body[0].value
                node.decorator_list.insert(0, dec)
        ast.fix_missing_locations(tree)

    future = "from __future__ import annotations"
    body_lines = [ln for ln in ast.unparse(tree).splitlines() if ln.strip() != future]
    parts = [
        '"""Generated tools (converted from source tool functions)."""',
        future,
        _decorator_shim(adapter),
    ]
    # Plugin-class-style targets still append wrapper classes for their tools.
    tail = ""
    if not function_style:
        tail = "\n\n" + "\n\n".join(_tool_plugin_class(t, adapter) for t in tools)
    module = "\n".join(parts) + "\n\n" + "\n".join(body_lines) + tail + "\n"
    return _rewrite_path_anchors(module, source_root)


def _stripped_output_path(rel: str, source_root: str | None) -> str:
    """Map a source path into the consolidated tree (strip src/, tools/ -> plugins/)."""
    segs = rel.replace("\\", "/").split("/")
    if source_root and segs and segs[0] == source_root:
        segs = segs[1:]
    if segs and segs[0] == "tools":
        segs = ["plugins"] + segs[1:]
    return "/".join(segs)


def generate_from_paths(
    ir: IR,
    conversion: ConversionResult,
    input_root: str,
    output_path: str,
    config: Config | None = None,
    adapter: TargetAdapter | None = None,
    generator: TargetGenerator | None = None,
) -> GenerationResult:
    """Generate the consolidated package, carry SUPPORT modules (with rewired
    imports) so cross-module references resolve, and copy neutral assets.

    Support modules = source .py that define no node/router/tool and aren't the
    state/graph/entrypoint modules (e.g. nlp/, memory/, llm.py, tool_wrapper.py).
    They are copied to the consolidated tree with `src.*` imports rewritten.
    """
    from converter.parser import extract_state_class_names

    config = config or Config()
    result = generate(ir, conversion, output_path, config, adapter, generator)

    exclude = set(config.extraction_exclude_dirs)
    source_root = _source_root(ir)
    node_files = _node_files(ir)
    converted_tool_files = {t.source_file for t in ir.tools}
    state_names = set(ir.state_class_names)
    skip_basenames = {"README.md", "requirements.txt"}  # regenerated

    # Regenerate each plugin module from its FULL source so module-level classes
    # and singletons the tool functions rely on are preserved (fixes NameErrors
    # like `_default = VectorStore()` missing from the stripped plugin module).
    adapter = adapter or get_target_adapter(ir.metadata.target_framework or "maf")
    tools_by_file: dict[str, list[ToolSpec]] = {}
    for tool in ir.tools:
        if tool.source_file:
            tools_by_file.setdefault(tool.source_file, []).append(tool)
    for source_file, tools in tools_by_file.items():
        src = os.path.join(input_root, source_file)
        if not os.path.exists(src):
            continue
        try:
            full = _plugin_module_from_source(
                open(src, encoding="utf-8").read(), tools, adapter, source_root
            )
        except SyntaxError:
            continue
        out_rel = os.path.join("plugins", f"{_plugin_module_name(source_file)}.py")
        # Overwrite the stripped plugin module written by generate().
        if out_rel.replace("\\", "/") in {p.replace("\\", "/") for p in result.written_files}:
            result.written_files.remove(
                next(p for p in result.written_files if p.replace("\\", "/") == out_rel.replace("\\", "/"))
            )
        _write_python(result, out_rel, full)

    for entry in ir.files:
        rel = entry.relative_path
        segs = rel.replace("\\", "/").split("/")
        if any(s in exclude for s in segs):
            continue

        if not rel.endswith(".py"):
            # Neutral asset (prompts, etc.) -- copy verbatim, stripping src/.
            if os.path.basename(rel) in skip_basenames:
                continue
            if _is_secret_file(rel):
                continue  # never copy secrets (.env, keys, credentials) into output
            src = os.path.join(input_root, rel)
            if not os.path.exists(src):
                continue
            out_rel = _stripped_output_path(rel, source_root)
            dst = os.path.join(result.output_root, *out_rel.split("/"))
            os.makedirs(os.path.dirname(dst) or result.output_root, exist_ok=True)
            shutil.copy2(src, dst)
            result.copied_files.append(out_rel)
            continue

        # --- Python: decide whether it's a SUPPORT module worth carrying ---
        base = os.path.basename(rel)
        if base == "__init__.py" or base in _ENTRYPOINT_MODULES:
            continue
        if rel in converted_tool_files or rel in node_files:
            continue  # tools -> plugins (generated); node files -> inlined
        src = os.path.join(input_root, rel)
        if not os.path.exists(src):
            continue
        source = open(src, encoding="utf-8").read()
        if any(tok in source for tok in _GRAPH_ASSEMBLY_TOKENS):
            continue  # graph-assembly module -> replaced by orchestrator
        try:
            if any(n in state_names for n in extract_state_class_names(source)):
                continue  # state module -> agent_context.py
        except SyntaxError:
            continue

        out_rel = _stripped_output_path(rel, source_root)
        try:
            rewritten = _rewrite_support_module(source, source_root)
        except SyntaxError:
            rewritten = source
        _write_python(result, out_rel, rewritten)

    # Re-derive requirements now that support modules are present (so their local
    # module names are excluded from the dependency list).
    result.written_files = list(dict.fromkeys(result.written_files))
    _write_requirements(result, ir, adapter or get_target_adapter(ir.metadata.target_framework or "maf"))
    result.written_files = list(dict.fromkeys(result.written_files))

    return result
