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


def _state_field_decl(f: StateField) -> str:
    if f.is_append_only:
        return (
            f"    {f.name}: list = field(default_factory=list)"
            f"  # was Annotated[{f.type}, add]; use .append()"
        )
    base = f.type or "Any"
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
    return ast.unparse(tree)


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


def _hitl_block(
    ir: IR,
    node,
    unit: ConversionUnit | None,
    ctx_class: str,
    has_audit_log: bool,
) -> str:
    """HITL node: active AUTO-APPROVE + a commented-out real approval flow.

    The agent runs end-to-end today (auto-approve). To enforce real human
    approval, the user deletes the auto-approve lines and uncomments the block --
    which holds the Gemini-generated flow (llm mode) or the original/template
    flow (manual mode).
    """
    header = f"def {node.name}(ctx: {ctx_class}) -> {ctx_class}:"

    active = [
        "    # AUTO-APPROVE (prototype): keeps the agent runnable without a human.",
        "    # To enforce real human approval, delete the auto-approve lines below",
        '    # and uncomment the "REAL HUMAN APPROVAL" block underneath.',
    ]
    if has_audit_log:
        active.append(
            f"    ctx.audit_log.append(\"auto-approved: '{node.name}' "
            "(no human in the loop)\")"
        )
    active.append("    return ctx")

    if unit and unit.generated_code:
        real_src = unit.generated_code.rstrip()
        origin = "Gemini-generated; review then uncomment"
    else:
        # Manual: emit a HumanApprovalRequired template (NOT the raw source, which
        # would reintroduce interrupt()/langgraph tokens). The original logic is
        # recorded in MIGRATION_REPORT.md for reference.
        real_src = (
            'if not getattr(ctx, "approved", False):\n'
            f'    raise HumanApprovalRequired({{"node": "{node.name}"}})\n'
            "# on resume, the caller sets the decision on ctx and returns ctx\n"
            "# (the original approval logic is in MIGRATION_REPORT.md)"
        )
        origin = "manual: implement, then uncomment"

    return "\n".join(
        [
            header,
            "\n".join(active),
            "",
            f"    # --- REAL HUMAN APPROVAL ({origin}) ---",
            _comment_lines(real_src, indent=4),
            "    # --- end real human approval ---",
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


def _loop_cap_constant(ir: IR) -> str:
    """A real cap constant name if one looks like a loop bound, else a literal."""
    for name in ir.config.constants:
        upper = name.upper()
        if any(tok in upper for tok in ("RETRIES", "RETRY", "MAX", "ITER", "LIMIT")):
            return name
    return "10  # TODO: confirm the real cap constant"


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
        lines = [
            "def run(ctx):",
            "    guard = 0",
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
        "        def _wrap(fn):\n"
        "            return fn\n"
        "        return _wrap\n"
    )


def _plugin_module_name(source_file: str | None) -> str:
    base = os.path.basename(source_file or "skills.py")
    return base[:-3] if base.endswith(".py") else "skills"


def _tool_plain_function(tool: ToolSpec) -> str:
    sig = tool.signature or ", ".join(p.name for p in tool.params)
    ret = f" -> {tool.returns}" if tool.returns else ""
    lines = [f"def {tool.name}({sig}){ret}:"]
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
    for tool in tools:
        parts.append("\n" + _tool_plain_function(tool))
    for tool in tools:
        parts.append("\n" + _tool_plugin_class(tool, adapter))
    return "\n".join(parts) + "\n"


def _entrypoint(adapter: TargetAdapter, target: str) -> str:
    ctx = adapter.context_class_name
    return (
        f'"""Generated entrypoint for the converted {target} agent."""\n'
        "from __future__ import annotations\n\n"
        f"from agent_context import {ctx}\n"
        "from orchestrator import run\n\n\n"
        "def main():\n"
        f"    ctx = {ctx}()\n"
        "    ctx = run(ctx)\n"
        "    print(ctx)\n"
        "    return ctx\n\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(
    ir: IR,
    conversion: ConversionResult,
    output_path: str,
    config: Config | None = None,
    adapter: TargetAdapter | None = None,
) -> GenerationResult:
    """Render and write ONE coherent MAF package. Returns a `GenerationResult`.

    Layout: agent_context.py, config.py, plugins/<mod>.py (+__init__), orchestrator.py,
    main.py. The original LangGraph tree is NOT reproduced -- the output is a
    self-contained target package.
    """
    config = config or Config()
    adapter = adapter or get_target_adapter(ir.metadata.target_framework or "maf")
    target = (ir.metadata.target_framework or "maf").upper()
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
            _validate_target(result, rel, content, [adapter.tool_decorator(), "class "])
            names = ", ".join(sorted({t.name for t in tools}))
            tool_import_lines.append(f"from plugins.{mod} import {names}")

    # 2. Context dataclass (+ backward-compat aliases for the source state names).
    fields_block = (
        "\n".join(_state_field_decl(f) for f in ir.state) if ir.state else "    pass"
    )
    ctx_rendered = env.get_template("agent_context.py.jinja").render(
        context_class=adapter.context_class_name,
        fields_block=fields_block,
        typing_imports=_typing_imports(ir),
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
    _validate_target(result, "agent_context.py", ctx_rendered, ["@dataclass"])

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
    has_hitl = any(u.rule_id == "R-08" for u in conversion.units)
    has_checkpointer = any(u.rule_id == "R-15" for u in conversion.units)
    extra_defs_parts = []
    if has_hitl:
        extra_defs_parts.append(
            "\n\nclass HumanApprovalRequired(Exception):\n"
            '    """Raised where the source paused for human-in-the-loop approval."""'
        )
    if has_checkpointer:
        extra_defs_parts.append(
            "\n\n# TODO: wire persistence layer (source used a checkpointer/saver)"
        )

    config_import = ""
    if ir.config.constants:
        config_import = "from config import " + ", ".join(ir.config.constants)

    helper_block = _helper_functions(ir, adapter)
    node_block = _node_functions(ir, adapter, conversion)
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
    orch_imports = "\n".join(dict.fromkeys([*rewritten, *tool_import_lines])).strip("\n")

    orch_rendered = env.get_template("orchestrator.py.jinja").render(
        context_class=adapter.context_class_name,
        config_import=config_import,
        imports=orch_imports,
        preamble="\n".join(ir.preamble),
        extra_defs="".join(extra_defs_parts),
        helper_functions=helper_block,
        node_functions=node_block,
        run_block=_run_block(ir, workflow_unit),
    )
    # Final pass: rewrite any remaining function-level source imports in bodies.
    if source_root:
        orch_rendered = _rewrite_imports_in_text(orch_rendered, source_root, inlined)
    _write_python(result, "orchestrator.py", orch_rendered)
    _validate_target(result, "orchestrator.py", orch_rendered, ["def run"])

    # 5. Fresh entrypoint importing the converted modules.
    _write_python(result, "main.py", _entrypoint(adapter, target))

    # 6. requirements.txt. (generate_from_paths re-writes it after support modules
    #    are carried, so their local modules are excluded from deps.)
    _write_requirements(result, ir, adapter)

    return result


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
) -> GenerationResult:
    """Generate the consolidated package, carry SUPPORT modules (with rewired
    imports) so cross-module references resolve, and copy neutral assets.

    Support modules = source .py that define no node/router/tool and aren't the
    state/graph/entrypoint modules (e.g. nlp/, memory/, llm.py, tool_wrapper.py).
    They are copied to the consolidated tree with `src.*` imports rewritten.
    """
    from converter.parser import extract_state_class_names

    config = config or Config()
    result = generate(ir, conversion, output_path, config, adapter)

    exclude = set(config.extraction_exclude_dirs)
    source_root = _source_root(ir)
    node_files = _node_files(ir)
    converted_tool_files = {t.source_file for t in ir.tools}
    state_names = set(ir.state_class_names)
    skip_basenames = {"README.md", "requirements.txt"}  # regenerated

    for entry in ir.files:
        rel = entry.relative_path
        segs = rel.replace("\\", "/").split("/")
        if any(s in exclude for s in segs):
            continue

        if not rel.endswith(".py"):
            # Neutral asset (prompts, etc.) -- copy verbatim, stripping src/.
            if os.path.basename(rel) in skip_basenames:
                continue
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
