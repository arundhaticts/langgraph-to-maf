"""Generate INSTALL.md and ARCHITECTURE.md for the converted agent.

These ship inside the output folder so the converted MAF agent is self-describing
and runnable. Both are built from the IR + conversion result (single source of
truth); nothing re-reads the source.
"""

from __future__ import annotations

import os

from converter.config import Config, ConversionMode
from converter.contracts import IR, ConversionResult, NodeRole, Tier

# Rough stdlib set so INSTALL only suggests third-party installs.
_STDLIB = frozenset(
    {
        "os", "sys", "re", "json", "math", "time", "typing", "dataclasses",
        "collections", "itertools", "functools", "pathlib", "operator",
        "datetime", "enum", "abc", "logging", "random", "asyncio", "io",
        "contextlib", "types", "textwrap", "uuid", "hashlib", "subprocess",
    }
)

# Best-effort import-root -> pip package name.
_PIP_NAME = {
    "langchain_openai": "langchain-openai",
    "langchain_core": "langchain-core",
    "langchain": "langchain",
    "semantic_kernel": "semantic-kernel",
    "google": "google-genai",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}


def _third_party_packages(ir: IR) -> list[str]:
    roots: list[str] = []
    for line in ir.imports:
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        if tokens[0] == "import" and len(tokens) > 1:
            root = tokens[1].split(".")[0]
        elif tokens[0] == "from" and len(tokens) > 1:
            root = tokens[1].split(".")[0]
        else:
            continue
        if root and root not in _STDLIB and root not in roots:
            roots.append(root)
    return [_PIP_NAME.get(r, r) for r in roots]


def _plugin_files(ir: IR) -> list[str]:
    files = []
    for tool in ir.tools:
        rel = tool.source_file if (tool.source_file or "").endswith(".py") else "skills.py"
        if rel not in files:
            files.append(rel)
    return files


def build_install_md(ir: IR, config: Config | None = None) -> str:
    config = config or Config()
    target = (ir.metadata.target_framework or "maf").upper()
    packages = _third_party_packages(ir)
    pip_line = "pip install semantic-kernel " + " ".join(packages) if packages else "pip install semantic-kernel"

    entry_class = "AgentContext"
    lines = [
        f"# Install & Run — Converted {target} Agent",
        "",
        "This agent was produced by the Framework Conversion Utility "
        f"(source: {ir.metadata.source_framework or 'unknown'} → target: {target}).",
        "",
        "## 1. Requirements",
        "- Python 3.10+",
        "",
        "## 2. Install dependencies",
        "```bash",
        pip_line,
        "```",
        "> `semantic-kernel` provides `@kernel_function`. The remaining packages are "
        "the agent's own dependencies, carried over from the source.",
        "",
        "## 3. Run",
        "```python",
        "from agent_context import AgentContext",
        "from orchestrator import run",
        "",
        f"ctx = {entry_class}(",
        "    # initialise the fields you need, e.g.:",
    ]
    for field in ir.state[:6]:
        default = "[]" if field.is_append_only else "None"
        lines.append(f"    {field.name}={default},")
    lines += [
        ")",
        "ctx = run(ctx)",
        "print(ctx)",
        "```",
        "",
        "```bash",
        "python orchestrator.py   # if you add an entry point at the bottom",
        "```",
        "",
        "## 4. Human-in-the-loop (HITL)",
    ]
    hitl_nodes = [
        n.name
        for n in (ir.workflow.nodes if ir.workflow else [])
        if n.role is NodeRole.HITL
    ]
    if hitl_nodes:
        lines += [
            f"The step(s) {', '.join('`' + n + '`' for n in hitl_nodes)} **auto-approve** "
            "by default so the agent runs end-to-end. To enforce real human approval, "
            "open `orchestrator.py`, delete the auto-approve lines in that function, "
            'and uncomment the "REAL HUMAN APPROVAL" block. See `MIGRATION_REPORT.md`.',
        ]
    else:
        lines.append("No human-in-the-loop steps were detected.")
    lines += [
        "",
        "## 5. What to review",
        "See `MIGRATION_REPORT.md` — anything under **Needs review** or "
        "**Manual action required** should be checked before deploying.",
        "",
    ]
    return "\n".join(lines)


def build_architecture_md(
    ir: IR, conversion: ConversionResult, config: Config | None = None
) -> str:
    config = config or Config()
    target = (ir.metadata.target_framework or "maf").upper()
    workflow = ir.workflow
    mode = "LLM-assisted (you review)" if config.allow_llm_fallback else "manual (you implement)"

    tier_counts: dict[str, int] = {}
    for u in conversion.units:
        tier_counts[u.tier.value] = tier_counts.get(u.tier.value, 0) + 1

    hitl_nodes = [n.name for n in (workflow.nodes if workflow else []) if n.role is NodeRole.HITL]

    lines = [
        f"# Architecture — Converted {target} Agent",
        "",
        f"- **Source framework:** {ir.metadata.source_framework or 'unknown'}",
        f"- **Target framework:** {target}",
        f"- **Hard-parts mode:** {mode}",
        f"- **Orchestration pattern:** {workflow.pattern.value if workflow else 'n/a'}",
        "",
        "## File layout",
        "| File | Purpose |",
        "|---|---|",
        "| `agent_context.py` | State schema → `AgentContext` dataclass |",
    ]
    for pf in _plugin_files(ir):
        lines.append(f"| `{pf}` | Tools → `@kernel_function` skill plugins |")
    lines += [
        "| `orchestrator.py` | Graph → node/router functions + `run(ctx)` |",
        "| `config.py` | Carried-over constants |",
        "| `README.md` | Regenerated agent docs (target vocabulary) |",
        "| `MIGRATION_REPORT.md` | Auto-converted / needs-review / manual items |",
        "",
        "## Components",
        f"- **Context:** `AgentContext` with {len(ir.state)} field(s); append-only "
        "list fields use `.append()`.",
        f"- **Skills:** {len(ir.tools)} tool(s) converted to plugin classes.",
        f"- **Orchestration:** `run(ctx)` drives "
        f"{len(workflow.nodes) if workflow else 0} node(s).",
    ]
    if hitl_nodes:
        lines.append(
            f"- **HITL:** {', '.join('`' + n + '`' for n in hitl_nodes)} — auto-approve "
            "active; real approval flow kept commented in `orchestrator.py`."
        )
    lines += [
        "",
        "## How each part was resolved",
        "| Tier | Meaning | Count |",
        "|---|---|---|",
        f"| Tier 1 | Deterministic rules | {tier_counts.get(Tier.TIER1.value, 0)} |",
        f"| Tier 2 | README-assisted | {tier_counts.get(Tier.TIER2.value, 0)} |",
        f"| Tier 3 | LLM (Gemini) | {tier_counts.get(Tier.TIER3.value, 0)} |",
        f"| Unresolved | Left for manual work | {tier_counts.get(Tier.UNRESOLVED.value, 0)} |",
        "",
        "## Source → target mapping",
        "| Source (LangGraph) | Target (MAF) |",
        "|---|---|",
        "| `TypedDict` state | `AgentContext` dataclass |",
        "| `@tool` function | `@kernel_function` plugin method |",
        "| graph node `def n(state)->dict` | `def n(ctx)->AgentContext` |",
        "| `add_conditional_edges` | `if/elif` on a router label |",
        "| generation/validation loop | `while` with a router break |",
        "| `interrupt()` (HITL) | auto-approve + commented approval flow |",
        "",
    ]
    return "\n".join(lines)


def write_docs(
    ir: IR,
    conversion: ConversionResult,
    output_path: str,
    config: Config | None = None,
) -> list[str]:
    """Write INSTALL.md and ARCHITECTURE.md into the output folder."""
    os.makedirs(output_path, exist_ok=True)
    written = []
    for name, content in (
        ("INSTALL.md", build_install_md(ir, config)),
        ("ARCHITECTURE.md", build_architecture_md(ir, conversion, config)),
    ):
        path = os.path.join(output_path, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(name)
    return written
