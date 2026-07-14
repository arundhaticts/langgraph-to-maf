"""Module 8 -- README generator.

Renders the output `README.md` from the IR using the target framework's
vocabulary (R-13: `## Tools` -> `## Skills`). It describes the converted agent
*as it actually is* -- if a HITL step became a stub, the README says so.

Rendered from `readme_maf.md.jinja` and the `TargetAdapter.README_VOCAB`. Works
from the IR (single source of truth); the `ConversionResult` is used only to know
which hard mappings became stubs.
"""

from __future__ import annotations

import os

from jinja2 import Environment, FileSystemLoader

from converter.adapters import get_target_adapter
from converter.adapters.base import TargetAdapter
from converter.config import Config
from converter.contracts import (
    IR,
    ConversionResult,
    NodeRole,
)

_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=True
    )


def _title(ir: IR, agent_name: str | None) -> str:
    if agent_name:
        return agent_name
    return "Converted Agent"


def build_readme(
    ir: IR,
    conversion: ConversionResult | None = None,
    config: Config | None = None,
    adapter: TargetAdapter | None = None,
    agent_name: str | None = None,
) -> str:
    """Render the output README markdown."""
    config = config or Config()
    adapter = adapter or get_target_adapter(ir.metadata.target_framework or "maf")
    framework = (ir.metadata.target_framework or "maf").upper()

    skills = [
        {
            "plugin_class": adapter.plugin_class_name(t.name),
            "method": adapter.method_name(t.name),
            "description": t.docstring or t.name,
        }
        for t in ir.tools
    ]

    context_fields = [
        {"name": f.name, "type": f.type or "Any", "append_only": f.is_append_only}
        for f in ir.state
    ]

    # HITL: prefer the conversion result (accurate), else infer from node roles.
    hitl_nodes: list[str] = []
    has_checkpointer = False
    if conversion is not None:
        hitl_nodes = [
            u.source_ref for u in conversion.units if u.rule_id == "R-08"
        ]
        has_checkpointer = any(u.rule_id == "R-15" for u in conversion.units)
    elif ir.workflow:
        hitl_nodes = [n.name for n in ir.workflow.nodes if n.role is NodeRole.HITL]

    workflow = ir.workflow
    context = {
        "title": _title(ir, agent_name),
        "purpose": ir.metadata.description or "N/A",
        "framework": framework,
        "skills_heading": adapter.README_VOCAB.get("tools_heading", "Tools"),
        "skills": skills,
        "context_heading": adapter.README_VOCAB.get("state_heading", "State"),
        "context_fields": context_fields,
        "workflow_pattern": workflow.pattern.value if workflow else "unknown",
        "workflow_description": (
            (workflow.readme_description if workflow else None)
            or "See the generated orchestrator.py."
        ),
        "has_hitl": bool(hitl_nodes),
        "hitl_nodes": ", ".join(f"`{n}`" for n in hitl_nodes),
        "has_checkpointer": has_checkpointer,
        "config_constants": ir.config.constants,
        "temperature": ir.config.temperature,
    }
    return _env().get_template("readme_maf.md.jinja").render(**context)


def write_readme(content: str, output_path: str) -> str:
    """Write README.md into the output folder; returns the path written."""
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
