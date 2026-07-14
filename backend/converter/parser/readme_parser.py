"""Module 3 -- README parser.

Splits the README on level-2 (`## `) headers and maps the known sections onto
`ReadmeSections`. Unlike the code parser, this is markdown, so regex is fine here
(the "no regex" rule applies only to Python source).

Rules (Section 11, Module 3):
- Tools bullets:  `` - `name`: description ``      -> ReadmeToolEntry
- State bullets:  `` - `name` (type): description `` -> ReadmeStateEntry
- Workflow block stored verbatim (Tier 2 input -- never pre-parsed)
- Missing sections are recorded as a warning, never a hard stop
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from converter.config import Config
from converter.contracts import (
    ReadmeSections,
    ReadmeStateEntry,
    ReadmeToolEntry,
)

logger = logging.getLogger(__name__)

# A level-2 header: exactly two hashes then a space (not `###`).
_H2 = re.compile(r"^##[ \t]+(.+?)\s*$")

# `- `name`: description`  /  `* `name` - description`
_TOOL_BULLET = re.compile(r"^\s*[-*]\s+`([^`]+)`\s*[:\-]?\s*(.*)$")

# `- `name` (type): description`
_STATE_BULLET = re.compile(r"^\s*[-*]\s+`([^`]+)`\s*\(([^)]*)\)\s*[:\-]?\s*(.*)$")


def _split_sections(text: str) -> dict[str, str]:
    """Return {header_text: body_text} for each level-2 section, in order."""
    sections: dict[str, str] = {}
    current_header: Optional[str] = None
    buffer: list[str] = []

    def flush() -> None:
        if current_header is not None:
            sections[current_header] = "\n".join(buffer).strip("\n")

    for line in text.splitlines():
        match = _H2.match(line)
        # Guard against `###` (level-3): those belong inside the current body.
        if match and not line.startswith("###"):
            flush()
            current_header = match.group(1).strip()
            buffer = []
        else:
            if current_header is not None:
                buffer.append(line)
    flush()
    return sections


def _find_section(sections: dict[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup."""
    for header, body in sections.items():
        if header.lower() == name.lower():
            return body
    return None


def _parse_tool_bullets(body: Optional[str]) -> list[ReadmeToolEntry]:
    entries: list[ReadmeToolEntry] = []
    if not body:
        return entries
    for line in body.splitlines():
        match = _TOOL_BULLET.match(line)
        if match:
            entries.append(
                ReadmeToolEntry(name=match.group(1).strip(), description=match.group(2).strip())
            )
    return entries


def _parse_state_bullets(body: Optional[str]) -> list[ReadmeStateEntry]:
    entries: list[ReadmeStateEntry] = []
    if not body:
        return entries
    for line in body.splitlines():
        match = _STATE_BULLET.match(line)
        if match:
            entries.append(
                ReadmeStateEntry(
                    name=match.group(1).strip(),
                    type=match.group(2).strip(),
                    description=match.group(3).strip(),
                )
            )
            continue
        # Fall back to a plain tool-style bullet (name, no explicit type).
        fallback = _TOOL_BULLET.match(line)
        if fallback:
            entries.append(
                ReadmeStateEntry(
                    name=fallback.group(1).strip(),
                    type="",
                    description=fallback.group(2).strip(),
                )
            )
    return entries


def parse_readme(text: str, config: Config | None = None) -> ReadmeSections:
    """Parse README markdown into `ReadmeSections`."""
    config = config or Config()
    sections = _split_sections(text)

    result = ReadmeSections(raw_sections=dict(sections))
    result.purpose = _find_section(sections, "Purpose")
    result.framework = _find_section(sections, "Framework")
    result.tools = _parse_tool_bullets(_find_section(sections, "Tools"))
    result.workflow_description = _find_section(sections, "Workflow")  # verbatim
    result.state = _parse_state_bullets(_find_section(sections, "State"))
    result.configuration = _find_section(sections, "Configuration")
    result.dependencies = _find_section(sections, "Dependencies")

    present = {h.lower() for h in sections}
    for required in config.required_readme_sections:
        if required.lower() not in present:
            result.missing_sections.append(required)
            logger.warning("README missing '## %s' section", required)

    return result


def parse_readme_file(path: str, config: Config | None = None) -> ReadmeSections:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_readme(fh.read(), config)
