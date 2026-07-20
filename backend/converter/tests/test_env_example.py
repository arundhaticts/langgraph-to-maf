"""Tests for .env.example emission in the converted output.

When the source reads environment variables (os.getenv / os.environ[...]), the
converted package must ship a .env.example documenting each one so the team knows
which secrets to provision. When the source reads none, no file is emitted.
"""

from __future__ import annotations

import io
import zipfile

from service import convert_folder

# A minimal CrewAI source that reads two environment variables.
SOURCE_WITH_ENV = '''
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

API_KEY = os.environ.get("MY_API_KEY")
REGION = os.getenv("SERVICE_REGION")


@tool
def fetch(url: str) -> dict:
    """Fetch a URL."""
    return {"url": url, "key": API_KEY, "region": REGION}


worker = Agent(
    role="Fetcher",
    goal="Fetch data from the configured service.",
    backstory="A reliable data courier.",
)

fetch_task = Task(
    description="Fetch the configured endpoint.",
    expected_output="The fetched payload.",
    agent=worker,
)

crew = Crew(agents=[worker], tasks=[fetch_task], process=Process.sequential)
'''

# A source that reads NO environment variables.
SOURCE_NO_ENV = '''
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


worker = Agent(role="Adder", goal="Add numbers.", backstory="A calculator.")
add_task = Task(description="Add the inputs.", expected_output="The sum.", agent=worker)
crew = Crew(agents=[worker], tasks=[add_task], process=Process.sequential)
'''

README = "# Env Agent\n## Purpose\nReads config from the environment.\n"


def _convert(source: str) -> zipfile.ZipFile:
    files = [
        {"path": "agent/README.md", "content": README},
        {"path": "agent/agent.py", "content": source},
    ]
    raw, _ = convert_folder(files, "manual", target="crewai", source="crewai")
    return zipfile.ZipFile(io.BytesIO(raw))


def test_env_example_emitted_with_detected_vars():
    zf = _convert(SOURCE_WITH_ENV)
    assert ".env.example" in zf.namelist(), f"got {zf.namelist()}"
    body = zf.read(".env.example").decode("utf-8")
    # Each detected variable appears as a placeholder line.
    assert "MY_API_KEY=" in body
    assert "SERVICE_REGION=" in body
    # It is a template, not a real secret file (no real values).
    assert "<your-value-here>" in body


def test_env_example_absent_when_no_env_vars():
    zf = _convert(SOURCE_NO_ENV)
    assert ".env.example" not in zf.namelist()
