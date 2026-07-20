"""End-to-end integration tests for CrewAI conversion.

Covers the gaps reported when the test-optimiser agent was converted to CrewAI:
- prompts/ templates are emitted (agent + task prompts, editable).
- plugins/ tool functions are COMPLETE (runnable, no bare NotImplementedError).
- CrewAI is the PRIMARY orchestration (Crew present; run_crew tries the Crew
  first and only falls back to the offline run()).
- the whole package compiles and ships an offline run() fast-path.
"""

from __future__ import annotations

import ast
import io
import zipfile

from service import convert_folder

# A multi-agent CrewAI source with real agents, tasks (with a dependency), and a
# tool carrying real logic -- representative of a converted production agent.
CREWAI_SOURCE = '''
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool


@tool
def fetch_metrics(suite: str) -> dict:
    """Fetch flaky-test metrics for a suite."""
    return {"suite": suite, "flaky": 3}


@tool
def score_tests(metrics: dict) -> list:
    """Score tests by flakiness."""
    return sorted(metrics.items())


researcher = Agent(
    role="Test Analyst",
    goal="Identify the flakiest tests in the suite.",
    backstory="A veteran QA engineer who lives in CI logs.",
)
optimiser = Agent(
    role="Optimisation Engineer",
    goal="Propose concrete fixes for the flakiest tests.",
    backstory="An expert at stabilising nondeterministic tests.",
)

analyse_task = Task(
    description="Analyse the suite and rank tests by flakiness.",
    expected_output="A ranked list of flaky tests.",
    agent=researcher,
)
optimise_task = Task(
    description="Propose fixes for the top flaky tests.",
    expected_output="A set of concrete remediation steps.",
    agent=optimiser,
    context=[analyse_task],
)

crew = Crew(
    agents=[researcher, optimiser],
    tasks=[analyse_task, optimise_task],
    process=Process.sequential,
)
'''

README = """# Test Optimiser
## Purpose
Finds flaky tests and proposes fixes.
## Tools
- `fetch_metrics`: fetch metrics
- `score_tests`: score tests
## Workflow
Analyse then optimise.
"""


def _convert_to_crewai() -> zipfile.ZipFile:
    files = [
        {"path": "agent/README.md", "content": README},
        {"path": "agent/agent.py", "content": CREWAI_SOURCE},
    ]
    raw = convert_folder(files, "manual", target="crewai", source="crewai")
    return zipfile.ZipFile(io.BytesIO(raw))


def test_crewai_output_compiles():
    zf = _convert_to_crewai()
    for name in zf.namelist():
        if name.endswith(".py"):
            src = zf.read(name).decode("utf-8")
            try:
                ast.parse(src)
            except SyntaxError as exc:  # pragma: no cover
                raise AssertionError(f"SyntaxError in {name}: {exc}") from exc


def test_crewai_emits_prompt_templates():
    zf = _convert_to_crewai()
    names = set(zf.namelist())
    prompt_files = {n for n in names if n.startswith("prompts/")}
    assert "prompts/README.md" in prompt_files, f"prompts/README.md missing; got {sorted(prompt_files)}"
    # Agent prompts derived from the source Agent(role/goal/backstory).
    assert any(n.startswith("prompts/agent_") for n in prompt_files), sorted(prompt_files)
    # Task prompts derived from the source Task(description/expected_output).
    assert any(n.startswith("prompts/task_") for n in prompt_files), sorted(prompt_files)
    # The real role/goal text must appear in an agent prompt.
    agent_blob = "\n".join(
        zf.read(n).decode("utf-8") for n in prompt_files if n.startswith("prompts/agent_")
    )
    assert "Test Analyst" in agent_blob
    assert "Optimisation Engineer" in agent_blob


def test_crewai_plugins_are_complete_no_stub_crash():
    zf = _convert_to_crewai()
    plugin_files = [n for n in zf.namelist() if n.startswith("plugins/") and n.endswith(".py")]
    assert plugin_files, "no plugin modules generated"
    blob = "\n".join(zf.read(n).decode("utf-8") for n in plugin_files)
    # No bare NotImplementedError -- tools must be runnable end-to-end.
    assert "raise NotImplementedError" not in blob
    # The real tool logic is carried over.
    assert "def fetch_metrics(" in blob
    assert "def score_tests(" in blob


def test_crewai_is_primary_orchestration():
    zf = _convert_to_crewai()
    orch = zf.read("orchestrator.py").decode("utf-8")
    # Genuine CrewAI constructs present.
    assert "Crew(" in orch
    assert "Task(" in orch
    assert "Process" in orch
    # The Crew is the primary path: run_crew kicks off the Crew and only falls
    # back to the offline run() on failure.
    assert "def run_crew(" in orch
    assert "build_crew().kickoff(" in orch
    # The offline fast-path still exists as the fallback.
    assert "def run(" in orch
    # Real agent role/goal from the source is embedded in the Crew.
    assert "Test Analyst" in orch or "Optimisation Engineer" in orch


def test_crewai_task_descriptions_are_rich():
    """Task descriptions carry the real intent, not just a one-line label."""
    zf = _convert_to_crewai()
    orch = zf.read("orchestrator.py").decode("utf-8")
    # The source task descriptions must survive into the Crew's Tasks.
    assert "rank tests by flakiness" in orch or "Propose fixes" in orch
