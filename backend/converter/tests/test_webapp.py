"""Tests for the web backend: the conversion service and the FastAPI endpoints."""

from __future__ import annotations

import io
import os
import sys
import zipfile

import pytest

# backend/ holds api.py and service.py (this test lives at backend/converter/tests).
_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BACKEND)

from service import convert_folder, convert_local_path, strip_common_top  # noqa: E402

README = """# Demo
## Purpose
A demo agent.
## Tools
- `read`: reads
## Workflow
Generate then loop back up to 3 times.
"""

AGENT = '''
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    coverage: float
    audit_log: Annotated[list, add]

@tool
def read(path: str) -> list:
    """reads"""
    return []

def generate(state):
    return {"coverage": state["coverage"] + 0.1}

g = StateGraph(State)
g.add_node("generate", generate)
g.set_entry_point("generate")
g.add_edge("generate", END)
'''


def _payload():
    # As a real folder upload would arrive: a common top-level folder.
    return [
        {"path": "my_agent/README.md", "content": README},
        {"path": "my_agent/agent.py", "content": AGENT},
    ]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def test_strip_common_top():
    files = strip_common_top(
        [{"path": "top/README.md", "content": "x"}, {"path": "top/a.py", "content": "y"}]
    )
    assert {f["path"] for f in files} == {"README.md", "a.py"}


def test_convert_manual_mode_returns_zip():
    zip_bytes = convert_folder(_payload(), "manual")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert {"agent_context.py", "orchestrator.py", "README.md",
            "MIGRATION_REPORT.md", "INSTALL.md", "ARCHITECTURE.md"} <= names


def test_convert_output_is_valid_python():
    import ast

    zip_bytes = convert_folder(_payload(), "manual")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith(".py"):
                ast.parse(zf.read(name).decode("utf-8"))


def test_convert_local_path_ignores_dependency_dirs(tmp_path):
    # A folder that also contains a fake .venv with many files -> scanner skips it.
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "agent.py").write_text(AGENT, encoding="utf-8")
    venv = tmp_path / ".venv" / "Lib" / "site-packages" / "junk"
    venv.mkdir(parents=True)
    for i in range(50):
        (venv / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")

    zip_bytes = convert_local_path(str(tmp_path), "manual")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert "orchestrator.py" in names
    # None of the .venv junk leaked into the converted output.
    assert not any("junk" in n or ".venv" in n for n in names)


def test_convert_local_path_bad_dir_raises():
    with pytest.raises(NotADirectoryError):
        convert_local_path("this/path/does/not/exist", "manual")


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "gemini_configured" in body


def test_convert_endpoint_returns_zip(client):
    r = client.post("/api/convert", json={"mode": "manual", "files": _payload()})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "orchestrator.py" in zf.namelist()


def test_convert_endpoint_empty_files_400(client):
    r = client.post("/api/convert", json={"mode": "manual", "files": []})
    assert r.status_code == 400


def test_convert_endpoint_bad_repo_400(client):
    # No README.md -> scanner hard-stops -> 400.
    r = client.post(
        "/api/convert",
        json={"mode": "manual", "files": [{"path": "x/agent.py", "content": "import langgraph"}]},
    )
    assert r.status_code == 400


def test_convert_path_endpoint(client, tmp_path):
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "agent.py").write_text(AGENT, encoding="utf-8")
    r = client.post("/api/convert-path", json={"mode": "manual", "path": str(tmp_path)})
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "orchestrator.py" in zf.namelist()


def test_convert_path_endpoint_bad_dir_400(client):
    r = client.post("/api/convert-path", json={"mode": "manual", "path": "nope/nope"})
    assert r.status_code == 400
