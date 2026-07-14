"""Intake node -- reads the suite via a tool and records it."""

from src.tools.repo_reader import detect_conventions, read_tests
from src.util import audit


def intake_node(state) -> dict:
    """Parse the suite and record conventions."""
    tests = read_tests(state["project_id"])
    conventions = detect_conventions(tests)
    return {
        "coverage": 0.0,
        "audit_log": [audit("intake", "parsed suite", count=len(tests))],
    }
