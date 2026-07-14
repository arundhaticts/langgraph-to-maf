"""Tools -- plain functions (NO @tool decorator), detected by living in tools/."""

import os


def read_tests(path) -> list:
    """Reads test files from a path."""
    if not os.path.isdir(path):
        return []
    return [f for f in os.listdir(path) if f.startswith("test_")]


def detect_conventions(tests: list) -> dict:
    """Detects suite conventions from the given tests."""
    return {"count": len(tests), "style": "pytest"}
