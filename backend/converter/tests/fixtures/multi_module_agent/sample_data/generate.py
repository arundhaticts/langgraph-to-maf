"""Sample-data generator -- noise that MUST be EXCLUDED (functions + fake constants)."""

GRAPH = {"nodes": 7}
SEED = 42
FILES = ["a.py", "b.py"]
GOLDEN = {"x": 1}
EXPECTED_FIELDS = ["project_id", "coverage"]


def write_ci_history(path):
    """Sample-data writer -- must NOT land in orchestrator.py or config.py."""
    return path


def make_fixtures(n=10):
    return list(range(n))
