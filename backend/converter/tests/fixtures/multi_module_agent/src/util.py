"""Utility helpers -- exercise signatures with defaults and **kwargs."""


def audit(node, event, level="info", **details):
    """A helper whose signature MUST be preserved (defaults + **kwargs)."""
    entry = {"node": node, "event": event, "level": level}
    entry.update(details)
    return entry


def configure_logging(level="INFO"):
    """Default arg must survive conversion."""
    return level.upper()
