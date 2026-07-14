"""Shared test fixtures.

Tests must be hermetic: never load the developer's real `.env` and never make a
live Gemini call. This autouse fixture strips the API key and blocks the dotenv
loader so a default `Config()` has no LLM access. Tests that exercise the Gemini
path inject a fake `client` (which bypasses the key check) or set their own .env.
"""

from __future__ import annotations

import pytest

import converter.config as config_mod


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", True)  # skip .env autoload
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    yield
