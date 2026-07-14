"""Tests for Tier 3 (Gemini) plumbing, the MAF knowledge pack, and .env loading."""

from __future__ import annotations

import converter.config as config_mod
import converter.engine.tier3_llm as tier3
from converter.config import Config, ConversionMode
from converter.contracts import (
    IR,
    FunctionSpec,
    IRMetadata,
    OrchestrationPattern,
    ToolParam,
    WorkflowSpec,
)
from converter.engine.tier3_llm import (
    load_framework_docs,
    resolve_hitl,
    resolve_with_llm,
)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeGemini:
    """Stand-in for genai.GenerativeModel with a canned response."""

    def __init__(self, text: str):
        self._text = text
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> _FakeResponse:
        self.prompts.append(prompt)
        return _FakeResponse(self._text)


def _ir() -> IR:
    return IR(
        metadata=IRMetadata(target_framework="maf"),
        workflow=WorkflowSpec(pattern=OrchestrationPattern.AGENT_DRIVEN),
    )


# ---------------------------------------------------------------------------
# Knowledge pack
# ---------------------------------------------------------------------------

def test_framework_docs_load():
    docs = load_framework_docs("maf", Config())
    assert "AgentContext" in docs
    assert "kernel_function" in docs
    assert "HumanApprovalRequired" in docs
    # vocabulary.json is bundled into the context too.
    assert "Microsoft Agent Framework" in docs


def test_missing_framework_docs_is_empty_not_error():
    assert load_framework_docs("nonexistent_fw", Config()) == ""


# ---------------------------------------------------------------------------
# Gemini resolver (with a fake client -- no network, no SDK)
# ---------------------------------------------------------------------------

def test_resolve_with_llm_parses_json():
    fake = _FakeGemini(
        '{"pattern": "loop_with_exit", "generated_code": "def run(ctx):\\n    return ctx",'
        ' "reasoning": "loop", "confidence": 0.88}'
    )
    result = resolve_with_llm(_ir(), _ir().workflow, Config(), client=fake)
    assert result is not None
    assert result.pattern == "loop_with_exit"
    assert "def run" in result.generated_code
    assert result.confidence == 0.88
    # The prompt included the MAF knowledge pack.
    assert "AgentContext" in fake.prompts[0]


def test_resolve_handles_fenced_json():
    fake = _FakeGemini('```json\n{"pattern": "x", "generated_code": "pass", "confidence": 0.5}\n```')
    result = resolve_with_llm(_ir(), _ir().workflow, Config(), client=fake)
    assert result is not None
    assert result.generated_code == "pass"


def test_resolve_hitl_uses_original_logic_in_prompt():
    fake = _FakeGemini('{"pattern": "hitl", "generated_code": "return ctx", "confidence": 0.9}')
    source = FunctionSpec(
        name="approve",
        params=[ToolParam("state")],
        body="decision = interrupt({})\nreturn {}",
    )
    result = resolve_hitl(_ir(), "approve", source, Config(), client=fake)
    assert result is not None
    assert result.generated_code == "return ctx"
    assert "interrupt(" in fake.prompts[0]  # original logic handed to the model


def test_deterministic_mode_never_calls_llm():
    # Approach 1: fallback disabled -> returns None even with a client present.
    fake = _FakeGemini('{"generated_code": "x", "confidence": 1.0}')
    config = Config(mode=ConversionMode.DETERMINISTIC)
    assert resolve_with_llm(_ir(), _ir().workflow, config, client=fake) is None


# ---------------------------------------------------------------------------
# SDK-preferred, REST-fallback resolution order
# ---------------------------------------------------------------------------

def test_prefers_sdk_when_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(
        tier3, "_call_gemini_sdk",
        lambda p, c: '{"pattern": "x", "generated_code": "sdk", "confidence": 0.9}',
    )
    rest_called = {"hit": False}

    def _rest(p, c):
        rest_called["hit"] = True
        return '{"generated_code": "rest", "confidence": 0.9}'

    monkeypatch.setattr(tier3, "_call_gemini_rest", _rest)

    result = resolve_with_llm(_ir(), _ir().workflow, Config())
    assert result.generated_code == "sdk"
    assert rest_called["hit"] is False  # REST not used when SDK answered


def test_falls_back_to_rest_when_sdk_absent(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tier3, "_call_gemini_sdk", lambda p, c: None)
    monkeypatch.setattr(
        tier3, "_call_gemini_rest",
        lambda p, c: '{"pattern": "x", "generated_code": "rest", "confidence": 0.9}',
    )

    result = resolve_with_llm(_ir(), _ir().workflow, Config())
    assert result.generated_code == "rest"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def test_dotenv_is_loaded(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=abc123\nGEMINI_MODEL=gemini-test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", False)

    config = Config()
    assert config.llm_api_key() == "abc123"
    assert config.resolved_model() == "gemini-test"


def test_empty_key_treated_as_absent(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config_mod, "_DOTENV_LOADED", False)

    assert Config().llm_api_key() is None
