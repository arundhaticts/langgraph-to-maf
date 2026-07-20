"""Central configuration: thresholds, tier cutoffs, and conversion mode.

The three approaches from the build plan are selected here via `ConversionMode`.
The rest of the pipeline reads this enum (through `main.PIPELINE_REGISTRY`) and
never hard-codes an approach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load KEY=VALUE lines from a `.env` file into os.environ (once).

    Zero-dependency loader (no python-dotenv needed). Searches the current
    working directory and the project root (parent of this package). Existing
    environment variables win -- `.env` never overrides an already-set value.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    _scan_dotenv()


def _scan_dotenv() -> None:
    """Read the .env candidates into os.environ (existing vars always win)."""
    # config.py is at backend/converter/config.py -> package_root = backend/.
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(package_root)  # .env typically lives here
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(package_root, ".env"),
        os.path.join(repo_root, ".env"),
    ]
    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            continue


class ConversionMode(str, Enum):
    """Which approach to run.

    - DETERMINISTIC (Approach 1): parser + IR + rules + templates only. Tier 3
      LLM fallback is DISABLED; unresolved patterns are flagged for manual work.
    - FULL_LLM (Approach 2): no parser/IR/rules/templates. Source + README +
      target docs go straight to the LLM.
    - HYBRID (Approach 3): deterministic for clean mappings, LLM only for the
      hard parts (HITL, checkpointing, complex orchestration). DEFAULT.
    """
    DETERMINISTIC = "deterministic"
    FULL_LLM = "full_llm"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Config:
    """Runtime configuration. Constructed once in main and threaded through."""

    mode: ConversionMode = ConversionMode.HYBRID

    # Tier 3: below this confidence, a decision is flagged for manual review
    # instead of auto-applied (Module 6).
    tier3_confidence_threshold: float = 0.70

    # Framework selection. `source_framework=None` -> auto-detect from imports.
    # `target_framework` picks the target adapter / knowledge pack.
    source_framework: str | None = None
    target_framework: str = "maf"

    # LLM settings for Tier 3 (Approach 2 & 3). Gemini today; swapped for the
    # local SLM later (Section 15) behind the same tier3 interface.
    llm_model: str = "gemini-2.0-flash"
    llm_api_key_env: str = "GEMINI_API_KEY"

    # Framework knowledge store root (relative to package), Tier 3 context.
    # Uploaded framework packs are stored here as `frameworks/<name>/`.
    frameworks_dir: str = "frameworks"

    # Input contract.
    required_readme_name: str = "README.md"   # case-sensitive
    required_readme_sections: tuple[str, ...] = (
        "Purpose", "Framework", "Tools", "Workflow",
        "State", "Configuration", "Dependencies",
    )

    # Directories whose Python files are NOT agent source and must be excluded
    # from component extraction (their functions/config/constants pollute the IR).
    extraction_exclude_dirs: tuple[str, ...] = (
        "tests", "test", "sample_data", "samples", "learning", "docs",
        "frontend", "logs", "outputs", "notebooks", "examples",
    )
    # Files (by basename) that are config modules -- the only place module-level
    # ALL_CAPS constants are treated as config (R-14). Prevents sweeping stray
    # constants from node/sample modules.
    config_module_names: tuple[str, ...] = ("config.py", "settings.py", "constants.py")
    # Folder name (any path segment) whose Python files are tools (R-01), even
    # without an @tool decorator.
    tools_dir_name: str = "tools"

    # Phase 11: after generating, run the runnable-validation subprocess checks
    # (import the agent_framework stub + build_workflow) and record them in
    # ACCEPTANCE.md. Off during the test suite to avoid per-test subprocess cost.
    validate_output: bool = False

    @property
    def allow_llm_fallback(self) -> bool:
        """Only the modes that are allowed to reach the LLM in Tier 3."""
        return self.mode in (ConversionMode.HYBRID, ConversionMode.FULL_LLM)

    def resolved_model(self) -> str:
        """The Gemini model, honouring a `GEMINI_MODEL` override from .env/env."""
        _load_dotenv_once()
        return os.environ.get("GEMINI_MODEL") or self.llm_model

    def llm_api_key(self) -> str | None:
        """The Gemini API key from the environment (loading `.env` if needed)."""
        _load_dotenv_once()
        value = os.environ.get(self.llm_api_key_env)
        if not value:
            # A .env edited AFTER the server started (or after the first one-time
            # load) is not in os.environ yet. Re-scan so a freshly-saved key is
            # picked up without a restart. Existing env vars still win, so this
            # can only ADD the missing key, never override a real one.
            _scan_dotenv()
            value = os.environ.get(self.llm_api_key_env)
        return value or None  # treat empty string (unfilled .env) as absent
