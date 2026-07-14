"""Real configuration constants (only these should land in the output config.py)."""

import os

MAX_GEN_RETRIES = 3
COVERAGE_FLOOR = 0.8
MAX_REVISE_ITERS = 5

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
