# Framework Conversion Utility

A code translator for AI agents. It reads an agent written in one framework
(e.g. **LangGraph**), re-describes what it does in a framework-neutral
**Intermediate Representation (IR)**, and re-writes that description using a
target framework's idioms (e.g. **MAF**).

The IR is the single source of truth: no stage after parsing looks at the
original source again, and no stage before code generation knows what the
target framework looks like.

---

## Three approaches, one codebase

This repository implements **Approach 3 (Hybrid)** end-to-end, but the
architecture is deliberately structured so that **Approach 1 (Deterministic)**
and **Approach 2 (Full LLM)** can be added *without touching the core modules*.

| Approach | Mode enum | Parser + IR | Tier 3 LLM | Templates | Status |
|---|---|---|---|---|---|
| 1 – Deterministic | `ConversionMode.DETERMINISTIC` | yes | no (unresolved → flagged) | full set | **droppable-in** |
| 2 – Full LLM | `ConversionMode.FULL_LLM` | no (bypassed) | whole file | none | **droppable-in** |
| 3 – Hybrid | `ConversionMode.HYBRID` | yes | hard parts only | clean parts | **implemented** |

### How the other two get added

Everything runs through a **pipeline strategy** selected by
`ConversionMode` (see [converter/config.py](converter/config.py) and the
`PIPELINE_REGISTRY` in [converter/main.py](converter/main.py)):

- **Approach 1** is almost free: it is the hybrid pipeline with the Tier 3 LLM
  fallback disabled. The conversion engine already accepts
  `allow_llm_fallback=False`, in which case unresolved patterns are flagged for
  manual review instead of sent to the LLM. Adding it is:
  1. register `ConversionMode.DETERMINISTIC` in `PIPELINE_REGISTRY`
  2. (optional) add `converter/pipeline/deterministic_pipeline.py` if you want
     it distinct from `hybrid` — otherwise reuse `HybridPipeline(allow_llm_fallback=False)`.

- **Approach 2** bypasses the parser/IR/engine entirely (source + README +
  target docs → LLM → output). Adding it is:
  1. create `converter/pipeline/full_llm_pipeline.py` implementing the
     `ConversionPipeline` protocol from [converter/pipeline/base.py](converter/pipeline/base.py)
  2. register `ConversionMode.FULL_LLM` in `PIPELINE_REGISTRY`.

  It still produces the same `MigrationReport` contract, so the report/README
  generators are reused unchanged.

No core module (scanner, parser, extractor, IR builder, generators, adapters)
needs to change to support any of the three approaches — they all consume and
produce the **frozen contracts** in [converter/contracts.py](converter/contracts.py).

---

## Repository layout

```
backend/            # all core functionality + the API
├── converter/      # the conversion package (scanner, parser, IR, engine, ...)
├── api.py          # FastAPI app (frontend calls this)
└── service.py      # conversion service shared by the API and tests
frontend/           # React (Vite) app that calls the backend over HTTP
examples/           # a sample LangGraph agent to try
```

## Usage — CLI

```bash
cd backend
python -m converter.main --input ../examples/sample_agent --output ../converted --mode hybrid
```

- `--input`  folder containing the source agent (must have `README.md` at root
  and at least one `.py` file, detectable as LangGraph)
- `--output` destination folder for the converted agent + `README.md`,
  `MIGRATION_REPORT.md`, `INSTALL.md`, `ARCHITECTURE.md`
- `--mode`   `hybrid` (default) | `deterministic` | `full_llm`

## Usage — Web UI (React + FastAPI)

```bash
# backend (from repo root)
python -m uvicorn api:app --app-dir backend --reload --port 8000

# frontend (separate terminal)
cd frontend
npm install
npm run dev        # opens http://localhost:5173, proxies /api to :8000
```

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md).

The converter validates syntax only — it does **not** execute the converted
agent. That is a manual step.

---

## Backend key entry points

- [backend/converter/contracts.py](backend/converter/contracts.py) — **frozen**
  dataclass contracts (`RepoManifest`, `ReadmeSections`, `ComponentInventory`,
  `IR`, `MigrationReport`).
- [backend/converter/config.py](backend/converter/config.py) — thresholds, tier
  cutoff, `ConversionMode`, `.env` loading.
- [backend/converter/main.py](backend/converter/main.py) — CLI entry point.
- [backend/converter/pipeline/](backend/converter/pipeline/) — pipeline strategies.
- [backend/converter/adapters/](backend/converter/adapters/) — framework adapters.
- [backend/converter/frameworks/](backend/converter/frameworks/) — Tier 3 knowledge store.

---

## Status

Scaffold. Contracts are frozen and fully defined; module bodies are stubbed with
signatures, docstrings, and `TODO`s ready for implementation.
