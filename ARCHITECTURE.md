# Architecture

The authoritative architecture reference for the Framework Conversion Utility.
For the end-to-end data flow and how to extend each seam, see
[ENGINEERING_GUIDE.md](ENGINEERING_GUIDE.md); for per-framework semantics, see
[FRAMEWORK_REFERENCE.md](FRAMEWORK_REFERENCE.md).

---

## System architecture overview

The system converts an AI-agent codebase from one framework to another by
routing everything through a **framework-neutral Intermediate Representation
(IR)**. Three tiers of layering enforce that:

1. **Delivery layer** — a React SPA and a FastAPI service, both thin. They move
   files in and a zip + metrics out; they contain no conversion logic.
2. **Pipeline layer** — an ordered sequence of stages
   (`converter/pipeline/hybrid_pipeline.py`) that scans, parses, builds the IR,
   converts, generates, verifies, and reports.
3. **Knowledge layer** — adapters (`converter/adapters/`), Tier-1 rules
   (`converter/engine/`), Jinja templates (`converter/templates/`), and Tier-3
   knowledge packs (`converter/frameworks/`). All framework-specific knowledge
   lives here, behind interfaces.

The invariant that holds the design together: **nothing after the parser reads
the original source, and nothing before code generation knows the target
framework.** The IR (`converter/contracts.py`) is the only thing that crosses
that boundary.

```mermaid
flowchart TB
    subgraph Delivery
      UI["React SPA<br/>frontend/src/App.jsx"]
      API["FastAPI<br/>backend/api.py"]
      SVC["Service<br/>backend/service.py"]
    end
    subgraph Pipeline["HybridPipeline (converter/pipeline)"]
      direction TB
      S1[scanner] --> S2[parser] --> S3[IR builder]
      S3 --> S4[IR validate + capability negotiation]
      S4 --> S5[Tier 1/2/3 engine]
      S5 --> S6[code generator]
      S6 --> S7[acceptance gate: verify.py]
      S7 --> S8[reports + readiness]
      S8 --> S9[LLM refinement]
    end
    subgraph Knowledge
      ADP[adapters registry]
      RULES[engine tier1_rules]
      TPL[Jinja templates]
      PACKS[frameworks/*/vocabulary.json + docs.md + examples]
    end
    UI -->|HTTP| API --> SVC --> S1
    S8 --> SVC -->|zip + X-* headers| UI
    S2 -.uses.-> ADP
    S5 -.uses.-> RULES
    S5 -.Tier 3.-> PACKS
    S5 -.Tier 3.-> GEM[(Gemini)]
    S6 -.uses.-> TPL
    S6 -.uses.-> ADP
```

## Architectural goals

- **Any-source → any-target without N×M code.** Achieved by the neutral IR: N
  source readers + M target writers, never N×M translators.
- **Frozen contracts.** `contracts.py` is explicitly frozen (Section 9 of the
  build plan). All three conversion modes and every adapter speak these exact
  dataclasses, so a change on one side never ripples.
- **Deterministic-first, LLM-only-where-needed.** Structure that can be
  converted by rule is (Tier 1/2); the LLM (Tier 3) is reserved for genuinely
  ambiguous orchestration and HITL, keeping output reproducible and cheap.
- **Runnable output, offline.** Generated packages import and run without the
  target SDK (import guards + offline `run()` +, for MAF, a bundled stub), so
  CI can smoke-test them immediately.
- **Honest reporting over silent success.** Every conversion emits Markdown
  reports bucketing work into auto / needs-review / manual, plus computed
  effort and accuracy metrics.
- **Onboarding a framework = adding files, not editing the core.**

## Frontend architecture

A single-page React 18 app built with Vite; the entire UI is one component,
`frontend/src/App.jsx` (no sub-components), with `frontend/src/styles.css`.

- **State** (React `useState`): input `method` (`path` | `upload`), `path`,
  `files`, `mode` (`llm` | `manual`), `frameworks`, `source`, `target`,
  custom-pack `packFiles`, transient flags (`reading`, `converting`), `status`,
  `result`, and a `log` array for the activity stream.
- **Framework discovery:** on mount it calls `GET /api/frameworks` and filters
  the result into Source and Target dropdowns (`f.source` / `f.target`). A
  sentinel value `__new__` ("New / custom framework…") switches on a
  `vocabulary.json` pack uploader and sets `target='dynamic'`.
- **Client-side pre-filtering:** `isSource()` drops dependency dirs
  (`IGNORE_DIRS`), binaries (`IGNORE_EXT`), and files over `MAX_FILE_BYTES`
  (1 MB) before upload — so path mode stays instant even for huge repos.
- **Conversion call:** `POST /api/convert-path` (path mode) or `/api/convert`
  (upload mode). The response is a zip blob auto-downloaded using the
  `X-Zip-Filename` header; KPI cards read `X-Target-Framework`,
  `X-Files-Converted`, `X-Conversion-Time`, `X-Readiness-Pct`,
  `X-Overall-Accuracy`, `X-Lowest/Highest-Accuracy`, `X-Total-Human-Time`.
- **Live activity log:** `startActivityStream()` simulates an 8-stage timed log
  (1300 ms interval) locally — it is *not* server-streamed, because the backend
  is a single synchronous endpoint (see *Service interactions*).

## Backend architecture

Two files, both thin:

- **`backend/api.py`** — the FastAPI app. Endpoints:
  `GET /api/frameworks` (dropdown data via `list_frameworks_detailed()`),
  `GET /api/health` (+ whether Gemini is configured), `POST /api/convert`
  (upload), `POST /api/convert-path` (read from disk). It builds the output zip
  filename (`_make_zip_filename` → `{input}-{target}.zip`), validates it, and
  attaches ~16 `X-*` headers (exposed via `Access-Control-Expose-Headers`). If
  `frontend/dist/` exists it also mounts the built SPA at `/`.
- **`backend/service.py`** — the conversion service used by both the API and the
  tests (no web imports, easy to test). `convert_folder` / `convert_local_path`
  write/resolve the input, call `_run_and_zip`, which runs `HybridPipeline`,
  reads the `readiness_metrics.json` sidecar (`_extract_report_summary`), walks
  the output into a zip, times the run (`_fmt_elapsed`), and returns
  `(zip_bytes, summary_dict)`.

Request models: `ConvertRequest`, `ConvertPathRequest` (pydantic). CORS is
open (`allow_origins=["*"]`).

## Conversion engine architecture

The pipeline is an ordered list of stages in
`converter/pipeline/hybrid_pipeline.py::HybridPipeline.run`. Each stage maps to
a numbered "Module" from the build plan:

Each row gives the **entry function** (`file::function`), the **helpers** that do
the work with **why each exists** and which file/folder it lives in, and the
**data in → out**. `run()` only wires these in order — it holds no conversion
logic itself. (`•` separates helpers within a cell.)

| Stage · Module | Entry point `file::function` | Helpers — why each exists | Data in → out |
|---|---|---|---|
| **1 · M1** scan+validate | `scanner/repo_scanner.py::scan_repo` | • `_iter_repo_files` (same file) — prune `_IGNORED_DIRS` so a repo carrying `.venv`/`node_modules` doesn't flood the scan<br>• `_classify_file` — tag README/py/requirements so each later stage picks the right handler<br>• `_detect_framework`→`_imported_roots` — read imports via **AST not regex**, so an unparseable file degrades instead of crashing detection<br>• `adapters/__init__.py::detect_source_framework` — registry-scored, so adding a `SourceAdapter` makes its framework detectable with zero scanner edits | folder path → `RepoManifest` (`input_root`, `files:[FileEntry]`, `detected_framework`, `readme_path`) |
| **2 · M3** parse README | `parser/readme_parser.py::parse_readme_file` | • section splitter (same file) — keep `workflow_description` **verbatim** so Tier 2 can keyword-classify the author's own prose | `manifest.readme_path` → `ReadmeSections` |
| **3 · M4** extract | `extractor/component_extractor.py::extract_components` | • `adapters/__init__.py::get_source_adapter().vocabulary()` — the **only** framework-specific input to the parser; swapping source = swapping the `SourceVocabulary`, no parser edits<br>• `parser/code_parser.py::extract_{tools,graph,state,config,functions,imports,preamble,state_class_names}` — one AST pass per concern, fully framework-agnostic<br>• `SourceAdapter.extract_graph/agents/tasks` — frameworks whose graph isn't builder-method calls (CrewAI `Crew/Task`, Strands `Agent`) override extraction<br>• `_merge_graph`/`_merge_config`/`_tools_from_module` (same file) — fold a multi-file repo into one inventory (dedup nodes/fields)<br>• `_cross_reference` — compare AST vs README, emit warnings; **AST wins**, drift is never silently applied<br>• `_assign_file_action` — decide per file: REWRITE / ADAPT / COPY_THROUGH | sources + `readme` → `ComponentInventory`; sets `FileEntry.file_action` in place |
| **4 · M5** build IR | `ir/ir_builder.py::build_ir` (+ `write_ir_json`) | • `_classify_roles` (`_build_adjacency`,`_nodes_in_cycle`,`_leads_to_end`,`_is_hitl`) — a node's `NodeRole` decides how it is generated (HITL/loop/branch differ)<br>• `_classify_pattern` — the `OrchestrationPattern` decides Tier 1 vs escalation<br>• `_analyze_node` — record per-node reads/writes so the validator can catch use of an undeclared state field<br>• `_back_edges`/`_flatten_edges`/`_loop_guards` — make implicit cycles explicit and find the iteration-cap constant<br>• `_build_hitl_points`/`_interrupt_payload` — capture the **verbatim** `interrupt(...)` payload for the approval flow<br>• `_classify_mode` — single-agent vs graph-workflow output shape<br>• `_detect_entrypoint` — emit a CLI or FastAPI `main.py`<br>• `detect_target_version` — record which target-SDK surface the conversion targeted | `ComponentInventory`+`ReadmeSections`+`RepoManifest` → `IR`; `ir.json` checkpoint to cwd |
| **4.5 · Phase 2** validate IR | `ir/validator.py::validate_ir` | • whole-IR checks — surface source inconsistencies to the report instead of shipping silently-broken output | `IR` → `list[str]` issues (non-fatal, prefixed `[IR]`) |
| **4.6 · Phase 5b** negotiate | `engine/capability_negotiation.py::negotiate` (+ `negotiation_summary`) | • `TargetAdapter.capability_matrix()` (via `get_target_adapter`) — learn LOSSY/UNSUPPORTED constructs **before** generation so stubs are tagged, not discovered at runtime | `IR`+matrix → `[CapabilityNegotiation]` → `[CAP-*]` issues |
| **5 · M6** convert | `engine/conversion_engine.py::convert` | • `tier1_rules.py::{tools,state,config,checkpointer,workflow}_rules` — deterministic, reproducible, free for clean mappings<br>• `tier2_readme.py::classify_from_readme` — fall back to the human's prose when structure is ambiguous<br>• `tier3_llm.py::resolve_with_llm/resolve_hitl` (`load_framework_docs`,`_call_gemini`) — spend the LLM **only** on the hard parts (orchestration/HITL)<br>• `_resolve_workflow`/`_resolve_hitl_nodes` (same file) — enforce the Tier 1→2→3 escalation order | `IR` → `ConversionResult` (`[ConversionUnit]`) |
| **6 · M7** generate | `generator/code_generator.py::generate_from_paths`→`generate` | • `targets/__init__.py::get_target_generator` → `TargetGenerator.*` — keep framework-specific code shape out of the agnostic core<br>• `get_target_adapter` → `TargetAdapter` — read idioms (decorator, context class) instead of hardcoding them<br>• `body_porter.py::{port_node_function,port_plain_function,plugin_method_body}` — the hard deterministic bit: rewrite `state["x"]`→`ctx.x` while porting bodies<br>• `templates/*.jinja` via `_env()` — render deterministic files from data, keeping code emission out of Python string soup | `IR`+`ConversionResult`+source root → `GenerationResult` + output tree |
| **6.5 · Phase 11** accept gate | `verify.py::verify_output` (+ `verify_runnable`, `write_acceptance`) | • 7 checks (compile / residue / requirements / IR coverage / orchestrator tokens / loop reachability / opt-in subprocess) — prove the package is coherent before shipping; **never raises** (a failed check is data, not an exception) | `IR`+`GenerationResult` → `AcceptanceReport` → `ACCEPTANCE.md`; `[ACCEPTANCE]` warnings |
| **7 · M8** README | `generator/readme_generator.py::build_readme` (+ `write_readme`) | • `TargetAdapter.README_VOCAB` (R-13) — headings use target vocabulary (`## Tools`→`## Skills`) so the output reads native | `IR`+`ConversionResult` → `README.md` |
| **8 · M9** migration report | `generator/report_generator.py::build_report` (+ `write_report`) | • unit bucketing — split decisions into auto / needs-review / manual so a human sees exactly what to check | → `MIGRATION_REPORT.md` (the returned `report`) |
| **9 · —** docs | `generator/docs_generator.py::write_docs` | • template render — ship self-describing output | → `INSTALL.md` + `ARCHITECTURE.md` |
| **10 · —** readiness | `generator/readiness_report.py::generate_readiness_report` | • `collect_facts`/`_scan_code_stubs` — gather deterministic grounding (stubs, HITL, orphan tools) so metrics don't depend on the LLM<br>• `_build_work_rows`/`_build_acc_rows`/`compute_readiness_metrics` — compute effort/accuracy from the report's own rows so numbers are exact and reproducible<br>• `_call_gemini` vs `_fallback_report`/`_fallback_banner` — LLM prose when a key works, honest deterministic body otherwise (banner states the real reason)<br>• `validate_metrics` — the **only hard gate**: abort rather than ship missing/invalid numbers | → `READINESS_REPORT.md` + `readiness_metrics.json` (read by `service.py`) |
| **11 · —** refinement | `generator/llm_refinement.py::run_llm_refinement` (+ `write_refinement_log`) | • `_build_prompt` — feed the **actual** gate failures + current code back to the LLM<br>• `_apply_patches`/`_validate_python` — apply only patches that `ast.parse` cleanly, and only to pre-existing files (never introduce broken/foreign files)<br>• re-invokes `verify.py::verify_output` — gate-closed loop: repair until green or cap; skipped when `not config.allow_llm_fallback` | → `REFINEMENT_LOG.md` (+ re-runs the gate if files changed) |

**Mode branching** is not inside `run()`; it is expressed by
`allow_llm_fallback` (constructor arg, derived from `Config.allow_llm_fallback`,
true only for `HYBRID`/`FULL_LLM`). Stage 5's `convert()` and Stage 11's
refinement early-return when it is false. The mode→pipeline mapping is
`PIPELINE_REGISTRY` in `converter/main.py`.

```mermaid
flowchart LR
    IN[input_path] --> M1[scan] --> M3[readme] --> M4[extract]
    M4 --> M5[build IR] --> P2[validate IR] --> P5b[capability negotiation]
    P5b --> M6[convert Tier 1/2/3] --> M7[generate]
    M7 --> P11[acceptance gate] --> M8a[README] --> M9[migration report]
    M9 --> DOCS[install/arch docs] --> RDY[readiness + gate]
    RDY --> REF[LLM refinement] --> OUT[(output package)]
    M5 -->|ir.json| DISK[(cwd)]
```

## Framework discovery architecture

Frameworks are discovered from three sources, unioned by
`adapters/__init__.py::list_frameworks_detailed()`:

1. `SOURCE_ADAPTERS` registry (built-in source readers),
2. `TARGET_ADAPTERS` registry (built-in target writers),
3. on-disk `frameworks/<name>/vocabulary.json` packs.

Each entry becomes `{name, display_name, source: bool, target: bool}`. A pack's
`supports_source` / `supports_target` keys and `display_name` feed the UI
directly, so **dropping a folder with a `vocabulary.json` makes a framework
appear in the API and UI with no code change**. Built-in class names take
precedence over on-disk packs of the same name.

`detect_source_framework(imported_roots)` picks the source when `--source` is
omitted: the registered `SourceAdapter` with the highest positive `detect()`
score wins (registry-driven, no hardcoded signature table).

## Framework abstraction layer

Two adapter families, both ABCs in `adapters/base.py`:

- **`SourceAdapter`** — owns how to *read* a framework: `import_signatures()`,
  `source_packages()` (dropped from output requirements), `vocabulary()` (a
  `SourceVocabulary` whose defaults are the LangGraph values), `detect()`, and
  optional `extract_agents()` / `extract_tasks()` / `extract_graph()` overrides
  for frameworks whose graph isn't graph-builder method calls (CrewAI, Strands).
- **`TargetAdapter`** — owns target *idioms*: `plugin_class_name()`,
  `method_name()`, `context_class_name`, `tool_style()`, `tool_decorator*()`,
  `runtime_requirements()`, `capability_matrix()`, `README_VOCAB`.

Paired with each target adapter is a **`TargetGenerator`**
(`generator/targets/base.py`, registry `TARGET_GENERATORS`) that emits the
framework-specific code: `workflow_block`, `entrypoint`, `smoke_test`,
`sdk_stub_files`, `extra_files`, `orchestrator_must_tokens`. The generator core
(`code_generator.py`) is a framework-agnostic dispatcher; it reads idioms off
the adapter and delegates framework-specific emission to the generator.

`DynamicTargetAdapter` is the "any target" seam: it is built at runtime from an
uploaded `vocabulary.json` (tool decorator, context class, requirements,
capability map, reject tokens), and pairs with the MAF generator as a fallback.

## Template rendering architecture

Deterministic (Tier 1/2) output is rendered with Jinja2 from
`converter/templates/` (`Environment(FileSystemLoader, keep_trailing_newline=True)`
in `code_generator.py::_env`). Four templates:

- `agent_context.py.jinja` — the state model (pydantic `BaseModel` with
  `advance(**updates)`; reducer fields extend lists, others replace).
- `orchestrator.py.jinja` — the orchestrator skeleton; its final slot
  `maf_workflow` receives *any* target's `workflow_block` (the variable name is
  legacy; it is not MAF-specific).
- `plugin_class.py.jinja` — Semantic-Kernel-style plugin classes (only for the
  `plugin_class` tool style; function-style targets skip it).
- `readme_maf.md.jinja` — the output README (title, purpose, skills, workflow,
  HITL, context fields, config).

Tier-3 LLM `generated_code` is not templated; it is stitched directly into
`orchestrator.py` (when a workflow unit carries `generated_code`, the
deterministic `maf_block` is replaced by it and the must-token check relaxes to
`["def run"]`).

## LLM integration architecture

Tier 3 uses Google Gemini, entirely optional and degrade-safe
(`engine/tier3_llm.py`):

- **Resolution order in `_call_gemini`:** injected client (tests) →
  `google-genai` SDK (`from google import genai`, with `truststore` TLS injection
  for corporate proxies) → stdlib `urllib` REST POST to the Generative Language
  API. Any failure returns `None`; nothing raises.
- **Gating:** returns `None` immediately if `not config.allow_llm_fallback` or no
  key/client — so deterministic mode never calls out.
- **Grounding:** `load_framework_docs(target)` reads the target's knowledge pack
  (`docs.md`, `vocabulary.json`, all `examples/*.py`) and prepends an
  "AUTHORITATIVE — this wins over prior knowledge" header before the prompt.
- **Where it fires:** only for orchestration shape Tiers 1/2 can't resolve
  (`resolve_with_llm`) and HITL nodes (`resolve_hitl`). Results are parsed into a
  frozen `Tier3Result(pattern, generated_code, reasoning, confidence)`; below
  `config.tier3_confidence_threshold` (0.70) the unit is flagged
  `needs_review`.
- **Second LLM stage:** `generator/llm_refinement.py` (Stage 11) is a
  gate-closed repair loop (max 3 iterations) that feeds the acceptance-gate
  failures + readiness report + current files back to Gemini, applies only
  patches that `ast.parse` cleanly and target files that already existed, and
  re-runs the gate. Writes `REFINEMENT_LOG.md`.

## Report generation architecture

Reports are the product's "explainability" layer:

- **`MIGRATION_REPORT.md`** (`report_generator.py::build_report`) — three
  buckets: *Auto-converted* (Tier 1/2 + confident Tier 3), *Needs review*
  (Tier 3 below threshold + `[VALIDATION]` warnings), *Manual action required*
  (R-08/R-15 stubs, unresolved workflow, `[SYNTAX]` ast failures).
- **`ACCEPTANCE.md`** (`verify.py`) — the acceptance gate's checks as data.
- **`READINESS_REPORT.md` + `readiness_metrics.json`**
  (`readiness_report.py`) — deterministically computed effort ranges
  (`recommended = (low + 2·high)/3`, per-item capped) and accuracy metrics
  (average/highest/lowest, confidence band, production-readiness label). The
  markdown may be LLM-authored, but the metrics and the summary block are always
  computed deterministically and validated by `validate_metrics()` (the
  pipeline's one hard gate). The JSON sidecar is what `service.py` reads to
  populate UI headers — no Markdown re-parsing.
- **`INSTALL.md` / `ARCHITECTURE.md`** (`docs_generator.py`) — so the output is
  self-describing.

## Configuration architecture

A single frozen dataclass `Config` (`config.py`) carries every knob: `mode`,
`source_framework`, `target_framework`, `tier3_confidence_threshold` (0.70),
`llm_model` / `llm_api_key_env` (`GEMINI_API_KEY`), `frameworks_dir`,
`required_readme_sections`, `extraction_exclude_dirs`, `validate_output`
(subprocess runnable checks; off in tests). Derived: `allow_llm_fallback`
(property), `resolved_model()` (honors `GEMINI_MODEL`), `llm_api_key()` (empty
string treated as absent). `.env` is loaded once by a dependency-free loader
where existing env vars always win.

## Validation architecture

Validation is layered and mostly *non-fatal* (recorded as data, surfaced in
reports) with exactly one hard gate:

```mermaid
flowchart TD
    A[IR validation - Stage 4.5] -->|ir_issues| R[reports]
    B[capability negotiation - Stage 4.6] -->|CAP-LOSSY / CAP-UNSUPPORTED| R
    C[ast.parse per generated file] -->|SYNTAX ERROR banner + syntax_errors| R
    D[_validate_target must-tokens] -->|validation_warnings| R
    E[acceptance gate verify.py - Stage 6.5] -->|ACCEPTANCE warnings| R
    F[validate_metrics - Stage 10] -->|ValueError| STOP[pipeline aborts]
```

- **IR validation** — `validate_ir()` produces non-fatal issues.
- **Capability negotiation** — LOSSY / UNSUPPORTED constructs become warnings.
- **Syntax** — every generated `.py` is `ast.parse`d in
  `code_generator._write_python`; failures get a `# SYNTAX ERROR` banner and are
  listed (still written, so a human can fix them).
- **Must-tokens** — `_validate_target` checks framework constructs are present
  (e.g. `WorkflowBuilder`, `@executor` for MAF; `StateGraph` for LangGraph;
  `Agent`, `_HAVE_STRANDS` for Strands).
- **Acceptance gate** — `verify.py` runs 7 checks (all `.py` compile; no
  source-framework residue; clean `requirements.txt`; IR coverage; required
  orchestrator tokens; loop reachability; opt-in subprocess run). Never raises.
- **Metrics gate** — `validate_metrics()` is the only hard failure: it aborts if
  the readiness metrics are missing/invalid.

## Logging architecture

Logging is intentionally **minimal**. There is no central logger config, no
handlers, and no structured logging layer. The one production logger is in
`parser/readme_parser.py` (`logger.warning` for a missing README section). The
CLI uses `print()` for user output; the dominant error idiom is *swallow and
degrade* (broad `except: pass` / `return None`) so the pipeline never crashes on
an optional step. **Diagnostics are delivered through the generated Markdown
reports** (`MIGRATION_REPORT`, `ACCEPTANCE`, `READINESS_REPORT`,
`REFINEMENT_LOG`), not through logs. This is a deliberate trade-off (see below)
and a known area for hardening.

## Component responsibilities

| Component | Responsibility |
|---|---|
| `frontend/src/App.jsx` | Collect input, call the API, render KPIs/activity, download the zip. |
| `backend/api.py` | HTTP surface, zip naming/headers, static SPA mount. |
| `backend/service.py` | Run the pipeline, package the zip, extract metrics. |
| `converter/scanner.py` | Discover + validate source files, ignore dependency dirs. |
| `converter/parser/` | README parsing + AST extraction into inventory. |
| `converter/ir/` + `contracts.py` | Assemble + validate the neutral IR. |
| `converter/engine/` | Tier 1/2/3 conversion decisions + capability negotiation. |
| `converter/generator/` | Emit target code, port logic, verify, generate reports. |
| `converter/generator/targets/` | Framework-specific code emission. |
| `converter/adapters/` | Framework knowledge behind interfaces + registry. |
| `converter/templates/` | Deterministic code/README rendering. |
| `converter/frameworks/` | Tier 3 knowledge packs + `vocabulary.json`. |

## Service interactions

```mermaid
sequenceDiagram
    participant UI as React SPA
    participant API as FastAPI (api.py)
    participant SVC as service.py
    participant PIPE as HybridPipeline
    participant FS as Output dir (temp)

    UI->>API: GET /api/frameworks
    API-->>UI: [{name, display_name, source, target}]
    UI->>API: POST /api/convert-path {mode, path, source, target}
    API->>SVC: convert_local_path(...)
    SVC->>PIPE: run(input_path, out_dir)
    PIPE->>FS: write package + reports + readiness_metrics.json
    PIPE-->>SVC: MigrationReport
    SVC->>FS: read readiness_metrics.json, walk + zip
    SVC-->>API: (zip_bytes, summary)
    API-->>UI: 200 zip + X-* headers (Content-Disposition, X-Readiness-Pct, ...)
    UI->>UI: auto-download {input}-{target}.zip, render KPI cards
```

The whole pipeline runs **synchronously inside the request**. There is no job
queue; the SPA's activity log is a client-side simulation, not a server stream.

## Dependency relationships

```mermaid
flowchart TD
    APIpy[api.py] --> SVCpy[service.py] --> PIPE[pipeline]
    PIPE --> SCAN[scanner] & PARSE[parser] & IRB[ir] & ENG[engine] & GEN[generator]
    ENG --> ADP[adapters] & FRPACKS[frameworks packs] & TIER3[tier3_llm]
    GEN --> ADP & TPL[templates] & BODY[body_porter] & VERIFY[verify]
    ADP --> CONTR[contracts.py]
    ENG --> CONTR
    GEN --> CONTR
    PARSE --> CONTR
    IRB --> CONTR
```

Everything points at `contracts.py`; nothing in the engine or generator imports
the delivery layer. The adapters and Tier-3 packs are the only places with
framework literals.

## Design decisions and rationale

- **Neutral IR (why, not just what):** direct framework-to-framework translation
  is O(N×M) and every new framework destabilises the others. The IR makes it
  O(N+M) and lets the test matrix guarantee "improving one target can't regress
  another" (`test_all_frameworks_matrix.py`).
- **Frozen contracts:** freezing `contracts.py` lets three conversion modes and
  every adapter evolve independently — the compile-time-ish guarantee that the
  seams won't drift.
- **AST-only parsing, no regex on Python:** regex over source is brittle; the
  parser walks the real AST and is driven purely by a `SourceVocabulary`, so a
  new source is a data change, not new parsing code.
- **Tiered engine:** deterministic rules are reproducible, free, and reviewable;
  the LLM is expensive and non-deterministic. Using the LLM only for what rules
  can't resolve keeps most output stable and auditable.
- **Offline-runnable output (MAF SDK stub):** shipping a pure-Python stub +
  import guards means CI can import/smoke-test the converted package with no SDK,
  and the deterministic `run()` still works — critical because the converter
  never executes agents itself.
- **Reports as the observability layer:** since runtime is never verified, the
  product's trust comes from explicit, computed reports rather than a green
  checkmark.
- **Metrics computed deterministically, JSON sidecar for the UI:** avoids
  re-parsing Markdown and guarantees the manager-facing numbers are exact and
  gate-validated.

## Trade-offs and limitations

- **Synchronous conversion.** Simple and stateless, but a long conversion blocks
  the worker; no progress streaming (the UI simulates it). Fine for local/single
  use; needs a queue for multi-tenant scale.
- **Minimal logging.** Great for "never crash," poor for post-hoc debugging of a
  silently-degraded step. Reports partially compensate.
- **No runtime verification.** Syntax + import + acceptance gate only. A
  converted agent can compile and still misbehave at runtime (ported business
  logic, or a non-target LLM client that isn't auto-rewritten to the target's
  provider).
- **LLM semantic limits.** Tier 3 can produce plausible-but-wrong orchestration;
  the confidence threshold and `needs_review` bucket mitigate but don't
  eliminate this.
- **Open CORS.** Convenient in dev, unsafe in production.

## Scalability considerations

- Conversion is CPU-bound and stateless per request; horizontal scaling is just
  more uvicorn workers behind a load balancer — but each request holds a worker
  for the full pipeline. For throughput, move `HybridPipeline.run` behind a task
  queue and stream progress over websockets/SSE instead of the current
  client-side simulation.
- Path mode avoids upload cost entirely (backend reads disk, skips
  `extraction_exclude_dirs`), so large monorepos convert quickly; upload mode is
  bounded by browser pre-filtering (`MAX_FILE_BYTES`, ignore lists).
- Tier 3 latency dominates when enabled; deterministic mode is near-instant.

## Security considerations

- **CORS `*`** — lock down before public exposure.
- **Path mode reads arbitrary server paths** — only expose to trusted callers;
  treat the path as untrusted input for traversal.
- **Secrets never copied** — `code_generator._is_secret_file` blocks `.env`,
  keys, and credentials from the output; a `.env.example` is emitted instead.
- **LLM data egress** — Tier 3 sends source snippets to Gemini; deterministic
  mode keeps everything local. Choose the mode accordingly for sensitive code.
- **Generated code is not sandboxed** — the converter never runs it, but a human
  who does should review the `MIGRATION_REPORT.md` first.

## Extensibility model

Adding a framework is additive — new files against the frozen contracts, no core
edits — with four levels of investment:

- **Level 0 (data only):** drop `frameworks/<name>/vocabulary.json` → appears in
  the UI, served as a `DynamicTargetAdapter` with the MAF generator.
- **Level 1:** add a `TargetAdapter` (idioms, capability matrix).
- **Level 2:** add a `TargetGenerator` (framework-specific code shape).
- **Level 3:** add a `SourceAdapter` (read a new source framework).

Register in the `SOURCE_ADAPTERS` / `TARGET_ADAPTERS` / `TARGET_GENERATORS`
dicts, add the pair to `test_all_frameworks_matrix.py`, and the discovery layer
does the rest. Full recipes with code are in
[ENGINEERING_GUIDE.md](ENGINEERING_GUIDE.md) and
[FRAMEWORK_AUTHORING.md](FRAMEWORK_AUTHORING.md).
