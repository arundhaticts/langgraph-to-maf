# Backend

All core functionality of the Framework Conversion Utility lives here:

- `converter/` — the conversion package (scanner → parser → IR → engine →
  generators), adapters, templates, and the Tier 3 framework knowledge store.
- `api.py` — the FastAPI app the frontend calls.
- `service.py` — the conversion service (folder-of-files → converted zip),
  shared by the API and the tests.

## Run the API

From the repo root:

```bash
python -m uvicorn api:app --app-dir backend --reload --port 8000
```

Endpoints:

| Method | Path           | Purpose |
|--------|----------------|---------|
| GET    | `/api/health`  | readiness + whether `GEMINI_API_KEY` is set |
| POST   | `/api/convert` | `{mode, files:[{path,content}]}` → converted agent zip |
| GET    | `/`            | the built React app if `../frontend/dist` exists (optional) |

`mode` is `"llm"` (Gemini writes the hard parts, you review) or `"manual"`
(hard parts left as commented templates).

## Run the CLI

```bash
cd backend
python -m converter.main --input ../examples/sample_agent --output ../converted --mode hybrid
```

## Tests

From the repo root (pytest is configured for this layout):

```bash
python -m pytest -q
```

## Tier 3 (Gemini)

Optional. Set `GEMINI_API_KEY` in a `.env` at the repo root (auto-loaded). Tier 3
calls Gemini via the `google-genai` SDK when installed, else the stdlib REST API.
Absent a key, the hard parts degrade to reviewable stubs — no crash.
