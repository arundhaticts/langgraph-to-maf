# Frontend

React (Vite) UI for the Framework Conversion Utility. It calls the backend
(FastAPI) over HTTP — upload an agent folder, choose how the hard parts are
handled, and download the converted MAF agent.

## Develop (hot reload)

```bash
npm install
npm run dev            # http://localhost:5173
```

`vite.config.js` proxies `/api` to the backend at `http://127.0.0.1:8000`, so run
the backend too:

```bash
# from the repo root
python -m uvicorn api:app --app-dir backend --reload --port 8000
```

## Build (production)

```bash
npm run build         # -> dist/
```

Serve `dist/` with any static host, or let the backend serve it (the API mounts
`../frontend/dist` at `/` if present).

## Pointing at a separate backend

If the backend runs on a different origin, set its URL at build time:

```bash
VITE_API_BASE=https://your-backend.example.com npm run build
```

Otherwise requests go to the same origin (`/api/...`), which works with the dev
proxy or when the backend serves the built app.

## Notes

- Dependency folders (`.venv`, `node_modules`, `site-packages`, …) and binaries
  are filtered out in the browser before upload — only agent source is sent.
- A spinner shows while the folder is read (“uploading…”) and while converting.
