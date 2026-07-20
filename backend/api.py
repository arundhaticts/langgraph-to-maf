"""FastAPI backend for the Framework Conversion Utility.

All core functionality lives in this `backend/` folder (the `converter` package
plus this API). The React app in the sibling `frontend/` folder calls these
endpoints over HTTP.

Endpoints:
    GET  /api/health  -> readiness + whether Gemini (Tier 3) is configured
    POST /api/convert -> {mode, files:[{path, content}]} -> converted agent zip
    GET  /            -> the built React app if frontend/dist exists (optional)

Run (from the repo root):
    python -m uvicorn api:app --app-dir backend --reload --port 8000
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make `converter` and `service` importable regardless of how uvicorn is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converter.adapters import list_frameworks_detailed  # noqa: E402
from converter.config import Config  # noqa: E402
from converter.scanner import ScannerError  # noqa: E402
from service import convert_folder, convert_local_path  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
# Optional convenience: serve the built frontend if present (sibling folder).
_DIST = os.path.join(os.path.dirname(_HERE), "frontend", "dist")

app = FastAPI(title="Framework Conversion Utility", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadedFile(BaseModel):
    path: str
    content: str = ""


class ConvertRequest(BaseModel):
    mode: str = "llm"                 # "llm" | "manual"
    files: list[UploadedFile]
    source: str | None = None        # source framework (None -> auto-detect)
    target: str = "maf"              # target framework name (chosen in the UI)
    # Optional legacy path: upload a target-framework pack folder (with a
    # vocabulary.json). Normally the UI just picks `target` from the frameworks/
    # folder on disk and no upload is needed.
    framework_files: list[UploadedFile] = []


class ConvertPathRequest(BaseModel):
    mode: str = "llm"                 # "llm" | "manual"
    path: str                        # local folder path (backend reads from disk)
    source: str | None = None
    target: str = "maf"
    framework_files: list[UploadedFile] = []


@app.get("/api/frameworks")
def frameworks() -> dict:
    """All known frameworks with capabilities, for the source/target dropdowns.

    Each entry: {name, display_name, source, target}. The UI filters `source:true`
    into the Source dropdown and `target:true` into the Target dropdown.
    """
    return {"frameworks": list_frameworks_detailed()}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "gemini_configured": bool(Config().llm_api_key())}


@app.post("/api/convert")
def convert(request: ConvertRequest) -> Response:
    if not request.files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    payload = [f.model_dump() for f in request.files]
    pack = [f.model_dump() for f in request.framework_files]
    try:
        zip_bytes, summary = convert_folder(
            payload, request.mode, request.target, pack, source=request.source
        )
    except (ScannerError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface failures to the UI
        raise HTTPException(status_code=500, detail=str(exc))
    folder_name = _folder_name_from_files(request.files)
    return _zip_response(zip_bytes, folder_name, request.target, summary)


@app.post("/api/convert-path")
def convert_path(request: ConvertPathRequest) -> Response:
    """Convert a folder already on disk (no upload). Backend must be local.

    Best for large folders: nothing is uploaded and the scanner skips .venv /
    node_modules / __pycache__, so a 60k-file folder is handled instantly.
    """
    if not request.path.strip():
        raise HTTPException(status_code=400, detail="No folder path provided.")
    pack = [f.model_dump() for f in request.framework_files]
    try:
        zip_bytes, summary = convert_local_path(
            request.path, request.mode, request.target, pack, source=request.source
        )
    except (ScannerError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    folder_name = os.path.basename(request.path.rstrip("/\\").rstrip()) or "agent"
    return _zip_response(zip_bytes, folder_name, request.target, summary)


def _folder_name_from_files(files) -> str:
    if not files:
        return "agent"
    first = files[0].path.replace("\\", "/").split("/")[0]
    return first or "agent"


def _make_zip_filename(folder_name: str, target: str) -> str:
    """Build the output zip name: {input_folder_name}-{target_framework}.zip.

    Sanitises each part to a safe slug (letters/digits/hyphens) and joins them
    with a single hyphen -- never a generic, internal, or temp-dir name.
    """
    import re

    def _slug(text: str, fallback: str) -> str:
        s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-").lower()
        return s or fallback

    return f"{_slug(folder_name, 'agent')}-{_slug(target, 'converted')}.zip"


def _zip_response(
    zip_bytes: bytes,
    folder_name: str = "agent",
    target: str = "converted",
    summary: dict | None = None,
) -> Response:
    filename = _make_zip_filename(folder_name, target)
    # Validation gate: the name MUST follow {input}-{target}.zip exactly.
    if not filename.endswith(".zip") or filename.count("-") < 1:
        raise HTTPException(
            status_code=500,
            detail=f"Output zip name '{filename}' does not follow "
            "'{input_folder_name}-{target_framework}.zip'.",
        )
    s = summary or {}
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Zip-Filename": filename,
        "X-Target-Framework": target,
        "X-Files-Converted": s.get("files_converted", ""),
        "X-Conversion-Time": s.get("conversion_time", ""),
        "X-Total-Human-Time": s.get("total_time", ""),
        "X-Overall-Accuracy": s.get("accuracy", ""),
        "X-Highest-Accuracy": s.get("highest_accuracy", ""),
        "X-Lowest-Accuracy": s.get("lowest_accuracy", ""),
        "X-Readiness-Pct": s.get("readiness_pct", ""),
        "X-Production-Readiness": s.get("production_readiness", ""),
        "X-Confidence": s.get("confidence", ""),
        "X-Confidence-Range": s.get("confidence_range", ""),
        "X-Effort-Low": s.get("low_end_effort", ""),
        "X-Effort-High": s.get("high_end_effort", ""),
        "X-Effort-Avg-High": s.get("avg_high_effort", ""),
    }
    headers["Access-Control-Expose-Headers"] = ", ".join(
        k for k in headers if k.startswith("X-")
    )
    return Response(content=zip_bytes, media_type="application/zip", headers=headers)


# Serve the built React app if it exists (after `npm run build`).
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
else:  # pragma: no cover - friendly hint before the first build
    @app.get("/")
    def _needs_build() -> dict:
        return {
            "message": "Backend API is running. Start the frontend separately.",
            "frontend_dev": "cd frontend && npm install && npm run dev",
            "api": ["GET /api/health", "POST /api/convert"],
        }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
