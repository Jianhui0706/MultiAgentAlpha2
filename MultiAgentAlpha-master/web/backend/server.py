from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(BACKEND_ROOT), str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env before anything else
for env_file in [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]:
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.runs import router as runs_router
from api.ws import router as ws_router

app = FastAPI(title="MultiAgentAlpha API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router)
app.include_router(ws_router)

# Serve generated report images (matplotlib charts embedded in markdown reports)
_reports_dir = PROJECT_ROOT / "reports"
_reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/report-assets", StaticFiles(directory=str(_reports_dir)), name="report-assets")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("WEB_BACKEND_RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=reload_enabled)
