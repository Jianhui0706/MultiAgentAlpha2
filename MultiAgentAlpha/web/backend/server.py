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

def _install_benign_finalizer_filter() -> None:
    """静默 openai SDK httpx 客户端的无害 GC 终结器噪音。

    openai 的 ``SyncHttpxClientWrapper.__del__`` 会读取 ``self.is_closed`` →
    httpx 的 ``self._state``。当某个 OpenAI 客户端内部的 httpx 客户端在
    ``__init__`` 中途失败（``_state`` 尚未赋值）后被垃圾回收时，``__del__`` 会抛
    ``AttributeError: ... has no attribute '_state'``。Python 通过
    “Exception ignored in __del__” 打印它但**会忽略**——不影响任何功能（请求照常
    200）。这里只过滤这一类（AttributeError + ``_state`` + 调用栈经过 httpx/openai）
    的 unraisable 噪音，其余异常仍交给默认 hook，绝不掩盖真实错误。
    """
    default_hook = sys.unraisablehook

    def _hook(unraisable: "sys.UnraisableHookArgs") -> None:  # type: ignore[name-defined]
        exc = unraisable.exc_value
        if isinstance(exc, AttributeError) and "_state" in str(exc):
            tb = unraisable.exc_traceback
            while tb is not None:
                filename = tb.tb_frame.f_code.co_filename
                if "httpx" in filename or "openai" in filename:
                    return  # 已确认是 httpx/openai 终结器噪音，丢弃
                tb = tb.tb_next
        default_hook(unraisable)

    sys.unraisablehook = _hook


_install_benign_finalizer_filter()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.runs import router as runs_router
from api.factor_validation import router as factor_validation_router
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
app.include_router(factor_validation_router)
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
