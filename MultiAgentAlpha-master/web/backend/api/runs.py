from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from trading_agents.infra.run_store import RunStore
from services import run_reader, runner
from services.ws_manager import ws_manager

router = APIRouter(prefix="/api")

_DB_PATH = RunStore.default_path(PROJECT_ROOT)

_DATE_FIELD_CANDIDATES = (
    "datetime",
    "date",
    "trade_date",
    "trading_date",
    "cal_date",
    "timestamp",
    "time",
    "dt",
    "ds",
)


def _resolve_db_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _db() -> Path:
    env = os.getenv("RUN_STORE_PATH", "").strip()
    return _resolve_db_path(env) if env else _DB_PATH


def _read_dbs() -> list[Path]:
    env = os.getenv("RUN_STORE_PATH", "").strip()
    if env:
        path = _resolve_db_path(env)
        return [path] if path.exists() else []
    if _DB_PATH.exists():
        return [_DB_PATH]

    candidates = [_DB_PATH]
    artifacts_dir = PROJECT_ROOT / "artifacts"
    if artifacts_dir.exists():
        candidates.extend(sorted(artifacts_dir.glob("run_store*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True))

    seen: set[Path] = set()
    dbs: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        dbs.append(path)
    return dbs


def _find_db_for_run(run_id: str) -> Path | None:
    for db_path in _read_dbs():
        if run_reader.get_run(db_path, run_id) is not None:
            return db_path
    return None


def _factor_score(factor: dict[str, Any]) -> float:
    reward_score = factor.get("reward_score")
    if isinstance(reward_score, (int, float)):
        return float(reward_score)
    sharpe = factor.get("sharpe")
    if isinstance(sharpe, (int, float)):
        return float(sharpe) * 0.1
    return 0.0


def _rank_top_factors(factors: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(factors, key=_factor_score, reverse=True)[: max(1, int(limit))]
    for rank, factor in enumerate(ranked, start=1):
        factor["rank"] = rank
    return ranked


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1_048_576:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1_048_576:.1f} MB"


def _parse_datetime_values(values: Any) -> Any:
    import pandas as pd

    raw = pd.Series(values).dropna()
    if raw.empty:
        raise ValueError("no non-empty date values")

    text = raw.astype(str).str.strip()
    compact_digits = text.str.fullmatch(r"\d{8}")
    if bool(compact_digits.mean() >= 0.8):
        dt = pd.to_datetime(text.where(compact_digits), format="%Y%m%d", errors="coerce")
    elif pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        max_value = numeric.dropna().abs().max()
        if pd.isna(max_value):
            dt = pd.to_datetime(raw, errors="coerce")
        elif max_value >= 10**17:
            dt = pd.to_datetime(numeric, unit="ns", errors="coerce")
        elif max_value >= 10**14:
            dt = pd.to_datetime(numeric, unit="us", errors="coerce")
        elif max_value >= 10**11:
            dt = pd.to_datetime(numeric, unit="ms", errors="coerce")
        elif max_value >= 10**9:
            dt = pd.to_datetime(numeric, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(raw, errors="coerce")
    else:
        dt = pd.to_datetime(raw, errors="coerce")

    dt = pd.Series(dt).dropna()
    if dt.empty:
        raise ValueError("no parseable date values")
    return dt


def _detect_datetime_series(df: Any) -> tuple[Any, str]:
    import pandas as pd

    if isinstance(df.index, pd.DatetimeIndex):
        return _parse_datetime_values(df.index), "index"

    index_names = list(getattr(df.index, "names", []) or [])
    for candidate in _DATE_FIELD_CANDIDATES:
        for index_name in index_names:
            if index_name and str(index_name).lower() == candidate:
                return _parse_datetime_values(df.index.get_level_values(index_name)), f"index:{index_name}"

    lower_to_column = {str(col).lower(): col for col in df.columns}
    for candidate in _DATE_FIELD_CANDIDATES:
        column = lower_to_column.get(candidate)
        if column is not None:
            return _parse_datetime_values(df[column]), f"column:{column}"

    fuzzy_columns = [
        col for col in df.columns
        if any(token in str(col).lower() for token in ("date", "datetime", "timestamp", "time"))
    ]
    for column in fuzzy_columns:
        try:
            return _parse_datetime_values(df[column]), f"column:{column}"
        except ValueError:
            continue

    return _parse_datetime_values(df.index), "index"


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for db_path in _read_dbs():
        for run in run_reader.list_runs(db_path):
            runs.setdefault(str(run.get("run_id", "")), run)
    return sorted(runs.values(), key=lambda item: str(item.get("started_at", "")), reverse=True)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    db_path = _find_db_for_run(run_id)
    if db_path is None:
        raise HTTPException(404, "run store not found")
    run = run_reader.get_run(db_path, run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    run["loops"] = run_reader.get_loops(db_path, run_id)
    return run


@router.get("/runs/{run_id}/factors")
def get_factors(run_id: str) -> list[dict[str, Any]]:
    db_path = _find_db_for_run(run_id)
    if db_path is None:
        return []
    return run_reader.get_factors(db_path, run_id)


@router.get("/runs/{run_id}/lineage")
def get_lineage(run_id: str) -> list[dict[str, Any]]:
    db_path = _find_db_for_run(run_id)
    if db_path is None:
        return []
    return run_reader.get_trajectories(db_path, run_id)


@router.get("/runs/{run_id}/progress")
def get_progress(run_id: str, limit: int = 200) -> list[dict[str, Any]]:
    db_path = _find_db_for_run(run_id)
    run = run_reader.get_run(db_path, run_id) if db_path is not None else None
    log_dir = run["log_dir"] if run else str(PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id)
    return run_reader.get_progress_events(log_dir, limit=limit)


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    result = runner.request_stop_run(run_id)
    if result.get("status") != "not_active":
        return result

    db_path = _find_db_for_run(run_id)
    run = run_reader.get_run(db_path, run_id) if db_path is not None else None
    if run is None:
        raise HTTPException(404, f"run {run_id} is not active")

    if run.get("status") == "running":
        summary = {
            "run_id": run_id,
            "status": "cancelled",
            "error": "run was marked running but is not active in the current backend process",
        }
        RunStore(db_path).finish_run(run_id=run_id, status="cancelled", summary=summary)
        progress_file = Path(str(run.get("log_dir", ""))) / "progress.jsonl"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.open("a", encoding="utf-8").write(
            json.dumps(
                {
                    "stage": "run_cancelled",
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": summary["error"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return {"run_id": run_id, "status": "cancelled", "stop_requested": True, "was_stale": True}

    return result


class StartRunRequest(BaseModel):
    run_name: str = ""
    loop_count: int = 1
    evolution_mode: bool = False
    evolution_profile: str = "quantaalpha"
    evolution_iterations: int = 5
    evolution_num_initial_hypotheses: int = 10
    research_report_pdf: str = ""
    initial_hypothesis: str = ""
    panel_data_path: str = ""
    panel_data_paths: list[str] = []
    start_date: str | None = None      # optional date range filter, e.g. "2022-03"
    end_date: str | None = None
    train_sharpe_threshold: float | None = None


@router.post("/runs")
async def start_run(req: StartRunRequest) -> dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["FACTOR_LIBRARY_RUN_ID"] = run_id

    sync_cb = ws_manager.make_sync_callback(run_id)

    # Resolve effective panel path: multi-select wins, fallback to single, then default
    effective_paths = req.panel_data_paths or ([req.panel_data_path] if req.panel_data_path else [])
    panel_path = effective_paths[0] if effective_paths else str(PROJECT_ROOT / "data" / "panel_data.parquet")
    params = {
        "run_name": req.run_name or run_id,
        "panel_data_path": panel_path,
        "panel_data_paths": effective_paths,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "loop_count": req.loop_count,
        "evolution_mode": req.evolution_mode,
        "evolution_profile": req.evolution_profile or "quantaalpha",
        "evolution_iterations": req.evolution_iterations,
        "evolution_num_initial_hypotheses": req.evolution_num_initial_hypotheses,
        "research_report_pdf": req.research_report_pdf,
        "initial_hypothesis": req.initial_hypothesis,
        "run_store_path": str(_db()),
    }
    if req.train_sharpe_threshold is not None:
        os.environ["TRAIN_SHARPE_THRESHOLD"] = str(req.train_sharpe_threshold)

    runner.launch_run(run_id=run_id, params=params, progress_callback=sync_cb)
    return {"run_id": run_id, "status": "started"}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    upload_dir = PROJECT_ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    return {"path": str(dest), "filename": file.filename}


@router.get("/data-files/preview")
def preview_data_file(path: str) -> dict[str, Any]:
    """Return date range and available years for a panel parquet/csv file."""
    import sys
    from pathlib import Path as _Path
    _proj = _Path(__file__).resolve().parents[3]
    _src = _proj / "src"
    for p in (str(_proj), str(_src)):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import pandas as pd
        fp = _Path(path)
        if not fp.exists():
            raise HTTPException(404, f"文件不存在：{path}")
        if fp.suffix == ".csv":
            df = pd.read_csv(fp, nrows=0)  # just get columns first
            df = pd.read_csv(fp)
        else:
            df = pd.read_parquet(fp)

        dt, date_source = _detect_datetime_series(df)

        years = sorted(dt.dt.year.unique().tolist())
        return {
            "path": path,
            "start_date": str(dt.min().date()),
            "end_date": str(dt.max().date()),
            "years": years,
            "row_count": len(df),
            "date_source": date_source,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"读取文件失败：{exc}") from exc


@router.get("/factors/top")
def get_top_factors(limit: int = 30) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    per_store_limit = max(1, int(limit))
    for db_path in _read_dbs():
        factors.extend(run_reader.get_top_factors(db_path, limit=per_store_limit))
    return _rank_top_factors(factors, limit=limit)


@router.get("/data-files")
def list_data_files() -> list[dict[str, str]]:
    """List all .parquet and .csv files under PROJECT_ROOT/data (recursively)."""
    data_root_env = os.getenv("DATA_ROOT", "").strip()
    data_root = Path(data_root_env) if data_root_env else PROJECT_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, str]] = []
    for ext in ("*.parquet", "*.csv"):
        for f in sorted(data_root.rglob(ext)):
            results.append({
                "path": str(f),
                "name": f.name,
                "relative": str(f.relative_to(data_root)),
                "size_mb": _format_file_size(f.stat().st_size),
            })
    return results


@router.get("/runs/{run_id}/report")
def get_report(run_id: str) -> dict[str, str]:
    db_path = _find_db_for_run(run_id)
    run = run_reader.get_run(db_path, run_id) if db_path is not None else None
    if run is None:
        raise HTTPException(404, "run not found")
    report_dir = PROJECT_ROOT / "reports"
    candidates = sorted(report_dir.glob(f"*{run_id}*.md"), reverse=True) if report_dir.exists() else []
    if not candidates:
        raise HTTPException(404, "report not found")
    report_file = candidates[0]
    # Compute the web-accessible base URL for images embedded in the markdown.
    # Images are stored relative to the report file's parent directory and
    # served under /report-assets/ which is mounted at PROJECT_ROOT/reports/.
    try:
        subdir = report_file.parent.relative_to(report_dir)
        subdir_str = str(subdir).replace("\\", "/")
        assets_base = f"/report-assets/{subdir_str}" if subdir_str not in (".", "") else "/report-assets"
    except ValueError:
        assets_base = "/report-assets"
    return {
        "content": report_file.read_text(encoding="utf-8"),
        "path": str(report_file),
        "assets_base": assets_base,
    }
