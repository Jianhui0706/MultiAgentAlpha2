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

from trading_agents.infra.research_report_loader import build_combined_research_report_payload, build_research_report_payload
from trading_agents.infra.run_store import RunStore
from services import run_reader, runner
from services.ws_manager import ws_manager
from main import _research_report_progress_fields

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

# Mirror services.runner._SYMBOL_FIELD_CANDIDATES so coverage analysis derives
# symbols the same way an actual run does.
_SYMBOL_FIELD_CANDIDATES = (
    "symbol",
    "instrument",
    "ts_code",
    "code",
    "ticker",
    "sec_code",
    "security",
)

# Cross-sectional quintile backtest needs at least this many instruments per
# timestamp to form non-degenerate layers (matches n_layers in the engine).
_MIN_CROSS_SECTION = 5

# Per-file symbol/date-range cache for the New Run date picker. Panel parquet
# files are large but immutable between selections, so we key on
# (path, mtime, size) and skip re-reading when the user re-selects the same files.
import threading

_SYMBOL_RANGES_CACHE: dict[str, tuple[float, int, list[dict[str, Any]]]] = {}
_SYMBOL_RANGES_CACHE_LOCK = threading.Lock()


def _csv_time_bounds(fp: Path) -> dict[str, Any] | None:
    """Fast start/end for a single-symbol, time-sorted CSV.

    Panel CSVs are sorted by time, so the min/max timestamp are just the first
    and last data rows. We read the header, the first data line, and the file's
    tail (via a seek) instead of parsing the entire datetime column — O(1) per
    file regardless of size. Returns ``None`` (caller falls back to a full read)
    when the file carries a symbol column (needs per-symbol grouping) or the time
    column can't be located. min/max of the two endpoints guards against either
    sort direction.
    """
    import csv as _csv
    import pandas as pd

    with open(fp, "rb") as f:
        header_raw = f.readline()
        if not header_raw:
            return None
        columns = next(_csv.reader([header_raw.decode("utf-8-sig", errors="replace")]), [])
        date_col = _resolve_date_col(columns)
        sym_col = _find_col_name(columns, _SYMBOL_FIELD_CANDIDATES)
        if date_col is None or sym_col is not None:
            return None  # let the caller group by symbol / handle headerless files
        date_idx = columns.index(date_col)

        first_data = b""
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip():
                first_data = line
                break
        if not first_data:
            return None

        f.seek(0, os.SEEK_END)
        size = f.tell()
        tail_size = min(size, 65536)
        f.seek(size - tail_size)
        tail = f.read()

    last_data = next((seg for seg in reversed(tail.splitlines()) if seg.strip()), b"")
    if not last_data:
        return None

    def _field(raw: bytes) -> str | None:
        row = next(_csv.reader([raw.decode("utf-8", errors="replace")]), [])
        return row[date_idx] if date_idx < len(row) else None

    raw_values = [v for v in (_field(first_data), _field(last_data)) if v]
    if not raw_values:
        return None
    dt = _parse_dt_aligned(pd.Series(raw_values)).dropna()
    if dt.empty:
        return None
    start_dt, end_dt = dt.min(), dt.max()
    return {
        "symbol": fp.stem.split("_")[0].upper(),
        "start_month": start_dt.strftime("%Y-%m"),
        "end_month": end_dt.strftime("%Y-%m"),
        "start_datetime": start_dt.isoformat(),
        "end_datetime": end_dt.isoformat(),
        "rows": 0,  # not counted on the fast path; the coverage UI doesn't use it
    }


def _read_dt_symbol_columns(fp: Path) -> Any:
    """Read only the datetime (+ symbol) columns from a data file.

    The coverage/preview UI only needs the time axis and instrument id, but panel
    parquet files carry dozens of feature columns. Decoding all of them just to
    compute a min/max date is what makes the date picker slow. We prune to the
    datetime + symbol columns (index columns included) so pyarrow/pandas only
    touch what we actually use. Falls back to a full read if pruning fails.
    """
    import pandas as pd

    if fp.suffix.lower() == ".csv":
        header = pd.read_csv(fp, nrows=0)
        date_col = _resolve_date_col(header.columns)
        sym_col = _find_col_name(header.columns, _SYMBOL_FIELD_CANDIDATES)
        usecols = [c for c in (date_col, sym_col) if c is not None]
        return pd.read_csv(fp, usecols=usecols) if usecols else pd.read_csv(fp)

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(fp)
        names = list(pf.schema_arrow.names)  # includes index columns (datetime/symbol)
        date_col = _resolve_date_col(names)
        sym_col = _find_col_name(names, _SYMBOL_FIELD_CANDIDATES)
        wanted = [c for c in (date_col, sym_col) if c is not None]
        if wanted:
            return pf.read(columns=wanted).to_pandas()
    except Exception:
        pass
    return pd.read_parquet(fp)


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
    factor_reward_score = factor.get("factor_reward_score")
    if isinstance(factor_reward_score, (int, float)):
        return float(factor_reward_score)
    reward_score = factor.get("reward_score")
    if isinstance(reward_score, (int, float)):
        return float(reward_score)
    sharpe = factor.get("sharpe")
    if isinstance(sharpe, (int, float)):
        return float(sharpe) * 0.1
    return 0.0


def _factor_rank_key(factor: dict[str, Any]) -> tuple[int, float]:
    return (1 if factor.get("admission_passed") is not False else 0, _factor_score(factor))


def _rank_top_factors(factors: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(factors, key=_factor_rank_key, reverse=True)[: max(1, int(limit))]
    for rank, factor in enumerate(ranked, start=1):
        factor["rank"] = rank
    return ranked


def _metric_value(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _reward_breakdown(metrics: dict[str, Any], *, expression: str = "") -> dict[str, Any]:
    from trading_agents.infra.factor_library_manager import FactorLibraryManager

    factor_entry = {"factor_expression": expression} if expression else None
    return FactorLibraryManager.compute_reward_breakdown(metrics, factor_entry=factor_entry)


def _progress_trajectories(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("stage") not in {"evolution_task_started", "evolution_task_finished"}:
            continue
        trajectory_id = str(event.get("trajectory_id", "") or "").strip()
        if not trajectory_id:
            continue
        metrics = event.get("metrics", {}) if isinstance(event.get("metrics", {}), dict) else {}
        current = by_id.get(trajectory_id, {})
        by_id[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "phase": str(event.get("phase", current.get("phase", "")) or ""),
            "round_idx": int(event.get("round_idx", current.get("round_idx", 0)) or 0),
            "direction_id": int(event.get("direction_id", current.get("direction_id", 0)) or 0),
            "parent_ids": list(event.get("parent_ids", current.get("parent_ids", [])) or []),
            "hypothesis": str(event.get("hypothesis", current.get("hypothesis", "")) or ""),
            "reward_score": event.get("reward_score", current.get("reward_score")),
            "metrics": dict(metrics or current.get("metrics", {})),
        }
    return sorted(by_id.values(), key=lambda item: (int(item.get("round_idx", 0)), int(item.get("direction_id", 0))))


def _progress_factors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    for event in events:
        if event.get("stage") != "evolution_task_finished":
            continue
        trajectory_id = str(event.get("trajectory_id", "") or "").strip()
        factor_summaries = event.get("factor_summaries", [])
        if not trajectory_id or not isinstance(factor_summaries, list):
            continue
        for idx, item in enumerate(factor_summaries):
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {}
            factor_name = str(item.get("factor_name", item.get("name", f"factor_{idx}")) or f"factor_{idx}")
            expression = str(item.get("expression", item.get("factor_expression", "")) or "")
            factors.append(
                {
                    "factor_id": f"{trajectory_id}:{factor_name}",
                    "loop_index": int(event.get("loop_index", 0) or 0),
                    "trajectory_id": trajectory_id,
                    "factor_name": factor_name,
                    "expression": expression,
                    "description": str(item.get("description", "") or ""),
                    "formulation": str(item.get("formulation", "") or ""),
                    "implementation_status": str(item.get("implementation_status", item.get("status", "generated")) or "generated"),
                    "reward_score": event.get("reward_score", item.get("reward_score")),
                    "factor_reward_score": item.get("reward_score"),
                    "reward_breakdown": item.get("reward_breakdown")
                    if isinstance(item.get("reward_breakdown"), dict)
                    else _reward_breakdown(metrics, expression=expression),
                    "phase": str(event.get("phase", "") or ""),
                    "round_idx": int(event.get("round_idx", 0) or 0),
                    "sharpe": _metric_value(metrics, "sharpe", "Sharpe"),
                    "IC": _metric_value(metrics, "IC"),
                    "rank_ic": _metric_value(metrics, "Rank IC", "RankIC"),
                    "icir": _metric_value(metrics, "ICIR"),
                    "rank_icir": _metric_value(metrics, "Rank ICIR", "RankICIR"),
                    "coverage": _metric_value(metrics, "coverage"),
                    "turnover": _metric_value(metrics, "turnover"),
                    "ls_return": _metric_value(metrics, "long_short_mean_return", "ls_return"),
                    "ann_return": _metric_value(metrics, "annualized_return", "ann_return"),
                    "yearly_returns": metrics.get("yearly_returns") if isinstance(metrics.get("yearly_returns"), dict) else None,
                    "quality_gate_passed": _metric_value(metrics, "quality_gate_passed"),
                    "quality_gate_threshold": _metric_value(metrics, "quality_gate_threshold"),
                    "valid_cross_section_mean": _metric_value(metrics, "valid_cross_section_mean"),
                    "cross_section_universe_size": _metric_value(metrics, "cross_section_universe_size"),
                    "ic_win_rate": _metric_value(metrics, "ic_win_rate"),
                    "rank_ic_win_rate": _metric_value(metrics, "rank_ic_win_rate"),
                    "drawdown": _metric_value(metrics, "drawdown", "max_drawdown"),
                    "long_short_positive_ratio": _metric_value(metrics, "long_short_positive_ratio", "win_rate"),
                    "layer_1": _metric_value(metrics, "layer_1_mean_return"),
                    "layer_2": _metric_value(metrics, "layer_2_mean_return"),
                    "layer_3": _metric_value(metrics, "layer_3_mean_return"),
                    "layer_4": _metric_value(metrics, "layer_4_mean_return"),
                    "layer_5": _metric_value(metrics, "layer_5_mean_return"),
                    "layer_monotonicity": _metric_value(metrics, "layer_monotonicity"),
                    "ls_cumulative_series": metrics.get("ls_cumulative_series") if isinstance(metrics.get("ls_cumulative_series"), list) else None,
                    "layer_1_cumulative_series": metrics.get("layer_1_cumulative_series") if isinstance(metrics.get("layer_1_cumulative_series"), list) else None,
                    "layer_2_cumulative_series": metrics.get("layer_2_cumulative_series") if isinstance(metrics.get("layer_2_cumulative_series"), list) else None,
                    "layer_3_cumulative_series": metrics.get("layer_3_cumulative_series") if isinstance(metrics.get("layer_3_cumulative_series"), list) else None,
                    "layer_4_cumulative_series": metrics.get("layer_4_cumulative_series") if isinstance(metrics.get("layer_4_cumulative_series"), list) else None,
                    "layer_5_cumulative_series": metrics.get("layer_5_cumulative_series") if isinstance(metrics.get("layer_5_cumulative_series"), list) else None,
                    "benchmark_cumulative_series": metrics.get("benchmark_cumulative_series") if isinstance(metrics.get("benchmark_cumulative_series"), list) else None,
                    "ic_cumulative_series": metrics.get("ic_cumulative_series") if isinstance(metrics.get("ic_cumulative_series"), list) else None,
                    "rank_ic_cumulative_series": metrics.get("rank_ic_cumulative_series") if isinstance(metrics.get("rank_ic_cumulative_series"), list) else None,
                    "hypothesis": str(event.get("hypothesis", "") or ""),
                    "feedback_summary": "",
                }
            )
    return run_reader.dedupe_factors_by_expression(factors)


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1_048_576:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1_048_576:.1f} MB"


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
    loops = run_reader.get_loops(db_path, run_id)
    trajectories = run_reader.get_trajectories(db_path, run_id)
    run["loops"] = loops
    run["loop_count"] = len(loops)
    run["trajectory_count"] = len(trajectories)
    run["factor_count"] = sum(len(loop.get("factors", []) or []) for loop in loops)
    return run


@router.get("/runs/{run_id}/factors")
def get_factors(run_id: str) -> list[dict[str, Any]]:
    db_path = _find_db_for_run(run_id)
    if db_path is None:
        return []
    factors = run_reader.get_factors(db_path, run_id)
    if factors:
        return factors
    run = run_reader.get_run(db_path, run_id)
    log_dir = run["log_dir"] if run else str(PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id)
    return _progress_factors(run_reader.get_progress_events(log_dir, limit=2000))


@router.get("/runs/{run_id}/lineage")
def get_lineage(run_id: str) -> list[dict[str, Any]]:
    db_path = _find_db_for_run(run_id)
    if db_path is None:
        return []
    trajectories = run_reader.get_trajectories(db_path, run_id)
    if trajectories:
        return trajectories
    run = run_reader.get_run(db_path, run_id)
    log_dir = run["log_dir"] if run else str(PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id)
    return _progress_trajectories(run_reader.get_progress_events(log_dir, limit=2000))


@router.get("/runs/{run_id}/progress")
def get_progress(run_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    db_path = _find_db_for_run(run_id)
    run = run_reader.get_run(db_path, run_id) if db_path is not None else None
    log_dir = run["log_dir"] if run else str(PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id)
    return run_reader.get_progress_events(log_dir, limit=limit)


def _report_paths_from_args(args: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in [args.get("research_report_pdf"), *list(args.get("research_report_pdfs", []) or [])]:
        text = str(item or "").strip()
        if text and text not in paths:
            paths.append(text)
    return paths


def _dataset_summary_for_report(run: dict[str, Any] | None, events: list[dict[str, Any]]) -> dict[str, Any]:
    args = dict((run or {}).get("args", {}) or {})
    prediction_horizon = str(args.get("prediction_horizon", "24h") or "24h")
    market = next((event for event in events if event.get("stage") == "market_payload_finished"), {})
    return {
        "feature_columns": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_quote",
            "returns_1h",
            "returns_4h",
            "returns_12h",
            "returns_24h",
            "high_low_spread",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ],
        "validation_target_column": f"fwd_returns_{runner._canonical_horizon(prediction_horizon) or '24h'}",
        "rows": int(market.get("row_count", 0) or 0),
        "symbol_count": int(market.get("symbol_count", 0) or 0),
    }


@router.get("/runs/{run_id}/research-report")
def get_research_report_detail(run_id: str) -> dict[str, Any]:
    db_path = _find_db_for_run(run_id)
    run = run_reader.get_run(db_path, run_id) if db_path is not None else None
    log_dir = run["log_dir"] if run else str(PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id)
    events = run_reader.get_progress_events(log_dir, limit=3000)
    args = dict((run or {}).get("args", {}) or {})
    report_paths = _report_paths_from_args(args)
    if not report_paths:
        session = next((event for event in events if event.get("stage") == "session_started"), {})
        report_paths = _report_paths_from_args(session)
    dataset_summary = _dataset_summary_for_report(run, events)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, report_path in enumerate(report_paths, start=1):
        try:
            payload = build_research_report_payload(report_path, dataset_summary=dataset_summary)
            report = dict(payload.get("research_report", {}))
            fields = _research_report_progress_fields(report)
            source = dict(report.get("source", {}))
            source_name = str(source.get("file_name", Path(report_path).name))
            suggestions = []
            for item in list(fields.get("report_suggested_hypotheses", [])):
                if isinstance(item, dict):
                    row = dict(item)
                    row["source_report_name"] = source_name
                    row["source_report_index"] = index
                    suggestions.append(row)
            reports.append(
                {
                    "index": index,
                    "path": str(report_path),
                    "source": source,
                    "source_report_name": source_name,
                    "summary": fields.get("report_summary", {}),
                    "summary_text": fields.get("report_summary_text", ""),
                    "initial_seed": fields.get("report_initial_seed", {}),
                    "suggested_hypotheses": suggestions,
                    "factor_extraction": report.get("report_factor_extraction", {}),
                    "report_text": str(report.get("report_text", "")),
                    "cleaning_stats": report.get("cleaning_stats", {}),
                }
            )
        except Exception as exc:
            errors.append({"path": str(report_path), "error": str(exc)})

    combined: dict[str, Any] = {}
    if len(report_paths) > 1:
        try:
            payload = build_combined_research_report_payload(report_paths, dataset_summary=dataset_summary)
            report = dict(payload.get("research_report", {}))
            fields = _research_report_progress_fields(report)
            combined = {
                "summary": fields.get("report_summary", {}),
                "summary_text": fields.get("report_summary_text", ""),
                "initial_seed": fields.get("report_initial_seed", {}),
                "suggested_hypotheses": fields.get("report_suggested_hypotheses", []),
                "source_reports": report.get("source_reports", []),
                "report_text": str(report.get("report_text", "")),
                "synthesis_status": report.get("multi_report_synthesis_status", ""),
            }
        except Exception as exc:
            errors.append({"path": "combined_reports", "error": str(exc)})

    initial_event = next((event for event in events if event.get("stage") == "direction_decomposer_finished"), {})
    return {
        "run_id": run_id,
        "report_count": len(report_paths),
        "reports": reports,
        "combined": combined,
        "errors": errors,
        "direction_decomposer": {
            "count": initial_event.get("initial_hypothesis_count", initial_event.get("direction_count", 0)),
            "initial_hypotheses": initial_event.get("initial_hypotheses", []),
        },
        "factor_reference_chain": [
            "研报内容会注入 DirectionDecomposerAgent 和 ExperimentDesignerAgent。",
            "ExperimentDesignerAgent 会把研报假说转成 experiment_spec 和 qlib_factor_experiment。",
            "FactorCoderAgent 会实现这些实验规格；通常不直接读取完整研报原文。",
        ],
    }


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


@router.post("/runs/stop-all")
def stop_all_runs(force: bool = True) -> dict[str, Any]:
    """Force-clear every run that is currently marked 'running'.

    Robust against stuck '终止中' runs and orphaned runs left over from a previous
    backend process: it collects running run_ids from *every* run-store DB as well
    as from in-memory tracking, then force-cancels each one. force_stop_run marks
    the run cancelled in the DB regardless of whether a live thread exists.
    """
    # Map each running run_id to the DB it lives in, so force_stop updates the
    # correct store (there can be several run_store*.sqlite files).
    running: dict[str, str] = {}
    for db_path in _read_dbs():
        for run in run_reader.list_runs(db_path):
            if str(run.get("status", "")) == "running":
                rid = str(run.get("run_id", ""))
                running.setdefault(rid, str(db_path))
    for rid in runner.active_run_ids():
        running.setdefault(rid, "")
    running.pop("", None)

    results = [runner.force_stop_run(rid, run_store_path=path or None) for rid, path in running.items()]
    return {"stopped_count": len(results), "runs": results}


@router.post("/runs/{run_id}/force-stop")
def force_stop_run(run_id: str) -> dict[str, Any]:
    db_path = _find_db_for_run(run_id)
    return runner.force_stop_run(run_id, run_store_path=str(db_path) if db_path else None)


@router.delete("/factors")
def clear_all_factors() -> dict[str, Any]:
    """Hide factors from the leaderboard while preserving run history."""
    run_store_hidden = 0

    # 1) Run-store DBs — these feed the leaderboard.
    for db_path in _read_dbs():
        try:
            run_store_hidden += RunStore(db_path).hide_all_factors_from_leaderboard()
        except Exception:
            pass

    # 2) FactorLibraryManager JSON file (legacy library).
    legacy_hidden = 0
    try:
        from trading_agents.infra.factor_library_manager import FactorLibraryManager
        lib_path = FactorLibraryManager.default_library_path()
        if lib_path.exists():
            mgr = FactorLibraryManager(lib_path)
            bucket = mgr.data.get("factors", {})
            if isinstance(bucket, dict):
                for factor in bucket.values():
                    if not isinstance(factor, dict):
                        continue
                    metadata = factor.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    if not metadata.get("leaderboard_hidden"):
                        metadata["leaderboard_hidden"] = True
                        metadata["leaderboard_hidden_at"] = datetime.now().isoformat()
                        factor["metadata"] = metadata
                        legacy_hidden += 1
                if legacy_hidden:
                    mgr._save()
    except Exception:
        pass

    total = run_store_hidden + legacy_hidden
    return {
        "hidden": total,
        "run_store_hidden": run_store_hidden,
        "legacy_hidden": legacy_hidden,
        "message": f"已从排行榜隐藏 {total} 个因子，历史记录仍保留",
    }


@router.post("/factors/{run_id}/{factor_id}/hide")
@router.delete("/factors/{run_id}/{factor_id}")
def hide_factor_from_leaderboard(run_id: str, factor_id: str) -> dict[str, Any]:
    db_path = _find_db_for_run(run_id)
    if db_path is None:
        raise HTTPException(404, f"run {run_id} not found")

    removed = RunStore(db_path).hide_factor_from_leaderboard(run_id=run_id, factor_id=factor_id)
    hidden = int(removed.get("hidden", 0) or 0)
    factor_name = str(removed.get("factor_name", "") or "")
    expression = str(removed.get("expression", "") or "")
    legacy_hidden = 0

    try:
        from trading_agents.infra.factor_library_manager import FactorLibraryManager

        lib_path = FactorLibraryManager.default_library_path()
        if lib_path.exists():
            mgr = FactorLibraryManager(lib_path)
            bucket = mgr.data.get("factors", {})
            if isinstance(bucket, dict):
                for key, factor in list(bucket.items()):
                    if not isinstance(factor, dict):
                        continue
                    metadata = factor.get("metadata", {}) if isinstance(factor.get("metadata", {}), dict) else {}
                    same_run = not run_id or str(metadata.get("run_id", "")) == run_id
                    same_id = str(key) == factor_id or str(factor.get("factor_id", "")) == factor_id
                    same_name = factor_name and str(factor.get("factor_name", "")) == factor_name
                    same_expr = expression and str(factor.get("factor_expression", factor.get("expression", ""))) == expression
                    if same_id or (same_run and same_name and same_expr):
                        metadata["leaderboard_hidden"] = True
                        metadata["leaderboard_hidden_at"] = datetime.now().isoformat()
                        factor["metadata"] = metadata
                        legacy_hidden += 1
                if legacy_hidden:
                    mgr._save()
    except Exception:
        legacy_hidden = 0

    if hidden <= 0 and legacy_hidden <= 0:
        raise HTTPException(404, f"factor {factor_id} not found")
    total = hidden + legacy_hidden
    return {
        "hidden": total,
        "run_store_hidden": hidden,
        "legacy_hidden": legacy_hidden,
        "factor_id": factor_id,
        "run_id": run_id,
        "message": f"已从排行榜隐藏 {total} 个因子，历史记录仍保留",
    }


class StartRunRequest(BaseModel):
    run_name: str = ""
    loop_count: int = 1
    evolution_mode: bool = False
    evolution_profile: str = "quantaalpha"
    evolution_iterations: int = 5
    evolution_num_initial_hypotheses: int = 10
    evolution_mutation_tasks_per_round: int | None = None
    evolution_crossover_tasks_per_round: int | None = None
    research_report_pdf: str = ""
    research_report_pdfs: list[str] = []
    initial_hypothesis: str = ""
    panel_data_path: str = ""
    panel_data_paths: list[str] = []
    start_date: str | None = None      # optional date range filter, e.g. "2022-03"
    end_date: str | None = None
    train_sharpe_threshold: float | None = None
    train_coverage_threshold: float | None = None
    prediction_horizon: str = "24h"   # e.g. "4h", "24h" — determines fwd_returns target label
    bar_frequency: str = "auto"       # e.g. "auto", "5min", "60min", "1d" — overrides bar-size auto-detection


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
        "research_report_pdfs": req.research_report_pdfs,
        "initial_hypothesis": req.initial_hypothesis,
        "run_store_path": str(_db()),
    }
    if req.evolution_mutation_tasks_per_round is not None:
        params["evolution_mutation_tasks_per_round"] = req.evolution_mutation_tasks_per_round
    if req.evolution_crossover_tasks_per_round is not None:
        params["evolution_crossover_tasks_per_round"] = req.evolution_crossover_tasks_per_round
    # Sharpe 入库阈值：新建运行填了就用填的值，没填则默认 1.5。
    _sharpe_threshold = req.train_sharpe_threshold if req.train_sharpe_threshold is not None else 1.5
    os.environ["TRAIN_SHARPE_THRESHOLD"] = str(_sharpe_threshold)
    params["train_sharpe_threshold"] = _sharpe_threshold
    if req.train_coverage_threshold is not None:
        os.environ["TRAIN_COVERAGE_THRESHOLD"] = str(req.train_coverage_threshold)
        params["train_coverage_threshold"] = req.train_coverage_threshold
    params["prediction_horizon"] = req.prediction_horizon or "24h"
    params["bar_frequency"] = req.bar_frequency or "auto"

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


def _resolve_date_col(columns: Any) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in _DATE_FIELD_CANDIDATES:
        if cand in lower:
            return lower[cand]
    for c in columns:
        if any(tok in str(c).lower() for tok in ("date", "datetime", "timestamp", "time")):
            return c
    return None


def _find_col_name(columns: Any, candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _parse_dt_aligned(series: Any) -> Any:
    """Row-aligned datetime parse (does NOT drop NA, unlike _parse_datetime_values).

    Falls back to epoch interpretation for numeric columns so Binance-style
    millisecond timestamps still resolve.
    """
    import pandas as pd
    s = pd.Series(series)
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() < 0.5 and pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        mx = float(num.abs().max()) if num.notna().any() else 0.0
        unit = "ns" if mx >= 1e17 else "us" if mx >= 1e14 else "ms" if mx >= 1e11 else "s"
        dt = pd.to_datetime(num, unit=unit, errors="coerce")
    return dt


def _range_dict(symbol: Any, dt: Any) -> dict[str, Any]:
    import pandas as pd

    clean_dt = dt.dropna()
    months = clean_dt.dt.strftime("%Y-%m")
    start_dt = clean_dt.min()
    end_dt = clean_dt.max()
    return {
        "symbol": str(symbol),
        "start_month": str(months.min()),
        "end_month": str(months.max()),
        "start_datetime": start_dt.isoformat() if pd.notna(start_dt) else "",
        "end_datetime": end_dt.isoformat() if pd.notna(end_dt) else "",
        "rows": int(len(months)),
    }


def _file_symbol_ranges(path: str) -> list[dict[str, Any]]:
    """Return [{symbol, start_month, end_month, rows}] for one data file.

    Reads only the date (+ symbol) columns to stay cheap on large panel files,
    and derives the symbol from the filename when no symbol column exists — the
    same convention an actual run uses (services.runner._load_single_file).
    Results are cached per (path, mtime, size) so re-selecting files is instant.
    """
    import pandas as pd
    fp = Path(path)
    if not fp.exists():
        raise HTTPException(404, f"文件不存在：{path}")

    stat = fp.stat()
    cache_key = str(fp.resolve())
    cached = _SYMBOL_RANGES_CACHE.get(cache_key)
    if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    def _store(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with _SYMBOL_RANGES_CACHE_LOCK:
            _SYMBOL_RANGES_CACHE[cache_key] = (stat.st_mtime, stat.st_size, result)
        return result

    # Fast path: single-symbol, time-sorted CSV — read only first/last row.
    if fp.suffix.lower() == ".csv":
        fast = _csv_time_bounds(fp)
        if fast is not None:
            return _store([fast])

    df = _read_dt_symbol_columns(fp)

    # Symbol carried on the index (standard panel parquet: datetime + symbol).
    idx_names = [str(n) for n in (getattr(df.index, "names", []) or []) if n]
    idx_low = {n.lower(): n for n in idx_names}
    if "symbol" in idx_low:
        sym = pd.Series(df.index.get_level_values(idx_low["symbol"]).astype(str))
        if "datetime" in idx_low:
            dt = _parse_dt_aligned(df.index.get_level_values(idx_low["datetime"]))
        else:
            dcol = _resolve_date_col(df.columns)
            dt = _parse_dt_aligned(df[dcol]) if dcol else _parse_dt_aligned(df.index)
        frame = pd.DataFrame({"sym": sym.values, "dt": pd.Series(dt).values})
        frame = frame[frame["dt"].notna()]
        return _store([_range_dict(s, g["dt"]) for s, g in frame.groupby("sym")])

    date_col = _resolve_date_col(df.columns)
    sym_col = _find_col_name(df.columns, _SYMBOL_FIELD_CANDIDATES)
    dt = _parse_dt_aligned(df[date_col]) if date_col is not None else _parse_dt_aligned(df.index)

    if sym_col is not None:
        frame = pd.DataFrame({"sym": df[sym_col].astype(str).values, "dt": pd.Series(dt).values})
        frame = frame[frame["dt"].notna()]
        if frame.empty:
            return _store([])
        return _store([_range_dict(s, g["dt"]) for s, g in frame.groupby("sym")])

    # No symbol column → one symbol per file, derived from the filename stem.
    symbol = fp.stem.split("_")[0].upper()
    dt = pd.Series(dt)
    if dt.notna().sum() == 0:
        return _store([])
    return _store([_range_dict(symbol, dt)])


def _sample_coverage_points(
    per_symbol: list[dict[str, Any]],
    *,
    max_points: int = 96,
) -> list[dict[str, Any]]:
    import pandas as pd

    starts = pd.to_datetime([item.get("start_datetime") for item in per_symbol], errors="coerce")
    ends = pd.to_datetime([item.get("end_datetime") for item in per_symbol], errors="coerce")
    if starts.isna().all() or ends.isna().all():
        return []
    union_start = starts.min()
    union_end = ends.max()
    if pd.isna(union_start) or pd.isna(union_end):
        return []
    if union_start >= union_end:
        points = [union_start]
    else:
        point_count = max(2, min(max_points, len(per_symbol) * 4, max_points))
        points = list(pd.date_range(union_start, union_end, periods=point_count))
    sampled: list[dict[str, Any]] = []
    for point in points:
        count = int(((starts <= point) & (ends >= point)).sum())
        sampled.append({"timestamp": point.isoformat(), "count": count})
    return sampled


class CoverageRequest(BaseModel):
    paths: list[str]


@router.post("/data-files/coverage")
def data_files_coverage(req: CoverageRequest) -> dict[str, Any]:
    """Cross-sectional instrument availability across the selected data files.

    Crypto/equity universes have misaligned histories (instruments list/delist at
    different times). This drives the New Run date picker: it reports the union
    span and a sampled timeline of how many instruments are simultaneously
    available, so the picker can recommend the first datetime where the
    cross-section is thick enough (>= _MIN_CROSS_SECTION) for a non-degenerate
    quintile backtest. Only the fields the picker consumes are returned.
    """
    paths = [str(p).strip() for p in (req.paths or []) if str(p).strip()]
    if not paths:
        raise HTTPException(400, "未提供数据文件")

    per_symbol: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    # File reads are independent and release the GIL during parquet decode, so
    # read them concurrently to keep the date picker responsive on multi-file
    # selections.
    from concurrent.futures import ThreadPoolExecutor

    ranges_by_path: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as executor:
        futures = {executor.submit(_file_symbol_ranges, p): p for p in paths}
        for future, p in futures.items():
            try:
                ranges_by_path[p] = future.result()
            except HTTPException:
                raise
            except Exception as exc:  # malformed file shouldn't abort the whole preview
                errors.append({"path": p, "error": str(exc)})

    for p in paths:
        ranges = ranges_by_path.get(p)
        if ranges is None:
            continue  # read failure already recorded in errors
        if not ranges:
            errors.append({"path": p, "error": "无法解析时间列"})
            continue
        per_symbol.extend(ranges)

    if not per_symbol:
        raise HTTPException(400, "所有文件都无法解析时间，请检查数据格式")

    union_start_datetime = min(r["start_datetime"] for r in per_symbol if r.get("start_datetime"))
    union_end_datetime = max(r["end_datetime"] for r in per_symbol if r.get("end_datetime"))
    sampled_coverage = _sample_coverage_points(per_symbol)
    recommended_start_datetime = next(
        (
            point["timestamp"]
            for point in sampled_coverage
            if int(point.get("count", 0) or 0) >= _MIN_CROSS_SECTION
        ),
        union_start_datetime,
    )

    return {
        "union_start_datetime": union_start_datetime,
        "union_end_datetime": union_end_datetime,
        "sampled_coverage": sampled_coverage,
        "recommended_start_datetime": recommended_start_datetime,
        "recommended_end_datetime": union_end_datetime,
        "min_cross_section": _MIN_CROSS_SECTION,
        "total_symbols": len({r["symbol"] for r in per_symbol}),
        "errors": errors,
    }


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


class PreprocessRequest(BaseModel):
    panel_data_paths: list[str]
    prediction_horizon: str = "24h"
    bar_frequency: str = "auto"
    output_name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    all_fwd_horizons: bool = False
    # Multi-select list of horizons to generate, e.g. ["5min","30min","4h","24h"].
    # When provided it takes precedence over prediction_horizon/all_fwd_horizons.
    target_horizons: list[str] | None = None


@router.post("/preprocess")
def preprocess_data(req: PreprocessRequest) -> dict[str, Any]:
    """Generate fwd_returns_* + feature columns from raw panel(s) and save to
    data/processed/. This is the explicit preprocessing step that the main run
    flow no longer performs (the run now fails fast if the target label is absent).
    """
    if not req.panel_data_paths:
        raise HTTPException(400, "请至少选择一个数据文件")
    try:
        return runner.preprocess_panel_to_processed(
            paths=req.panel_data_paths,
            prediction_horizon=req.prediction_horizon or "24h",
            bar_frequency=req.bar_frequency or "auto",
            output_name=req.output_name,
            start_date=req.start_date,
            end_date=req.end_date,
            all_fwd_horizons=req.all_fwd_horizons,
            target_horizons=req.target_horizons,
        )
    except Exception as exc:
        raise HTTPException(400, f"预处理失败：{exc}")


@router.get("/runs/{run_id}/report")
def get_report(run_id: str) -> dict[str, str]:
    db_path = _find_db_for_run(run_id)
    run = run_reader.get_run(db_path, run_id) if db_path is not None else None
    if run is None:
        raise HTTPException(404, "run not found")
    report_dir = PROJECT_ROOT / "reports"
    # 匹配两种报告布局：
    #   1) 旧版扁平结构：reports/factor_mining_report_<run_id>_<ts>.md
    #   2) 新版子目录结构：reports/factor_mining_report_<run_id>_<ts>/factor_mining_report.md
    # 第二种里 .md 文件名不含 run_id，run_id 在父目录名上，因此需要额外用 "<run_id>*/*.md" 匹配。
    if report_dir.exists():
        candidates = sorted(
            [*report_dir.glob(f"*{run_id}*.md"), *report_dir.glob(f"*{run_id}*/*.md")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    else:
        candidates = []
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
