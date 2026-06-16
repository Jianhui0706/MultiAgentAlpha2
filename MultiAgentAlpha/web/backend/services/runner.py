from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)


_active_runs: dict[str, dict[str, Any]] = {}
_active_runs_lock = threading.Lock()


def cleanup_signal_cache() -> dict[str, Any]:
    """删除因子信号 pickle 缓存（factor_library/cache/*.pkl）。

    这些 pickle 只在「同一次运行内」被 BacktestRunner 读取一次（FactorCoder→Runner
    的信号传递）；运行结束后没有任何消费者，且跨运行复用率≈0（缓存键含带时间戳的
    面板路径）。因子本身存在 all_factors_library.json / factor_library.jsonl /
    artifacts/run_store.sqlite 等，与此缓存无关——删缓存不会丢任何因子。
    缺失的 pickle 会在下次回测时按需重算（_load_signal_from_cache 未命中即回退）。

    设环境变量 KEEP_SIGNAL_CACHE=1 可保留缓存（调试用）。返回清理统计。
    """
    if os.getenv("KEEP_SIGNAL_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"skipped": True, "reason": "KEEP_SIGNAL_CACHE", "removed_files": 0, "freed_bytes": 0}
    cache_dir = PROJECT_ROOT / "factor_library" / "cache"
    removed = 0
    freed = 0
    try:
        if cache_dir.exists():
            for pkl in cache_dir.glob("*.pkl"):
                try:
                    freed += pkl.stat().st_size
                    pkl.unlink()
                    removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return {"skipped": False, "removed_files": removed, "freed_bytes": freed}
_force_stopped_runs: set[str] = set()  # run IDs that were force-cancelled; thread checks this to avoid overwriting DB status

_DATE_FIELD_CANDIDATES = (
    "datetime",
    "date",
    "trade_date",
    "trading_date",
    "cal_date",
    "timestamp",
    "time",
    "open_time",   # Binance kline CSV
    "dt",
    "ds",
)
_SYMBOL_FIELD_CANDIDATES = (
    "symbol",
    "instrument",
    "ts_code",
    "code",
    "ticker",
    "sec_code",
    "security",
)


class RunCancelled(RuntimeError):
    """Raised when a user requests a running workflow to stop."""


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


def _find_column(df: Any, candidates: tuple[str, ...]) -> Any | None:
    lower_to_column = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        column = lower_to_column.get(candidate)
        if column is not None:
            return column
    return None


def _load_single_file(path: str) -> "Any":
    """Load one parquet or CSV file into a DataFrame."""
    import pandas as pd
    from pathlib import Path as _Path
    fp = _Path(path)
    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
        # Binance-style kline CSV: if no symbol column, derive from filename
        if "symbol" not in df.columns and "ts_code" not in df.columns:
            stem = fp.stem  # e.g. "BTCUSDT_5m" or "BTCUSDT"
            symbol = stem.split("_")[0].upper()
            df["symbol"] = symbol
        return df
    return pd.read_parquet(fp)


def _file_summary(path: str, df: "Any") -> dict:
    """Return a per-file summary dict for the panel_data_loaded progress event."""
    import pandas as pd
    from pathlib import Path as _Path
    fp = _Path(path)
    # Try to detect date column for date range
    date_col = None
    for candidate in _DATE_FIELD_CANDIDATES:
        if candidate in df.columns:
            date_col = candidate
            break
    # If multi-indexed with datetime level, use that
    if date_col is None and hasattr(df.index, "names") and "datetime" in (df.index.names or []):
        dates = pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce")
        start = str(dates.min())[:19] if len(dates) else "?"
        end = str(dates.max())[:19] if len(dates) else "?"
    elif date_col:
        try:
            dates = _parse_datetime_values(df[date_col].dropna().values)
            start = str(dates.min())[:19]
            end = str(dates.max())[:19]
        except Exception:
            start = end = "?"
    else:
        start = end = "?"
    # Symbol(s) in this file
    symbol_col = None
    for candidate in _SYMBOL_FIELD_CANDIDATES:
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col:
        symbols = sorted(df[symbol_col].dropna().unique().tolist())
    elif hasattr(df.index, "names") and "symbol" in (df.index.names or []):
        symbols = sorted(df.index.get_level_values("symbol").unique().tolist())
    else:
        symbols = []
    return {
        "filename": fp.name,
        "rows": len(df),
        "symbols": symbols[:20],   # cap list length
        "symbol_count": len(symbols),
        "start": start,
        "end": end,
    }


def _load_panel_from_paths(paths: list[str]) -> tuple["Any", list[dict]]:
    """Load one or many parquet/CSV files and concatenate into a single DataFrame.

    Returns (combined_df, file_summaries) where file_summaries is a list of
    per-file metadata dicts suitable for emitting in a progress event.
    """
    import pandas as pd
    if not paths:
        raise ValueError("No panel data paths provided")
    summaries: list[dict] = []
    frames = []
    for p in paths:
        df = _load_single_file(p)
        summaries.append(_file_summary(p, df))
        frames.append(df)
    combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return combined, summaries


def _adapter_can_read_panel_directly(df: "Any") -> bool:
    index_names = {n for n in (getattr(df.index, "names", []) or []) if n}
    return {"datetime", "symbol"}.issubset(index_names) or {"datetime", "symbol"}.issubset(set(df.columns))


def _should_reuse_preprocessed_panel(
    *,
    paths: list[str],
    df: "Any",
    target_col: str,
    start_date: str | None,
    end_date: str | None,
) -> bool:
    return (
        len(paths) == 1
        and not start_date
        and not end_date
        and target_col in df.columns
        and _adapter_can_read_panel_directly(df)
    )


def _panel_bar_minutes(df: "Any") -> int | None:
    import pandas as pd

    try:
        index_names = {n for n in (getattr(df.index, "names", []) or []) if n}
        if "datetime" in index_names:
            dt = pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce")
        elif "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            return None
        unique_dt = pd.Series(dt).dropna().drop_duplicates().sort_values()
        diffs = unique_dt.diff().dropna()
        if diffs.empty:
            return None
        return max(1, int(diffs.median().total_seconds() / 60))
    except Exception:
        return None


def _detect_bar_minutes(panel: Any) -> int:
    """Infer bar size in minutes from the datetime index (median gap between unique timestamps)."""
    import pandas as pd
    try:
        dt_sorted = pd.to_datetime(panel.index.get_level_values("datetime")).sort_values().drop_duplicates()
        diffs = dt_sorted.diff().dropna()
        return max(1, int(diffs.median().total_seconds() / 60))
    except Exception:
        return 60  # fallback: hourly


def _parse_bar_frequency(bar_frequency: str) -> int | None:
    """Convert a bar_frequency string to minutes, or None for auto-detect.

    Accepted formats: "auto", "1min", "5min", "15min", "30min", "60min", "240min", "1d", "1h", "4h"
    """
    import re as _re
    s = str(bar_frequency).strip().lower()
    if not s or s == "auto":
        return None
    m = _re.match(r"(\d+)\s*min$", s)
    if m:
        return max(1, int(m.group(1)))
    m = _re.match(r"(\d+)\s*h$", s)
    if m:
        return max(1, int(m.group(1)) * 60)
    if s in {"1d", "daily", "day"}:
        return 1440
    return None


def _horizon_to_minutes(value: str) -> int | None:
    """Parse a prediction-horizon string into total minutes.

    Accepts minute / hour / day forms: "5min", "30m", "4h", "1d". Returns None for
    unparseable / non-positive values. Used to support sub-hour forward-return
    horizons (5/10/15/20/30/60 min) alongside the existing hour-based ones.
    """
    import re as _re
    m = _re.match(r"^\s*(\d+)\s*(min|m|h|d)\s*$", str(value).strip().lower())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        return None
    if unit in ("min", "m"):
        return n
    if unit == "h":
        return n * 60
    return n * 1440  # d


def _canonical_horizon(value: str) -> str | None:
    """Canonical fwd_returns_* column suffix for a horizon.

    < 60min stays "<m>min"; exact multiples of 60min collapse to "<h>h" so
    "60min"→"1h", "240min"→"4h", "1440min"/"1d"→"24h" — keeping one column per
    horizon and matching the existing fwd_returns_24h naming. Non-hour minute
    horizons keep the minute form (e.g. "90min").
    """
    minutes = _horizon_to_minutes(value)
    if minutes is None:
        return None
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def _resolve_target_horizons(
    target_horizons: list[str] | None,
    prediction_horizon: str,
    *,
    all_fwd_horizons: bool = False,
) -> list[str]:
    """Resolve the list of canonical horizons whose fwd columns should be computed.

    Priority: explicit multi-select target_horizons > all_fwd_horizons summary set >
    [prediction_horizon, 24h] default. De-duplicated, order preserved.
    """
    if target_horizons:
        requested = list(target_horizons)
    elif all_fwd_horizons:
        requested = ["1h", "4h", "12h", "24h"]
    else:
        requested = [prediction_horizon, "24h"]
    resolved: list[str] = []
    for item in requested:
        canon = _canonical_horizon(item)
        if canon and canon not in resolved:
            resolved.append(canon)
    return resolved or ["24h"]


def _standardize_panel_dataframe(
    df: Any,
    prediction_horizon: str = "24h",
    bar_frequency: str = "auto",
    *,
    compute_targets: bool = True,
    target_horizons: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Normalize raw DataFrame into (datetime, symbol) multi-index panel.

    prediction_horizon: "4h" | "24h" | "12h" | "1h" — determines which fwd_returns_* column
    is the required target label. target_horizons can request a summary panel with
    several fwd_returns_* labels; all feature columns are derived with the correct bar-frequency.

    bar_frequency: "auto" uses median-gap auto-detection; otherwise overrides bar size
    (e.g. "5min", "60min", "1h", "4h", "1d").

    compute_targets: True (preprocessing tool) computes missing fwd_returns_* columns.
    False (main run flow) only normalizes/derives features and REQUIRES the target column
    to already exist — if it doesn't, the caller is expected to fail the run instead of
    silently fabricating labels.
    """
    import pandas as pd

    from trading_agents.factor_runtime import downcast_float_columns

    # changed = 任何标准化(含重建索引)；content_changed = 真正改了【列内容】(改名/补特征/算标签)，
    # 不含纯索引重建。后者由回测适配器在加载时自行重建，因此不需要为它另存副本。
    info: dict[str, Any] = {"changed": False, "content_changed": False, "original_rows": int(len(df)), "prediction_horizon": prediction_horizon, "bar_frequency_input": bar_frequency}
    panel = df.copy()
    index_names = list(getattr(panel.index, "names", []) or [])
    has_standard_index = "datetime" in index_names and "symbol" in index_names

    if not has_standard_index:
        date_column = _find_column(panel, _DATE_FIELD_CANDIDATES)
        symbol_column = _find_column(panel, _SYMBOL_FIELD_CANDIDATES)
        if date_column is None:
            raise ValueError(f"could not detect date column; tried {', '.join(_DATE_FIELD_CANDIDATES)}")
        if symbol_column is None:
            raise ValueError(f"could not detect symbol column; tried {', '.join(_SYMBOL_FIELD_CANDIDATES)}")

        parsed_dates = _parse_datetime_values(panel[date_column])
        if len(parsed_dates) != len(panel):
            parsed_dates = pd.to_datetime(panel[date_column], errors="coerce")
        if pd.Series(parsed_dates).isna().any():
            raise ValueError(f"date column {date_column} contains unparseable values")

        panel = panel.assign(datetime=parsed_dates.values, symbol=panel[symbol_column].astype(str).values)
        drop_columns = [col for col in (date_column, symbol_column) if col in panel.columns and col not in ("datetime", "symbol")]
        panel = panel.drop(columns=drop_columns)
        panel = panel.set_index(["datetime", "symbol"]).sort_index()
        info.update({"changed": True, "date_source": str(date_column), "symbol_source": str(symbol_column)})
    else:
        panel = panel.sort_index()

    if "volume" not in panel.columns and "vol" in panel.columns:
        panel["volume"] = panel["vol"]
        info["changed"] = True
        info["content_changed"] = True
        info["volume_source"] = "vol"

    _bar_override = _parse_bar_frequency(bar_frequency)
    bar_minutes = _bar_override if _bar_override is not None else _detect_bar_minutes(panel)
    bars_per_hour = max(1, round(60 / bar_minutes))
    info["bar_minutes"] = bar_minutes
    info["bars_per_hour"] = bars_per_hour
    info["bar_frequency_source"] = "manual" if _bar_override is not None else "auto_detected"

    if "close" not in panel.columns:
        raise ValueError("panel is missing 'close' column; cannot derive forward returns or features")
    close = pd.to_numeric(panel["close"], errors="coerce")
    g_close = close.groupby(level="symbol")

    # Determine which fwd columns are needed: requested horizons + the run's target.
    # Horizons are canonicalized (minutes/hours) and offsets are computed directly
    # from bar_minutes — supports sub-hour windows (5/10/.../30min) and avoids the
    # old bars_per_hour rounding bug on coarse (≥1h) bars.
    target_canon = _canonical_horizon(prediction_horizon) or "24h"
    target_col = f"fwd_returns_{target_canon}"
    needed_canon = _resolve_target_horizons(target_horizons, prediction_horizon)
    if target_canon not in needed_canon:
        needed_canon = [*needed_canon, target_canon]
    precomputed: list[str] = []
    if compute_targets:
        for canon in needed_canon:
            col = f"fwd_returns_{canon}"
            if col not in panel.columns:
                minutes = _horizon_to_minutes(canon) or (24 * 60)
                bars = max(1, round(minutes / bar_minutes))
                panel[col] = g_close.shift(-bars) / close - 1.0
                panel[col] = panel[col].where(panel[col].notna())
                precomputed.append(col)
                info["changed"] = True
                info["content_changed"] = True
        if precomputed:
            info["target_source"] = f"derived from close (bar={bar_minutes}min, computed={precomputed})"

    # Verify required target column exists. When compute_targets=False this is the strict
    # gate for the main run flow: missing label → raise so the run fails instead of
    # backtesting on fabricated/empty labels.
    if target_col not in panel.columns:
        raise ValueError(
            f"target column '{target_col}' not found in data. "
            f"Available fwd_returns_* columns: {[c for c in panel.columns if c.startswith('fwd_')]}. "
            f"请先在「数据预处理」页面生成带 {target_col} 的数据。"
        )

    # Lookback feature columns — frequency-aware.
    for hours, name in [(1, "returns_1h"), (4, "returns_4h"), (12, "returns_12h"), (24, "returns_24h")]:
        if name not in panel.columns:
            panel[name] = g_close.pct_change(bars_per_hour * hours)
            info["content_changed"] = True

    if "high" in panel.columns and "low" in panel.columns:
        if "high_low_spread" not in panel.columns:
            panel["high_low_spread"] = (
                pd.to_numeric(panel["high"], errors="coerce") - pd.to_numeric(panel["low"], errors="coerce")
            ) / close.replace(0, float("nan"))
            info["content_changed"] = True

    vol_col = "volume" if "volume" in panel.columns else ("vol" if "vol" in panel.columns else None)
    if vol_col:
        gv = pd.to_numeric(panel[vol_col], errors="coerce").groupby(level="symbol")
        for hours, name in [(1, "volume_change_1h"), (4, "volume_change_4h"), (24, "volume_change_24h")]:
            if name not in panel.columns:
                panel[name] = gv.pct_change(bars_per_hour * hours)
                info["content_changed"] = True
        if "volume_quote" not in panel.columns:
            if "amount" in panel.columns:
                panel["volume_quote"] = pd.to_numeric(panel["amount"], errors="coerce")
            else:
                panel["volume_quote"] = pd.to_numeric(panel[vol_col], errors="coerce")
            info["content_changed"] = True

    # float64→float32 降精度：减小内存与落盘体积；下游 IC/因子计算本就用 float32。
    downcast_float_columns(panel)
    return panel, info


def _apply_time_window(df: Any, start_date: str | None, end_date: str | None) -> Any:
    """Filter a (datetime, symbol) panel to [start_date, end_date] inclusive.

    Accepts YYYY-MM (month), YYYY-MM-DD (day), or full datetime strings; month/day
    forms are expanded to cover the whole month/day at the end bound.
    """
    import pandas as pd
    if not (start_date or end_date):
        return df
    dt_series = pd.to_datetime(df.index.get_level_values("datetime"))
    mask = pd.Series([True] * len(df), index=df.index)
    if start_date:
        mask &= dt_series >= pd.Timestamp(start_date)
    if end_date:
        end_text = str(end_date).strip()
        if len(end_text) <= 7:
            end_ts = pd.Timestamp(end_text) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        elif len(end_text) <= 10 and "T" not in end_text and " " not in end_text:
            end_ts = pd.Timestamp(end_text) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            end_ts = pd.Timestamp(end_text)
        mask &= dt_series <= end_ts
    return df[mask.values]


def preprocess_panel_to_processed(
    *,
    paths: list[str],
    prediction_horizon: str = "24h",
    bar_frequency: str = "auto",
    output_name: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    all_fwd_horizons: bool = False,
    target_horizons: list[str] | None = None,
) -> dict[str, Any]:
    """Standalone data preprocessing: load raw panel(s), derive fwd_returns_* and
    feature columns, and save the result as a parquet under data/processed/.

    This is the explicit preprocessing step (invoked from the 数据预处理 page) that
    replaces in-run auto-computation of labels. Returns a summary describing the
    saved file so the frontend can show it.

    start_date/end_date optionally restrict the saved panel to a sub-window. fwd
    labels are computed on the FULL data first, then the window is cropped, so the
    forward-return labels near the window end remain valid (they look ahead into
    data outside the saved window).
    """
    import re
    import pandas as pd
    from datetime import datetime as _dt

    # Resolve which fwd horizons to generate. Priority: explicit multi-select
    # target_horizons > legacy all_fwd_horizons summary set > the single
    # prediction_horizon (+24h). All canonicalized (minute/hour aware).
    resolved_horizons = _resolve_target_horizons(
        target_horizons, prediction_horizon, all_fwd_horizons=all_fwd_horizons
    )
    # The "primary" horizon used for filename + summary target_column display.
    horizon = (_canonical_horizon(prediction_horizon) if not target_horizons else resolved_horizons[0]) or "24h"
    is_multi = len(resolved_horizons) > 1
    df, file_summaries = _load_panel_from_paths(paths)
    target_col = f"fwd_returns_{horizon}"
    expected_target_cols = [f"fwd_returns_{item}" for item in resolved_horizons]
    expected_target_cols = list(dict.fromkeys(expected_target_cols))
    already_present_cols = [col for col in expected_target_cols if col in df.columns]
    had_target = target_col in df.columns

    df, info = _standardize_panel_dataframe(
        df,
        prediction_horizon=horizon,
        bar_frequency=bar_frequency,
        compute_targets=True,
        target_horizons=resolved_horizons,
    )

    # Compute fwd on full data above, THEN crop to the requested window.
    df = _apply_time_window(df, start_date, end_date)
    if df.empty:
        raise ValueError("所选时间区间内没有数据，请放宽时间范围。")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", str(output_name).strip()) if output_name else ""
    file_stem = safe_name or f"panel_{'multi' if is_multi else horizon}_{stamp}"
    filename = file_stem if file_stem.endswith(".parquet") else f"{file_stem}.parquet"
    out_path = processed_dir / filename
    df.to_parquet(str(out_path))

    fwd_cols = [c for c in df.columns if str(c).startswith("fwd_")]
    feature_cols = [c for c in df.columns if not str(c).startswith("fwd_")]
    try:
        symbol_count = int(pd.Index(df.index.get_level_values("symbol")).nunique())
        dt_index = pd.to_datetime(df.index.get_level_values("datetime"))
        dt_start, dt_end = str(dt_index.min()), str(dt_index.max())
    except Exception:
        symbol_count, dt_start, dt_end = 0, "", ""

    return {
        "saved_to": str(out_path),
        "filename": out_path.name,
        "relative": str(out_path.relative_to(PROJECT_ROOT / "data")),
        "prediction_horizon": horizon,
        "target_column": target_col,
        "target_columns": expected_target_cols,
        "target_already_present": bool(had_target),
        "target_already_present_columns": already_present_cols,
        "all_fwd_horizons": bool(is_multi),
        "target_horizons": resolved_horizons,
        "bar_minutes": info.get("bar_minutes"),
        "bars_per_hour": info.get("bars_per_hour"),
        "bar_frequency_source": info.get("bar_frequency_source"),
        "target_source": info.get("target_source", ""),
        "rows": int(len(df)),
        "symbol_count": symbol_count,
        "datetime_start": dt_start,
        "datetime_end": dt_end,
        "fwd_columns": fwd_cols,
        "feature_columns": feature_cols,
        "source_files": file_summaries,
    }


def get_active_run(run_id: str) -> dict[str, Any] | None:
    with _active_runs_lock:
        run = _active_runs.get(run_id)
        return dict(run) if run is not None else None


def request_stop_run(run_id: str) -> dict[str, Any]:
    with _active_runs_lock:
        run = _active_runs.get(run_id)
        if run is None:
            return {"run_id": run_id, "status": "not_active", "stop_requested": False}
        if run.get("status") != "running":
            return {"run_id": run_id, "status": run.get("status"), "stop_requested": bool(run.get("stop_requested"))}
        run["stop_requested"] = True
        run["stop_requested_at"] = datetime.now().isoformat()
        callback = run.get("progress_callback")

    if callable(callback):
        callback(
            {
                "stage": "run_stop_requested",
                "run_id": run_id,
                "message": "Stop requested by user. The workflow will stop at the next cancellation checkpoint.",
            },
            check_cancel=False,
        )
    return {"run_id": run_id, "status": "stopping", "stop_requested": True}


def active_run_ids() -> list[str]:
    """run_ids currently tracked in this process (any status)."""
    with _active_runs_lock:
        return list(_active_runs.keys())


def force_stop_run(run_id: str, run_store_path: str | None = None) -> dict[str, Any]:
    """Forcefully cancel a run regardless of thread state.

    Works even when the server was restarted and _active_runs is empty but the DB
    still shows the run as 'running' (orphaned run).  Always attempts to mark the
    run cancelled in RunStore regardless of whether it is currently tracked.
    ``run_store_path`` overrides the DB to update (caller knows which DB the
    orphaned run lives in); otherwise the run's own / default store is used.
    """
    from trading_agents.infra.run_store import RunStore
    import traceback as _tb
    with _active_runs_lock:
        run = _active_runs.pop(run_id, None)

    # Write a cancelled progress event (only possible when the thread is live).
    if run is not None:
        callback = run.get("progress_callback")
        if callable(callback):
            try:
                callback(
                    {"stage": "run_cancelled", "run_id": run_id, "error": "force-cancelled by user"},
                    check_cancel=False,
                )
            except Exception:
                pass

    # Record that this run was force-stopped so the background thread (if still
    # alive waiting on LLM timeout) won't overwrite the DB status with "failed".
    with _active_runs_lock:
        _force_stopped_runs.add(run_id)

    # Always mark as cancelled in the persistent store, even for orphaned runs
    # (server restarted while run was in-progress → run missing from _active_runs
    # but DB still shows 'running').
    try:
        store_path = run_store_path or (run.get("run_store_path") if run else None) or str(
            RunStore.default_path(PROJECT_ROOT)
        )
        RunStore(store_path).finish_run(
            run_id=run_id,
            status="cancelled",
            summary={"run_id": run_id, "status": "cancelled", "error": "force-cancelled by user"},
        )
    except Exception:
        _tb.print_exc()

    return {"run_id": run_id, "status": "cancelled", "forced": True}


def stop_all_runs(*, force: bool = False) -> dict[str, Any]:
    """Gracefully stop all running runs.  If force=True, also force-cancel any
    run that is already in 'stopping' state (stop_requested but still alive)."""
    with _active_runs_lock:
        all_active = list(_active_runs.items())

    results = []
    for rid, r in all_active:
        if r.get("status") != "running":
            continue
        if force and r.get("stop_requested"):
            results.append(force_stop_run(rid))
        else:
            results.append(request_stop_run(rid))
            if force:
                # Also mark stop_requested so next call can force them.
                pass
    return {"stopped_count": len(results), "runs": results}


def launch_run(
    *,
    run_id: str,
    params: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None],
) -> None:
    """Launch the workflow in a background thread."""
    from trading_agents.domains import CryptoCrossSectionDomainAdapter
    from trading_agents.infra.run_store import RunStore
    from trading_agents.workflows import AlphaFactorMiningWorkflow, EvolutionConfig
    from trading_agents.infra.llm import GatewayModelClient, LLMGateway, RetryPolicy
    from trading_agents.infra.llm.providers import OpenAICompatibleProvider
    from main import (
        _append_jsonl,
        _compact_progress_payload,
        _enrich_progress_payload,
        _factor_id_for_storage,
        _first_dict,
        _research_report_progress_fields,
        _save_loop_trace,
    )

    panel_data_paths_raw: list[str] = params.get("panel_data_paths") or []
    panel_data_path = params.get("panel_data_path", str(PROJECT_ROOT / "data" / "panel_data.parquet"))
    # Collect all paths: multi-path list wins over single path
    all_panel_paths: list[str] = panel_data_paths_raw if panel_data_paths_raw else [panel_data_path]
    prediction_horizon: str = str(params.get("prediction_horizon") or "24h").strip().lower() or "24h"
    bar_frequency: str = str(params.get("bar_frequency") or "auto").strip().lower() or "auto"
    loop_count = int(params.get("loop_count", 1))
    evolution_mode = bool(params.get("evolution_mode", False))
    evolution_profile = str(params.get("evolution_profile", "quantaalpha") or "quantaalpha").strip().lower() or "quantaalpha"
    research_report_pdf = str(params.get("research_report_pdf", "") or "")
    research_report_pdfs: list[str] = []
    for item in [research_report_pdf, *list(params.get("research_report_pdfs", []) or [])]:
        text = str(item or "").strip()
        if text and text not in research_report_pdfs:
            research_report_pdfs.append(text)
    initial_hypothesis = str(params.get("initial_hypothesis", "") or "").strip()
    run_store_path = str(params.get("run_store_path", RunStore.default_path(PROJECT_ROOT)))
    start_date: str | None = params.get("start_date")  # e.g. "2022-03"
    end_date: str | None = params.get("end_date")      # e.g. "2023-09"
    train_coverage_threshold = params.get("train_coverage_threshold")
    if train_coverage_threshold is not None:
        os.environ["TRAIN_COVERAGE_THRESHOLD"] = str(train_coverage_threshold)
    log_dir = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id
    progress_file = log_dir / "progress.jsonl"
    run_store = RunStore(run_store_path)
    log_lock = threading.Lock()
    progress_sequence = 0
    run_store.start_run(run_id=run_id, log_dir=str(log_dir), args=dict(params))

    def _stop_requested() -> bool:
        with _active_runs_lock:
            return bool(_active_runs.get(run_id, {}).get("stop_requested"))

    def _check_cancelled() -> None:
        if _stop_requested():
            raise RunCancelled(f"run {run_id} cancelled by user")

    def _emit_progress(payload: dict[str, Any], *, check_cancel: bool = True) -> None:
        nonlocal progress_sequence
        if check_cancel:
            _check_cancelled()
        enriched = _compact_progress_payload(_enrich_progress_payload(dict(payload)))
        progress_sequence += 1
        enriched.setdefault("sequence", progress_sequence)
        enriched.setdefault("run_id", run_id)
        enriched.setdefault("timestamp", datetime.now().isoformat())
        with log_lock:
            _append_jsonl(progress_file, enriched)
        progress_callback(enriched)

    _active_runs[run_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "stop_requested": False,
        "progress_callback": _emit_progress,
        "run_store_path": str(run_store_path),
    }

    _emit_progress(
        {
            "stage": "session_started",
            "loop_count": loop_count,
            "evolution_mode": evolution_mode,
            "evolution_profile": evolution_profile,
            "evolution_iterations": int(params.get("evolution_iterations", 5)),
            "evolution_num_initial_hypotheses": int(params.get("evolution_num_initial_hypotheses", 10)),
            "research_report_pdf": research_report_pdf or None,
            "research_report_pdfs": research_report_pdfs,
        }
    )

    def _run() -> None:
        session = None
        try:
            _check_cancelled()
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("LLM_MODEL", "")
            base_url = os.getenv("OPENAI_BASE_URL", None) or None
            try:
                request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
            except (TypeError, ValueError):
                request_timeout = 120.0
            provider = OpenAICompatibleProvider(
                api_key=api_key,
                base_url=base_url,
                timeout=request_timeout,
            )
            # 更稳的重试：瞬时 502/服务不可用/限流时指数退避重试，扛过 provider 短暂故障。
            # max_retries=5 → 最多 4 次重试，退避 2s→4s→8s→8s（max_wait 封顶 8s），约 22s 窗口。
            # 可用 LLM_MAX_RETRIES / LLM_RETRY_WAIT 环境变量覆盖。
            try:
                _max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
            except (TypeError, ValueError):
                _max_retries = 5
            try:
                _retry_wait = float(os.getenv("LLM_RETRY_WAIT", "2.0"))
            except (TypeError, ValueError):
                _retry_wait = 2.0
            gateway = LLMGateway(
                provider=provider,
                retry_policy=RetryPolicy(max_retries=max(1, _max_retries), wait_seconds=_retry_wait),
                default_model=model,
            )
            model_client = GatewayModelClient(gateway)

            # Set per-thread backtest config BEFORE building the workflow, so the
            # QlibFactorBacktestEngine reads the correct target column (matching the
            # selected prediction horizon) at construction time. If set afterwards the
            # engine would fall back to the yaml default (fwd_returns_4h) and reject
            # panels preprocessed for a different horizon → invalid_panel_or_target_column.
            # Canonicalize the horizon (minute/hour aware) so the target column and
            # annualization match the fwd_returns_* column the preprocessor wrote
            # (e.g. "60min"→fwd_returns_1h). The engine later overrides annualization
            # from the actual panel step, so this is only the initial value.
            _horizon_canon = _canonical_horizon(prediction_horizon) or "24h"
            _target_minutes = _horizon_to_minutes(_horizon_canon) or (24 * 60)
            _annualization = round(365 * 24 * 60 / _target_minutes)  # 24h→365, 4h→2190, 30min→17520
            from trading_agents.agents.backtest_runner_agent import set_thread_backtest_config
            set_thread_backtest_config(
                target_column=f"fwd_returns_{_horizon_canon}",
                annualization_factor=float(_annualization),
            )

            workflow = AlphaFactorMiningWorkflow(
                model_client=model_client,
                progress_callback=_emit_progress,
            )

            # Normalize user-provided market tables into the workflow panel contract.
            _check_cancelled()
            import pandas as pd
            df, file_summaries = _load_panel_from_paths(all_panel_paths)
            _emit_progress({
                "stage": "panel_data_loaded",
                "file_count": len(file_summaries),
                "files": file_summaries,
                "total_rows": sum(s["rows"] for s in file_summaries),
            })

            # 严格校验：主流程不再自动伪造预测标签。若输入数据缺少目标 fwd 列，
            # 直接让本次运行失败，避免在错误/空标签上乱回测。
            # 需要预测标签的数据请先在「数据预处理」页面生成并存入 data/processed/。
            target_col = f"fwd_returns_{_horizon_canon}"
            if target_col not in df.columns:
                available_fwd = [str(c) for c in df.columns if str(c).startswith("fwd_")]
                _emit_progress({
                    "stage": "preprocessing_failed",
                    "reason": (
                        f"输入数据缺少预测标签列 {target_col}，已终止本次运行。"
                        f"请先在「数据预处理」页面用该数据生成带 {target_col} 的文件，"
                        f"再到 data/processed 选择它运行。"
                    ),
                    "prediction_horizon": prediction_horizon,
                    "available_fwd_columns": available_fwd,
                }, check_cancel=False)
                raise RuntimeError(
                    f"missing target column {target_col}; available fwd columns={available_fwd}. "
                    f"请先在「数据预处理」生成带 {target_col} 的数据后再运行。"
                )

            # 判断原始文件能否被回测适配器直接读取：磁盘上已是 (datetime, symbol) 多级索引，
            # 或这两列以普通列存在（适配器加载时会自行重建索引）。需在标准化【之前】判断。
            _reuse_preprocessed_input = _should_reuse_preprocessed_panel(
                paths=all_panel_paths,
                df=df,
                target_col=target_col,
                start_date=start_date,
                end_date=end_date,
            )
            _input_adapter_ready = _adapter_can_read_panel_directly(df)

            # fwd 已存在 → 只做标准化（建立 MultiIndex / 派生特征），不重算标签。
            if _reuse_preprocessed_input:
                normalization_info = {
                    "changed": False,
                    "content_changed": False,
                    "original_rows": int(len(df)),
                    "prediction_horizon": prediction_horizon,
                    "bar_frequency_input": bar_frequency,
                    "bar_minutes": _panel_bar_minutes(df),
                    "source": "preprocessed_file_reused",
                }
            else:
                df, normalization_info = _standardize_panel_dataframe(
                    df, prediction_horizon=prediction_horizon, bar_frequency=bar_frequency, compute_targets=False
                )

            _cropped = False
            if start_date or end_date:
                dt_series = pd.to_datetime(df.index.get_level_values("datetime"))
                mask = pd.Series([True] * len(df), index=df.index)
                if start_date:
                    mask &= dt_series >= pd.Timestamp(start_date)
                if end_date:
                    end_text = str(end_date).strip()
                    if len(end_text) <= 7:
                        end_ts = pd.Timestamp(end_text) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    elif len(end_text) <= 10 and "T" not in end_text and " " not in end_text:
                        end_ts = pd.Timestamp(end_text) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    else:
                        end_ts = pd.Timestamp(end_text)
                    mask &= dt_series <= end_ts
                original_row_count = len(df)
                df = df[mask.values]
                _cropped = len(df) != original_row_count
                _emit_progress({
                    "stage": "data_filter_applied",
                    "original_rows": original_row_count,
                    "filtered_rows": len(df),
                    "start_date": start_date,
                    "end_date": end_date,
                    "cropped": _cropped,
                })

            # 只有确实需要时才另存到 data/processed/：多文件拼接、时间裁剪、或标准化
            # 真正改动了【列内容】(补特征 / 改列名 / 算标签)。若输入是【单个、未裁剪、
            # 已含 fwd、且适配器能直接读取】的文件，则直接复用原文件、不复制——纯索引
            # 重建由回测适配器在加载时完成，数据不变，无需多存一份。
            _reuse_input = (
                _reuse_preprocessed_input
                or (
                    len(all_panel_paths) == 1
                    and not _cropped
                    and _input_adapter_ready
                    and not bool(normalization_info.get("content_changed", True))
                )
            )
            tmp_path = None  # no temp file to clean up
            if _reuse_input:
                effective_path = str(all_panel_paths[0])
                _preprocess_msg = f"已校验数据包含 {target_col}，直接复用所选文件（未另存）"
            else:
                processed_dir = PROJECT_ROOT / "data" / "processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                processed_path = processed_dir / f"panel_{_horizon_canon}_{run_id}.parquet"
                df.to_parquet(str(processed_path))
                effective_path = str(processed_path)
                _preprocess_msg = f"数据已处理并另存 {processed_path.name}（已校验 {target_col}）"

            # 主流程不再计算标签，只校验：数据已含目标 fwd 列并完成标准化。
            _emit_progress({
                "stage": "preprocessing_complete",
                "prediction_horizon": prediction_horizon,
                "bar_minutes": normalization_info.get("bar_minutes"),
                "saved_to": effective_path,
                "was_computed": False,
                "reused_input": _reuse_input,
                "message": _preprocess_msg,
            })

            _emit_progress({
                "stage": "panel_data_normalized",
                "panel_data_path": str(panel_data_path),
                "effective_panel_data_path": effective_path,
                "row_count": len(df),
                **normalization_info,
            })

            _check_cancelled()
            adapter = CryptoCrossSectionDomainAdapter(effective_path, target_column=f"fwd_returns_{_horizon_canon}")
            payload = adapter.build_initial_payload()
            dataset_summary = payload.get("domain_dataset_summary", {})
            _emit_progress(
                {
                    "stage": "market_payload_finished",
                    "panel_data_path": str(panel_data_path),
                    "feature_count": len(list(dataset_summary.get("feature_columns", []))) if isinstance(dataset_summary, dict) else 0,
                    "target_count": len(list(dataset_summary.get("target_columns", []))) if isinstance(dataset_summary, dict) else 0,
                    "row_count": int(dataset_summary.get("rows", 0) or 0) if isinstance(dataset_summary, dict) else 0,
                    "symbol_count": int(dataset_summary.get("symbol_count", 0) or 0) if isinstance(dataset_summary, dict) else 0,
                    "datetime_start": str(dataset_summary.get("datetime_start", "")) if isinstance(dataset_summary, dict) else "",
                    "datetime_end": str(dataset_summary.get("datetime_end", "")) if isinstance(dataset_summary, dict) else "",
                }
            )

            _check_cancelled()
            if research_report_pdfs:
                _emit_progress(
                    {
                        "stage": "research_report_payload_started",
                        "report_count": len(research_report_pdfs),
                        "research_report_pdfs": research_report_pdfs,
                    }
                )
                if len(research_report_pdfs) > 1:
                    payload = workflow.build_multi_report_driven_initial_payload(
                        initial_shared_payload=payload,
                        research_report_pdfs=research_report_pdfs,
                    )
                else:
                    payload = workflow.build_report_driven_initial_payload(
                        initial_shared_payload=payload,
                        research_report_pdf=research_report_pdfs[0],
                    )
                research_report_payload = payload.get("research_report", {})
                _emit_progress(
                    {
                        "stage": "research_report_payload_finished",
                        "report_count": len(research_report_pdfs),
                        "multi_report_synthesis_status": str(
                            research_report_payload.get("multi_report_synthesis_status", "")
                        )
                        if isinstance(research_report_payload, dict)
                        else "",
                        **_research_report_progress_fields(research_report_payload),
                    }
                )
            if initial_hypothesis:
                payload["initial_hypothesis"] = initial_hypothesis

            _check_cancelled()
            if evolution_mode:
                evolution_iterations = max(1, int(params.get("evolution_iterations", 5)))
                profile_is_quantaalpha = evolution_profile == "quantaalpha"
                default_initial_directions = 10 if profile_is_quantaalpha else 5
                default_mutation_children = 10 if profile_is_quantaalpha else 3
                default_crossover_children = 10 if profile_is_quantaalpha else 2
                effective_initial_directions = max(
                    1,
                    int(params.get("evolution_num_initial_hypotheses", default_initial_directions)),
                )
                effective_mutation_tasks = max(
                    0,
                    int(params.get("evolution_mutation_tasks_per_round", default_mutation_children)),
                )
                effective_crossover_tasks = max(
                    0,
                    int(params.get("evolution_crossover_tasks_per_round", default_crossover_children)),
                )
                effective_max_rounds = max(
                    1,
                    int(params.get(
                        "evolution_max_rounds",
                        evolution_iterations * 2 + 1 if profile_is_quantaalpha else 6,
                    )),
                )
                session = workflow.run_evolution_session(
                    initial_shared_payload=payload,
                    initial_direction=initial_hypothesis or None,
                    evolution_config=EvolutionConfig(
                        profile=evolution_profile,
                        iterations=evolution_iterations,
                        phase_schedule="mutation_then_crossover" if profile_is_quantaalpha else "legacy",
                        max_rounds=effective_max_rounds,
                        num_initial_directions=effective_initial_directions,
                        # 初始假说资格赛（宁缺毋滥）：每个名额最多重生 10 次，
                        # 任一主线因子训练窗 Sharpe>1.0 或 abs(IC)>0.02 即可进入后续进化。
                        bootstrap_refill_enabled=bool(params.get("evolution_bootstrap_refill_enabled", True)),
                        bootstrap_max_refill_rounds=max(0, int(params.get("evolution_bootstrap_max_refill_rounds", 10))),
                        bootstrap_min_sharpe=float(params.get("evolution_bootstrap_min_sharpe", 1.0)),
                        bootstrap_min_abs_ic=float(params.get("evolution_bootstrap_min_abs_ic", 0.02)),
                        mutation_tasks_per_round=effective_mutation_tasks,
                        crossover_tasks_per_round=effective_crossover_tasks,
                        children_per_mutation_phase=effective_mutation_tasks,
                        children_per_crossover_phase=effective_crossover_tasks,
                        factor_expressions_per_hypothesis=2,
                        evolved_directions_per_round=max(1, int(params.get("evolution_evolved_directions_per_round", 5))),
                        mutation_parent_top_n=max(1, int(params.get("evolution_mutation_parent_top_n", 10))),
                        crossover_parent_top_n=max(1, int(params.get("evolution_crossover_parent_top_n", 10))),
                        mutation_top_n=max(0, int(params.get("evolution_mutation_top_n", 3))),
                        crossover_group_size=max(1, int(params.get("evolution_crossover_group_size", 2))),
                        crossover_n_groups=max(0, int(params.get("evolution_crossover_n_groups", 2))),
                        parallel_enabled=bool(params.get("evolution_parallel_enabled", True)),
                        max_workers=max(1, int(params.get("evolution_max_workers", 5))),
                    ),
                )
            else:
                session = workflow.run_loop_session(
                    initial_shared_payload=payload,
                    loop_count=loop_count,
                )
            _check_cancelled()
            for idx, trace in enumerate(session.loop_traces, start=1):
                loop_summary, compact_final_context, agent_records, _debug_path = _save_loop_trace(log_dir, idx, trace)
                run_store.record_loop(
                    run_id=run_id,
                    loop_index=idx,
                    summary=loop_summary,
                    final_context=compact_final_context,
                )
                for agent_record in agent_records:
                    run_store.record_agent(
                        run_id=run_id,
                        loop_index=idx,
                        agent_index=int(agent_record.get("agent_index", 0) or 0),
                        payload=agent_record,
                    )
                trajectory_id = str(loop_summary.get("trajectory_id", ""))
                for factor in list(loop_summary.get("factors", []) or []):
                    if not isinstance(factor, dict):
                        continue
                    run_store.upsert_factor(
                        run_id=run_id,
                        loop_index=idx,
                        trajectory_id=trajectory_id,
                        factor_id=_factor_id_for_storage(trajectory_id, factor),
                        factor=factor,
                        metrics=_first_dict(factor.get("metrics", {})),
                    )
            for trajectory in session.trajectory_pool.get_all():
                run_store.upsert_trajectory(run_id=run_id, trajectory=trajectory)
            with _active_runs_lock:
                _force_stopped = run_id in _force_stopped_runs
                _run_entry = _active_runs.get(run_id)
                if _run_entry is not None:
                    _run_entry["status"] = session.status
            if not _force_stopped:
                run_store.finish_run(run_id=run_id, status=session.status, summary={"run_id": run_id, "status": session.status})
            # 生成 Markdown 报告 + 图表 png（与 CLI main.py 行为对齐）。
            # 失败不影响主流程，仅记录一条进度事件。
            try:
                from scripts.generate_report import generate_report

                report_md_file, _ = generate_report(log_dir=log_dir, no_pdf=True)
                _emit_progress(
                    {
                        "stage": "report_generated",
                        "run_id": run_id,
                        "report_path": str(report_md_file) if report_md_file else None,
                    },
                    check_cancel=False,
                )
            except Exception as report_exc:
                _emit_progress(
                    {
                        "stage": "report_generation_failed",
                        "run_id": run_id,
                        "error": str(report_exc),
                    },
                    check_cancel=False,
                )
        except RunCancelled as exc:
            with _active_runs_lock:
                _run_entry = _active_runs.get(run_id)
                if _run_entry is not None:
                    _run_entry["status"] = "cancelled"
                    _run_entry["error"] = str(exc)
                _force_stopped = run_id in _force_stopped_runs
            if not _force_stopped:
                run_store.finish_run(run_id=run_id, status="cancelled", summary={"run_id": run_id, "status": "cancelled", "error": str(exc)})
            _emit_progress({"stage": "run_cancelled", "run_id": run_id, "error": str(exc)}, check_cancel=False)
        except Exception as exc:
            with _active_runs_lock:
                _run_entry = _active_runs.get(run_id)
                if _run_entry is not None:
                    _run_entry["status"] = "failed"
                    _run_entry["error"] = str(exc)
                _force_stopped = run_id in _force_stopped_runs
            if not _force_stopped:
                run_store.finish_run(run_id=run_id, status="failed", summary={"run_id": run_id, "status": "failed", "error": str(exc)})
            _emit_progress({"stage": "session_failed", "run_id": run_id, "error": str(exc)}, check_cancel=False)
        finally:
            # processed panel is kept in data/processed/ for audit.
            # 清理因子信号 pickle 缓存：仅在运行内用于 coder→runner 传递，运行后无消费者
            # 且会无界堆积（曾撑到上百 GB）。删它不影响因子库本身；KEEP_SIGNAL_CACHE=1 可保留。
            try:
                _cache_stats = cleanup_signal_cache()
                if not _cache_stats.get("skipped") and _cache_stats.get("removed_files"):
                    print(
                        f"[runner] cleaned signal cache: removed {_cache_stats['removed_files']} pkl, "
                        f"freed {_cache_stats['freed_bytes'] / 1e9:.2f} GB"
                    )
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True, name=f"run-{run_id}")
    thread.start()
