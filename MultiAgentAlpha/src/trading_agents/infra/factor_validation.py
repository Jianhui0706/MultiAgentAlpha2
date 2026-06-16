from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_agents.agents.backtest_runner_agent import QlibFactorBacktestEngine
from trading_agents.factor_runtime import calculate_factor_expression, downcast_float_columns


@dataclass(frozen=True)
class FactorValidationRequest:
    panel_data_path: str
    expression: str
    factor_name: str = "manual_factor"
    target_column: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    rolling_window: int = 60
    rolling_icir_mode: str = "ewm"  # "ewm" (近期指数加权，默认) | "simple" (等权)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _engine() -> QlibFactorBacktestEngine:
    config_path = _project_root() / "src" / "trading_agents" / "configs" / "qlib" / "backtest_engine.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"invalid backtest config: {config_path}")
    return QlibFactorBacktestEngine(config, config_path)


def _finite(value: object, digits: int = 8) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return round(number, digits)


def _pct(value: object, digits: int = 2) -> str:
    return f"{_finite(float(value) * 100.0, digits):.{digits}f}%"


def _metric(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if not np.isfinite(number):
        return "-"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.{digits}f}"


def _annualized_from_returns(returns: pd.Series, annualization_factor: float) -> float:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    total = float((1.0 + clean).prod() - 1.0)
    periods = max(1, len(clean))
    base = max(1e-12, 1.0 + total)
    return _finite(base ** (float(annualization_factor) / periods) - 1.0)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _series_points(series: pd.Series, *, digits: int = 8, max_points: int = 500) -> list[dict[str, Any]]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    step = max(1, len(clean) // max_points)
    sampled = clean.iloc[::step]
    return [
        {"timestamp": pd.Timestamp(index).isoformat(), "value": _finite(value, digits)}
        for index, value in sampled.items()
        if np.isfinite(float(value))
    ]


def _frame_points(frame: pd.DataFrame, *, digits: int = 8, max_points: int = 500) -> list[dict[str, Any]]:
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if clean.empty:
        return []
    step = max(1, len(clean) // max_points)
    sampled = clean.iloc[::step]
    points: list[dict[str, Any]] = []
    for index, row in sampled.iterrows():
        point: dict[str, Any] = {"timestamp": pd.Timestamp(index).isoformat()}
        for column, value in row.items():
            if pd.notna(value) and np.isfinite(float(value)):
                point[str(column)] = _finite(value, digits)
        points.append(point)
    return points


def _load_panel(path: str) -> pd.DataFrame:
    panel_path = Path(path)
    if not panel_path.is_absolute():
        panel_path = _project_root() / panel_path
    if not panel_path.exists():
        raise FileNotFoundError(f"panel data not found: {panel_path}")
    if panel_path.suffix.lower() == ".csv":
        panel = pd.read_csv(panel_path)
    else:
        panel = pd.read_parquet(panel_path)
    panel = downcast_float_columns(panel)
    if not isinstance(panel.index, pd.MultiIndex):
        date_col = next((c for c in ("datetime", "date", "timestamp", "time") if c in panel.columns), None)
        symbol_col = next((c for c in ("symbol", "instrument", "ticker", "code") if c in panel.columns), None)
        if date_col and symbol_col:
            panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
            panel = panel.dropna(subset=[date_col, symbol_col]).set_index([date_col, symbol_col])
    if not isinstance(panel.index, pd.MultiIndex) or "datetime" not in panel.index.names or "symbol" not in panel.index.names:
        raise ValueError("panel data must use a MultiIndex with datetime and symbol levels")
    panel = panel.sort_index()
    return panel


def _fwd_column_sort_key(column: str) -> tuple[float, str]:
    match = re.match(r"^fwd_returns_(\d+(?:\.\d+)?)(min|m|h|d)$", column)
    if not match:
        return (float("inf"), column)
    value = float(match.group(1))
    unit = match.group(2)
    multiplier = {"min": 1.0 / 60.0, "m": 1.0 / 60.0, "h": 1.0, "d": 24.0}.get(unit, 1.0)
    return (value * multiplier, column)


def inspect_panel(path: str) -> dict[str, Any]:
    panel = _load_panel(path)
    datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
    fwd_columns = sorted(
        [str(c) for c in panel.columns if str(c).startswith("fwd_returns_")],
        key=_fwd_column_sort_key,
    )
    feature_columns = [str(c) for c in panel.columns if not str(c).startswith("fwd_")]
    return {
        "path": str(Path(path)),
        "rows": int(len(panel)),
        "symbol_count": int(panel.index.get_level_values("symbol").nunique()),
        "datetime_start": pd.Timestamp(datetimes.min()).isoformat() if len(datetimes) else None,
        "datetime_end": pd.Timestamp(datetimes.max()).isoformat() if len(datetimes) else None,
        "fwd_columns": fwd_columns,
        "feature_columns": feature_columns,
        "columns": [str(c) for c in panel.columns],
    }


def _slice_panel(panel: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not start and not end:
        return panel
    datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
    mask = pd.Series(True, index=panel.index)
    if start:
        mask &= datetimes >= pd.Timestamp(start)
    if end:
        mask &= datetimes <= pd.Timestamp(end)
    return panel.loc[mask.to_numpy()].copy()


def _check_expression(expression: str) -> None:
    referenced_vars = re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", expression)
    lookahead = [name for name in referenced_vars if name.startswith("fwd_")]
    if lookahead:
        raise ValueError(f"expression references future label columns: {lookahead}")


def _split_timestamp(index: pd.Index, ratio: float = 0.75) -> pd.Timestamp | None:
    if index.empty:
        return None
    datetimes = pd.Index(pd.to_datetime(index, errors="coerce")).dropna().unique().sort_values()
    if len(datetimes) < 2:
        return None
    split_idx = max(1, min(len(datetimes) - 1, int(len(datetimes) * ratio)))
    return pd.Timestamp(datetimes[split_idx])


def _ic_stats(ic: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "mean": 0.0,
            "std": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "positive_ratio": 0.0,
            "period_count": 0,
        }
    mean = float(clean.mean())
    std = float(clean.std())
    return {
        "mean": _finite(mean),
        "std": _finite(std),
        "icir": _finite(mean / std) if abs(std) > 1e-12 else 0.0,
        "t_stat": _finite(mean / std * np.sqrt(len(clean))) if abs(std) > 1e-12 else 0.0,
        "positive_ratio": _finite(float((clean > 0).mean()), 4),
        "period_count": int(len(clean)),
    }


def _monthly_ic_table(ic: pd.Series, split_at: pd.Timestamp | None) -> list[dict[str, Any]]:
    clean = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if clean.empty:
        return []
    rows: list[dict[str, Any]] = []
    for month, group in clean.groupby(clean.index.to_period("M")):
        stats = _ic_stats(group)
        month_start = pd.Timestamp(month.start_time)
        rows.append({
            "month": str(month),
            "segment": "OS" if split_at is not None and month_start >= split_at else "IS",
            **stats,
        })
    return rows


def _is_os_stats(ic: pd.Series, split_at: pd.Timestamp | None) -> dict[str, Any]:
    clean = pd.to_numeric(ic, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if split_at is None or clean.empty:
        full = _ic_stats(clean)
        return {"split_at": None, "is": full, "os": _ic_stats(pd.Series(dtype=float)), "os_is_ratio": 0.0}
    is_stats = _ic_stats(clean[clean.index < split_at])
    os_stats = _ic_stats(clean[clean.index >= split_at])
    ratio = (
        _finite(float(os_stats["icir"]) / float(is_stats["icir"]))
        if abs(float(is_stats["icir"])) > 1e-12
        else 0.0
    )
    return {
        "split_at": split_at.isoformat(),
        "is": is_stats,
        "os": os_stats,
        "os_is_ratio": ratio,
    }


def _return_diagnostics(
    engine: QlibFactorBacktestEngine,
    signal: pd.Series,
    label: pd.Series,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.concat([signal.rename("signal"), label.rename("label")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return {
            "net_returns": pd.Series(dtype=float),
            "layer_returns": pd.DataFrame(),
            "yearly_annualized_returns": {},
            "layer_return_table": [],
        }
    names = list(frame.index.names)
    if "symbol" in names:
        names[names.index("symbol")] = "instrument"
        frame.index = frame.index.set_names(names)
    signal_wide = frame["signal"].unstack(level="instrument")
    label_wide = frame["label"].unstack(level="instrument")
    signal_mat = signal_wide.values.astype(np.float32)
    label_mat = label_wide.values.astype(np.float32)
    valid_mask = ~(np.isnan(signal_mat) | np.isnan(label_mat))
    valid_per_row = valid_mask.sum(axis=1)
    n_layers = 5
    thick_row = valid_per_row >= n_layers
    ranks = signal_wide.rank(axis=1, method="first").values
    layer_size = np.where(thick_row, valid_per_row / n_layers, 1.0)
    layer_mat = np.floor((ranks - 1.0) / layer_size[:, None]).astype(np.float32) + 1.0
    layer_mat = np.clip(layer_mat, 1, n_layers)
    layer_mat[~thick_row] = np.nan
    layer_mat[~valid_mask] = np.nan

    lr_mat = np.full((len(signal_wide.index), n_layers), np.nan)
    for idx in range(n_layers):
        mask = (layer_mat == (idx + 1)) & valid_mask
        cnt = mask.sum(axis=1).clip(1)
        lr_mat[:, idx] = np.where(
            mask.any(axis=1),
            np.nansum(np.where(mask, label_mat, np.nan), axis=1) / cnt,
            np.nan,
        )
    layer_returns = pd.DataFrame(lr_mat, index=signal_wide.index, columns=[f"G{i}" for i in range(1, n_layers + 1)])
    ls = (layer_returns["G5"] - layer_returns["G1"]).dropna().sort_index()

    high_mask = (layer_mat == n_layers) & valid_mask
    low_mask = (layer_mat == 1) & valid_mask
    n_high = high_mask.sum(axis=1, keepdims=True).clip(1)
    n_low = low_mask.sum(axis=1, keepdims=True).clip(1)
    weights = np.where(high_mask, 0.5 / n_high, np.where(low_mask, -0.5 / n_low, 0.0))
    weights[~valid_mask] = 0.0
    first_cost = np.nansum(np.abs(weights[0])) * engine.one_way_fee_rate
    diff_cost = np.abs(np.diff(weights, axis=0)).sum(axis=1) * engine.one_way_fee_rate
    turnover_cost = pd.Series(np.concatenate([[first_cost], diff_cost]), index=signal_wide.index)

    ls = engine._subsample_returns_to_horizon(ls)
    layer_returns = engine._subsample_frame_to_horizon(layer_returns)
    direction_sign = 1.0 if float(metrics.get("Rank IC", 0.0) or 0.0) > 0 else -1.0
    orientation = -1.0 if direction_sign < 0 else 1.0
    cost = turnover_cost.reindex(ls.index, fill_value=0.0)
    net_returns = pd.Series(ls.values.astype(np.float32) * orientation - cost.values.astype(np.float32), index=ls.index)
    annual_factor = float(getattr(engine, "_current_sharpe_annualization_factor", engine.annualization_factor))

    yearly_annualized = {
        str(int(year)): _annualized_from_returns(group, annual_factor)
        for year, group in net_returns.groupby(pd.to_datetime(net_returns.index).year)
    }
    layer_table: list[dict[str, Any]] = []
    for column in layer_returns.columns:
        clean = pd.to_numeric(layer_returns[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            layer_table.append({"group": column, "mean_return": 0.0, "annualized_return": 0.0, "cumulative_return": 0.0})
            continue
        layer_table.append({
            "group": column,
            "mean_return": _finite(clean.mean()),
            "annualized_return": _finite(float(clean.mean()) * annual_factor),
            "cumulative_return": _finite((1.0 + clean).prod() - 1.0),
        })
    return {
        "net_returns": net_returns,
        "layer_returns": layer_returns,
        "yearly_annualized_returns": yearly_annualized,
        "layer_return_table": layer_table,
        "compounded_annualized_return": _annualized_from_returns(net_returns, annual_factor),
    }


def _ic_diagnostics(engine: QlibFactorBacktestEngine, signal: pd.Series, label: pd.Series) -> dict[str, Any]:
    eval_frame = pd.concat([signal.rename("signal"), label.rename("label")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if eval_frame.empty:
        return {"ic_series": [], "rank_ic_series": [], "cumulative_ic": [], "monthly_ic": [], "rolling_icir": []}
    if isinstance(eval_frame.index, pd.MultiIndex):
        names = list(eval_frame.index.names)
        if "symbol" in names:
            names[names.index("symbol")] = "instrument"
            eval_frame.index = eval_frame.index.set_names(names)
    # 非重叠 IC：按预测窗口降采样，与主回测引擎 / reward 口径一致——否则重叠窗口
    # 的强自相关会让验证页的 ICIR / 滚动 ICIR 与运行口径不一致且偏乐观。
    ic = engine._subsample_ic_to_horizon(engine._cross_sectional_corr(eval_frame, rank=False)).sort_index()
    rank_ic = engine._subsample_ic_to_horizon(engine._cross_sectional_corr(eval_frame, rank=True)).sort_index()
    rolling_window = max(2, int(getattr(engine, "_manual_rolling_window", 60)))
    rolling_mode = str(getattr(engine, "_manual_rolling_icir_mode", "ewm")).strip().lower()
    _min_p = max(2, rolling_window // 3)
    if rolling_mode == "simple":
        # 等权滚动 ICIR：窗口内 IC 等权 mean/std。
        rolling_icir = ic.rolling(rolling_window, min_periods=_min_p).mean() / ic.rolling(
            rolling_window, min_periods=_min_p
        ).std()
    else:
        # 指数加权（EWM，默认）滚动 ICIR：近期 IC 权重指数级更高（span=滚动窗口），
        # 对近期表现赋予更高权重（会议 6.15 #7）。
        rolling_icir = ic.ewm(span=rolling_window, min_periods=_min_p).mean() / ic.ewm(
            span=rolling_window, min_periods=_min_p
        ).std()
    monthly_ic = ic.resample("MS").mean() if not ic.empty else pd.Series(dtype=float)
    cumulative = pd.DataFrame({"IC": ic.cumsum(), "Rank IC": rank_ic.cumsum()}).dropna(how="all")
    split_at = _split_timestamp(ic.index)
    return {
        "ic_series": _series_points(ic),
        "rank_ic_series": _series_points(rank_ic),
        "cumulative_ic": _frame_points(cumulative),
        "monthly_ic": _series_points(monthly_ic),
        "rolling_icir": _series_points(rolling_icir),
        "ic_stats": _ic_stats(ic),
        "rank_ic_stats": _ic_stats(rank_ic),
        "is_os_stats": _is_os_stats(ic, split_at),
        "monthly_ic_table": _monthly_ic_table(ic, split_at),
        "raw": {"ic": ic, "rank_ic": rank_ic, "monthly_ic": monthly_ic, "rolling_icir": rolling_icir},
    }


def _plot_outputs(report_dir: Path, diagnostics: dict[str, Any], return_diagnostics: dict[str, Any]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = report_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    raw = diagnostics["raw"]
    paths: dict[str, str] = {}

    def save(name: str) -> None:
        path = charts_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths[name.removesuffix(".png")] = str(path)

    ic = raw["ic"]
    rank_ic = raw["rank_ic"]
    if not ic.empty or not rank_ic.empty:
        plt.figure(figsize=(10, 4.5))
        if not ic.empty:
            ic.cumsum().plot(label="Cumulative IC", linewidth=1.5)
        if not rank_ic.empty:
            rank_ic.cumsum().plot(label="Cumulative Rank IC", linewidth=1.5)
        plt.axhline(0, color="#888", linewidth=0.8)
        plt.title("Cumulative IC")
        plt.legend()
        save("cumulative_ic.png")

        plt.figure(figsize=(10, 4.5))
        if not ic.empty:
            ic.plot(label="IC", linewidth=1.0, alpha=0.85)
        if not rank_ic.empty:
            rank_ic.plot(label="Rank IC", linewidth=1.0, alpha=0.75)
        plt.axhline(0, color="#888", linewidth=0.8)
        plt.title("IC Time Series")
        plt.legend()
        save("ic_timeseries.png")

    monthly_ic = raw["monthly_ic"]
    if not monthly_ic.empty:
        plt.figure(figsize=(10, 4.5))
        colors = ["#2f80ed" if v >= 0 else "#eb5757" for v in monthly_ic.values]
        labels = [pd.Timestamp(index).strftime("%Y-%m") for index in monthly_ic.index]
        plt.bar(labels, monthly_ic.values, color=colors)
        plt.axhline(0, color="#888", linewidth=0.8)
        plt.title("Monthly IC")
        plt.xticks(rotation=45, ha="right")
        save("monthly_ic.png")

    rolling_icir = raw["rolling_icir"]
    if not rolling_icir.empty:
        plt.figure(figsize=(10, 4.5))
        rolling_icir.plot(color="#7b61ff", linewidth=1.3)
        plt.axhline(0, color="#888", linewidth=0.8)
        plt.title("Rolling ICIR")
        save("rolling_icir.png")

    if not ic.empty:
        plt.figure(figsize=(8.5, 4.5))
        plt.hist(ic.values, bins=50, color="#4f8bd6", alpha=0.85)
        plt.axvline(float(ic.mean()), color="#d9534f", linewidth=1.4, label=f"Mean={ic.mean():.4f}")
        plt.axvline(0, color="#555", linewidth=0.8)
        plt.title("IC Distribution")
        plt.legend()
        save("ic_distribution.png")

    yearly = return_diagnostics.get("yearly_annualized_returns", {})
    if isinstance(yearly, dict) and yearly:
        plt.figure(figsize=(9, 4.5))
        labels = list(yearly.keys())
        values = [float(yearly[k]) for k in labels]
        colors = ["#2f80ed" if v >= 0 else "#eb5757" for v in values]
        plt.bar(labels, values, color=colors)
        plt.axhline(0, color="#888", linewidth=0.8)
        plt.title("Yearly Annualized Return")
        plt.gca().yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
        save("yearly_annualized_return.png")

    layer_returns = return_diagnostics.get("layer_returns")
    if isinstance(layer_returns, pd.DataFrame) and not layer_returns.empty:
        cumulative_layers = (1.0 + layer_returns.dropna(how="all")).cumprod()
        if not cumulative_layers.empty:
            plt.figure(figsize=(10, 4.8))
            for column in cumulative_layers.columns:
                cumulative_layers[column].plot(label=column, linewidth=1.2)
            plt.axhline(1.0, color="#888", linewidth=0.8)
            plt.title("Five-Group Cumulative Return")
            plt.legend(ncol=5, fontsize=8)
            save("five_group_cumulative_return.png")

    return paths


def _infer_frequency_text(panel: pd.DataFrame) -> str:
    datetimes = pd.Index(pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")).dropna().unique().sort_values()
    if len(datetimes) < 2:
        return "unknown"
    diffs = datetimes.to_series().diff().dropna()
    if diffs.empty:
        return "unknown"
    step = pd.Timedelta(diffs.mode().iloc[0])
    total_minutes = int(step.total_seconds() // 60)
    if total_minutes <= 0:
        return str(step)
    if total_minutes % 1440 == 0:
        return f"{total_minutes // 1440}d"
    if total_minutes % 60 == 0:
        return f"{total_minutes // 60}h"
    return f"{total_minutes}min"


def _evaluation_bullets(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    ic_stats = diagnostics.get("ic_stats", {}) if isinstance(diagnostics.get("ic_stats"), dict) else {}
    is_os = diagnostics.get("is_os_stats", {}) if isinstance(diagnostics.get("is_os_stats"), dict) else {}
    bullets: list[str] = []
    icir = abs(float(ic_stats.get("icir", 0.0) or 0.0))
    if icir >= 0.3:
        bullets.append(f"ICIR={_metric(ic_stats.get('icir'), 4)}，截面预测稳定性较强。")
    elif icir >= 0.15:
        bullets.append(f"ICIR={_metric(ic_stats.get('icir'), 4)}，截面预测稳定性中等。")
    else:
        bullets.append(f"ICIR={_metric(ic_stats.get('icir'), 4)}，截面预测稳定性偏弱，需要谨慎复核。")
    bullets.append(
        f"IC>0 占比为 {_pct(ic_stats.get('positive_ratio', 0.0), 1)}，有效期数 {int(ic_stats.get('period_count', 0) or 0):,}。"
    )
    t_stat = float(ic_stats.get("t_stat", 0.0) or 0.0)
    if abs(t_stat) >= 2.0:
        bullets.append(f"t 统计量={_metric(t_stat, 2)}，按 IC 时序口径具备统计显著性。")
    else:
        bullets.append(f"t 统计量={_metric(t_stat, 2)}，统计显著性不足。")
    if is_os.get("split_at"):
        ratio = float(is_os.get("os_is_ratio", 0.0) or 0.0)
        bullets.append(f"OS/IS ICIR 比例={_metric(ratio, 2)}x，用于观察样本外衰减或增强。")
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    bullets.append(f"多空组合净 Sharpe={_metric(sharpe, 4)}，年化收益率={_pct(metrics.get('annualized_return', 0.0), 2)}。")
    return bullets


def _format_ic_stat_table(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    ic_stats = diagnostics.get("ic_stats", {}) if isinstance(diagnostics.get("ic_stats"), dict) else {}
    rank_stats = diagnostics.get("rank_ic_stats", {}) if isinstance(diagnostics.get("rank_ic_stats"), dict) else {}
    return [
        {"metric": "IC mean", "value": _metric(ic_stats.get("mean"), 6), "note": "Pearson cross-section IC"},
        {"metric": "ICIR", "value": _metric(ic_stats.get("icir"), 4), "note": "IC mean / IC std"},
        {"metric": "t-stat", "value": _metric(ic_stats.get("t_stat"), 2), "note": "IC time-series significance"},
        {"metric": "IC std", "value": _metric(ic_stats.get("std"), 6), "note": "IC volatility"},
        {"metric": "IC > 0", "value": _pct(ic_stats.get("positive_ratio", 0.0), 1), "note": "Directional hit rate"},
        {"metric": "Valid periods", "value": f"{int(ic_stats.get('period_count', 0) or 0):,}", "note": "Non-null IC points"},
        {"metric": "Rank IC mean", "value": _metric(rank_stats.get("mean"), 6), "note": "Spearman-style rank IC"},
        {"metric": "Rank ICIR", "value": _metric(rank_stats.get("icir"), 4), "note": "Rank IC mean / std"},
    ]


def _configure_matplotlib_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    for font_path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            matplotlib.rcParams["font.family"] = [font_name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def _pdf_text_page(pdf: Any, title: str, lines: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", color="#1f2937")
    y = 0.89
    for line in lines:
        fig.text(0.08, y, line, fontsize=10.5, color="#111827", wrap=True)
        y -= 0.035 if line else 0.02
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_table_pages(
    pdf: Any,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
    *,
    rows_per_page: int = 22,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        rows = [["-", "-", "-"][: len(columns)]]
    for page_idx, start in enumerate(range(0, len(rows), rows_per_page)):
        chunk = rows[start:start + rows_per_page]
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        suffix = f" ({page_idx + 1})" if len(rows) > rows_per_page else ""
        ax.set_title(title + suffix, loc="left", fontsize=16, weight="bold", pad=18)
        table = ax.table(
            cellText=[[str(value) for value in row] for row in chunk],
            colLabels=columns,
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0.05, 1, 0.88],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.25)
        for (row_idx, _col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#1f77b4")
                cell.set_text_props(color="white", weight="bold")
            elif row_idx % 2 == 0:
                cell.set_facecolor("#f3f6fb")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _pdf_image_page(pdf: Any, title: str, image_path: str) -> None:
    import matplotlib.pyplot as plt

    path = Path(image_path)
    if not path.exists():
        return
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=18)
    ax.imshow(image)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    report_dir: Path,
    *,
    request: FactorValidationRequest,
    target_column: str,
    panel: pd.DataFrame,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    return_diagnostics: dict[str, Any],
    charts: dict[str, str],
) -> Path:
    datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
    is_os = diagnostics.get("is_os_stats", {}) if isinstance(diagnostics.get("is_os_stats"), dict) else {}
    yearly = return_diagnostics.get("yearly_annualized_returns", {})
    layer_table = return_diagnostics.get("layer_return_table", [])
    lines = [
        "# 固定因子验证报告",
        "",
        "## 指标定义与数据范围",
        "",
        f"- 指标名称：`{request.factor_name}`",
        f"- 因子表达式：`{request.expression}`",
        f"- 回测数据：`{request.panel_data_path}`",
        f"- 目标收益列：`{target_column}`",
        f"- 数据频率：{_infer_frequency_text(panel)}",
        f"- 数据范围：{pd.Timestamp(datetimes.min()).isoformat()} ~ {pd.Timestamp(datetimes.max()).isoformat()}",
        f"- 样本规模：{len(panel):,} 行，{panel.index.get_level_values('symbol').nunique():,} 个标的",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    core_rows = [
        ("IC", _metric(metrics.get("IC"), 6)),
        ("ICIR", _metric(metrics.get("ICIR"), 4)),
        ("Rank IC", _metric(metrics.get("Rank IC"), 6)),
        ("Rank ICIR", _metric(metrics.get("Rank ICIR"), 4)),
        ("Sharpe", _metric(metrics.get("sharpe"), 4)),
        ("Drawdown", _pct(metrics.get("drawdown", 0.0), 2)),
        ("Coverage", _pct(metrics.get("coverage", 0.0), 2)),
        ("Turnover", _metric(metrics.get("turnover"), 4)),
        ("Annualized return", _pct(metrics.get("annualized_return", 0.0), 2)),
        ("Compounded annualized return", _pct(return_diagnostics.get("compounded_annualized_return", 0.0), 2)),
    ]
    for key, value in core_rows:
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## IS / OS 详细统计", "", "| 指标 | 样本内 IS | 样本外 OS |", "| --- | ---: | ---: |"])
    is_stats = is_os.get("is", {}) if isinstance(is_os.get("is"), dict) else {}
    os_stats = is_os.get("os", {}) if isinstance(is_os.get("os"), dict) else {}
    for key, label in [("mean", "IC 均值"), ("icir", "ICIR"), ("t_stat", "t 统计量"), ("std", "IC 标准差"), ("positive_ratio", "IC>0 占比"), ("period_count", "有效期数")]:
        left = _pct(is_stats.get(key, 0.0), 1) if key == "positive_ratio" else (f"{int(is_stats.get(key, 0) or 0):,}" if key == "period_count" else _metric(is_stats.get(key), 6 if key in {"mean", "std"} else 4))
        right = _pct(os_stats.get(key, 0.0), 1) if key == "positive_ratio" else (f"{int(os_stats.get(key, 0) or 0):,}" if key == "period_count" else _metric(os_stats.get(key), 6 if key in {"mean", "std"} else 4))
        lines.append(f"| {label} | {left} | {right} |")

    lines.extend(["", "## 月度 IC 明细", "", "| 月份 | 分段 | IC 均值 | IC 标准差 | ICIR | IC>0 占比 | 有效期数 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in diagnostics.get("monthly_ic_table", []):
        if isinstance(row, dict):
            lines.append(
                f"| {row.get('month')} | {row.get('segment')} | {_metric(row.get('mean'), 5)} | {_metric(row.get('std'), 5)} | {_metric(row.get('icir'), 4)} | {_pct(row.get('positive_ratio', 0.0), 1)} | {int(row.get('period_count', 0) or 0):,} |"
            )

    lines.extend(["", "## 收益表现", "", f"- 年化收益率：{_pct(metrics.get('annualized_return', 0.0), 2)}"])
    lines.append(f"- 复利年化收益率：{_pct(return_diagnostics.get('compounded_annualized_return', 0.0), 2)}")
    lines.append(f"- 最大回撤：{_pct(metrics.get('drawdown', 0.0), 2)}")

    lines.extend(["", "### 分年年化收益率", "", "| 年份 | 年化收益率 |", "| --- | ---: |"])
    if isinstance(yearly, dict) and yearly:
        for year, value in yearly.items():
            lines.append(f"| {year} | {_pct(value, 2)} |")

    lines.extend(["", "### 五分组收益率", "", "| 分组 | 平均单期收益 | 年化收益率 | 累计收益率 |", "| --- | ---: | ---: | ---: |"])
    if isinstance(layer_table, list):
        for row in layer_table:
            if isinstance(row, dict):
                lines.append(
                    f"| {row.get('group')} | {_pct(row.get('mean_return', 0.0), 4)} | {_pct(row.get('annualized_return', 0.0), 2)} | {_pct(row.get('cumulative_return', 0.0), 2)} |"
                )

    lines.extend(["", "## 图表", ""])
    labels = {
        "cumulative_ic": "累积 IC",
        "ic_timeseries": "IC 时序",
        "monthly_ic": "月度 IC 柱状图",
        "ic_distribution": "IC 分布直方图",
        "rolling_icir": "滚动 ICIR",
        "yearly_annualized_return": "分年年化收益率",
        "five_group_cumulative_return": "五分组累计收益",
    }
    for key, label in labels.items():
        if key in charts:
            rel = Path(charts[key]).relative_to(report_dir).as_posix()
            lines.extend([f"### {label}", "", f"![{label}]({rel})", ""])
    lines.extend(["", "## 综合评价", ""])
    for bullet in _evaluation_bullets(metrics, diagnostics):
        lines.append(f"- {bullet}")
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_pdf_report(
    report_dir: Path,
    *,
    request: FactorValidationRequest,
    target_column: str,
    panel: pd.DataFrame,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    return_diagnostics: dict[str, Any],
    charts: dict[str, str],
) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    _configure_matplotlib_font()
    datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
    pdf_path = report_dir / "factor_validation_report.pdf"
    is_os = diagnostics.get("is_os_stats", {}) if isinstance(diagnostics.get("is_os_stats"), dict) else {}
    is_stats = is_os.get("is", {}) if isinstance(is_os.get("is"), dict) else {}
    os_stats = is_os.get("os", {}) if isinstance(is_os.get("os"), dict) else {}
    yearly = return_diagnostics.get("yearly_annualized_returns", {})
    layer_table = return_diagnostics.get("layer_return_table", [])

    with PdfPages(pdf_path) as pdf:
        _pdf_text_page(pdf, "固定因子验证报告", [
            f"指标名称：{request.factor_name}",
            f"因子表达式：{request.expression}",
            f"目标收益列：{target_column}",
            f"数据频率：{_infer_frequency_text(panel)}",
            f"数据范围：{pd.Timestamp(datetimes.min()).isoformat()} ~ {pd.Timestamp(datetimes.max()).isoformat()}",
            f"样本规模：{len(panel):,} 行 / {panel.index.get_level_values('symbol').nunique():,} 个标的",
            f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
            "",
            "本报告参照截面 IC 测试报告结构生成：核心指标概览、IS/OS 统计、月度 IC 明细、IC 图表、收益分解、五分组收益和综合评价。",
        ])
        _pdf_table_pages(pdf, "核心指标概览", ["指标", "数值", "说明"], [
            ["IC", _metric(metrics.get("IC"), 6), "Pearson 截面 IC"],
            ["ICIR", _metric(metrics.get("ICIR"), 4), "IC 均值 / 标准差"],
            ["Rank IC", _metric(metrics.get("Rank IC"), 6), "Rank/Spearman 截面 IC"],
            ["Rank ICIR", _metric(metrics.get("Rank ICIR"), 4), "Rank IC 均值 / 标准差"],
            ["Sharpe", _metric(metrics.get("sharpe"), 4), "交易成本后多空组合"],
            ["年化收益率", _pct(metrics.get("annualized_return", 0.0), 2), "算术年化口径"],
            ["复利年化收益率", _pct(return_diagnostics.get("compounded_annualized_return", 0.0), 2), "累计收益折算"],
            ["最大回撤", _pct(metrics.get("drawdown", 0.0), 2), "多空净值曲线"],
            ["覆盖率", _pct(metrics.get("coverage", 0.0), 2), "有效信号 / 面板样本"],
            ["换手率", _metric(metrics.get("turnover"), 4), "截面排名变化代理"],
        ])
        _pdf_table_pages(pdf, "IS / OS 详细统计", ["指标", "样本内 IS", "样本外 OS"], [
            ["IC 均值", _metric(is_stats.get("mean"), 6), _metric(os_stats.get("mean"), 6)],
            ["ICIR", _metric(is_stats.get("icir"), 4), _metric(os_stats.get("icir"), 4)],
            ["t 统计量", _metric(is_stats.get("t_stat"), 2), _metric(os_stats.get("t_stat"), 2)],
            ["IC 标准差", _metric(is_stats.get("std"), 6), _metric(os_stats.get("std"), 6)],
            ["IC>0 占比", _pct(is_stats.get("positive_ratio", 0.0), 1), _pct(os_stats.get("positive_ratio", 0.0), 1)],
            ["有效期数", f"{int(is_stats.get('period_count', 0) or 0):,}", f"{int(os_stats.get('period_count', 0) or 0):,}"],
            ["OS / IS ICIR", _metric(is_os.get("os_is_ratio"), 2) + "x", "-"],
        ])
        monthly_rows = [
            [
                row.get("month"),
                row.get("segment"),
                _metric(row.get("mean"), 5),
                _metric(row.get("std"), 5),
                _metric(row.get("icir"), 4),
                _pct(row.get("positive_ratio", 0.0), 1),
                f"{int(row.get('period_count', 0) or 0):,}",
            ]
            for row in diagnostics.get("monthly_ic_table", [])
            if isinstance(row, dict)
        ]
        _pdf_table_pages(pdf, "月度 IC 明细", ["月份", "分段", "IC均值", "IC标准差", "ICIR", "IC>0", "期数"], monthly_rows, rows_per_page=24)
        yearly_rows = [[year, _pct(value, 2)] for year, value in (yearly.items() if isinstance(yearly, dict) else [])]
        _pdf_table_pages(pdf, "分年年化收益率", ["年份", "年化收益率"], yearly_rows)
        layer_rows = [
            [row.get("group"), _pct(row.get("mean_return", 0.0), 4), _pct(row.get("annualized_return", 0.0), 2), _pct(row.get("cumulative_return", 0.0), 2)]
            for row in layer_table
            if isinstance(row, dict)
        ] if isinstance(layer_table, list) else []
        _pdf_table_pages(pdf, "五分组收益率", ["分组", "平均单期收益", "年化收益率", "累计收益率"], layer_rows)
        for key, title in [
            ("ic_timeseries", "IC 时序图"),
            ("cumulative_ic", "累积 IC"),
            ("monthly_ic", "月度 IC 柱状图"),
            ("ic_distribution", "IC 分布直方图"),
            ("rolling_icir", "滚动 ICIR（稳定性）"),
            ("yearly_annualized_return", "分年年化收益率"),
            ("five_group_cumulative_return", "五分组累计收益"),
        ]:
            if key in charts:
                _pdf_image_page(pdf, title, charts[key])
        _pdf_text_page(pdf, "综合评价", _evaluation_bullets(metrics, diagnostics))
    return pdf_path


def run_validation(request: FactorValidationRequest, *, reports_root: str | Path | None = None) -> dict[str, Any]:
    expression = request.expression.strip()
    if not expression:
        raise ValueError("factor expression is required")
    _check_expression(expression)

    panel = _load_panel(request.panel_data_path)
    target_column = request.target_column or next((str(c) for c in panel.columns if str(c).startswith("fwd_returns_")), "")
    if not target_column or target_column not in panel.columns:
        available_targets = sorted(
            [str(c) for c in panel.columns if str(c).startswith("fwd_returns_")],
            key=_fwd_column_sort_key,
        )
        raise ValueError(
            f"target column not found: {target_column}. "
            f"Available fwd_returns_* columns: {available_targets or 'none'}"
        )
    panel = _slice_panel(panel, request.start_datetime, request.end_datetime)
    if panel.empty:
        raise ValueError("selected time window produced an empty panel")

    engine = _engine()
    engine.target_column = target_column
    engine._configure_sharpe_sampling(panel)
    setattr(engine, "_manual_rolling_window", max(2, int(request.rolling_window or 60)))
    setattr(engine, "_manual_rolling_icir_mode", str(getattr(request, "rolling_icir_mode", "ewm") or "ewm").strip().lower())

    label = pd.to_numeric(panel[target_column], errors="coerce")
    feature_panel = panel.drop(columns=[c for c in panel.columns if str(c).startswith("fwd_")], errors="ignore")
    signal = calculate_factor_expression(expression, feature_panel, request.factor_name or "manual_factor")
    if signal is None or signal.dropna().empty:
        raise ValueError("factor calculation returned empty signal")

    metrics = engine._calculate_factor_metrics_bundle(
        factor_name=request.factor_name or "manual_factor",
        expression=expression,
        signal_series=signal,
        label_series=label,
        panel=panel,
    )
    diagnostics = _ic_diagnostics(engine, signal, label)
    return_diagnostics = _return_diagnostics(engine, signal, label, metrics)

    reports_root_path = Path(reports_root) if reports_root else _project_root() / "reports"
    report_id = "factor_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = reports_root_path / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    charts = _plot_outputs(report_dir, diagnostics, return_diagnostics)
    report_path = _write_report(
        report_dir,
        request=request,
        target_column=target_column,
        panel=panel,
        metrics=metrics,
        diagnostics=diagnostics,
        return_diagnostics=return_diagnostics,
        charts=charts,
    )
    pdf_path = _write_pdf_report(
        report_dir,
        request=request,
        target_column=target_column,
        panel=panel,
        metrics=metrics,
        diagnostics=diagnostics,
        return_diagnostics=return_diagnostics,
        charts=charts,
    )

    payload = {
        "report_id": report_id,
        "report_dir": str(report_dir),
        "report_path": str(report_path),
        "pdf_path": str(pdf_path),
        "metrics": _json_safe(metrics),
        "diagnostics": {k: v for k, v in diagnostics.items() if k != "raw"},
        "return_diagnostics": _json_safe({
            "yearly_annualized_returns": return_diagnostics.get("yearly_annualized_returns", {}),
            "layer_return_table": return_diagnostics.get("layer_return_table", []),
            "compounded_annualized_return": return_diagnostics.get("compounded_annualized_return", 0.0),
        }),
        "charts": charts,
        "data_range": {
            "start": pd.Timestamp(panel.index.get_level_values("datetime").min()).isoformat(),
            "end": pd.Timestamp(panel.index.get_level_values("datetime").max()).isoformat(),
            "rows": int(len(panel)),
            "symbols": int(panel.index.get_level_values("symbol").nunique()),
            "target_column": target_column,
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.make_archive(str(report_dir), "zip", root_dir=report_dir)
    return payload
