"""
分组收益曲线绘图脚本
从 factor library / evolution log 中提取 top 因子，重新计算因子值，绘制 5 分组累积收益曲线。

用法:
    python scripts/plot_group_returns.py                  # 自动用最近一次 run 的 log
    python scripts/plot_group_returns.py --sharpe-min 0.2  # 只画 sharpe >= 0.2 的因子
    python scripts/plot_group_returns.py --top 5           # 只画 sharpe 最高的 5 个因子
    python scripts/plot_group_returns.py --log path/to/evolution_debug.log
    python scripts/plot_group_returns.py --out results/group_returns.png
"""

import sys
import os
import re
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 路径设置 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trading_agents.factor_runtime.factor_functions import calculate_factor_expression  # noqa: E402
from trading_agents.factor_runtime import factor_functions as _ff  # noqa: E402


def _MAX(a, b=None):
    """Element-wise max(a, b) when 2 args; cross-sectional max when 1 arg."""
    if b is None:
        return _ff.MAX(a)
    if isinstance(b, (int, float)):
        return a.clip(lower=b)
    return pd.concat([a, b], axis=1).max(axis=1)


def _MIN(a, b=None):
    """Element-wise min(a, b) when 2 args; cross-sectional min when 1 arg."""
    if b is None:
        return _ff.MIN(a)
    if isinstance(b, (int, float)):
        return a.clip(upper=b)
    return pd.concat([a, b], axis=1).min(axis=1)


def _patched_calculate_factor_expression(
    expression: str, panel: pd.DataFrame, factor_name: str
) -> pd.Series | None:
    """calculate_factor_expression patched with 2-arg MAX/MIN and fixed TS_QUANTILE order."""
    orig_max, orig_min, orig_tsq = _ff.MAX, _ff.MIN, _ff.TS_QUANTILE

    # 捕获原始函数到局部变量，避免 patch 后递归调用自身
    _orig_tsq = orig_tsq

    def _TS_QUANTILE_fixed(series, q_or_n, n_or_q=None):
        """
        Expressions write TS_QUANTILE(series, quantile, window).
        Function signature is TS_QUANTILE(series, n_window, quantile).
        Auto-detect by value: first numeric arg < 1 → it's quantile.
        """
        if n_or_q is None:
            return _orig_tsq(series, int(q_or_n), 0.5)
        if isinstance(q_or_n, float) and 0 < q_or_n < 1:
            return _orig_tsq(series, int(n_or_q), q_or_n)
        return _orig_tsq(series, int(q_or_n), n_or_q)

    try:
        _ff.MAX         = _MAX                  # type: ignore[attr-defined]
        _ff.MIN         = _MIN                  # type: ignore[attr-defined]
        _ff.TS_QUANTILE = _TS_QUANTILE_fixed    # type: ignore[attr-defined]
        return calculate_factor_expression(expression, panel, factor_name)
    finally:
        _ff.MAX         = orig_max              # type: ignore[attr-defined]
        _ff.MIN         = orig_min              # type: ignore[attr-defined]
        _ff.TS_QUANTILE = orig_tsq              # type: ignore[attr-defined]

PANEL_PATH   = ROOT / "data" / "panel_data.parquet"
LABEL_COL    = "fwd_returns_24h"
N_LAYERS     = 5
DEFAULT_LOG  = ROOT / "log"
RESAMPLE_FREQ = "1D"   # 降采样到日频，避免小时复利爆炸


# ── Step 1: 从 log 提取因子 metrics + expression ─────────────────────────────

def _fix_json_numbers(s: str) -> str:
    s = s.replace(": Infinity",  ": 9e308")
    s = s.replace(":-Infinity", ":-9e308")
    s = s.replace(": NaN",      ": null")
    return s


def _extract_json_object(text: str, start: int) -> dict | None:
    """从 start 位置（必须是 '{'）向后数括号，提取完整 JSON 对象。"""
    depth = 0
    for i in range(start, min(start + 200_000, len(text))):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def collect_factor_metrics(log_text: str) -> dict[str, dict]:
    """遍历 log，收集每个因子的最高 sharpe 对应的 metrics。"""
    text = _fix_json_numbers(log_text)
    pattern = re.compile(r'"per_factor_metrics":\s*\{"')
    all_factors: dict[str, dict] = {}

    for m in pattern.finditer(text):
        brace_start = text.index("{", m.start() + len('"per_factor_metrics": '))
        obj = _extract_json_object(text, brace_start)
        if not obj:
            continue
        for fname, metrics in obj.items():
            if not isinstance(metrics, dict) or "sharpe" not in metrics:
                continue
            # 过滤掉 Infinity 替换后的异常值
            sharpe = float(metrics["sharpe"])
            if not math.isfinite(sharpe):
                continue
            if sharpe > all_factors.get(fname, {}).get("sharpe", -1e9):
                all_factors[fname] = metrics

    return all_factors


def collect_factor_expressions(log_text: str, factor_names: list[str]) -> dict[str, str]:
    """从 log 里为每个因子名找到 expression 字段。"""
    expressions: dict[str, str] = {}
    for fname in factor_names:
        start = 0
        while True:
            idx = log_text.find(fname, start)
            if idx < 0:
                break
            ctx = log_text[max(0, idx - 1500) : idx + 1500]
            m = re.search(r'"expression":\s*"([^"]{5,600})"', ctx)
            if m:
                expressions[fname] = m.group(1)
                break
            start = idx + 1
    return expressions


# ── Step 2: 计算因子值 + 分层收益 ────────────────────────────────────────────

def _assign_layers(factor_frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """向量化分层：pivot → rank → cut → melt，避免 groupby-apply 慢循环。"""
    sig = factor_frame["signal"]
    dt  = sig.index.get_level_values("datetime")
    sym = sig.index.get_level_values("symbol")

    # 横截面百分位 rank（0..1）
    pct_rank = sig.groupby(level="datetime").rank(pct=True, method="first")

    # 分位切割
    bins = np.linspace(0, 1, n + 1)
    bins[0] = -1e-9   # 包含最小值
    bins[-1] = 1 + 1e-9
    layer = np.searchsorted(bins, pct_rank.values, side="right").clip(1, n)

    factor_frame = factor_frame.copy()
    factor_frame["layer"] = layer
    return factor_frame


def _resample_to_daily(
    signal: pd.Series,
    label: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    小时 → 日频。信号取日内最后值；标签取日内均值。
    用 unstack/resample/stack 避免 groupby-apply。
    """
    sig_wide = signal.unstack("symbol")
    lbl_wide = label.unstack("symbol")

    sig_daily = sig_wide.resample("1D").last().dropna(how="all")
    lbl_daily = lbl_wide.resample("1D").mean().dropna(how="all")

    sig_out = sig_daily.stack(future_stack=True).rename(signal.name)
    lbl_out = lbl_daily.stack(future_stack=True).rename(label.name)
    return sig_out, lbl_out


def compute_group_returns(
    signal: pd.Series,
    label: pd.Series,
    n_layers: int = N_LAYERS,
    resample: bool = True,
) -> pd.DataFrame | None:
    """
    返回 DataFrame，index=datetime（日频），columns=layer 1..N + 'long_short'
    每列是当期等权均值收益（未累积）。
    """
    if resample:
        signal, label = _resample_to_daily(signal, label)

    frame = pd.concat(
        [signal.rename("signal"), label.rename("label")], axis=1
    ).dropna()
    if frame.empty:
        return None

    frame = _assign_layers(frame, n_layers)
    layer_rt = (
        frame.groupby([frame.index.get_level_values("datetime"), "layer"])["label"]
        .mean()
        .unstack("layer")
        .sort_index()
    )
    if layer_rt.empty or 1 not in layer_rt.columns or n_layers not in layer_rt.columns:
        return None

    layer_rt["long_short"] = layer_rt[n_layers] - layer_rt[1]

    # 清洗 Inf / 极端离群值
    layer_rt = layer_rt.replace([np.inf, -np.inf], np.nan)
    clip_q = 0.005
    for col in layer_rt.columns:
        lo = layer_rt[col].quantile(clip_q)
        hi = layer_rt[col].quantile(1 - clip_q)
        layer_rt[col] = layer_rt[col].clip(lo, hi)

    return layer_rt


def cumulative(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod() - 1.0


# ── Step 3: 绘图 ──────────────────────────────────────────────────────────────

LAYER_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
LS_COLOR     = "#17becf"


def plot_factor_group_returns(
    factor_name: str,
    metrics: dict,
    layer_returns: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    sharpe   = metrics.get("sharpe", float("nan"))
    rank_icir = metrics.get("Rank ICIR", metrics.get("Rank_ICIR", float("nan")))

    n = layer_returns.shape[1] - 1  # exclude long_short
    for i, col in enumerate(range(1, n + 1)):
        if col not in layer_returns.columns:
            continue
        cum = cumulative(layer_returns[col]) * 100
        ax.plot(cum.index, cum.values, color=LAYER_COLORS[i % len(LAYER_COLORS)],
                linewidth=1.2, label=f"G{col}")

    if "long_short" in layer_returns.columns:
        cum_ls = cumulative(layer_returns["long_short"]) * 100
        ax.plot(cum_ls.index, cum_ls.values, color=LS_COLOR,
                linewidth=1.8, linestyle="--", label="L-S")

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(
        f"{factor_name}\nSharpe={sharpe:.3f}  RankICIR={rank_icir:.3f}",
        fontsize=8, pad=4,
    )
    # 对数纵轴（当累积收益跨越多个量级时更易读）
    all_vals = []
    for col in layer_returns.columns:
        cum = cumulative(layer_returns[col]) * 100
        all_vals.extend(cum.dropna().tolist())
    if all_vals and max(all_vals) > 200:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left", ncol=3)


def build_figure(
    results: list[tuple[str, dict, pd.DataFrame]],
    out_path: Path,
) -> None:
    n = len(results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Factor Group Return Curves (5 quantiles)", fontsize=12, y=1.01)

    for idx, (fname, metrics, layer_rt) in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        plot_factor_group_returns(fname, metrics, layer_rt, ax)

    # hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def find_latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("evolution_debug_*.log"), reverse=True)
    return logs[0] if logs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制因子分组收益曲线")
    parser.add_argument("--log",        type=Path, default=None, help="evolution_debug log 路径")
    parser.add_argument("--panel",      type=Path, default=PANEL_PATH)
    parser.add_argument("--sharpe-min", type=float, default=0.2, help="最低 Sharpe 阈值（默认 0.2）")
    parser.add_argument("--top",        type=int,   default=None, help="只取 sharpe 最高的 N 个因子")
    parser.add_argument("--label-col",  type=str,   default=LABEL_COL)
    parser.add_argument("--n-layers",   type=int,   default=N_LAYERS)
    parser.add_argument("--date-from",  type=str,   default=None,
                        help="只使用此日期之后的数据，格式 YYYY-MM-DD（默认：最早日期）")
    parser.add_argument("--out",        type=Path,  default=ROOT / "results" / "group_returns.png")
    args = parser.parse_args()

    # ── 找 log ────────────────────────────────────────────────────────────────
    log_path = args.log or find_latest_log(DEFAULT_LOG)
    if log_path is None or not log_path.exists():
        sys.exit(f"[error] Log not found. Specify --log <path>")
    print(f"[info] Using log: {log_path}")

    with open(log_path, encoding="utf-8", errors="replace") as f:
        log_text = f.read()

    # ── 提取因子 metrics ───────────────────────────────────────────────────────
    all_metrics = collect_factor_metrics(log_text)
    print(f"[info] Collected metrics for {len(all_metrics)} factors")

    # 筛选 & 排序
    selected = sorted(
        [(name, m) for name, m in all_metrics.items()
         if math.isfinite(m.get("sharpe", float("nan"))) and m["sharpe"] >= args.sharpe_min],
        key=lambda x: x[1]["sharpe"],
        reverse=True,
    )
    if args.top:
        selected = selected[: args.top]

    if not selected:
        sys.exit(
            f"[error] No factors found with sharpe >= {args.sharpe_min}. "
            f"Max sharpe in log: {max((m.get('sharpe',0) for m in all_metrics.values()), default='N/A')}"
        )

    print(f"[info] Selected {len(selected)} factors to plot:")
    for name, m in selected:
        print(f"       sharpe={m['sharpe']:.4f}  RankICIR={m.get('Rank ICIR', 0):.4f}  {name}")

    # ── 提取表达式 ────────────────────────────────────────────────────────────
    factor_names = [n for n, _ in selected]
    expressions  = collect_factor_expressions(log_text, factor_names)
    print(f"[info] Found expressions for {len(expressions)}/{len(factor_names)} factors")

    missing_expr = [n for n in factor_names if n not in expressions]
    if missing_expr:
        print(f"[warn] No expression found for: {missing_expr}")

    # ── 加载 panel data ───────────────────────────────────────────────────────
    panel_path = args.panel
    if not panel_path.exists():
        sys.exit(f"[error] Panel data not found: {panel_path}")
    print(f"[info] Loading panel data from {panel_path} ...")
    panel = pd.read_parquet(panel_path)
    print(f"[info] Panel shape (raw): {panel.shape}")

    if args.date_from:
        cutoff = pd.Timestamp(args.date_from)
        panel = panel[panel.index.get_level_values("datetime") >= cutoff]
        print(f"[info] Panel after date filter ({args.date_from}): {panel.shape}")

    if panel.empty:
        sys.exit("[error] Panel is empty after date filter.")

    if args.label_col not in panel.columns:
        sys.exit(f"[error] Label column '{args.label_col}' not in panel. Available: {list(panel.columns)}")
    label = panel[args.label_col]

    # ── 计算分组收益 ──────────────────────────────────────────────────────────
    results: list[tuple[str, dict, pd.DataFrame]] = []
    for fname, metrics in selected:
        if fname not in expressions:
            print(f"[skip] {fname}: no expression")
            continue
        expr = expressions[fname]
        print(f"[calc] {fname} ...")
        signal = _patched_calculate_factor_expression(expr, panel, fname)
        if signal is None or signal.dropna().empty:
            print(f"[skip] {fname}: factor computation failed or all-NaN")
            continue
        layer_rt = compute_group_returns(signal, label, n_layers=args.n_layers)
        if layer_rt is None:
            print(f"[skip] {fname}: could not build layer returns")
            continue
        results.append((fname, metrics, layer_rt))
        print(f"       OK  dates={len(layer_rt)}  long_short_cum={cumulative(layer_rt['long_short']).iloc[-1]*100:.2f}%")

    if not results:
        sys.exit("[error] No factors could be computed. Check expressions and panel columns.")

    # ── 输出图像 ──────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(results, args.out)


if __name__ == "__main__":
    main()
