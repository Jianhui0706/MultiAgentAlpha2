from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_group_returns import DEFAULT_LOG, collect_factor_expressions, collect_factor_metrics, find_latest_log


def build_selected_factors_payload(log_path: Path, sharpe_min: float) -> dict[str, object]:
    log_text = log_path.read_text(encoding="utf-8")
    all_metrics = collect_factor_metrics(log_text)
    selected_items = sorted(
        [
            (name, metrics)
            for name, metrics in all_metrics.items()
            if math.isfinite(metrics.get("sharpe", float("nan"))) and metrics["sharpe"] >= sharpe_min
        ],
        key=lambda item: item[1]["sharpe"],
        reverse=True,
    )
    expressions = collect_factor_expressions(log_text, [name for name, _ in selected_items])

    return {
        "log_path": str(log_path),
        "sharpe_min": sharpe_min,
        "count": len(selected_items),
        "factors": [
            {
                "factor_name": name,
                "expression": expressions.get(name),
                "metrics": metrics,
            }
            for name, metrics in selected_items
        ],
    }


def export_selected_factors(log_path: Path, out_path: Path, sharpe_min: float) -> dict[str, object]:
    payload = build_selected_factors_payload(log_path=log_path, sharpe_min=sharpe_min)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="从日志提取满足 Sharpe 阈值的因子表达式和指标")
    parser.add_argument("--log", type=Path, default=None, help="evolution_debug log 路径")
    parser.add_argument("--sharpe-min", type=float, default=0.2, help="最低 Sharpe 阈值（默认 0.2）")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "selected_factors.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    log_path = args.log or find_latest_log(DEFAULT_LOG)
    if log_path is None or not log_path.exists():
        raise SystemExit("[error] No log file found.")

    payload = export_selected_factors(log_path=log_path, out_path=args.out, sharpe_min=args.sharpe_min)
    print(f"[extract] Saved {payload['count']} factors to {args.out}")


if __name__ == "__main__":
    main()
