from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from trading_agents.infra.factor_validation import FactorValidationRequest, run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed manual factor validation backtest.")
    parser.add_argument("--panel-data-path", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--factor-name", default="manual_factor")
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--start-datetime", default=None)
    parser.add_argument("--end-datetime", default=None)
    parser.add_argument("--rolling-window", type=int, default=60)
    args = parser.parse_args()

    result = run_validation(
        FactorValidationRequest(
            panel_data_path=args.panel_data_path,
            expression=args.expression,
            factor_name=args.factor_name,
            target_column=args.target_column,
            start_datetime=args.start_datetime,
            end_datetime=args.end_datetime,
            rolling_window=args.rolling_window,
        )
    )
    metric_keys = [
        "IC",
        "ICIR",
        "Rank IC",
        "Rank ICIR",
        "sharpe",
        "drawdown",
        "coverage",
        "turnover",
        "annualized_return",
    ]
    print(json.dumps({
        "report_id": result["report_id"],
        "report_path": result["report_path"],
        "pdf_path": result.get("pdf_path"),
        "download_zip": str(Path(result["report_dir"]).with_suffix(".zip")),
        "metrics": {key: result["metrics"].get(key) for key in metric_keys if key in result["metrics"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
