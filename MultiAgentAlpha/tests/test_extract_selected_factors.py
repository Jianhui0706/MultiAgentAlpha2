from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_selected_factors import export_selected_factors


def test_export_selected_factors_writes_sharpe_filtered_results(tmp_path: Path) -> None:
    log_path = tmp_path / "sample.log"
    out_path = tmp_path / "selected_factors.json"
    log_path.write_text(
        "\n".join(
            [
                '{"payload": {"factor": {"factor_name": "alpha", "expression": "RANK($close)"}}}',
                '{"payload": {"backtest_report": {"per_factor_metrics": {"alpha": {"sharpe": 0.1, "Rank ICIR": 0.05}}}}}',
                '{"payload": {"factor": {"factor_name": "alpha", "expression": "RANK($close)"}}}',
                '{"payload": {"backtest_report": {"per_factor_metrics": {"alpha": {"sharpe": 0.3, "Rank ICIR": 0.15}}}}}',
                '{"payload": {"factor": {"factor_name": "beta", "expression": "RANK($open)"}}}',
                '{"payload": {"backtest_report": {"per_factor_metrics": {"beta": {"sharpe": 0.19, "Rank ICIR": 0.09}}}}}',
            ]
        ),
        encoding="utf-8",
    )

    export_selected_factors(log_path=log_path, out_path=out_path, sharpe_min=0.2)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["log_path"] == str(log_path)
    assert payload["sharpe_min"] == 0.2
    assert payload["count"] == 1
    assert payload["factors"] == [
        {
            "factor_name": "alpha",
            "expression": "RANK($close)",
            "metrics": {"Rank ICIR": 0.15, "sharpe": 0.3},
        }
    ]


def test_export_selected_factors_keeps_highest_sharpe_per_factor(tmp_path: Path) -> None:
    log_path = tmp_path / "sample.log"
    out_path = tmp_path / "selected_factors.json"
    log_path.write_text(
        "\n".join(
            [
                '{"payload": {"factor": {"factor_name": "gamma", "expression": "RANK($high)"}}}',
                '{"payload": {"backtest_report": {"per_factor_metrics": {"gamma": {"sharpe": 0.45, "coverage": 0.91}}}}}',
                '{"payload": {"backtest_report": {"per_factor_metrics": {"gamma": {"sharpe": 0.21, "coverage": 0.82}}}}}',
            ]
        ),
        encoding="utf-8",
    )

    export_selected_factors(log_path=log_path, out_path=out_path, sharpe_min=0.2)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["factors"][0]["factor_name"] == "gamma"
    assert payload["factors"][0]["expression"] == "RANK($high)"
    assert payload["factors"][0]["metrics"] == {"coverage": 0.91, "sharpe": 0.45}
