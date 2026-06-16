from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.infra.factor_pool_manager import FactorPoolConfig, FactorPoolManager
import trading_agents.infra.factor_pool_manager as factor_pool_module


def test_factor_pool_admits_by_rank_ic_and_rejects_correlated_signal(tmp_path, monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    symbols = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    panel = pd.DataFrame({"fwd_returns_4h": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2]}, index=index)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path)
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "metadata": {"created_at": "now", "last_updated": "now", "total_factors": 3, "version": "1.0"},
                "factors": {
                    "f1": {
                        "factor_id": "f1",
                        "factor_name": "FactorOne",
                        "factor_expression": "expr_one",
                        "backtest_results": {"Rank IC": 0.04},
                        "train_filter_decision": {"passed": True},
                    },
                    "f2": {
                        "factor_id": "f2",
                        "factor_name": "FactorTwo",
                        "factor_expression": "expr_two",
                        "backtest_results": {"Rank IC": 0.03},
                        "train_filter_decision": {"passed": True},
                    },
                    "f3": {
                        "factor_id": "f3",
                        "factor_name": "FactorThree",
                        "factor_expression": "expr_three",
                        "backtest_results": {"Rank IC": 0.02},
                        "train_filter_decision": {"passed": True},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_calculate(expression: str, _panel: pd.DataFrame, _factor_name: str):
        if expression == "expr_three":
            return pd.Series([1, -1, -1, 1, 1, -1, -1, 1], index=index)
        return pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=index)

    monkeypatch.setattr(factor_pool_module, "calculate_factor_expression", fake_calculate)
    payload = FactorPoolManager(
        library_path=library_path,
        panel_data_path=panel_path,
        output_dir=tmp_path,
        config=FactorPoolConfig(corr_threshold=0.7, max_fraction=1.0),
    ).build()

    assert payload["status"] == "finished"
    assert [item["factor_id"] for item in payload["selected_factors"]] == ["f1", "f3"]
    assert payload["rejected_factors"][0]["factor_id"] == "f2"
    assert payload["rejected_factors"][0]["rejection_reason"] == "correlation_threshold"
    assert (tmp_path / "factor_pool.json").exists()
    assert (tmp_path / "combined_factors_df.parquet").exists()
