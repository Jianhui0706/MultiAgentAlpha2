from __future__ import annotations

from trading_agents.infra.holdout_validation import build_holdout_validation_payload


def test_holdout_validation_payload_summarizes_train_and_validation_sharpe() -> None:
    payload = build_holdout_validation_payload(
        candidates=[
            {
                "factor_id": "factor-a",
                "factor_name": "FactorA",
                "factor_expression": "RANK($close)",
                "train_metrics": {"sharpe": 2.5, "Rank IC": 0.12},
            }
        ],
        validation_metrics_by_factor={
            "FactorA": {"sharpe": 1.4, "Rank IC": 0.03},
        },
        metric="sharpe",
        threshold=2.0,
        train_window={"start": "2021-01-01", "end": "2025-06-01"},
        validation_window={"start": "2025-06-01", "end": None},
    )

    assert payload["status"] == "finished"
    assert payload["factor_count"] == 1
    assert payload["factors"] == [
        {
            "factor_id": "factor-a",
            "factor_name": "FactorA",
            "factor_expression": "RANK($close)",
            "train_sharpe": 2.5,
            "validation_sharpe": 1.4,
            "train_metrics": {"sharpe": 2.5, "Rank IC": 0.12},
            "validation_metrics": {"sharpe": 1.4, "Rank IC": 0.03},
            "passed_train_threshold": True,
            "passed_validation_threshold": False,
        }
    ]
    assert payload["validation_window"] == {"start": "2025-06-01", "end": None}
