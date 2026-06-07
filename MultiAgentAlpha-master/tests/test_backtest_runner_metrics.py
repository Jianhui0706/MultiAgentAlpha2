from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from trading_agents.agents.backtest_runner_agent import QlibFactorBacktestEngine


def _engine() -> QlibFactorBacktestEngine:
    return QlibFactorBacktestEngine(
        {"metrics": {"corr_method": "spearman"}},
        Path("src/trading_agents/configs/qlib/backtest_engine.yaml"),
    )


def _sample_window_engine() -> QlibFactorBacktestEngine:
    return QlibFactorBacktestEngine(
        {
            "metrics": {"corr_method": "spearman"},
            "sample_windows": {
                "enabled": True,
                "train": {"start": "2021-01-01", "end": "2025-06-01"},
                "validation": {"start": "2025-06-01", "end": None},
                "train_filter": {"metric": "sharpe", "min_value": 2.0},
            },
        },
        Path("src/trading_agents/configs/qlib/backtest_engine.yaml"),
    )


def _ratio_sample_window_engine() -> QlibFactorBacktestEngine:
    return QlibFactorBacktestEngine(
        {
            "metrics": {"corr_method": "spearman"},
            "sample_windows": {
                "enabled": True,
                "split_mode": "chronological_ratio",
                "train_ratio": 0.75,
                "train_filter": {"metric": "sharpe", "min_value": 2.0},
            },
        },
        Path("src/trading_agents/configs/qlib/backtest_engine.yaml"),
    )


def _robustness_engine() -> QlibFactorBacktestEngine:
    return QlibFactorBacktestEngine(
        {
            "metrics": {"corr_method": "spearman", "annualization_factor": 12},
            "robustness_filter": {
                "enabled": True,
                "market_beta_abs_max": 0.20,
                "residual_sharpe_min": 0.50,
                "yearly_positive_ratio_min": 0.50,
                "max_year_positive_contribution_share": 0.80,
                "regime_positive_sharpe_count_min": 2,
            },
        },
        Path("src/trading_agents/configs/qlib/backtest_engine.yaml"),
    )


def test_sample_window_mask_uses_inclusive_start_and_exclusive_end() -> None:
    index = pd.MultiIndex.from_product(
        [
            [
                "2020-12-31 23:00:00",
                "2021-01-01 00:00:00",
                "2025-05-31 23:00:00",
                "2025-06-01 00:00:00",
                "2026-01-01 00:00:00",
            ],
            ["BTC"],
        ],
        names=["datetime", "symbol"],
    )
    panel = pd.DataFrame({"fwd_returns_4h": range(len(index))}, index=index)
    engine = _sample_window_engine()

    train_mask = engine._build_sample_window_mask(panel, engine.train_window)
    validation_mask = engine._build_sample_window_mask(panel, engine.validation_window)

    train_dates = pd.to_datetime(panel.index[train_mask].get_level_values("datetime")).tolist()
    validation_dates = pd.to_datetime(panel.index[validation_mask].get_level_values("datetime")).tolist()

    assert train_dates == [
        pd.Timestamp("2021-01-01 00:00:00"),
        pd.Timestamp("2025-05-31 23:00:00"),
    ]
    assert validation_dates == [
        pd.Timestamp("2025-06-01 00:00:00"),
        pd.Timestamp("2026-01-01 00:00:00"),
    ]


def test_ratio_sample_windows_use_first_75_percent_for_train() -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.date_range("2026-01-01", periods=8, freq="D"),
            ["BTC", "ETH"],
        ],
        names=["datetime", "symbol"],
    )
    panel = pd.DataFrame({"fwd_returns_4h": range(len(index))}, index=index)
    engine = _ratio_sample_window_engine()

    engine._ensure_sample_windows_resolved(panel)
    train_mask = engine._build_sample_window_mask(panel, engine.train_window)
    validation_mask = engine._build_sample_window_mask(panel, engine.validation_window)

    train_dates = pd.Index(pd.to_datetime(panel.index[train_mask].get_level_values("datetime"))).unique()
    validation_dates = pd.Index(pd.to_datetime(panel.index[validation_mask].get_level_values("datetime"))).unique()

    assert train_dates.tolist() == list(pd.date_range("2026-01-01", periods=6, freq="D"))
    assert validation_dates.tolist() == list(pd.date_range("2026-01-07", periods=2, freq="D"))
    assert engine._format_sample_window(engine.train_window) == {
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-07T00:00:00",
    }
    assert engine._format_sample_window(engine.validation_window) == {
        "start": "2026-01-07T00:00:00",
        "end": None,
    }
    assert engine._sample_window_resolution["mode"] == "chronological_ratio"
    assert engine._sample_window_resolution["train_datetime_count"] == 6
    assert engine._sample_window_resolution["validation_datetime_count"] == 2


def test_train_filter_uses_only_train_metrics_during_mining() -> None:
    engine = _sample_window_engine()

    passed = engine._build_train_filter_decision({"sharpe": 2.01})
    failed_at_threshold = engine._build_train_filter_decision({"sharpe": 2.0})

    assert passed["passed"] is True
    assert passed["window"] == "train"
    assert passed["threshold"] == 2.0
    assert passed["comparison"] == ">"
    assert failed_at_threshold["passed"] is False


def test_train_filter_threshold_can_be_overridden_by_env(monkeypatch) -> None:
    monkeypatch.setenv("TRAIN_SHARPE_THRESHOLD", "0.5")
    engine = _sample_window_engine()

    passed = engine._build_train_filter_decision({"sharpe": 0.51})
    failed_at_threshold = engine._build_train_filter_decision({"sharpe": 0.5})

    assert engine.train_filter_min_value == 0.5
    assert passed["threshold"] == 0.5
    assert passed["passed"] is True
    assert failed_at_threshold["passed"] is False


def test_constant_cross_section_is_skipped_without_warning() -> None:
    index = pd.MultiIndex.from_product(
        [["2026-01-01", "2026-01-02"], ["BTC", "ETH", "SOL"]],
        names=["datetime", "symbol"],
    )
    panel = pd.DataFrame({"fwd_returns_24h": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01]}, index=index)
    signal = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 3.0], index=index)
    label = panel["fwd_returns_24h"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = _engine()._calculate_single_factor_metrics(signal, label, panel)

    assert not [item for item in caught if item.category.__name__ == "ConstantInputWarning"]
    assert metrics["coverage"] == 1.0
    assert metrics["IC"] == -1.0
    assert metrics["Rank IC"] == -1.0


def test_all_constant_cross_sections_return_zero_ic() -> None:
    index = pd.MultiIndex.from_product(
        [["2026-01-01", "2026-01-02"], ["BTC", "ETH"]],
        names=["datetime", "symbol"],
    )
    panel = pd.DataFrame({"fwd_returns_24h": [0.01, 0.02, 0.03, 0.04]}, index=index)
    signal = pd.Series([1.0, 1.0, 2.0, 2.0], index=index)
    label = panel["fwd_returns_24h"]

    metrics = _engine()._calculate_single_factor_metrics(signal, label, panel)

    assert metrics["IC"] == 0.0
    assert metrics["ICIR"] == 0.0
    assert metrics["Rank IC"] == 0.0
    assert metrics["Rank ICIR"] == 0.0


def test_market_beta_robustness_filter_rejects_market_driven_returns() -> None:
    engine = _robustness_engine()
    index = pd.date_range("2021-01-01", periods=12, freq="M")
    benchmark = pd.Series([0.01, 0.02, 0.03, -0.01, 0.02, 0.03, 0.04, -0.02, 0.03, 0.04, 0.05, -0.01], index=index)
    portfolio = benchmark * 0.8

    metrics = engine._calculate_market_neutral_robustness(
        oriented_net_returns=portfolio,
        benchmark_returns=benchmark,
    )
    decision = engine._build_robustness_filter_decision(metrics)

    assert metrics["market_beta"] > 0.7
    assert decision["passed"] is False
    assert any("market_beta" in reason for reason in decision["reasons"])


def test_market_neutral_robustness_filter_accepts_residual_signal() -> None:
    engine = _robustness_engine()
    index = pd.date_range("2021-01-01", periods=12, freq="M")
    benchmark = pd.Series([0.03, -0.02, 0.02, -0.01, 0.04, -0.03, 0.02, -0.02, 0.01, -0.01, 0.03, -0.02], index=index)
    portfolio = pd.Series([0.030, 0.020, 0.025, 0.018, 0.028, 0.022, 0.030, 0.024, 0.026, 0.020, 0.029, 0.023], index=index)

    metrics = engine._calculate_market_neutral_robustness(
        oriented_net_returns=portfolio,
        benchmark_returns=benchmark,
    )
    decision = engine._build_robustness_filter_decision(metrics)

    assert abs(metrics["market_beta"]) <= 0.20
    assert metrics["residual_sharpe"] >= 0.50
    assert decision["passed"] is True
