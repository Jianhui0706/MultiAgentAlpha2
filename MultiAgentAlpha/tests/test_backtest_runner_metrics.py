from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from trading_agents.agents.backtest_runner_agent import BacktestRunnerAgent, QlibFactorBacktestEngine
from trading_agents.schemas import AgentContext, SharedContext


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


def test_qlib_provider_uri_relative_path_resolves_under_project_root() -> None:
    engine = _engine()

    provider_uri = engine._resolve_qlib_provider_uri("data/qlib/crypto_data")

    assert Path(provider_uri) == Path.cwd().resolve() / "data" / "qlib" / "crypto_data"


def test_qlib_provider_uri_env_override_wins(monkeypatch) -> None:
    engine = _engine()
    override = Path.cwd().resolve() / "custom_qlib_data"
    monkeypatch.setenv("QLIB_PROVIDER_URI", str(override))

    provider_uri = engine._resolve_qlib_provider_uri("data/qlib/crypto_data")

    assert Path(provider_uri) == override


def test_backtest_runner_failed_report_uses_engine_backtest_period() -> None:
    agent = BacktestRunnerAgent("backtest_runner_agent")
    result = agent.run(
        private_context=AgentContext(),
        shared_context=SharedContext(
            payload={
                "factor_implementation": {"source": "test"},
                "calculation_report": {"total_factors": 1, "generated_factors": 1, "failed_factors": []},
                "domain_dataset_summary": {"path": ""},
            }
        ),
        model_client=None,  # type: ignore[arg-type]
    )

    report = result.shared_updates["backtest_report"]
    assert report["status"] == "failed"
    assert report["backtest_period"] == {"start": None, "end": None}
    assert report["metrics_semantics"] == "invalid_due_to_backtest_failure"


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


def test_coverage_filter_rejects_low_coverage_before_retention() -> None:
    engine = _engine()

    passed = engine._build_coverage_filter_decision({"coverage": 0.9})
    failed = engine._build_coverage_filter_decision({"coverage": 0.8999})

    assert passed["passed"] is True
    assert passed["comparison"] == ">="
    assert passed["threshold"] == 0.9
    assert failed["passed"] is False


def test_coverage_filter_threshold_can_be_overridden_by_env(monkeypatch) -> None:
    monkeypatch.setenv("TRAIN_COVERAGE_THRESHOLD", "0.8")
    engine = _engine()

    passed = engine._build_coverage_filter_decision({"coverage": 0.8})
    failed = engine._build_coverage_filter_decision({"coverage": 0.7999})

    assert engine.coverage_filter_min_value == 0.8
    assert passed["threshold"] == 0.8
    assert passed["passed"] is True
    assert failed["passed"] is False


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


def test_ic_is_pearson_and_rank_ic_is_spearman_style() -> None:
    index = pd.MultiIndex.from_product(
        [["2026-01-01"], ["A", "B", "C", "D", "E"]],
        names=["datetime", "symbol"],
    )
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    label = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0], index=index)
    panel = pd.DataFrame({"fwd_returns_4h": label}, index=index)

    metrics = _engine()._calculate_single_factor_metrics(signal, label, panel)

    assert round(metrics["IC"], 6) == round(signal.corr(label), 6)
    assert metrics["Rank IC"] == 1.0
    assert metrics["IC"] != metrics["Rank IC"]


def test_matrix_ic_and_icir_match_cross_sectional_definitions() -> None:
    engine = _engine()
    times = pd.date_range("2026-01-01", periods=3, freq="D")
    symbols = ["A", "B", "C", "D", "E"]
    index = pd.MultiIndex.from_product([times, symbols], names=["datetime", "symbol"])
    signal_values = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ])
    label_values = np.array([
        [1.0, 4.0, 9.0, 16.0, 25.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 5.0, 4.0, 8.0],
    ])
    signal = pd.Series(signal_values.ravel(), index=index)
    label = pd.Series(label_values.ravel(), index=index)

    metrics = engine._calculate_factor_metrics_bundle_matrix(
        factor_name="ic_defs",
        expression="x",
        signal_series=signal,
        label_series=label,
        panel=pd.DataFrame({"fwd_returns_24h": label}, index=index),
    )

    expected_ic = []
    expected_rank_ic = []
    for signal_row, label_row in zip(signal_values, label_values):
        s = pd.Series(signal_row)
        y = pd.Series(label_row)
        expected_ic.append(float(s.corr(y, method="pearson")))
        expected_rank_ic.append(float(s.rank().corr(y.rank(), method="pearson")))
    expected_ic = pd.Series(expected_ic)
    expected_rank_ic = pd.Series(expected_rank_ic)

    assert math.isclose(metrics["IC"], float(expected_ic.mean()), rel_tol=1e-6)
    assert math.isclose(metrics["ICIR"], float(expected_ic.mean() / expected_ic.std()), rel_tol=1e-6)
    assert math.isclose(metrics["Rank IC"], float(expected_rank_ic.mean()), rel_tol=1e-6)
    assert math.isclose(metrics["Rank ICIR"], float(expected_rank_ic.mean() / expected_rank_ic.std()), rel_tol=1e-6)


def test_ic_subsampling_uses_actual_timestamps_for_irregular_panels() -> None:
    engine = _engine()
    engine.target_column = "fwd_returns_4h"
    times = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-01-01 00:30:00",
        "2026-01-01 01:00:00",
        "2026-01-01 01:30:00",
        "2026-01-01 02:00:00",
        "2026-01-01 06:00:00",
    ])
    index = pd.MultiIndex.from_product([times, ["BTC", "ETH", "SOL"]], names=["datetime", "symbol"])
    panel = pd.DataFrame({"fwd_returns_4h": 0.0}, index=index)

    engine._configure_sharpe_sampling(panel)
    ic_series = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=times)
    sampled = engine._subsample_ic_to_horizon(ic_series)

    assert engine._current_sharpe_sampling_step == 8
    assert sampled.index.tolist() == [times[0], times[-1]]
    assert sampled.tolist() == [0.1, 0.6]


def test_matrix_sharpe_uses_sample_standard_deviation() -> None:
    engine = QlibFactorBacktestEngine(
        {"metrics": {"annualization_factor": 1, "transaction_cost": {"one_way_fee_rate": 0.0}}},
        Path("src/trading_agents/configs/qlib/backtest_engine.yaml"),
    )
    engine._current_sharpe_annualization_factor = 1.0
    engine._current_sharpe_sampling_step = 1
    engine._current_target_horizon = None
    times = pd.date_range("2026-01-01", periods=3, freq="D")
    symbols = ["S1", "S2", "S3", "S4", "S5"]
    index = pd.MultiIndex.from_product([times, symbols], names=["datetime", "symbol"])
    desired_ls = np.array([0.01, 0.02, 0.04], dtype=float)
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * len(times), index=index)
    labels = []
    for value in desired_ls:
        labels.extend([0.0, 0.0, 0.0, 0.0, float(value)])
    label = pd.Series(labels, index=index)

    metrics = engine._calculate_factor_metrics_bundle_matrix(
        factor_name="sample_std",
        expression="x",
        signal_series=signal,
        label_series=label,
        panel=pd.DataFrame({"fwd_returns_24h": label}, index=index),
    )

    expected = float(desired_ls.mean() / desired_ls.std(ddof=1))
    population_sharpe = float(desired_ls.mean() / desired_ls.std(ddof=0))
    assert math.isclose(metrics["sharpe"], expected, rel_tol=1e-6)
    assert not math.isclose(metrics["sharpe"], population_sharpe, rel_tol=1e-6)


def test_matrix_return_sampling_uses_actual_timestamps_for_irregular_panels() -> None:
    engine = _engine()
    engine.target_column = "fwd_returns_4h"
    times = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-01-01 00:30:00",
        "2026-01-01 01:00:00",
        "2026-01-01 01:30:00",
        "2026-01-01 02:00:00",
        "2026-01-01 06:00:00",
    ])
    symbols = ["S1", "S2", "S3", "S4", "S5"]
    index = pd.MultiIndex.from_product([times, symbols], names=["datetime", "symbol"])
    expected_ls_by_time = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=float)
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * len(times), index=index)
    labels = []
    for value in expected_ls_by_time:
        labels.extend([0.0, 0.0, 0.0, 0.0, float(value)])
    label = pd.Series(labels, index=index)
    panel = pd.DataFrame({"fwd_returns_4h": label}, index=index)

    engine._configure_sharpe_sampling(panel)
    metrics = engine._calculate_factor_metrics_bundle_matrix(
        factor_name="irregular_sampling",
        expression="x",
        signal_series=signal,
        label_series=label,
        panel=panel,
    )

    assert engine._current_sharpe_sampling_step == 8
    assert math.isclose(metrics["long_short_mean_return"], float(np.mean([0.01, 0.06])), rel_tol=1e-6)


def test_thin_cross_section_excluded_from_layers_but_keeps_ic() -> None:
    """可用标的数 < n_layers(5) 的薄横截面：分层/多空剔除（=0），IC 仍计算。

    旧逻辑会把薄横截面全塞 L1，产生退化的非零分层值污染指标；新逻辑剔除这些
    时间点（layer_mat 置 NaN），分层/多空干净归零，而 IC（≥2 个标的即有定义）不受影响。
    """
    times = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]

    def _bundle(n_inst: int) -> dict:
        syms = [f"S{i}" for i in range(n_inst)]
        index = pd.MultiIndex.from_product([times, syms], names=["datetime", "symbol"])
        sig = [float(i) for _ in times for i in range(n_inst)]
        lab = [float(i) * 0.1 for _ in times for i in range(n_inst)]  # 与 signal 单调同向
        return _engine()._calculate_factor_metrics_bundle_matrix(
            factor_name="f",
            expression="x",
            signal_series=pd.Series(sig, index=index),
            label_series=pd.Series(lab, index=index),
            panel=pd.DataFrame({"label": lab}, index=index),
        )

    thick = _bundle(8)   # 8 个标的 → 足够分 5 层
    assert thick["long_short_mean_return"] > 0
    assert thick["layer_5_mean_return"] > thick["layer_1_mean_return"]

    thin = _bundle(3)    # 3 个标的 → 不足 5 层
    assert thin["long_short_mean_return"] == 0.0
    assert thin.get("layer_1_mean_return", 0.0) == 0.0
    assert thin.get("layer_5_mean_return", 0.0) == 0.0
    assert thin["IC"] > 0.5  # IC 不被薄横截面剔除


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
