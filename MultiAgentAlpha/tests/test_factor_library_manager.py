from __future__ import annotations

import json
from pathlib import Path

from trading_agents.infra.factor_library_manager import FactorLibraryManager
from trading_agents.schemas import ExperimentFactorTask, QlibFactorExperiment


def test_compute_reward_breakdown_matches_reward_score() -> None:
    metrics = {
        "sharpe": 1.5,
        "IC": 0.03,
        "Rank IC": -0.02,
        "ICIR": 0.06,
        "turnover": 0.2,
        "drawdown": 0.1,
    }

    score = FactorLibraryManager.compute_reward_score(metrics, {"factor_expression": "RANK($close)"})
    breakdown = FactorLibraryManager.compute_reward_breakdown(metrics, {"factor_expression": "RANK($close)"})

    assert isinstance(breakdown["terms"], list)
    assert {term["name"] for term in breakdown["terms"]} >= {"sharpe_progress", "ic_progress", "icir_progress"}
    assert breakdown["score"] == sum(term["contribution"] for term in breakdown["terms"])
    assert abs(breakdown["score"] - score) < 1e-12


def test_factor_library_writes_per_factor_metrics_and_combined_metrics() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_001",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
            ExperimentFactorTask(factor_name="FactorB", expression="RANK($volume)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
                {"factor_name": "FactorB", "expression": "RANK($volume)", "code": "code_b"},
            ]
        },
        metrics={"IC": 0.3, "Rank IC": 0.4},
        per_factor_metrics={
            "FactorA": {"IC": 0.1, "Rank IC": 0.2},
            "FactorB": {"IC": -0.1, "Rank IC": -0.2},
        },
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        data = json.loads(library_path.read_text(encoding="utf-8"))
        factors = list(data["factors"].values())
        by_name = {item["factor_name"]: item for item in factors}

        assert by_name["FactorA"]["backtest_results"] == {"IC": 0.1, "Rank IC": 0.2}
        assert by_name["FactorB"]["backtest_results"] == {"IC": -0.1, "Rank IC": -0.2}
        assert by_name["FactorA"]["combined_backtest_results"] == {"IC": 0.3, "Rank IC": 0.4}
        assert by_name["FactorB"]["combined_backtest_results"] == {"IC": 0.3, "Rank IC": 0.4}
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_skips_validation_decayed_factors_and_stores_both_windows() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_decay_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_decay",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="KeepFactor", expression="RANK($close)"),
            ExperimentFactorTask(factor_name="DecayFactor", expression="RANK($volume)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "KeepFactor", "expression": "RANK($close)", "code": "code_a"},
                {"factor_name": "DecayFactor", "expression": "RANK($volume)", "code": "code_b"},
            ]
        },
        backtest_report={
            "validation_decay_filter": {
                "enabled": True,
                "metric": "reward_score",
                "min_retention": 0.5,
                "filtered_factor_names": ["DecayFactor"],
                "decisions": {
                    "KeepFactor": {"passed": True, "retention_ratio": 0.9, "train_reward": 0.2, "validation_reward": 0.18},
                    "DecayFactor": {"passed": False, "retention_ratio": 0.1, "train_reward": 0.2, "validation_reward": 0.02},
                },
            },
            "per_factor_validation_metrics": {
                "KeepFactor": {"Rank IC": 0.18, "sharpe": 2.2},
                "DecayFactor": {"Rank IC": 0.02, "sharpe": 0.3},
            },
        },
        per_factor_metrics={
            "KeepFactor": {"Rank IC": 0.2, "sharpe": 2.5},
            "DecayFactor": {"Rank IC": 0.2, "sharpe": 2.5},
        },
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        data = json.loads(library_path.read_text(encoding="utf-8"))
        by_name = {item["factor_name"]: item for item in data["factors"].values()}

        # Decayed factor is not admitted; the retained factor keeps both windows.
        assert set(by_name) == {"KeepFactor"}
        keep = by_name["KeepFactor"]
        assert keep["backtest_results"] == {"Rank IC": 0.2, "sharpe": 2.5}
        assert keep["validation_results"] == {"Rank IC": 0.18, "sharpe": 2.2}
        assert keep["validation_decay"]["passed"] is True
        # Combined reward sits between train and validation reward (0.75/0.25 blend).
        assert keep["validation_reward_score"] is not None
        assert min(keep["train_reward_score"], keep["validation_reward_score"]) <= keep["combined_reward_score"] <= max(
            keep["train_reward_score"], keep["validation_reward_score"]
        )
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_stamps_backtest_period_on_saved_factors() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_period_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_period",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
            ]
        },
        backtest_report={
            "backtest_period": {
                "start": "2020-01-01T00:00:00",
                "end": "2023-12-31T23:00:00",
            }
        },
        per_factor_metrics={"FactorA": {"IC": 0.1, "Rank IC": 0.2}},
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        data = json.loads(library_path.read_text(encoding="utf-8"))
        factor = list(data["factors"].values())[0]

        expected = {"start": "2020-01-01T00:00:00", "end": "2023-12-31T23:00:00"}
        assert factor["backtest_period"] == expected
        assert factor["metadata"]["backtest_period"] == expected
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_backtest_period_falls_back_to_sample_windows() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_period_fallback_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_period_fallback",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
            ]
        },
        backtest_report={
            "sample_windows": {
                "train": {"start": "2020-01-01T00:00:00", "end": "2022-06-30T00:00:00"},
                "validation": {"start": "2022-06-30T00:00:00", "end": "2023-12-31T23:00:00"},
            }
        },
        per_factor_metrics={"FactorA": {"IC": 0.1, "Rank IC": 0.2}},
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        data = json.loads(library_path.read_text(encoding="utf-8"))
        factor = list(data["factors"].values())[0]

        assert factor["backtest_period"] == {
            "start": "2020-01-01T00:00:00",
            "end": "2023-12-31T23:00:00",
        }
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_keeps_only_train_sharpe_candidates_for_holdout() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_train_filter_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_002",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
            ExperimentFactorTask(factor_name="FactorB", expression="RANK($volume)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
                {"factor_name": "FactorB", "expression": "RANK($volume)", "code": "code_b"},
            ]
        },
        backtest_report={
            "train_filter": {
                "enabled": True,
                "retained_factor_names": ["FactorA"],
                "filtered_factor_names": ["FactorB"],
            }
        },
        metrics={"IC": 0.3, "Rank IC": 0.4, "sharpe": 2.5},
        per_factor_metrics={
            "FactorA": {"IC": 0.1, "Rank IC": 0.2, "sharpe": 2.5},
            "FactorB": {"IC": 0.2, "Rank IC": 0.3, "sharpe": 1.5},
        },
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        data = json.loads(library_path.read_text(encoding="utf-8"))
        factors = list(data["factors"].values())

        assert [item["factor_name"] for item in factors] == ["FactorA"]
        assert factors[0]["backtest_results"] == {"IC": 0.1, "Rank IC": 0.2, "sharpe": 2.5}
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_does_not_create_empty_file_when_all_factors_filtered() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_empty_filter_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_empty",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
            ]
        },
        backtest_report={
            "train_filter": {
                "enabled": True,
                "retained_factor_names": [],
                "filtered_factor_names": ["FactorA"],
            }
        },
        per_factor_metrics={
            "FactorA": {"IC": 0.1, "Rank IC": 0.2, "sharpe": 1.0},
        },
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        assert not library_path.exists()
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_skips_coverage_filtered_factors() -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_coverage_filter_test.json")
    if library_path.exists():
        library_path.unlink()
    experiment = QlibFactorExperiment(
        experiment_id="exp_coverage",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
            ExperimentFactorTask(factor_name="FactorB", expression="RANK($volume)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
                {"factor_name": "FactorB", "expression": "RANK($volume)", "code": "code_b"},
            ]
        },
        backtest_report={
            "coverage_filter": {
                "enabled": True,
                "filtered_factor_names": ["FactorB"],
                "decisions": {
                    "FactorA": {"passed": True, "value": 0.95, "threshold": 0.9},
                    "FactorB": {"passed": False, "value": 0.85, "threshold": 0.9},
                },
            }
        },
        per_factor_metrics={
            "FactorA": {"IC": 0.1, "Rank IC": 0.2, "coverage": 0.95},
            "FactorB": {"IC": 0.2, "Rank IC": 0.3, "coverage": 0.85},
        },
    )

    try:
        FactorLibraryManager(library_path).add_experiment(experiment, panel_data_path="data/panel.parquet")

        data = json.loads(library_path.read_text(encoding="utf-8"))
        factors = list(data["factors"].values())

        assert [item["factor_name"] for item in factors] == ["FactorA"]
        assert factors[0]["coverage_filter_decision"]["passed"] is True
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_lists_train_sharpe_candidates_for_holdout(monkeypatch) -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_candidates_test.json")
    if library_path.exists():
        library_path.unlink()
    monkeypatch.setenv("FACTOR_LIBRARY_RUN_ID", "run-current")
    experiment = QlibFactorExperiment(
        experiment_id="exp_003",
        target_hypothesis="test hypothesis",
        factors=[
            ExperimentFactorTask(factor_name="FactorA", expression="RANK($close)"),
            ExperimentFactorTask(factor_name="FactorB", expression="RANK($volume)"),
        ],
        factor_implementation={
            "implementations": [
                {"factor_name": "FactorA", "expression": "RANK($close)", "code": "code_a"},
                {"factor_name": "FactorB", "expression": "RANK($volume)", "code": "code_b"},
            ]
        },
        per_factor_metrics={
            "FactorA": {"sharpe": 2.5, "Rank IC": 0.2},
            "FactorB": {"sharpe": 1.5, "Rank IC": 0.3},
        },
    )

    try:
        manager = FactorLibraryManager(library_path)
        manager.add_experiment(experiment, panel_data_path="data/panel.parquet")

        candidates = FactorLibraryManager(library_path).list_train_metric_candidates(metric="sharpe", threshold=2.0)

        assert [item["factor_name"] for item in candidates] == ["FactorA"]
        assert candidates[0]["train_metrics"]["sharpe"] == 2.5
        assert candidates[0]["factor_expression"] == "RANK($close)"
        assert candidates[0]["run_id"] == "run-current"
        assert FactorLibraryManager(library_path).list_train_metric_candidates(
            metric="sharpe",
            threshold=2.0,
            run_ids={"run-current"},
        )[0]["factor_name"] == "FactorA"
        assert FactorLibraryManager(library_path).list_train_metric_candidates(
            metric="sharpe",
            threshold=2.0,
            run_ids={"other-run"},
        ) == []
        assert FactorLibraryManager(library_path).list_train_metric_candidates(
            metric="sharpe",
            threshold=2.0,
            run_ids=set(),
        ) == []
        assert FactorLibraryManager(library_path).list_train_metric_candidates(
            metric="sharpe",
            threshold=2.0,
            experiment_ids=set(),
        ) == []
    finally:
        if library_path.exists():
            library_path.unlink()


def test_factor_library_run_filter_prevents_historical_experiment_id_collision(monkeypatch) -> None:
    library_path = Path(__file__).with_name("__tmp_factor_library_run_filter_test.json")
    if library_path.exists():
        library_path.unlink()
    historical = QlibFactorExperiment(
        experiment_id="repeatable_exp_id",
        target_hypothesis="historical",
        factors=[ExperimentFactorTask(factor_name="HistoricalFactor", expression="RANK($close)")],
        factor_implementation={
            "implementations": [
                {"factor_name": "HistoricalFactor", "expression": "RANK($close)", "code": "code_old"},
            ]
        },
        per_factor_metrics={"HistoricalFactor": {"sharpe": 3.0}},
    )
    current = QlibFactorExperiment(
        experiment_id="repeatable_exp_id",
        target_hypothesis="current",
        factors=[ExperimentFactorTask(factor_name="CurrentFactor", expression="RANK($volume)")],
        factor_implementation={
            "implementations": [
                {"factor_name": "CurrentFactor", "expression": "RANK($volume)", "code": "code_new"},
            ]
        },
        per_factor_metrics={"CurrentFactor": {"sharpe": 2.5}},
    )

    try:
        manager = FactorLibraryManager(library_path)
        monkeypatch.setenv("FACTOR_LIBRARY_RUN_ID", "old-run")
        manager.add_experiment(historical, panel_data_path="data/panel.parquet")
        monkeypatch.setenv("FACTOR_LIBRARY_RUN_ID", "current-run")
        FactorLibraryManager(library_path).add_experiment(current, panel_data_path="data/panel.parquet")

        by_experiment = FactorLibraryManager(library_path).list_train_metric_candidates(
            metric="sharpe",
            threshold=2.0,
            experiment_ids={"repeatable_exp_id"},
        )
        current_only = FactorLibraryManager(library_path).list_train_metric_candidates(
            metric="sharpe",
            threshold=2.0,
            experiment_ids={"repeatable_exp_id"},
            run_ids={"current-run"},
        )

        assert {item["factor_name"] for item in by_experiment} == {"HistoricalFactor", "CurrentFactor"}
        assert [item["factor_name"] for item in current_only] == ["CurrentFactor"]
    finally:
        if library_path.exists():
            library_path.unlink()
