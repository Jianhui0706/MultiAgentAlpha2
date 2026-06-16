from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

from trading_agents.infra.run_store import RunStore
from trading_agents.schemas import RoundPhase, StrategyTrajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "web" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import runs as runs_api  # noqa: E402


def _write_factor_store(db_path: Path, *, run_id: str, factor_id: str, sharpe: float) -> None:
    store = RunStore(db_path)
    store.start_run(run_id=run_id, log_dir=str(db_path.parent / run_id), args={"run_name": run_id})
    gate_sharpe = max(float(sharpe), 1.1)
    store.upsert_factor(
        run_id=run_id,
        loop_index=1,
        trajectory_id=f"trajectory-{run_id}",
        factor_id=factor_id,
        factor={
            "factor_name": factor_id,
            "expression": "RANK($close)",
            "description": "",
            "implementation_status": "generated",
        },
        metrics={
            "sharpe": gate_sharpe,
            "IC": 0.031,
            "ICIR": 0.6,
            "Rank ICIR": 0.7,
            "valid_cross_section_mean": 40.0,
            "cross_section_universe_size": 60.0,
            "ic_win_rate": 0.6,
            "rank_ic_win_rate": 0.6,
            "drawdown": 0.1,
            "long_short_positive_ratio": 0.6,
        },
    )


def test_factor_top_discovers_timestamped_run_stores(monkeypatch, tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    older = artifacts_dir / "run_store_older.sqlite"
    newer = artifacts_dir / "run_store_newer.sqlite"
    _write_factor_store(older, run_id="run-low", factor_id="FactorLow", sharpe=1.0)
    _write_factor_store(newer, run_id="run-high", factor_id="FactorHigh", sharpe=5.0)

    monkeypatch.delenv("RUN_STORE_PATH", raising=False)
    monkeypatch.setattr(runs_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runs_api, "_DB_PATH", artifacts_dir / "run_store.sqlite")

    top = runs_api.get_top_factors(limit=2)

    assert [item["run_id"] for item in top] == ["run-high", "run-low"]
    assert [item["rank"] for item in top] == [1, 2]
    assert runs_api.get_factors("run-high")[0]["factor_name"] == "FactorHigh"


def test_factor_top_prefers_unified_run_store_when_present(monkeypatch, tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    unified = artifacts_dir / "run_store.sqlite"
    legacy = artifacts_dir / "run_store_legacy.sqlite"
    _write_factor_store(unified, run_id="run-unified", factor_id="FactorUnified", sharpe=1.0)
    _write_factor_store(legacy, run_id="run-legacy", factor_id="FactorLegacy", sharpe=5.0)

    monkeypatch.delenv("RUN_STORE_PATH", raising=False)
    monkeypatch.setattr(runs_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runs_api, "_DB_PATH", unified)

    top = runs_api.get_top_factors(limit=5)

    assert [item["run_id"] for item in top] == ["run-unified"]


def test_run_factors_include_trajectory_reward(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(run_id="run-reward", log_dir=str(tmp_path / "run-reward"), args={"run_name": "run-reward"})
    trajectory = StrategyTrajectory(
        trajectory_id="original_000_000",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        hypothesis="reward test",
        backtest_metrics={"Rank IC": 0.05, "Rank ICIR": 0.5, "sharpe": 1.5, "coverage": 1.0},
    )
    store.upsert_trajectory(run_id="run-reward", trajectory=trajectory)
    store.upsert_factor(
        run_id="run-reward",
        loop_index=1,
        trajectory_id=trajectory.trajectory_id,
        factor_id="FactorReward",
        factor={
            "factor_name": "FactorReward",
            "expression": "RANK($close)",
            "implementation_status": "generated",
        },
        metrics={"sharpe": 1.5},
    )

    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    factors = runs_api.get_factors("run-reward")

    assert factors[0]["factor_name"] == "FactorReward"
    assert isinstance(factors[0]["reward_score"], float)
    assert isinstance(factors[0]["factor_reward_score"], float)
    assert isinstance(factors[0]["reward_breakdown"], dict)
    assert isinstance(factors[0]["reward_breakdown"]["terms"], list)


def test_hide_single_factor_preserves_history_and_metrics(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(run_id="run-delete", log_dir=str(tmp_path / "run-delete"), args={"run_name": "run-delete"})
    store.upsert_factor(
        run_id="run-delete",
        loop_index=1,
        trajectory_id="trajectory-delete",
        factor_id="factor-delete",
        factor={
            "factor_name": "DeleteMe",
            "expression": "RANK($close)",
            "implementation_status": "generated",
        },
        metrics={"sharpe": 1.8, "IC": 0.04},
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO factor_metrics (run_id, factor_id, metric_name, metric_value, metric_json)
            VALUES ('', 'factor-delete', 'legacy_metric', 1.0, NULL)
            """
        )

    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    result = runs_api.hide_factor_from_leaderboard("run-delete", "factor-delete")

    assert result["run_store_hidden"] == 1
    assert runs_api.get_top_factors(limit=10) == []
    factors = runs_api.get_factors("run-delete")
    assert [factor["factor_id"] for factor in factors] == ["factor-delete"]
    with sqlite3.connect(db_path) as conn:
        factor_count = conn.execute("SELECT COUNT(*) FROM factors WHERE factor_id = ?", ("factor-delete",)).fetchone()[0]
        metric_count = conn.execute("SELECT COUNT(*) FROM factor_metrics WHERE factor_id = ?", ("factor-delete",)).fetchone()[0]
        hidden = conn.execute("SELECT leaderboard_hidden FROM factors WHERE factor_id = ?", ("factor-delete",)).fetchone()[0]
    assert factor_count == 1
    assert metric_count > 0
    assert hidden == 1


def _leaderboard_metrics(
    *,
    reward_score: float,
    sharpe: float,
    ic: float = 0.031,
    drawdown: float = 0.10,
    valid_cross_section_mean: float = 40.0,
) -> dict[str, float]:
    return {
        "reward_score": reward_score,
        "sharpe": sharpe,
        "IC": ic,
        "ICIR": 0.6,
        "Rank ICIR": 0.7,
        "valid_cross_section_mean": valid_cross_section_mean,
        "cross_section_universe_size": 60.0,
        "ic_win_rate": 0.6,
        "rank_ic_win_rate": 0.6,
        "drawdown": drawdown,
        "long_short_positive_ratio": 0.6,
    }


def _write_leaderboard_factor(
    store: RunStore,
    *,
    run_id: str,
    factor_id: str,
    expression: str,
    reward_score: float,
    sharpe: float = 1.2,
    drawdown: float = 0.10,
    valid_cross_section_mean: float = 40.0,
) -> None:
    store.upsert_factor(
        run_id=run_id,
        loop_index=1,
        trajectory_id=f"trajectory-{factor_id}",
        factor_id=factor_id,
        factor={
            "factor_name": factor_id,
            "expression": expression,
            "description": "",
            "implementation_status": "generated",
        },
        metrics=_leaderboard_metrics(
            reward_score=reward_score,
            sharpe=sharpe,
            drawdown=drawdown,
            valid_cross_section_mean=valid_cross_section_mean,
        ),
    )


def test_run_factors_deduplicate_same_expression(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(run_id="run-dedup", log_dir=str(tmp_path / "run-dedup"), args={"run_name": "run-dedup"})
    for idx, sharpe in enumerate([1.0, 2.0]):
        trajectory_id = f"original_000_00{idx}"
        store.upsert_factor(
            run_id="run-dedup",
            loop_index=idx + 1,
            trajectory_id=trajectory_id,
            factor_id=f"legacy-trajectory-bound-id-{idx}",
            factor={
                "factor_name": f"DuplicateFactor{idx}",
                "expression": "RANK($close)",
                "implementation_status": "generated",
            },
            metrics={"sharpe": sharpe},
        )

    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    factors = runs_api.get_factors("run-dedup")

    assert len(factors) == 1
    assert factors[0]["factor_name"] == "DuplicateFactor1"
    assert factors[0]["sharpe"] == 2.0
    assert factors[0]["duplicate_expression_count"] == 2
    assert factors[0]["duplicate_trajectory_ids"] == ["original_000_000", "original_000_001"]


def test_progress_factors_deduplicate_same_expression_and_use_factor_reward() -> None:
    events = [
        {
            "stage": "evolution_task_finished",
            "trajectory_id": "original_000_000",
            "loop_index": 1,
            "phase": "original",
            "round_idx": 0,
            "reward_score": -0.50,
            "factor_summaries": [
                {
                    "factor_name": "DuplicateA",
                    "expression": "RANK($close)",
                    "reward_score": 0.80,
                    "metrics": {"sharpe": 1.0, "IC": 0.01},
                }
            ],
        },
        {
            "stage": "evolution_task_finished",
            "trajectory_id": "original_000_001",
            "loop_index": 2,
            "phase": "original",
            "round_idx": 0,
            "reward_score": -0.75,
            "factor_summaries": [
                {
                    "factor_name": "DuplicateB",
                    "expression": "rank( $close )",
                    "reward_score": 0.90,
                    "metrics": {"sharpe": 2.0, "IC": 0.02},
                }
            ],
        },
    ]

    factors = runs_api._progress_factors(events)

    assert len(factors) == 1
    assert factors[0]["factor_name"] == "DuplicateB"
    assert factors[0]["reward_score"] == -0.75
    assert factors[0]["factor_reward_score"] == 0.90
    assert factors[0]["duplicate_expression_count"] == 2
    assert factors[0]["duplicate_trajectory_ids"] == ["original_000_000", "original_000_001"]


def test_top_factors_use_run_scoped_sharpe_threshold(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(
        run_id="run-scoped-threshold",
        log_dir=str(tmp_path / "run-scoped-threshold"),
        args={"run_name": "run-scoped-threshold", "train_sharpe_threshold": 1.0},
    )
    store.upsert_trajectory(
        run_id="run-scoped-threshold",
        trajectory=StrategyTrajectory(
            trajectory_id="trajectory-scoped-threshold",
            phase=RoundPhase.ORIGINAL,
            round_idx=0,
            direction_id=0,
            hypothesis="stable threshold",
            backtest_metrics={"sharpe": 1.2},
        ),
    )
    store.upsert_factor(
        run_id="run-scoped-threshold",
        loop_index=1,
        trajectory_id="trajectory-scoped-threshold",
        factor_id="factor-scoped-threshold",
        factor={
            "factor_name": "ScopedThresholdFactor",
            "expression": "RANK($close)",
            "implementation_status": "generated",
        },
        metrics={
            "valid_cross_section_mean": 40.0,
            "cross_section_universe_size": 60.0,
            "IC": 0.031,
            "ic_win_rate": 0.60,
            "ICIR": 0.6,
            "Rank ICIR": 0.6,
            "rank_ic_win_rate": 0.60,
            "sharpe": 1.2,
            "drawdown": 0.10,
            "long_short_positive_ratio": 0.60,
        },
    )

    monkeypatch.setenv("TRAIN_SHARPE_THRESHOLD", "2.5")
    factors = runs_api.run_reader.get_top_factors(db_path, limit=30)

    assert len(factors) == 1
    assert factors[0]["factor_name"] == "ScopedThresholdFactor"


def test_top_factors_rank_admitted_before_qualified_reward(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(
        run_id="run-priority",
        log_dir=str(tmp_path / "run-priority"),
        args={"run_name": "run-priority", "train_sharpe_threshold": 1.0},
    )
    _write_leaderboard_factor(
        store,
        run_id="run-priority",
        factor_id="AdmittedLowReward",
        expression="RANK($close)",
        reward_score=0.10,
        sharpe=1.2,
    )
    _write_leaderboard_factor(
        store,
        run_id="run-priority",
        factor_id="QualifiedHighReward",
        expression="RANK($volume)",
        reward_score=99.0,
        sharpe=5.0,
        drawdown=0.60,
    )

    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    factors = runs_api.get_top_factors(limit=2)

    assert [factor["factor_id"] for factor in factors] == ["AdmittedLowReward", "QualifiedHighReward"]
    assert factors[0]["admission_passed"] is True
    assert factors[1]["admission_passed"] is False


def test_top_factors_replenish_after_single_hide(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(
        run_id="run-replenish",
        log_dir=str(tmp_path / "run-replenish"),
        args={"run_name": "run-replenish", "train_sharpe_threshold": 1.0},
    )
    for idx, reward_score in enumerate([3.0, 2.0, 1.0], start=1):
        _write_leaderboard_factor(
            store,
            run_id="run-replenish",
            factor_id=f"Factor{idx}",
            expression=f"RANK($close + {idx})",
            reward_score=reward_score,
            sharpe=1.2,
        )

    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    assert [factor["factor_id"] for factor in runs_api.get_top_factors(limit=2)] == ["Factor1", "Factor2"]

    runs_api.hide_factor_from_leaderboard("run-replenish", "Factor1")

    refreshed = runs_api.get_top_factors(limit=2)
    assert [factor["factor_id"] for factor in refreshed] == ["Factor2", "Factor3"]
    assert [factor["rank"] for factor in refreshed] == [1, 2]
    assert {factor["factor_id"] for factor in runs_api.get_factors("run-replenish")} == {"Factor1", "Factor2", "Factor3"}


def test_clear_all_factors_hides_leaderboard_without_deleting_history(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "artifacts" / "run_store.sqlite"
    db_path.parent.mkdir()
    store = RunStore(db_path)
    store.start_run(
        run_id="run-clear-hide",
        log_dir=str(tmp_path / "run-clear-hide"),
        args={"run_name": "run-clear-hide", "train_sharpe_threshold": 1.0},
    )
    _write_leaderboard_factor(
        store,
        run_id="run-clear-hide",
        factor_id="KeepHistory",
        expression="RANK($close)",
        reward_score=1.0,
        sharpe=1.2,
    )

    monkeypatch.setenv("RUN_STORE_PATH", str(db_path))
    assert [factor["factor_id"] for factor in runs_api.get_top_factors(limit=10)] == ["KeepHistory"]

    result = runs_api.clear_all_factors()

    assert result["run_store_hidden"] == 1
    assert runs_api.get_top_factors(limit=10) == []
    assert [factor["factor_id"] for factor in runs_api.get_factors("run-clear-hide")] == ["KeepHistory"]
