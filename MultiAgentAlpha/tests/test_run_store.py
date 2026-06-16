from __future__ import annotations

import sqlite3
import json
from datetime import datetime

from trading_agents.infra.run_store import RunStore
from trading_agents.schemas import LoopTrace, RoundPhase, StrategyTrajectory, TraceRecord

from main import _factor_id_for_storage, _save_loop_trace, _summarize_factors_across_rounds


def test_run_store_schema_is_idempotent_and_records_loop(tmp_path) -> None:
    db_path = tmp_path / "run_store.sqlite"
    store = RunStore(db_path)
    RunStore(db_path)

    store.start_run(run_id="run-1", log_dir=str(tmp_path), args={"loop_count": 1})
    final_context = {
        "hypothesis": "test hypothesis",
        "phase": "original",
        "round_idx": 1,
        "trajectory_id": "original_001_000",
        "qlib_factor_experiment": {
            "experiment_id": "exp-1",
            "target_hypothesis": "test hypothesis",
            "factors": [{"factor_name": "FactorA", "expression": "RANK($close)"}],
            "metrics": {"sharpe": 1.2},
            "per_factor_metrics": {"FactorA": {"sharpe": 1.2}},
        },
    }
    summary = {
        "status": "completed",
        "phase": "original",
        "round_idx": 1,
        "trajectory_id": "original_001_000",
        "parent_ids": [],
        "error_count": 0,
        "agent_count": 1,
        "elapsed_seconds": 0.1,
    }

    store.record_loop(run_id="run-1", loop_index=1, summary=summary, final_context=final_context)
    loaded = store.load_loop_compatible(run_id="run-1", loop_index=1)

    assert loaded is not None
    assert loaded["final_shared_context"]["qlib_factor_experiment"]["metrics"] == {"sharpe": 1.2}

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM loops").fetchone()[0] == 1


def test_run_store_keeps_same_factor_id_across_runs(tmp_path) -> None:
    db_path = tmp_path / "run_store.sqlite"
    store = RunStore(db_path)
    factor = {
        "factor_name": "RepeatedFactor",
        "expression": "RANK($close)",
        "implementation_status": "generated",
    }

    for run_id, sharpe in [("run-1", 1.0), ("run-2", 2.0)]:
        store.start_run(run_id=run_id, log_dir=str(tmp_path), args={"run_name": run_id})
        store.upsert_factor(
            run_id=run_id,
            loop_index=1,
            trajectory_id=f"trajectory-{run_id}",
            factor_id="same-factor-id",
            factor=factor,
            metrics={"sharpe": sharpe},
        )

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM factor_metrics").fetchone()[0] == 2
        rows = conn.execute(
            """
            SELECT f.run_id, m.metric_value
            FROM factors f
            JOIN factor_metrics m ON m.run_id = f.run_id AND m.factor_id = f.factor_id
            ORDER BY f.run_id
            """
        ).fetchall()

    assert rows == [("run-1", 1.0), ("run-2", 2.0)]


def test_factor_storage_id_uses_expression_not_trajectory() -> None:
    factor = {"factor_name": "SameExpression", "expression": " RANK( $close ) "}

    first = _factor_id_for_storage("original_000_001", factor)
    second = _factor_id_for_storage("original_000_002", {"factor_name": "OtherName", "expression": "RANK($close)"})

    assert first == second


def test_run_store_keeps_same_trajectory_id_across_runs(tmp_path) -> None:
    db_path = tmp_path / "run_store.sqlite"
    store = RunStore(db_path)

    for run_id, reward in [("run-1", 0.1), ("run-2", 0.2)]:
        trajectory = StrategyTrajectory(
            trajectory_id="original_000_000",
            direction_id=0,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
            hypothesis=run_id,
            backtest_metrics={"reward": reward},
        )
        store.upsert_trajectory(run_id=run_id, trajectory=trajectory)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, trajectory_id, reward_score FROM trajectories ORDER BY run_id"
        ).fetchall()

    assert len(rows) == 2
    assert [row[0] for row in rows] == ["run-1", "run-2"]


def test_light_loop_json_omits_full_snapshots(tmp_path) -> None:
    trace = LoopTrace(
        records=[
            TraceRecord(
                agent_name="agent_a",
                started_at=datetime(2026, 1, 1, 1, 0, 0),
                ended_at=datetime(2026, 1, 1, 1, 0, 1),
                elapsed_seconds=1.0,
                input_snapshot={"shared_context": {"huge": "x" * 100}, "private_context": {"secret": "y"}},
                output_snapshot={"shared_updates": {"metrics": {"sharpe": 1.0}}, "private_context": {"raw": "z"}},
            )
        ],
        final_shared_context={
            "hypothesis": "h",
            "qlib_factor_experiment": {
                "experiment_id": "exp",
                "factors": [{"factor_name": "FactorA", "expression": "RANK($close)"}],
                "metrics": {"sharpe": 1.0},
            },
        },
    )

    _save_loop_trace(tmp_path, 1, trace, debug_full_trace=False)
    text = (tmp_path / "loop_001.json").read_text(encoding="utf-8")

    assert "input_snapshot" not in text
    assert "output_snapshot" not in text
    assert "final_shared_context_keys" in text
    assert not (tmp_path / "loop_001.debug.json.gz").exists()


def test_summary_factor_index_keeps_factor_expressions_and_metrics(tmp_path) -> None:
    trace = LoopTrace(
        records=[],
        final_shared_context={
            "hypothesis": "h",
            "phase": "original",
            "trajectory_id": "original_001_000",
            "qlib_factor_experiment": {
                "experiment_id": "exp",
                "factors": [
                    {
                        "factor_name": "FactorA",
                        "description": "close rank",
                    }
                ],
                "factor_implementation": {
                    "implementations": [
                        {
                            "factor_name": "FactorA",
                            "expression": "RANK($close)",
                            "status": "generated",
                            "signal_cache_key": "abc123",
                        }
                    ]
                },
                "per_factor_metrics": {
                    "FactorA": {"sharpe": 1.2, "Rank IC": 0.03, "unused_large_field": "kept out"}
                },
            },
        },
    )

    _save_loop_trace(tmp_path, 1, trace, debug_full_trace=False)
    loop_payload = json.loads((tmp_path / "loop_001.json").read_text(encoding="utf-8"))
    rounds = [loop_payload["summary"]]
    factor_summary = _summarize_factors_across_rounds(rounds)

    assert loop_payload["summary"]["factors"][0]["factor_name"] == "FactorA"
    assert loop_payload["summary"]["factors"][0]["expression"] == "RANK($close)"
    assert factor_summary["factor_count"] == 1
    assert factor_summary["unique_factor_count"] == 1
    assert factor_summary["factors"][0]["metrics"] == {"sharpe": 1.2, "Rank IC": 0.03}


def test_debug_full_trace_writes_compressed_artifact(tmp_path) -> None:
    trace = LoopTrace(
        records=[
            TraceRecord(
                agent_name="agent_a",
                started_at=datetime(2026, 1, 1, 1, 0, 0),
                ended_at=datetime(2026, 1, 1, 1, 0, 1),
                input_snapshot={"shared_context": {"full": "kept"}},
                output_snapshot={"shared_updates": {"result": "kept"}},
            )
        ],
        final_shared_context={"hypothesis": "h"},
    )

    _save_loop_trace(tmp_path, 1, trace, debug_full_trace=True)

    assert (tmp_path / "loop_001.debug.json.gz").exists()
