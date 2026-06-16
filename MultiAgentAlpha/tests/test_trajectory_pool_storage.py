from __future__ import annotations

from trading_agents.schemas import RoundPhase, StrategyTrajectory, TrajectoryPool


def _trajectory(trajectory_id: str, hypothesis: str) -> StrategyTrajectory:
    return StrategyTrajectory(
        trajectory_id=trajectory_id,
        direction_id=1,
        round_idx=1,
        phase=RoundPhase.ORIGINAL,
        hypothesis=hypothesis,
        factors=[{"factor_name": "FactorA", "expression": "RANK($close)"}],
        backtest_metrics={"Rank IC": 0.1},
    )


def test_trajectory_pool_jsonl_appends_and_loads_by_id(tmp_path) -> None:
    save_path = tmp_path / "trajectory_pool.jsonl"
    pool = TrajectoryPool(save_path=str(save_path))

    pool.add(_trajectory("original_001_001", "first"))
    pool.add(_trajectory("original_001_002", "second"))

    lines = save_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    reloaded = TrajectoryPool(save_path=str(save_path))
    reloaded.load()

    assert [item.trajectory_id for item in reloaded.get_all()] == [
        "original_001_001",
        "original_001_002",
    ]


def test_trajectory_pool_jsonl_upserts_duplicate_id_on_load(tmp_path) -> None:
    save_path = tmp_path / "trajectory_pool.jsonl"
    pool = TrajectoryPool(save_path=str(save_path))

    pool.add(_trajectory("original_001_001", "first"))
    pool.add(_trajectory("original_001_001", "updated"))

    lines = save_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    reloaded = TrajectoryPool(save_path=str(save_path))
    reloaded.load()

    trajectories = reloaded.get_all()
    assert len(trajectories) == 1
    assert trajectories[0].trajectory_id == "original_001_001"
    assert trajectories[0].hypothesis == "updated"
