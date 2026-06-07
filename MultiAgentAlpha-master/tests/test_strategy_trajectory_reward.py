from __future__ import annotations

from trading_agents.schemas import RoundPhase, StrategyTrajectory, TrajectoryPool


def _trajectory(
    trajectory_id: str,
    *,
    rank_ic: float,
    drawdown: float = 0.0,
    complexity: dict[str, float] | None = None,
    redundancy: float = 0.0,
    corr: float = 0.0,
) -> StrategyTrajectory:
    factors = []
    if complexity:
        factors.append({"factor_name": "factor", "complexity": complexity})
    return StrategyTrajectory(
        trajectory_id=trajectory_id,
        direction_id=1,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        factors=factors,
        backtest_metrics={
            "Rank IC": rank_ic,
            "Rank ICIR": 0.25,
            "sharpe": 1.5,
            "coverage": 0.75,
            "drawdown": drawdown,
            "turnover": 0.5,
            "structural_similarity": redundancy,
            "max_abs_corr": corr,
        },
    )


def test_reward_score_uses_normalized_weighted_components() -> None:
    clean = _trajectory("clean", rank_ic=0.05)
    penalized = _trajectory(
        "penalized",
        rank_ic=0.05,
        drawdown=0.30,
        complexity={"complexity_score": 30.0, "max_complexity_score": 30.0},
        redundancy=1.0,
        corr=1.0,
    )

    assert round(clean.get_reward_score(), 6) == 0.525
    assert round(penalized.get_reward_score(), 6) == 0.225


def test_reward_components_expose_normalized_metric_table() -> None:
    trajectory = _trajectory(
        "components",
        rank_ic=0.025,
        drawdown=0.15,
        complexity={"complexity_score": 15.0, "max_complexity_score": 30.0},
        redundancy=0.25,
        corr=0.4,
    )

    components = trajectory.get_reward_components()

    assert components == {
        "norm_rank_ic": 0.5,
        "norm_rank_icir": 0.5,
        "norm_sharpe": 0.5,
        "norm_coverage": 0.75,
        "norm_drawdown": 0.5,
        "norm_turnover": 0.5,
        "norm_complexity": 0.5,
        "norm_redundancy": 0.25,
        "norm_corr": 0.4,
    }


def test_selection_score_uses_reward_not_raw_rank_ic() -> None:
    high_rank_high_risk = _trajectory(
        "high_rank_high_risk",
        rank_ic=0.05,
        drawdown=0.30,
        complexity={"complexity_score": 30.0, "max_complexity_score": 30.0},
        redundancy=1.0,
        corr=1.0,
    )
    lower_rank_clean = _trajectory("lower_rank_clean", rank_ic=0.04)
    pool = TrajectoryPool(trajectories=[high_rank_high_risk, lower_rank_clean])

    best = pool.get_best_trajectories(top_n=2)

    assert TrajectoryPool._selection_score(high_rank_high_risk) == high_rank_high_risk.get_reward_score()
    assert best[0].trajectory_id == "lower_rank_clean"
