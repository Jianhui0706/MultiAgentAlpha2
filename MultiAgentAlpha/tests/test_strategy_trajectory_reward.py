from __future__ import annotations

from trading_agents.schemas import RoundPhase, StrategyTrajectory, TrajectoryPool


def _trajectory(
    trajectory_id: str,
    *,
    sharpe: float = 1.5,
    ic: float = 0.03,
    rank_ic: float,
    rank_icir: float = 0.25,
    icir: float = 0.25,
    drawdown: float = 0.0,
    turnover: float = 0.5,
    coverage: float = 0.75,
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
            "IC": ic,
            "Rank IC": rank_ic,
            "ICIR": icir,
            "Rank ICIR": rank_icir,
            "sharpe": sharpe,
            "coverage": coverage,
            "drawdown": drawdown,
            "turnover": turnover,
            "structural_similarity": redundancy,
            "max_abs_corr": corr,
        },
    )


def test_reward_marks_factor_that_passes_all_hard_targets() -> None:
    passed = _trajectory(
        "passed",
        sharpe=1.6,
        ic=0.031,
        rank_ic=0.032,
        rank_icir=0.6,
        icir=0.6,
        turnover=0.2,
        coverage=1.0,
    )

    status = passed.get_target_status()

    assert status["passed"] is True
    assert status["sharpe_passed"] is True
    assert status["ic_passed"] is True
    assert status["rank_ic_passed"] is True
    assert passed.get_reward_components()["joint_target_bonus"] == 1.0
    assert passed.get_reward_score() > 3.0


def test_ic_only_candidate_does_not_beat_balanced_near_target_candidate() -> None:
    ic_only_low_sharpe = _trajectory(
        "ic_only_low_sharpe",
        sharpe=0.4,
        ic=0.06,
        rank_ic=0.06,
        rank_icir=0.6,
        icir=0.6,
        turnover=0.2,
        coverage=1.0,
    )
    balanced_near_target = _trajectory(
        "balanced_near_target",
        sharpe=1.45,
        ic=0.028,
        rank_ic=0.029,
        rank_icir=0.35,
        icir=0.35,
        turnover=0.2,
        coverage=1.0,
    )

    assert balanced_near_target.get_target_status()["passed"] is False
    assert ic_only_low_sharpe.get_target_status()["passed"] is False
    assert balanced_near_target.get_reward_score() > ic_only_low_sharpe.get_reward_score()


def test_high_sharpe_low_ic_candidate_is_not_marked_as_good_factor() -> None:
    sharpe_only = _trajectory(
        "sharpe_only",
        sharpe=1.8,
        ic=0.01,
        rank_ic=0.01,
        turnover=0.2,
        coverage=1.0,
    )

    status = sharpe_only.get_target_status()

    assert status["passed"] is False
    assert status["sharpe_passed"] is True
    assert status["ic_passed"] is False
    assert status["rank_ic_passed"] is False
    assert sharpe_only.get_reward_components()["joint_target_bonus"] == 0.0


def test_reward_score_uses_target_progress_and_secondary_penalties() -> None:
    clean = _trajectory("clean", sharpe=1.55, ic=0.031, rank_ic=0.032)
    penalized = _trajectory(
        "penalized",
        sharpe=1.55,
        ic=0.031,
        rank_ic=0.032,
        drawdown=0.30,
        complexity={"complexity_score": 30.0, "max_complexity_score": 30.0},
        redundancy=1.0,
        corr=1.0,
    )

    assert clean.get_reward_score() > penalized.get_reward_score()
    assert clean.get_reward_components()["joint_target_bonus"] == 1.0


def test_reward_components_expose_normalized_metric_table() -> None:
    trajectory = _trajectory(
        "components",
        sharpe=1.2,
        ic=-0.02,
        rank_ic=0.025,
        drawdown=0.15,
        complexity={"complexity_score": 15.0, "max_complexity_score": 30.0},
        redundancy=0.25,
        corr=0.4,
    )

    components = trajectory.get_reward_components()

    assert round(components["sharpe_progress"], 6) == 0.8
    assert round(components["ic_progress"], 6) == round(abs(-0.02) / 0.03, 6)
    assert round(components["rank_ic_progress"], 6) == round(0.025 / 0.03, 6)
    assert components["joint_target_bonus"] == 0.0
    assert components["coverage_score"] == 0.75
    assert components["drawdown_penalty"] == 0.5
    assert components["turnover_penalty"] == 0.5
    assert components["complexity_penalty"] == 0.5
    assert components["redundancy_penalty"] == 0.25
    assert components["corr_penalty"] == 0.4
    assert components["norm_rank_ic"] == 0.5
    assert components["norm_redundancy"] == 0.25


def test_selection_score_uses_reward_not_raw_rank_ic() -> None:
    high_rank_high_risk = _trajectory(
        "high_rank_high_risk",
        sharpe=1.55,
        ic=0.04,
        rank_ic=0.05,
        drawdown=0.30,
        complexity={"complexity_score": 30.0, "max_complexity_score": 30.0},
        redundancy=1.0,
        corr=1.0,
    )
    lower_rank_clean = _trajectory("lower_rank_clean", sharpe=1.55, ic=0.035, rank_ic=0.04)
    pool = TrajectoryPool(trajectories=[high_rank_high_risk, lower_rank_clean])

    best = pool.get_best_trajectories(top_n=2)

    assert TrajectoryPool._selection_score(high_rank_high_risk) == high_rank_high_risk.get_reward_score()
    assert best[0].trajectory_id == "lower_rank_clean"
