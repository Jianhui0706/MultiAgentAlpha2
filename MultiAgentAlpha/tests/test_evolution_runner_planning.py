from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.core import AgentContextPolicy, BaseAgent, ContextPolicyRegistry, ModelClient, Orchestrator
from trading_agents.core.orchestrator import SequentialLoopOrchestrator
from trading_agents.infra.factor_library_manager import FactorLibraryManager
from trading_agents.infra.in_memory_context_store import InMemoryContextStore
from trading_agents.schemas import AgentContext, AgentResult, LoopTrace, RoundPhase, SharedContext, StrategyTrajectory, TrajectoryPool
from trading_agents.workflows.evolution_runner import EvolutionConfig, EvolutionRunner, EvolutionTask
from trading_agents.workflows.loop_runtime import LoopPolicy, LoopSession


class FakePlannerClient(ModelClient):
    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        if "DirectionDecomposerAgent" in system_prompt:
            families = [
                ("price_reversal", ["$close"], ["price_reversal"], "short reversal"),
                ("volume_confirmation", ["$volume"], ["volume_confirmation"], "volume confirmation"),
                ("range_stress", ["$high", "$low"], ["range_stress"], "range stress"),
                ("spread_pressure", ["$spread"], ["spread_pressure"], "spread pressure"),
                ("volatility_filter", ["$close", "$volume"], ["volatility_filter"], "volatility filter"),
            ]
            return json.dumps(
                {
                    "initial_hypotheses": [
                        {
                            "direction_id": idx,
                            "title": f"branch_{name}",
                            "hypothesis": f"LLM-created initial hypothesis {idx}: {description}",
                            "thesis": f"test thesis {idx}: {description}",
                            "mechanism": f"{description} mechanism",
                            "expected_direction": "negative_forward_return",
                            "proxy_reconstruction_plan": {
                                "input_features": input_features,
                                "core_proxy_family": core_family,
                                "supporting_proxy_family": [],
                                "validation_target": "fwd_returns_4h",
                            },
                            "selected_parametric_specifications": [f"{name} specification"],
                            "hypothesis_evolution_state": {
                                "core_variables": [name],
                                "verifiable_anchors": {
                                    "input_features": input_features,
                                    "expression_patterns": [f"RANK({input_features[0]})"],
                                },
                                "refinement_stage": "initial_exploration",
                                "allowed_mutation_types": ["window_adjustment"],
                            },
                            "rationale": f"rationale {idx}",
                            "expected_signal_family": name,
                            "risk_notes": "test risk",
                            "diversity_axis": description,
                        }
                        for idx, (name, input_features, core_family, description) in enumerate(families)
                    ]
                },
                ensure_ascii=False,
            )
        if "EvolutionAgent" in system_prompt:
            return json.dumps(
                {
                    "mutation_tasks": [
                        {
                            "direction_id": 101,
                            "parent_ids": ["original_000_000"],
                            "new_direction": "LLM-created mutation direction using stronger volatility control",
                            "mutation_type": "risk_mutation",
                            "target_segment": "factor_expression",
                            "fault_diagnosis": "Parent signal is useful but lacks risk control.",
                            "preserve_segments": ["hypothesis_generation"],
                            "rewrite_instruction": "Add volatility control only in factor construction.",
                            "rationale": "Improve stability.",
                            "preserve": "Parent reversal mechanism.",
                            "change": "Add volatility control.",
                        }
                    ],
                    "crossover_tasks": [
                        {
                            "direction_id": 201,
                            "parent_ids": ["original_000_000", "original_000_001"],
                            "new_direction": "LLM-created crossover direction fusing reversal and volume confirmation",
                            "source_segments": [
                                {"parent_id": "original_000_000", "segment": "hypothesis_generation", "inherit": "reversal thesis"},
                                {"parent_id": "original_000_001", "segment": "factor_expression", "inherit": "volume confirmation"},
                            ],
                            "fusion_plan": "Fuse reversal thesis with volume confirmation.",
                            "fusion_logic": "Combine reversal mechanism with volume confirmation.",
                            "rationale": "Fuse complementary evidence.",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return "{}"


class DummyOrchestrator(Orchestrator):
    def run_one_loop(self, initial_shared_context: SharedContext | None = None) -> LoopTrace:
        return LoopTrace(final_shared_context=dict((initial_shared_context or SharedContext()).payload))


def _runner() -> EvolutionRunner:
    return EvolutionRunner(
        orchestrator=DummyOrchestrator(),
        model_client=FakePlannerClient(),
    )


def _trajectory(trajectory_id: str, direction_id: int) -> StrategyTrajectory:
    return StrategyTrajectory(
        trajectory_id=trajectory_id,
        direction_id=direction_id,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        hypothesis=f"hypothesis {direction_id}",
        factors=[
            {
                "factor_name": f"factor_{direction_id}",
                "expression": "RANK($close)",
                "description": "test factor",
            }
        ],
        backtest_metrics={"Rank IC": 0.03 + direction_id * 0.01, "Rank ICIR": 0.2, "coverage": 0.9},
        feedback="usable parent",
        extra_info={
            "segments": {
                "hypothesis_generation": {"summary": f"hypothesis {direction_id}", "reusable_anchors": ["$close"]},
                "factor_expression": {"summary": "RANK($close)", "reusable_anchors": ["RANK($close)"]},
            }
        },
    )


def test_bootstrap_quality_gate_rejects_only_when_sharpe_and_ic_both_fail() -> None:
    config = EvolutionConfig(bootstrap_min_sharpe=1.0, bootstrap_min_abs_ic=0.02)
    weak = StrategyTrajectory(
        trajectory_id="weak",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        backtest_metrics={"sharpe": 0.8, "IC": 0.01},
    )
    sharpe_ok = StrategyTrajectory(
        trajectory_id="sharpe_ok",
        direction_id=1,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        backtest_metrics={"sharpe": 1.1, "IC": 0.0},
    )
    ic_ok = StrategyTrajectory(
        trajectory_id="ic_ok",
        direction_id=2,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        backtest_metrics={"sharpe": 0.2, "IC": -0.03},
    )

    assert EvolutionRunner._bootstrap_quality_decision(weak, config=config)["rejected"] is True
    assert EvolutionRunner._bootstrap_quality_decision(sharpe_ok, config=config)["rejected"] is False
    assert EvolutionRunner._bootstrap_quality_decision(ic_ok, config=config)["rejected"] is False


def test_bootstrap_quality_gate_scans_per_factor_metrics() -> None:
    config = EvolutionConfig(bootstrap_min_sharpe=1.5, bootstrap_min_abs_ic=0.02)
    # 摘要指标（按 |RankIC| 选出的最佳因子）不达标，但另一个因子 Sharpe 过线 → 合格
    summary_weak = StrategyTrajectory(
        trajectory_id="summary_weak",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        backtest_metrics={"sharpe": 0.5, "IC": 0.01},
    )
    decision = EvolutionRunner._bootstrap_quality_decision(
        summary_weak,
        config=config,
        per_factor_metrics={
            "weak_factor": {"sharpe": 0.3, "IC": 0.005},
            "strong_factor": {"sharpe": 1.8, "IC": 0.001},
        },
    )
    assert decision["rejected"] is False
    assert decision["qualifying_factor_count"] == 1
    assert decision["qualifying_factors"][0]["factor_name"] == "strong_factor"

    # 提供了 per-factor 指标且全部不达标 → 拒绝（以 per-factor 为准，不回退摘要）
    summary_good = StrategyTrajectory(
        trajectory_id="summary_good",
        direction_id=1,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        backtest_metrics={"sharpe": 2.0, "IC": 0.05},
    )
    decision = EvolutionRunner._bootstrap_quality_decision(
        summary_good,
        config=config,
        per_factor_metrics={"weak_factor": {"sharpe": 0.2, "IC": 0.01}},
    )
    assert decision["rejected"] is True
    assert decision["reason"] == "no_factor_reached_min_sharpe_or_min_abs_ic"


def test_bootstrap_quality_gate_ignores_non_mainline_factors() -> None:
    """adjacent/counterfactual 对照因子不得替假说通过资格线。"""
    config = EvolutionConfig(bootstrap_min_sharpe=1.5, bootstrap_min_abs_ic=0.02)
    trajectory = StrategyTrajectory(
        trajectory_id="adjacent_carry",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
        backtest_metrics={"sharpe": 1.6, "IC": 0.05},
    )
    decision = EvolutionRunner._bootstrap_quality_decision(
        trajectory,
        config=config,
        per_factor_metrics={
            # 主线因子全弱
            "mainline_weak": {"sharpe": 0.3, "IC": 0.005, "mainline_factor": 1.0},
            # 搭车的 adjacent 因子很强，但不算数
            "adjacent_strong": {"sharpe": 1.8, "IC": 0.06, "mainline_factor": 0.0},
        },
    )
    assert decision["rejected"] is True
    assert decision["checked_factor_count"] == 1

    # 缺 mainline_factor 字段的旧数据按 mainline 兼容
    decision = EvolutionRunner._bootstrap_quality_decision(
        trajectory,
        config=config,
        per_factor_metrics={"legacy_factor": {"sharpe": 1.8, "IC": 0.01}},
    )
    assert decision["rejected"] is False


def test_bootstrap_defaults_use_relaxed_initial_qualification() -> None:
    config = EvolutionConfig()
    assert config.bootstrap_refill_enabled is True
    assert config.bootstrap_max_refill_rounds == 10
    assert config.bootstrap_min_sharpe == 1.0
    assert config.bootstrap_min_abs_ic == 0.02
    assert config.factor_expressions_per_hypothesis == 2


def test_bootstrap_rejected_trajectories_are_not_selected_as_mutation_parents() -> None:
    rejected = _trajectory("rejected", 0)
    rejected.extra_info["bootstrap_rejected"] = True
    rejected.backtest_metrics.update({"sharpe": 3.0, "IC": 0.1, "Rank IC": 0.1})
    kept = _trajectory("kept", 1)
    kept.backtest_metrics.update({"sharpe": 1.1, "IC": 0.03, "Rank IC": 0.03})
    pool = TrajectoryPool(trajectories=[rejected, kept])

    parents = pool.select_parents_for_mutation(top_n=2)

    assert [item.trajectory_id for item in parents] == ["kept"]


def test_round_zero_uses_llm_initial_hypothesis_generator() -> None:
    events: list[dict[str, object]] = []
    runner = EvolutionRunner(
        orchestrator=DummyOrchestrator(),
        model_client=FakePlannerClient(),
        progress_callback=events.append,
    )
    tasks = runner._build_tasks(
        round_idx=0,
        base_payload={"direction": "broad alpha mining direction"},
        initial_direction="broad alpha mining direction",
        config=EvolutionConfig(num_initial_directions=5),
        trajectory_pool=TrajectoryPool(),
    )

    assert len(tasks) == 5
    assert all(task.phase == RoundPhase.ORIGINAL for task in tasks)
    assert all(task.planner_mode == "llm" for task in tasks)
    assert tasks[0].direction == "LLM-created initial hypothesis 0: short reversal"
    assert tasks[0].start_agent_name == "experiment_designer_agent"
    assert tasks[0].frozen_prefix_payload["hypothesis"] == "LLM-created initial hypothesis 0: short reversal"
    assert tasks[0].frozen_prefix_payload["hypothesis_structured"]["selected_parametric_specifications"] == [
        "price_reversal specification"
    ]
    finished = [event for event in events if event.get("stage") == "direction_decomposer_finished"][-1]
    shown_hypotheses = finished.get("initial_hypotheses")
    assert isinstance(shown_hypotheses, list)
    assert shown_hypotheses[0]["title"] == "branch_price_reversal"
    assert shown_hypotheses[0]["core_proxy_family"] == "price_reversal"


class DuplicateInitialHypothesisClient(ModelClient):
    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        return json.dumps(
            {
                "initial_hypotheses": [
                    {
                        "direction_id": idx,
                        "title": f"duplicate_{idx}",
                        "hypothesis": f"Volume pressure reversal hypothesis variant {idx}",
                        "thesis": "Volume pressure should predict reversal.",
                        "mechanism": "Same pressure mechanism.",
                        "expected_direction": "negative_forward_return",
                        "proxy_reconstruction_plan": {
                            "input_features": ["$volume", "$close"],
                            "core_proxy_family": ["volume_pressure"],
                            "supporting_proxy_family": ["spread_pressure"],
                            "validation_target": "fwd_returns_4h",
                        },
                        "selected_parametric_specifications": ["volume pressure specification"],
                        "expected_signal_family": "volume_pressure",
                        "diversity_axis": "same volume pressure axis",
                    }
                    for idx in range(5)
                ]
            },
            ensure_ascii=False,
        )


def test_round_zero_deduplicates_initial_hypotheses_and_fills_shortfall() -> None:
    runner = EvolutionRunner(
        orchestrator=DummyOrchestrator(),
        model_client=DuplicateInitialHypothesisClient(),
    )
    tasks = runner._build_tasks(
        round_idx=0,
        base_payload={"direction": "broad alpha mining direction"},
        initial_direction="broad alpha mining direction",
        config=EvolutionConfig(num_initial_directions=5),
        trajectory_pool=TrajectoryPool(),
    )

    assert len(tasks) == 5
    assert tasks[0].planner_mode == "llm"
    assert [task.direction_id for task in tasks] == [0, 1, 2, 3, 4]
    assert len({task.direction for task in tasks}) == 5
    assert any(task.planner_mode == "fallback_rule" for task in tasks[1:])


def test_record_trajectory_penalizes_duplicate_factor_expression() -> None:
    session = LoopSession(
        policy=LoopPolicy(max_loops=1),
        trajectory_pool=TrajectoryPool(trajectories=[_trajectory("original_000_000", 0)]),
    )
    task = EvolutionTask(
        phase=RoundPhase.ORIGINAL,
        round_idx=0,
        direction_id=1,
        direction="duplicate expression branch",
        trajectory_id="original_000_001",
        parent_ids=[],
    )
    payload = {
        "phase": "original",
        "round_idx": 0,
        "trajectory_id": "original_000_001",
        "metrics": {"Rank IC": 0.05, "Rank ICIR": 0.5, "sharpe": 2.0, "coverage": 1.0},
        "experiment_spec": {
            "target_hypothesis": "duplicate expression hypothesis",
            "factors": [
                {
                    "factor_name": "DuplicateRankClose",
                    "expression": " RANK( $close ) ",
                }
            ],
        },
    }

    EvolutionRunner._record_trajectory(session, payload, task)

    trajectory = session.trajectory_pool.get_by_id("original_000_001")
    assert trajectory is not None
    assert trajectory.extra_info["reward_penalties"]["structural_similarity"] == 1.0
    assert trajectory.extra_info["reward_penalties"]["duplicate_against_trajectory_ids"] == ["original_000_000"]
    assert trajectory.get_reward_components()["norm_redundancy"] == 1.0


def test_build_novelty_context_summarizes_history(tmp_path: Path) -> None:
    trajectory = _trajectory("original_000_000", 0)
    trajectory.hypothesis_details = {
        "thesis": "Close reversal should predict forward returns.",
        "mechanism": "Short-term overreaction unwinds.",
        "expected_direction": "negative_forward_return",
        "proxy_reconstruction_plan": {
            "input_features": ["$close"],
            "core_proxy_family": ["price_reversal"],
            "supporting_proxy_family": ["range_stress"],
            "validation_target": "fwd_returns_4h",
        },
        "selected_parametric_specifications": ["close reversal specification"],
        "initialization_agent": {
            "expected_signal_family": "price_reversal",
            "diversity_axis": "short reversal axis",
        },
    }
    factor_library = FactorLibraryManager(tmp_path / "factor_library.json")
    factor_library.data["factors"] = {
        "stored_factor": {
            "factor_expression": "TS_MEAN($volume, 24)",
        }
    }

    context = EvolutionRunner._build_novelty_context(TrajectoryPool(trajectories=[trajectory]), factor_library)

    assert context["enabled"] is True
    assert context["recent_hypotheses"][0]["hypothesis"] == "hypothesis 0"
    assert context["recent_hypotheses"][0]["expected_signal_family"] == "price_reversal"
    assert "RANK($close)" in context["recent_factor_expressions"]
    assert "TS_MEAN($volume, 24)" in context["recent_factor_expressions"]
    assert "price_reversal" in context["explored_proxy_families"]


def test_evolved_round_executes_planned_mutation_and_crossover_branches() -> None:
    pool = TrajectoryPool(
        trajectories=[
            _trajectory("original_000_000", 0),
            _trajectory("original_000_001", 1),
        ]
    )
    tasks = _runner()._build_tasks(
        round_idx=1,
        base_payload={},
        initial_direction=None,
        config=EvolutionConfig(
            profile="legacy",
            mutation_tasks_per_round=1,
            crossover_tasks_per_round=1,
            evolved_directions_per_round=5,
            mutation_parent_top_n=2,
            crossover_parent_top_n=2,
        ),
        trajectory_pool=pool,
    )

    phases = [task.phase for task in tasks]
    assert len(tasks) == 2
    assert phases.count(RoundPhase.MUTATION) == 1
    assert phases.count(RoundPhase.CROSSOVER) == 1
    assert all(task.planner_mode == "llm" for task in tasks)
    assert tasks[0].phase == RoundPhase.MUTATION
    assert tasks[0].mutation_type == "risk_mutation"
    assert tasks[0].direction == "LLM-created mutation direction using stronger volatility control"
    assert tasks[0].parent_ids == ["original_000_000"]
    assert tasks[0].target_segment == "factor_expression"
    assert tasks[0].start_agent_name == "experiment_designer_agent"
    assert tasks[0].frozen_prefix_payload["hypothesis"] == "hypothesis 0"
    assert tasks[1].phase == RoundPhase.CROSSOVER
    assert tasks[1].direction == "LLM-created crossover direction fusing reversal and volume confirmation"
    assert tasks[1].parent_ids == ["original_000_000", "original_000_001"]
    assert tasks[1].fusion_logic == "Combine reversal mechanism with volume confirmation."
    assert len(tasks[1].source_segments) == 2


def test_quantaalpha_schedule_alternates_mutation_then_crossover() -> None:
    pool = TrajectoryPool(
        trajectories=[
            _trajectory("original_000_000", 0),
            _trajectory("original_000_001", 1),
        ]
    )
    config = EvolutionConfig(
        profile="quantaalpha",
        max_rounds=3,
        mutation_tasks_per_round=1,
        crossover_tasks_per_round=1,
        mutation_parent_top_n=2,
        crossover_parent_top_n=2,
    )

    round_one = _runner()._build_tasks(
        round_idx=1,
        base_payload={},
        initial_direction=None,
        config=config,
        trajectory_pool=pool,
    )
    round_two = _runner()._build_tasks(
        round_idx=2,
        base_payload={},
        initial_direction=None,
        config=config,
        trajectory_pool=pool,
    )

    assert {task.phase for task in round_one} == {RoundPhase.MUTATION}
    assert {task.phase for task in round_two} == {RoundPhase.CROSSOVER}


class _AppendAgent(BaseAgent):
    def run(self, *, private_context: AgentContext, shared_context: SharedContext, model_client: ModelClient) -> AgentResult:
        order = list(shared_context.payload.get("order", []))
        order.append(self.name)
        return AgentResult(shared_updates={"order": order})


def test_sequential_orchestrator_can_start_from_middle_agent() -> None:
    policy_registry = ContextPolicyRegistry()
    policy_registry.add_many(
        [
            AgentContextPolicy(agent_name=name, role="test", readable_shared_keys={"order"}, writable_shared_keys={"order"})
            for name in ["a", "b", "c"]
        ]
    )
    orchestrator = SequentialLoopOrchestrator(
        agents=[_AppendAgent("a"), _AppendAgent("b"), _AppendAgent("c")],
        context_store=InMemoryContextStore(),
        model_client=FakePlannerClient(),
        policy_registry=policy_registry,
    )

    trace = orchestrator.run_from_agent("b", SharedContext(payload={"order": []}))

    assert [record.agent_name for record in trace.records] == ["b", "c"]
    assert trace.final_shared_context["order"] == ["b", "c"]


class _ProgressAgent(BaseAgent):
    def run(self, *, private_context: AgentContext, shared_context: SharedContext, model_client: ModelClient) -> AgentResult:
        callback = private_context.payload.get("_progress_callback")
        assert callable(callback)
        callback({"stage": "custom_agent_progress", "agent_name": self.name})
        return AgentResult(shared_updates={"progress_seen": True})


def test_sequential_orchestrator_exposes_internal_progress_callback_temporarily() -> None:
    events: list[dict[str, object]] = []
    store = InMemoryContextStore()
    policy_registry = ContextPolicyRegistry()
    policy_registry.add(AgentContextPolicy(agent_name="progress_agent", role="test"))
    orchestrator = SequentialLoopOrchestrator(
        agents=[_ProgressAgent("progress_agent")],
        context_store=store,
        model_client=FakePlannerClient(),
        policy_registry=policy_registry,
        progress_callback=events.append,
    )

    trace = orchestrator.run_one_loop(SharedContext(payload={}))

    stages = [event.get("stage") for event in events]
    assert stages == ["agent_started", "custom_agent_progress", "agent_finished"]
    assert trace.final_shared_context == {}
    assert "_progress_callback" not in store.get_private_context("progress_agent").payload
