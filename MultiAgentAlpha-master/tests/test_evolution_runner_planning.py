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
from trading_agents.infra.in_memory_context_store import InMemoryContextStore
from trading_agents.schemas import AgentContext, AgentResult, LoopTrace, RoundPhase, SharedContext, StrategyTrajectory, TrajectoryPool
from trading_agents.workflows.evolution_runner import EvolutionConfig, EvolutionRunner


class FakePlannerClient(ModelClient):
    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        if "DirectionDecomposerAgent" in system_prompt:
            return json.dumps(
                {
                    "initial_hypotheses": [
                        {
                            "direction_id": idx,
                            "title": f"branch_{idx}",
                            "hypothesis": f"LLM-created initial hypothesis {idx}",
                            "thesis": f"test thesis {idx}",
                            "mechanism": "price-volume mechanism",
                            "expected_direction": "negative_forward_return",
                            "proxy_reconstruction_plan": {
                                "input_features": ["$close", "$volume"],
                                "core_proxy_family": ["price_volume"],
                                "supporting_proxy_family": [],
                                "validation_target": "fwd_returns_4h",
                            },
                            "selected_parametric_specifications": ["10-period high/low range"],
                            "hypothesis_evolution_state": {
                                "core_variables": ["price_volume"],
                                "verifiable_anchors": {
                                    "input_features": ["$close", "$volume"],
                                    "expression_patterns": ["TS_PCTCHANGE($close,*)"],
                                },
                                "refinement_stage": "initial_exploration",
                                "allowed_mutation_types": ["window_adjustment"],
                            },
                            "rationale": f"rationale {idx}",
                            "expected_signal_family": "test_signal",
                            "risk_notes": "test risk",
                            "diversity_axis": "price-volume/short-horizon/reversal",
                        }
                        for idx in range(5)
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


def test_round_zero_uses_llm_initial_hypothesis_generator() -> None:
    tasks = _runner()._build_tasks(
        round_idx=0,
        base_payload={"direction": "broad alpha mining direction"},
        initial_direction="broad alpha mining direction",
        config=EvolutionConfig(num_initial_directions=5),
        trajectory_pool=TrajectoryPool(),
    )

    assert len(tasks) == 5
    assert all(task.phase == RoundPhase.ORIGINAL for task in tasks)
    assert all(task.planner_mode == "llm" for task in tasks)
    assert tasks[0].direction == "LLM-created initial hypothesis 0"
    assert tasks[0].start_agent_name == "experiment_designer_agent"
    assert tasks[0].frozen_prefix_payload["hypothesis"] == "LLM-created initial hypothesis 0"
    assert tasks[0].frozen_prefix_payload["hypothesis_structured"]["selected_parametric_specifications"] == [
        "10-period high/low range"
    ]


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
