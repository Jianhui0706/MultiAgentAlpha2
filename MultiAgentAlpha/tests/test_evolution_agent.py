from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.evolution_agent import EvolutionAgent
from trading_agents.core import ModelClient
from trading_agents.schemas import AgentContext, SharedContext


class EvolutionClient(ModelClient):
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def test_evolution_agent_parses_mutation_and_crossover() -> None:
    payload = {
        "mutation_tasks": [
            {
                "direction_id": 101,
                "parent_ids": ["p1"],
                "new_direction": "fresh mutation direction",
                "mutation_type": "parameter_mutation",
                "target_gap": "improve_sharpe",
                "metric_objective": "raise net Sharpe above 1.5 while keeping abs(IC) above 0.03",
                "target_segment": "factor_expression",
                "fault_diagnosis": "window is too short",
                "preserve_segments": ["hypothesis_generation"],
                "rewrite_instruction": "Tune the rolling window only.",
                "rationale": "tune",
                "preserve": "core",
                "change": "window",
            }
        ],
        "crossover_tasks": [
            {
                "direction_id": 201,
                "parent_ids": ["p1", "p2"],
                "new_direction": "fresh crossover direction",
                "target_gap": "reduce_turnover",
                "metric_objective": "inherit signal from p1 and low-turnover structure from p2",
                "source_segments": [
                    {"parent_id": "p1", "segment": "hypothesis_generation", "inherit": "core thesis"},
                    {"parent_id": "p2", "segment": "factor_expression", "inherit": "proxy pattern"},
                ],
                "fusion_plan": "combine thesis and proxy",
                "fusion_logic": "fuse mechanisms",
                "rationale": "combine strengths",
            }
        ],
    }
    result = EvolutionAgent("evolution_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(payload={"round_idx": 1, "mutation_target_count": 1, "crossover_target_count": 1}),
        model_client=EvolutionClient(payload),
    )

    plan = result.shared_updates["evolution_agent_plan"]
    assert len(plan["mutation_tasks"]) == 1
    assert len(plan["crossover_tasks"]) == 1
    assert plan["mutation_tasks"][0]["new_direction"] == "fresh mutation direction"
    assert plan["mutation_tasks"][0]["target_gap"] == "improve_sharpe"
    assert plan["mutation_tasks"][0]["metric_objective"] == "raise net Sharpe above 1.5 while keeping abs(IC) above 0.03"
    assert plan["mutation_tasks"][0]["target_segment"] == "factor_expression"
    assert plan["mutation_tasks"][0]["preserve_segments"] == ["hypothesis_generation"]
    assert plan["crossover_tasks"][0]["parent_ids"] == ["p1", "p2"]
    assert plan["crossover_tasks"][0]["target_gap"] == "reduce_turnover"
    assert plan["crossover_tasks"][0]["metric_objective"] == "inherit signal from p1 and low-turnover structure from p2"
    assert plan["crossover_tasks"][0]["fusion_plan"] == "combine thesis and proxy"
    assert len(plan["crossover_tasks"][0]["source_segments"]) == 2
    assert result.shared_updates["evolution_planner_mode"] == "llm"


def test_evolution_agent_filters_empty_and_invalid_tasks() -> None:
    payload = {
        "mutation_tasks": [
            {"direction_id": 101, "parent_ids": ["p1"], "new_direction": ""},
            {"direction_id": 102, "parent_ids": [], "new_direction": "missing parent"},
            {"direction_id": 103, "parent_ids": ["p1"], "new_direction": "valid mutation"},
        ],
        "crossover_tasks": [
            {"direction_id": 201, "parent_ids": ["p1"], "new_direction": "not enough parents"},
            {"direction_id": 202, "parent_ids": ["p1", "p2"], "new_direction": "valid crossover"},
        ],
    }
    result = EvolutionAgent("evolution_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(payload={"round_idx": 1, "mutation_target_count": 2, "crossover_target_count": 2}),
        model_client=EvolutionClient(payload),
    )

    plan = result.shared_updates["evolution_agent_plan"]
    assert [item["new_direction"] for item in plan["mutation_tasks"]] == ["valid mutation"]
    assert [item["new_direction"] for item in plan["crossover_tasks"]] == ["valid crossover"]
