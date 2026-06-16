from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.experiment_designer_agent import ExperimentDesignerAgent
from trading_agents.core import ModelClient
from trading_agents.schemas import AgentContext, SharedContext


class SequentialClient(ModelClient):
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[tuple[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        self.prompts.append((system_prompt, user_prompt))
        if not self.payloads:
            raise AssertionError("No payload left for model call")
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


def _factor_payload(name: str, expression: str) -> dict[str, object]:
    return {
        "task_plan": ["build a simple feature-whitelisted expression"],
        "factors": [
            {
                "factor_name": name,
                "description": "simple test factor",
                "formulation": expression,
                "expression": expression,
                "variables": {},
            }
        ],
    }


def test_experiment_designer_rejects_unsupported_dataset_features_then_retries(monkeypatch) -> None:
    monkeypatch.setenv("LLM_VERIFIER_ENABLED", "false")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "1")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", "0")

    client = SequentialClient(
        [
            _factor_payload("BadVolume4h", "RANK($volume_change_4h)"),
            _factor_payload("GoodVolume1h", "RANK($volume_change_1h)"),
        ]
    )

    result = ExperimentDesignerAgent("experiment_designer_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "hypothesis": "Volume shock should predict reversal.",
                "domain_dataset_summary": {
                    "feature_columns": ["close", "volume_change_1h", "volume_change_24h"],
                    "validation_target_column": "fwd_returns_24h",
                },
            }
        ),
        model_client=client,
    )

    spec = result.shared_updates["experiment_spec"]
    assert spec["factors"][0]["factor_name"] == "GoodVolume1h"
    assert spec["factors"][0]["expression"] == "RANK($volume_change_1h)"
    assert any(
        record["stage"] == "dataset_feature_whitelist" and "$volume_change_4h" in record["feedback"]
        for record in spec["consistency_gate"]["quality_gate"]["records"]
    )
    assert "Runtime-supported dataset feature whitelist" in client.prompts[0][1]
