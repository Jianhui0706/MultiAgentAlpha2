from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.direction_decomposer_agent import DirectionDecomposerAgent
from trading_agents.core import ModelClient
from trading_agents.schemas import AgentContext, SharedContext


class DirectionClient(ModelClient):
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.user_prompt = ""

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        self.user_prompt = user_prompt
        return json.dumps(self.payload, ensure_ascii=False)


def test_direction_decomposer_parses_five_directions() -> None:
    payload = {
        "directions": [
            {
                "direction_id": idx,
                "title": f"branch {idx}",
                "direction": f"LLM direction {idx}",
                "rationale": "test",
                "expected_signal_family": "test_signal",
                "risk_notes": "risk",
            }
            for idx in range(5)
        ]
    }
    result = DirectionDecomposerAgent("direction_decomposer_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(payload={"direction": "broad", "target_direction_count": 5}),
        model_client=DirectionClient(payload),
    )

    directions = result.shared_updates["decomposed_directions"]
    assert len(directions) == 5
    assert directions[0]["direction_id"] == 0
    assert directions[0]["direction"] == "LLM direction 0"
    assert result.shared_updates["direction_decomposer_mode"] == "llm"


def test_direction_decomposer_accepts_partial_payload_and_reports_shortfall() -> None:
    payload = {
        "directions": [
            {
                "direction_id": 0,
                "direction": "only one direction",
            }
        ]
    }
    result = DirectionDecomposerAgent("direction_decomposer_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(payload={"direction": "broad", "target_direction_count": 5}),
        model_client=DirectionClient(payload),
    )

    assert len(result.shared_updates["decomposed_directions"]) == 1
    assert result.artifacts["shortfall"] == 4


def test_direction_decomposer_injects_parametric_specifications() -> None:
    payload = {
        "directions": [
            {
                "direction_id": 0,
                "direction": "test 10-period high-low stress",
                "diversity_axis": "high_low_range_stress",
            }
        ]
    }
    client = DirectionClient(payload)

    DirectionDecomposerAgent("direction_decomposer_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(
            payload={
                "direction": "broad",
                "target_direction_count": 5,
                "parametric_specifications": [
                    {
                        "name": "high_low_range_stress",
                        "specification": "10-period high/low range",
                    }
                ],
            }
        ),
        model_client=client,
    )

    assert "Parametric specifications" in client.user_prompt
    assert "10-period high/low range" in client.user_prompt


def test_initial_hypothesis_output_is_normalized_to_compat_directions() -> None:
    payload = {
        "initial_hypotheses": [
            {
                "direction_id": 0,
                "title": "range stress",
                "hypothesis": "10-period high-low range stress should condition reversal strength.",
                "thesis": "range stress changes short-term reversal reliability",
                "mechanism": "wide ranges indicate unstable participation",
                "selected_parametric_specifications": ["10-period high/low range"],
            }
        ]
    }

    result = DirectionDecomposerAgent("direction_decomposer_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(payload={"direction": "broad", "target_direction_count": 3}),
        model_client=DirectionClient(payload),
    )

    hypotheses = result.shared_updates["initial_hypotheses"]
    directions = result.shared_updates["decomposed_directions"]
    assert hypotheses[0]["hypothesis"] == "10-period high-low range stress should condition reversal strength."
    assert hypotheses[0]["selected_parametric_specifications"] == ["10-period high/low range"]
    assert directions[0]["direction"] == hypotheses[0]["hypothesis"]
    assert result.artifacts["initial_hypothesis_count"] == 1
