from __future__ import annotations

import json
from pathlib import Path

from trading_agents.agents.hypothesis.hypothesis_agent import HypothesisAgentV2
from trading_agents.schemas import AgentContext, SharedContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeModelClient:
    def __init__(self, *, fail_above_chars: int) -> None:
        self.fail_above_chars = fail_above_chars
        self.calls: list[dict[str, int | bool]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        total_chars = len(system_prompt) + len(user_prompt)
        self.calls.append(
            {
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "total_prompt_chars": total_chars,
                "json_mode": json_mode,
            }
        )
        if total_chars > self.fail_above_chars:
            raise RuntimeError("APIConnectionError: Connection error.")
        return json.dumps(
            {
                "hypothesis": "Price-volume divergence in crypto predicts short-term reversal.",
                "thesis": "Unconfirmed price momentum reflects weak conviction and mean reversion risk.",
                "mechanism": "Price extremes without matching volume support tend to reverse cross-sectionally.",
                "expected_direction": "negative_forward_return",
                "market_structure_dependencies": ["session-based trading"],
                "crypto_adaptation_summary": {
                    "transferable_core": "Divergence logic transfers across markets.",
                    "non_transferable_parts": ["equity session structure"],
                    "crypto_rewrite": "Use rolling windows instead of discrete sessions.",
                },
                "proxy_reconstruction_plan": {
                    "input_features": ["$returns_24h", "$volume_change_24h", "$high_low_spread"],
                    "core_proxy_family": ["price_volume_divergence"],
                    "supporting_proxy_family": ["volatility_regime"],
                    "validation_target": "fwd_returns_24h",
                },
                "concise_knowledge": "Unconfirmed momentum can reverse when conviction is weak.",
                "concise_observation": "Prompt was reduced enough to let the model respond.",
                "concise_justification": "Load shedding preserved the main thesis while reducing prompt size.",
                "concise_specification": "Use returns and volume change proxies with rolling-window divergence.",
                "hypothesis_evolution_state": {
                    "core_variables": ["price_momentum", "volume_change", "spread_regime"],
                    "verifiable_anchors": {
                        "input_features": ["$returns_24h", "$volume_change_24h", "$high_low_spread"],
                        "expression_patterns": [
                            "TS_PCTCHANGE($close,*)",
                            "TS_HIGH_HIGH_SPREAD($high_low_spread,*)",
                        ],
                    },
                    "refinement_stage": "parameter_tuning",
                    "allowed_mutation_types": ["window_adjustment", "threshold_adjustment"],
                },
            },
            ensure_ascii=False,
        )


def load_research_report_text() -> str:
    candidate = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / "20260424_114736" / "loop_003.json"
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        report_text = (
            payload.get("records", [{}])[0]
            .get("input_snapshot", {})
            .get("shared_context", {})
            .get("research_report", {})
            .get("report_text", "")
        )
        if report_text:
            return str(report_text)
    return "Example report.\n" * 2000


def main() -> None:
    report_text = load_research_report_text()
    history = [
        {
            "hypothesis": f"Historical hypothesis {idx}",
            "feedback_summary": "Weak signal with noisy IC.",
            "feedback_reason": "Need parameter tuning before resetting thesis.",
            "feedback_hypothesis_supported": True,
            "feedback_replace_best_result": False,
            "feedback_new_hypothesis": "Tune the same proxy family conservatively.",
            "hypothesis_structured": {
                "hypothesis_evolution_state": {
                    "core_variables": ["price_momentum", "volume_change", "spread_regime"],
                    "verifiable_anchors": {
                        "input_features": ["$returns_24h", "$volume_change_24h", "$high_low_spread"],
                        "expression_patterns": ["TS_PCTCHANGE($close,*)"],
                    },
                }
            },
        }
        for idx in range(6)
    ]

    agent = HypothesisAgentV2("hypothesis_agent")
    private_context = AgentContext(payload={})
    shared_context = SharedContext(
        payload={
            "scenario": "panel_data_cross_section_factor_mining",
            "direction": "discover cross-sectional alpha factors with robust turnover control",
            "rag_text": "dataset summary",
            "hypothesis_feedback_history": history,
            "research_report": {
                "report_text": report_text,
                "report_type": "pdf_research_report",
                "source": {"file_path": "materials/20260329-dingbeili.pdf"},
                "dataset_features": [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volume_quote",
                    "returns_24h",
                    "volume_change_24h",
                    "high_low_spread",
                ],
                "validation_target": "fwd_returns_24h",
            },
            "research_stage": "parameter_tuning",
            "research_stage_reason": "Third-round retry simulation.",
            "hypothesis_structured": {
                "hypothesis_evolution_state": {
                    "core_variables": ["price_momentum", "volume_change", "spread_regime"],
                    "verifiable_anchors": {
                        "input_features": ["$returns_24h", "$volume_change_24h", "$high_low_spread"],
                        "expression_patterns": ["TS_HIGH_HIGH_SPREAD($high_low_spread,*)"],
                    },
                }
            },
        }
    )
    model_client = FakeModelClient(fail_above_chars=15000)
    result = agent.run(
        private_context=private_context,
        shared_context=shared_context,
        model_client=model_client,
    )
    print("result_keys=", sorted(result.shared_updates.keys()))
    print("calls=", json.dumps(model_client.calls, ensure_ascii=False, indent=2))
    print("diagnostics=", json.dumps(private_context.payload.get("hypothesis_diagnostics", []), ensure_ascii=False, indent=2))
    print(
        "expression_patterns=",
        result.shared_updates["hypothesis_structured"]["hypothesis_evolution_state"]["verifiable_anchors"]["expression_patterns"],
    )


if __name__ == "__main__":
    main()
