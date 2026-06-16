from __future__ import annotations

import json

from trading_agents.agents.experiment_designer_agent import ExperimentDesignerAgent
from trading_agents.agents.hypothesis.hypothesis_agent import HypothesisAgentV2
from trading_agents.schemas import AgentContext, SharedContext


class SequencedModelClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        self.calls += 1
        if not self._responses:
            raise RuntimeError("No more fake responses configured.")
        return self._responses.pop(0)


def build_previous_hypothesis() -> dict[str, object]:
    previous = {
        "hypothesis": "Price-volume divergence predicts short-term reversal in crypto cross-section.",
        "thesis": "Price advances without volume quote confirmation tend to reverse.",
        "mechanism": "Weak participation breadth makes momentum unsustainable.",
        "expected_direction": "negative_forward_return",
        "market_structure_dependencies": ["session-based trading"],
        "crypto_adaptation_summary": {
            "transferable_core": "Price-volume divergence is transferable.",
            "non_transferable_parts": ["equity session structure"],
            "crypto_rewrite": "Use rolling price and volume quote momentum windows.",
        },
        "proxy_reconstruction_plan": {
            "input_features": ["$close", "$volume_quote", "$volume_change_24h"],
            "core_proxy_family": ["price_volume_divergence", "volume_confirmation_gap"],
            "supporting_proxy_family": ["volume_confirmation"],
            "validation_target": "fwd_returns_24h",
        },
        "concise_knowledge": "Price-volume divergence often signals reversal.",
        "concise_observation": "Previous loop used price and volume quote divergence.",
        "concise_justification": "Broad participation is required to sustain trends.",
        "concise_specification": "Use TS_PCTCHANGE on close and volume quote with short/medium windows.",
        "hypothesis_evolution_state": {
            "core_variables": ["price_momentum", "volume_quote_momentum", "divergence_gap"],
            "verifiable_anchors": {
                "input_features": ["$close", "$volume_quote", "$volume_change_24h"],
                "expression_patterns": [
                    "TS_PCTCHANGE($close,*)",
                    "TS_PCTCHANGE($volume_quote,*)",
                ],
            },
            "refinement_stage": "parameter_tuning",
            "allowed_mutation_types": [
                "window_adjustment",
                "threshold_adjustment",
                "normalization_change",
                "regime_filter",
                "signal_combination",
            ],
        },
    }
    return previous


def validate_hypothesis_continuity() -> dict[str, object]:
    previous = build_previous_hypothesis()
    drifted_response = json.dumps(
        {
            "hypothesis": "Momentum regime conflict across 1h/4h/24h windows predicts reversal.",
            "thesis": "Short-term acceleration with medium-term deceleration predicts exhaustion.",
            "mechanism": "Multi-horizon momentum conflict creates reversal.",
            "expected_direction": "negative_forward_return",
            "market_structure_dependencies": ["session-based trading"],
            "crypto_adaptation_summary": {
                "transferable_core": "Momentum conflict transfers.",
                "non_transferable_parts": ["equity sessions"],
                "crypto_rewrite": "Use multi-horizon returns.",
            },
            "proxy_reconstruction_plan": {
                "input_features": ["$returns_1h", "$returns_4h", "$returns_24h", "$volume_change_1h"],
                "core_proxy_family": ["momentum_regime_conflict", "multi_horizon_acceleration"],
                "supporting_proxy_family": ["volume_regime_shift"],
                "validation_target": "fwd_returns_24h",
            },
            "concise_knowledge": "Momentum conflict predicts exhaustion.",
            "concise_observation": "This intentionally drifts away from price-volume divergence.",
            "concise_justification": "Use multi-horizon returns.",
            "concise_specification": "Use returns_1h, returns_4h, returns_24h.",
            "hypothesis_evolution_state": {
                "core_variables": ["momentum_regime_conflict", "returns_1h", "returns_4h"],
                "verifiable_anchors": {
                    "input_features": ["$returns_1h", "$returns_4h", "$returns_24h"],
                    "expression_patterns": ["TS_PCTCHANGE($returns_4h,*)"],
                },
                "refinement_stage": "parameter_tuning",
                "allowed_mutation_types": ["window_adjustment", "threshold_adjustment"],
            },
        },
        ensure_ascii=False,
    )
    model_client = SequencedModelClient([drifted_response])
    agent = HypothesisAgentV2("hypothesis_agent")
    private_context = AgentContext(payload={})
    shared_context = SharedContext(
        payload={
            "scenario": "panel_data_cross_section_factor_mining",
            "direction": "report-driven crypto factor mining",
            "rag_text": "dataset summary",
            "research_report": {
                "report_text": "sample report text",
                "report_type": "pdf_research_report",
                "source": {"file_path": "materials/sample.pdf"},
                "dataset_features": ["close", "volume_quote", "volume_change_24h"],
                "validation_target": "fwd_returns_24h",
            },
            "research_stage": "parameter_tuning",
            "research_stage_reason": "Continue tuning same family.",
            "hypothesis_feedback_history": [
                {
                    "hypothesis": previous["hypothesis"],
                    "feedback_summary": "Weak but directionally aligned.",
                    "feedback_reason": "Keep tuning current family.",
                    "feedback_hypothesis_supported": False,
                    "feedback_replace_best_result": False,
                    "feedback_new_hypothesis": "",
                    "feedback_semantic_diagnosis": "parameter_not_tuned",
                    "hypothesis_structured": previous,
                }
            ],
            "hypothesis_structured": previous,
        }
    )
    result = agent.run(private_context=private_context, shared_context=shared_context, model_client=model_client)
    return {
        "result": result.shared_updates["hypothesis_structured"],
        "diagnostics": private_context.payload.get("hypothesis_diagnostics", []),
    }


def validate_experiment_continuity(hypothesis_structured: dict[str, object]) -> dict[str, object]:
    bad_payload = json.dumps(
        {
            "task_plan": ["Try a new proxy family."],
            "factors": [
                {
                    "factor_name": "drifted_factor",
                    "description": "Uses unrelated returns-only family.",
                    "formulation": "RANK($returns_1h)",
                    "expression": "RANK($returns_1h)",
                    "variables": {"$returns_1h": "short return"},
                },
                {
                    "factor_name": "drifted_factor_2",
                    "description": "Uses another unrelated returns-only family.",
                    "formulation": "RANK($returns_1h + 1e-8)",
                    "expression": "RANK($returns_1h + 1e-8)",
                    "variables": {"$returns_1h": "short return"},
                }
            ],
        },
        ensure_ascii=False,
    )
    good_payload = json.dumps(
        {
            "task_plan": ["Tune windows within the same price-volume divergence family."],
            "factors": [
                {
                    "factor_name": "pv_divergence_tuned",
                    "description": "Tuned price-volume divergence.",
                    "formulation": "RANK(TS_PCTCHANGE($close, 3) - TS_PCTCHANGE($volume_quote, 3))",
                    "expression": "RANK(TS_PCTCHANGE($close, 3) - TS_PCTCHANGE($volume_quote, 3))",
                    "variables": {
                        "$close": "price feature",
                        "$volume_quote": "quote volume feature",
                    },
                },
                {
                    "factor_name": "pv_divergence_volume_filter",
                    "description": "Price-volume divergence with lightweight volume filter.",
                    "formulation": "(TS_PCTCHANGE($close, 6) - TS_PCTCHANGE($volume_quote, 6)) / ($volume_change_24h + 1e-8)",
                    "expression": "(TS_PCTCHANGE($close, 6) - TS_PCTCHANGE($volume_quote, 6)) / ($volume_change_24h + 1e-8)",
                    "variables": {
                        "$close": "price feature",
                        "$volume_quote": "quote volume feature",
                        "$volume_change_24h": "volume filter",
                    },
                },
                {
                    "factor_name": "counterfactual_volume_lag_continuation",
                    "description": "Counterfactual using the same anchors: price leads volume in healthy continuation.",
                    "formulation": "RANK(TS_PCTCHANGE($close, 6)) * RANK(DELAY(TS_PCTCHANGE($volume_quote, 6), 2))",
                    "expression": "RANK(TS_PCTCHANGE($close, 6)) * RANK(DELAY(TS_PCTCHANGE($volume_quote, 6), 2))",
                    "variables": {
                        "$close": "price feature",
                        "$volume_quote": "quote volume feature",
                    },
                    "theory_alignment": "counterfactual",
                    "counterfactual_thesis": "Volume confirmation may lag price rather than refute the trend.",
                    "counterfactual_mechanism": "In healthy trends, price can move before volume catches up.",
                    "counterfactual_critique": "The main thesis may mistake normal confirmation delay for exhaustion.",
                    "counterfactual_expected_direction": "positive_forward_return",
                },
                {
                    "factor_name": "counterfactual_absorption_trend",
                    "description": "Counterfactual using volume-change anchor as informed absorption.",
                    "formulation": "RANK(TS_PCTCHANGE($close, 3)) * RANK($volume_change_24h)",
                    "expression": "RANK(TS_PCTCHANGE($close, 3)) * RANK($volume_change_24h)",
                    "variables": {
                        "$close": "price feature",
                        "$volume_change_24h": "volume change feature",
                    },
                    "theory_alignment": "counterfactual",
                    "counterfactual_thesis": "Volume pressure may represent informed absorption instead of exhaustion.",
                    "counterfactual_mechanism": "Informed flow absorbs supply and supports continuation.",
                    "counterfactual_critique": "The main thesis may misread absorption as distribution.",
                    "counterfactual_expected_direction": "positive_forward_return",
                },
            ],
        },
        ensure_ascii=False,
    )
    model_client = SequencedModelClient([bad_payload, good_payload, good_payload, good_payload])
    agent = ExperimentDesignerAgent("experiment_designer_agent")
    private_context = AgentContext(payload={})
    shared_context = SharedContext(
        payload={
            "scenario": "panel_data_cross_section_factor_mining",
            "hypothesis": str(hypothesis_structured.get("hypothesis", "")),
            "hypothesis_reasoning": str(hypothesis_structured.get("concise_justification", "")),
            "hypothesis_structured": hypothesis_structured,
            "research_report": {
                "report_text": "sample report text",
            },
            "domain_dataset_summary": {
                "feature_columns": [
                    "close",
                    "volume_quote",
                    "volume_change_24h",
                    "returns_1h",
                ]
            },
            "research_stage": "parameter_tuning",
            "research_stage_reason": "Continue tuning same family.",
            "hypothesis_feedback_history": [],
            "phase": "original",
            "round_idx": 2,
            "trajectory_id": "original_002_000",
            "parent_ids": [],
            "rag_text": "dataset summary",
        }
    )
    result = agent.run(private_context=private_context, shared_context=shared_context, model_client=model_client)
    return {
        "experiment_spec": result.shared_updates["experiment_spec"],
        "attempts": model_client.calls,
        "validation_feedback": private_context.payload.get("last_experiment_validation_feedback", []),
    }


def main() -> None:
    hypothesis_validation = validate_hypothesis_continuity()
    corrected_hypothesis = hypothesis_validation["result"]
    experiment_validation = validate_experiment_continuity(corrected_hypothesis)
    print("hypothesis_core_proxy_family=", corrected_hypothesis["proxy_reconstruction_plan"]["core_proxy_family"])
    print("hypothesis_continuity=", json.dumps(corrected_hypothesis.get("continuity_diagnostics", {}), ensure_ascii=False, indent=2))
    print("experiment_attempts=", experiment_validation["attempts"])
    print(
        "experiment_factor_names=",
        [item.get("factor_name") for item in experiment_validation["experiment_spec"]["factors"]],
    )
    print("experiment_feedback_tail=", json.dumps(experiment_validation["validation_feedback"][-2:], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
