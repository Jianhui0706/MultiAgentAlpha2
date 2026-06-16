from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.experiment_designer_agent import EvolutionNoveltyGate
from trading_agents.agents.experiment_designer_agent import ExperimentDesignerAgent
from trading_agents.agents.experiment_designer_agent import FactorSchemaNormalizer
from trading_agents.agents.tools import FactorExpressionToolkit
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
        "task_plan": ["generate one expression"],
        "factors": [
            {
                "factor_name": name,
                "description": "test factor",
                "formulation": expression,
                "expression": expression,
                "variables": {},
            }
        ],
    }


def _multi_factor_payload(items: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "task_plan": ["generate factor batch"],
        "factors": [
            {
                "factor_name": name,
                "description": "test factor",
                "formulation": expression,
                "expression": expression,
                "variables": {},
            }
            for name, expression in items
        ],
    }


def test_factor_schema_normalizer_extracts_list_and_rewrites_aliases() -> None:
    normalizer = FactorSchemaNormalizer(FactorExpressionToolkit())

    candidates = normalizer.extract_candidates(
        {
            "factors": [
                {
                    "factor_name": "RollingMaxClose",
                    "description": "rolling max",
                    "formulation": "MAX($close, 5)",
                    "expression": "MAX($close, 5)",
                    "variables": {},
                }
            ]
        }
    )

    assert candidates == [
        {
            "factor_name": "RollingMaxClose",
            "description": "rolling max",
            "formulation": "TS_MAX($close, 5)",
            "expression": "TS_MAX($close, 5)",
            "variables": {},
            "theory_alignment": "mainline",
            "counterfactual_reason": "",
            "counterfactual_thesis": "",
            "counterfactual_mechanism": "",
            "counterfactual_critique": "",
            "counterfactual_expected_direction": "",
        }
    ]


def test_evolution_novelty_gate_component_rejects_parent_copy() -> None:
    feedback, metrics = EvolutionNoveltyGate().check(
        phase="mutation",
        mutation_type="expression_mutation",
        expression="RANK($close)",
        expression_variables={"$close"},
        parent_expressions=["RANK($close)"],
        evolution_plan_summaries=[],
    )

    assert "identical" in feedback
    assert metrics["expression_similarity"] == 1.0


def test_experiment_designer_defaults_to_one_counterfactual_factor(monkeypatch) -> None:
    monkeypatch.delenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", raising=False)
    monkeypatch.delenv("COUNTERFACTUAL_FACTOR_COUNT", raising=False)

    agent = ExperimentDesignerAgent("experiment_designer_agent")

    assert agent._resolve_counterfactual_factor_count() == 1


def test_experiment_designer_injects_base_rules_then_targeted_retry_rules(monkeypatch) -> None:
    monkeypatch.setenv("LLM_VERIFIER_ENABLED", "false")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "1")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", "0")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_ADJACENT_FACTOR_COUNT", "0")

    client = SequentialClient(
        [
            _factor_payload("BadRollingMax", "MAX($volume_change_4h, 5)"),
            _factor_payload("GoodVolume1h", "RANK($volume_change_1h)"),
        ]
    )
    private_context = AgentContext()

    result = ExperimentDesignerAgent("experiment_designer_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "hypothesis": "volume stress should mean revert",
                "domain_dataset_summary": {
                    "feature_columns": ["close", "volume_change_1h", "volume_change_24h"],
                    "validation_target_column": "fwd_returns_4h",
                },
            }
        ),
        model_client=client,
    )

    assert result.shared_updates["experiment_spec"]["factors"][0]["expression"] == "RANK($volume_change_1h)"
    assert len(client.prompts) == 2

    first_user_prompt = client.prompts[0][1]
    second_user_prompt = client.prompts[1][1]

    assert "Project-local ExperimentDesigner rule context" in first_user_prompt
    assert "Function Rule: DELAY" in first_user_prompt
    assert "Function Rule: TS_MAX" in first_user_prompt

    assert "Project-local ExperimentDesigner rule context" in second_user_prompt
    assert "Function Rule: TS_MAX" in second_user_prompt
    assert "Function Rule: DELAY" not in second_user_prompt

    assert private_context.payload["retry_guidance_reason"] == (
        "targeted_retry_rules_from_previous_candidates_and_validation_feedback"
    )
    assert private_context.payload["last_experiment_skill_context_summary"]["mode"] == "retry"
    assert private_context.payload["last_experiment_skill_context_summary"]["function_rule_names"] == ["TS_MAX"]


def test_experiment_designer_uses_compact_report_summary_instead_of_raw_report_text(monkeypatch) -> None:
    monkeypatch.setenv("LLM_VERIFIER_ENABLED", "false")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "1")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", "0")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_ADJACENT_FACTOR_COUNT", "0")

    client = SequentialClient([_factor_payload("GoodCloseRank", "RANK($close)")])
    private_context = AgentContext()

    ExperimentDesignerAgent("experiment_designer_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "hypothesis": "price momentum divergence should mean revert",
                "domain_dataset_summary": {
                    "feature_columns": ["close", "volume", "high_low_spread"],
                    "validation_target_column": "fwd_returns_4h",
                },
                "research_report": {
                    "report_text": "VERY_LONG_REPORT_TEXT_SHOULD_NOT_BE_INJECTED " * 40,
                    "raw_report_text": "RAW_EXCERPT_SHOULD_NOT_APPEAR " * 40,
                    "source_reports": [
                        {
                            "thesis_summary": "Price extension without confirmation tends to mean revert.",
                            "mechanism_summary": "Weak participation after sharp price displacement signals exhaustion.",
                            "caveats_or_conflicts": [
                                "Single-signal usage is probabilistic and can have long-tail risk."
                            ],
                            "proxy_candidates": ["price_momentum", "volume_confirmation"],
                        }
                    ],
                },
                "hypothesis_structured": {
                    "thesis": "Price extension without confirmation tends to mean revert.",
                    "mechanism": "Weak participation after sharp price displacement signals exhaustion.",
                    "proxy_reconstruction_plan": {
                        "core_proxy_family": ["price_momentum", "volume_confirmation"],
                        "supporting_proxy_family": ["volatility_regime"],
                    },
                },
            }
        ),
        model_client=client,
    )

    system_prompt, user_prompt = client.prompts[0]
    combined_prompt = f"{system_prompt}\n{user_prompt}"

    assert "VERY_LONG_REPORT_TEXT_SHOULD_NOT_BE_INJECTED" not in combined_prompt
    assert "RAW_EXCERPT_SHOULD_NOT_APPEAR" not in combined_prompt
    assert "Price extension without confirmation tends to mean revert." in combined_prompt
    assert "Weak participation after sharp price displacement signals exhaustion." in combined_prompt
    assert "Single-signal usage is probabilistic and can have long-tail risk." in combined_prompt
    assert "price_momentum, volume_confirmation, volatility_regime" in combined_prompt
    assert private_context.payload["experiment_designer_prompt_compression"]["report_context_chars"] > 0


def test_experiment_designer_uses_batch_verifier_once_for_multiple_factors(monkeypatch) -> None:
    monkeypatch.delenv("FAST_MODE", raising=False)
    monkeypatch.setenv("LLM_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "2")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", "0")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_ADJACENT_FACTOR_COUNT", "0")

    client = SequentialClient(
        [
            _multi_factor_payload(
                [
                    ("CloseRank", "RANK($close)"),
                    ("VolumeRank", "RANK($volume)"),
                ]
            ),
            {
                "results": [
                    {
                        "factor_name": "CloseRank",
                        "passed": True,
                        "feedback": "ok",
                        "corrected_expression": "",
                    },
                    {
                        "factor_name": "VolumeRank",
                        "passed": True,
                        "feedback": "ok",
                        "corrected_expression": "",
                    },
                ]
            },
        ]
    )
    private_context = AgentContext()

    result = ExperimentDesignerAgent("experiment_designer_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "hypothesis": "simple cross-sectional ranks should be legal",
                "domain_dataset_summary": {
                    "feature_columns": ["close", "volume"],
                    "validation_target_column": "fwd_returns_4h",
                },
            }
        ),
        model_client=client,
    )

    assert len(client.prompts) == 2
    verifier_system_prompt, verifier_user_prompt = client.prompts[1]
    assert "batch LLM verifier" in verifier_system_prompt
    assert "\"factors\"" in verifier_user_prompt
    assert result.shared_updates["experiment_spec"]["consistency_gate"]["llm_call_count"] == 1
    assert result.shared_updates["experiment_spec"]["consistency_gate"]["quality_gate"]["mode"] == (
        "llm_batch_verifier_with_metric_advisory"
    )


def test_experiment_designer_compacts_duplication_feedback_in_retry_prompt(monkeypatch) -> None:
    monkeypatch.setenv("LLM_VERIFIER_ENABLED", "false")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "1")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", "0")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_ADJACENT_FACTOR_COUNT", "0")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_FEEDBACK_ITEM_LIMIT", "3")
    monkeypatch.setenv("EXPERIMENT_DESIGNER_FEEDBACK_CHAR_LIMIT", "500")

    client = SequentialClient(
        [
            _multi_factor_payload(
                [
                    ("BadOne", "RANK($volume_change_4h)"),
                    ("BadTwo", "RANK(TS_PERCENTILE($close, 10))"),
                ]
            ),
            _factor_payload("GoodTwo", "RANK($close)"),
        ]
    )
    private_context = AgentContext()

    ExperimentDesignerAgent("experiment_designer_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "hypothesis": "retry should only keep compact legality feedback",
                "domain_dataset_summary": {
                    "feature_columns": ["close"],
                    "validation_target_column": "fwd_returns_4h",
                },
            }
        ),
        model_client=client,
    )

    retry_user_prompt = client.prompts[1][1]
    assert "Unsupported dataset feature check failed for BadOne" in retry_user_prompt
    assert "TS_PERCENTILE" in retry_user_prompt
    assert len(retry_user_prompt) < 12000
