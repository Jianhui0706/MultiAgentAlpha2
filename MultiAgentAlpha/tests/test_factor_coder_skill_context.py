from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.factor_coder_agent import ExpressionImplementationBuilder
from trading_agents.agents.factor_coder_agent import ExpressionRuntimeEvaluator
from trading_agents.agents.factor_coder_agent import FactorCoderAgent
from trading_agents.agents.factor_coder_agent import FactorCoderEvaluationConfig
from trading_agents.agents.factor_coder_agent import ImplementationSelector
from trading_agents.agents.tools import ExpressionComplexityBudget
from trading_agents.agents.tools import FactorExpressionToolkit
from trading_agents.core import ModelClient
from trading_agents.schemas import AgentContext, SharedContext


class SequentialClient(ModelClient):
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[tuple[str, str, bool]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        self.prompts.append((system_prompt, user_prompt, json_mode))
        if not self.payloads:
            raise AssertionError("No payload left for model call")
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)


def test_factor_coder_injects_generation_metadata_and_targeted_retry_rules(monkeypatch) -> None:
    monkeypatch.setenv("FAST_MODE", "false")
    monkeypatch.setenv("FACTOR_CODER_LLM_EVAL_ENABLED", "true")

    client = SequentialClient(
        [
            "No comment found",
            {
                "final_decision": True,
                "final_feedback": "Expression accepted after runtime checks.",
            },
        ]
    )
    private_context = AgentContext()

    result = FactorCoderAgent("factor_coder_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "domain_dataset_summary": {
                    "feature_columns": ["close", "volume"],
                },
                "experiment_spec": {
                    "factors": [
                        {
                            "factor_name": "RollingMaxClose",
                            "description": "rolling max of close",
                            "formulation": "MAX($close, 5)",
                            "expression": "MAX($close, 5)",
                            "variables": {},
                        }
                    ]
                },
                "factor_coder_max_loop": 1,
                "factor_coder_fail_task_trial_limit": 1,
                "factor_coder_llm_eval_enabled": True,
            }
        ),
        model_client=client,
    )

    assert result.shared_updates["calculation_report"]["generated_factors"] == 1
    assert private_context.payload["factor_coder_generation_skill_context_summary"]["mode"] == "generation"
    assert "DELAY" in private_context.payload["factor_coder_generation_skill_context_summary"]["function_rule_names"]

    assert private_context.payload["last_factor_coder_skill_context_summary"]["mode"] == "retry"
    assert private_context.payload["last_factor_coder_skill_context_summary"]["function_rule_names"] == ["TS_MAX"]

    user_prompts = [prompt[1] for prompt in client.prompts]
    assert any("Project-local FactorCoder rule context" in prompt for prompt in user_prompts)
    assert any("Function Rule: TS_MAX" in prompt for prompt in user_prompts)
    assert all("Function Rule: DELAY" not in prompt for prompt in user_prompts)


def test_factor_coder_default_accepts_local_gate_without_llm(monkeypatch) -> None:
    monkeypatch.setenv("FAST_MODE", "false")
    monkeypatch.delenv("FACTOR_CODER_LLM_EVAL_ENABLED", raising=False)

    client = SequentialClient([])
    private_context = AgentContext()

    result = FactorCoderAgent("factor_coder_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "domain_dataset_summary": {
                    "feature_columns": ["close", "volume"],
                },
                "experiment_spec": {
                    "factors": [
                        {
                            "factor_name": "CloseRank",
                            "description": "rank close",
                            "formulation": "RANK($close)",
                            "expression": "RANK($close)",
                            "variables": {},
                        }
                    ]
                },
                "factor_coder_max_loop": 2,
            }
        ),
        model_client=client,
    )

    report = result.shared_updates["calculation_report"]
    assert report["generated_factors"] == 1
    assert report["max_loop"] == 2
    assert report["llm_eval_enabled"] is False
    assert report["diagnostics"]["llm_call_count"] == 0
    assert report["diagnostics"]["local_accept_count"] == 1
    assert client.prompts == []


def test_factor_coder_emits_fine_grained_progress_events(monkeypatch) -> None:
    monkeypatch.setenv("FAST_MODE", "false")
    monkeypatch.delenv("FACTOR_CODER_LLM_EVAL_ENABLED", raising=False)
    events: list[dict[str, object]] = []
    private_context = AgentContext(payload={"_progress_callback": events.append})

    result = FactorCoderAgent("factor_coder_agent").run(
        private_context=private_context,
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "domain_dataset_summary": {
                    "feature_columns": ["close"],
                },
                "experiment_spec": {
                    "factors": [
                        {
                            "factor_name": "CloseRank",
                            "description": "rank close",
                            "formulation": "RANK($close)",
                            "expression": "RANK($close)",
                            "variables": {},
                        }
                    ]
                },
                "factor_coder_max_loop": 1,
            }
        ),
        model_client=SequentialClient([]),
    )

    stages = [str(event.get("stage", "")) for event in events]
    assert result.shared_updates["calculation_report"]["generated_factors"] == 1
    assert "factor_coder_started" in stages
    assert "factor_coder_config_ready" in stages
    assert "factor_coder_factor_started" in stages
    assert "factor_coder_value_started" in stages
    assert "factor_coder_factor_finished" in stages
    assert "factor_coder_finished" in stages
    factor_event = next(event for event in events if event.get("stage") == "factor_coder_factor_started")
    assert factor_event["factor_name"] == "CloseRank"
    assert factor_event["factor_index"] == 1
    assert factor_event["factor_total"] == 1


def test_factor_coder_components_build_evaluate_and_select_local_accept() -> None:
    builder = ExpressionImplementationBuilder()
    evaluator = ExpressionRuntimeEvaluator(
        implementation_builder=builder,
        complexity_budget=ExpressionComplexityBudget(
            max_expression_length=180,
            max_function_count=8,
            max_nesting_depth=4,
            max_input_feature_count=3,
            max_operator_count=8,
            max_complexity_score=30,
        ),
        toolkit=FactorExpressionToolkit(),
    )
    implementation = builder.build_first_pass({"factor_name": "CloseRank", "expression": "RANK($close)"})

    result = evaluator.local_evaluate(
        factor={"factor_name": "CloseRank", "description": "rank close"},
        implementation=implementation,
        reference_expressions=[],
        panel_data=None,
        supported_feature_variables={"$close"},
        panel_data_path="",
        allowed_input_features={"$close"},
        allowed_runtime_functions={"RANK"},
    )

    selected = ImplementationSelector().select(
        local_result=result,
        config=FactorCoderEvaluationConfig(
            fast_mode=False,
            fast_path_enabled=False,
            llm_eval_enabled=False,
        ),
    )
    assert selected is result
    assert result["acceptable"] is True
    assert result["quality_gate"]["stage"] == "expression_quality_local_gate"


def test_factor_coder_rejects_duplicate_library_expression_in_local_gate() -> None:
    builder = ExpressionImplementationBuilder()
    evaluator = ExpressionRuntimeEvaluator(
        implementation_builder=builder,
        complexity_budget=ExpressionComplexityBudget(
            max_expression_length=180,
            max_function_count=8,
            max_nesting_depth=4,
            max_input_feature_count=3,
            max_operator_count=8,
            max_complexity_score=30,
        ),
        toolkit=FactorExpressionToolkit(),
    )
    implementation = builder.build_first_pass({"factor_name": "CloseRankCopy", "expression": "RANK($close)"})

    result = evaluator.local_evaluate(
        factor={"factor_name": "CloseRankCopy", "description": "rank close duplicate"},
        implementation=implementation,
        reference_expressions=[],
        library_expressions=[" RANK( $close ) "],
        panel_data=None,
        supported_feature_variables={"$close"},
        panel_data_path="",
        allowed_input_features={"$close"},
        allowed_runtime_functions={"RANK"},
    )

    assert result["acceptable"] is False
    assert result["quality_gate"]["stage"] == "factor_library_duplicate_hard_gate"
    assert result["quality_gate"]["passed"] is False
    assert "Duplicate factor expression rejected" in result["feedback"]["final_feedback"]


def test_factor_coder_run_filters_duplicate_library_expression(monkeypatch, tmp_path) -> None:
    library_path = tmp_path / "all_factors_library.json"
    library_path.write_text(
        json.dumps(
            {
                "metadata": {"created_at": "now", "last_updated": "now", "total_factors": 1, "version": "1.0"},
                "factors": {
                    "existing": {
                        "factor_id": "existing",
                        "factor_name": "ExistingCloseRank",
                        "factor_expression": "RANK($close)",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FACTOR_LIBRARY_PATH", str(library_path))
    monkeypatch.setenv("FAST_MODE", "false")
    monkeypatch.delenv("FACTOR_CODER_LLM_EVAL_ENABLED", raising=False)

    result = FactorCoderAgent("factor_coder_agent").run(
        private_context=AgentContext(),
        shared_context=SharedContext(
            payload={
                "scenario": "panel_data_cross_section_factor_mining",
                "domain_dataset_summary": {"feature_columns": ["close"]},
                "experiment_spec": {
                    "factors": [
                        {
                            "factor_name": "CloseRankCopy",
                            "description": "rank close duplicate",
                            "formulation": "RANK($close)",
                            "expression": "RANK($close)",
                            "variables": {},
                        }
                    ]
                },
                "factor_coder_max_loop": 1,
                "factor_coder_fail_task_trial_limit": 1,
            }
        ),
        model_client=SequentialClient([]),
    )

    report = result.shared_updates["calculation_report"]
    assert report["generated_factors"] == 0
    assert report["failed_factors"] == ["CloseRankCopy"]
    record = result.shared_updates["factor_implementation"]["quality_gate"]["records"][0]
    assert record["stage"] == "factor_library_duplicate_hard_gate"


def test_factor_coder_rejects_over_complex_expression_before_llm(monkeypatch) -> None:
    monkeypatch.setenv("FAST_MODE", "false")
    agent = FactorCoderAgent("factor_coder_agent")
    expression = (
        "RANK("
        "TS_ZSCORE($close,24)+TS_ZSCORE($volume,24)+TS_ZSCORE($open,24)+"
        "TS_ZSCORE($high,24)+TS_ZSCORE($low,24)+TS_ZSCORE($returns_24h,24)+"
        "TS_ZSCORE($volume_change_24h,24)"
        ")"
    )
    client = SequentialClient([])

    result = agent._evaluate_one_factor(
        factor={"factor_name": "TooComplex", "description": "too many inputs"},
        implementation={
            "factor_name": "TooComplex",
            "expression": expression,
            "code": agent._build_code(expression, "TooComplex"),
        },
        reference_expressions=[],
        prompts={},
        model_client=client,
        scenario="test",
        panel_data=None,
        supported_feature_variables=set(),
    )

    quality_gate = result["quality_gate"]
    assert result["acceptable"] is False
    assert quality_gate["stage"] == "expression_complexity_hard_gate"
    assert quality_gate["passed"] is False
    assert client.prompts == []
