from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, StrictUndefined

from trading_agents.agents.tools import FactorExpressionToolkit
from trading_agents.agents.tools import ExperimentDesignerSkillLoader
from trading_agents.agents.tools import ExpressionComplexityBudget
from trading_agents.agents.tools import evaluate_expression_complexity
from trading_agents.agents.tools import get_supported_runtime_functions
from trading_agents.agents.tools import normalize_window_function_aliases
from trading_agents.agents.tools import validate_expression_symbols
from trading_agents.agents.tools.factor_expression_quality import evaluate_expression
from trading_agents.agents.tools.factor_expression_quality import is_expression_acceptable
from trading_agents.agents.tools.factor_expression_quality import is_parsable
from trading_agents.core import BaseAgent, ModelClient
from trading_agents.factor_runtime import calculate_factor_expression
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


@dataclass(frozen=True)
class FactorCoderFeedback:
    execution_feedback: str
    code_feedback: str
    value_feedback: str
    final_decision: bool
    final_feedback: str
    value_generated_flag: bool
    final_decision_based_on_gt: bool


@dataclass(frozen=True)
class FactorCoderEvaluationConfig:
    fast_mode: bool
    fast_path_enabled: bool
    llm_eval_enabled: bool


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class ExpressionImplementationBuilder:
    def build_code(self, expression: str, factor_name: str) -> str:
        escaped_expression = expression.replace("\\", "\\\\").replace('"', '\\"')
        escaped_factor_name = factor_name.replace("\\", "\\\\").replace('"', '\\"')
        return (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "def calculate_factor(df):\n"
            f"    expr = \"{escaped_expression}\"\n"
            f"    name = \"{escaped_factor_name}\"\n"
            "    return {\"factor_name\": name, \"expression\": expr}"
        )

    def extract_expr_from_code(self, code: str) -> str:
        for raw_line in code.splitlines():
            line = raw_line.strip()
            if not line.startswith("expr ="):
                continue
            value = line.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            if value.startswith("'") and value.endswith("'") and len(value) >= 2:
                return value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        return ""

    def build_first_pass(self, factor: dict[str, object]) -> dict[str, object]:
        factor_name = str(factor.get("factor_name", "")).strip()
        expression = normalize_window_function_aliases(str(factor.get("expression", "")).strip())
        return {
            "factor_name": factor_name,
            "expression": expression,
            "code": self.build_code(expression, factor_name),
            "status": "generated",
            "source": "template_first_pass",
        }


class ExpressionRuntimeEvaluator:
    def __init__(
        self,
        *,
        implementation_builder: ExpressionImplementationBuilder,
        complexity_budget: ExpressionComplexityBudget,
        toolkit: FactorExpressionToolkit,
    ) -> None:
        self.implementation_builder = implementation_builder
        self.complexity_budget = complexity_budget
        self.tools = toolkit

    @staticmethod
    def extract_expression_variables(expression: str) -> set[str]:
        return {match.group(0) for match in re.finditer(r"\$[A-Za-z_][A-Za-z0-9_]*", expression)}

    @staticmethod
    def supported_expression_variables(
        *,
        domain_dataset_summary: dict[str, object],
        panel_data: pd.DataFrame | None,
    ) -> set[str]:
        feature_columns = domain_dataset_summary.get("feature_columns", [])
        if isinstance(feature_columns, list) and feature_columns:
            return {f"${str(item).strip().lstrip('$')}" for item in feature_columns if str(item).strip()}
        if panel_data is not None:
            return {
                f"${str(column).strip().lstrip('$')}"
                for column in panel_data.columns
                if str(column).strip() and not str(column).startswith("fwd_")
            }
        return set()

    @staticmethod
    def unsupported_variable_feedback(
        *,
        factor_name: str,
        unsupported_variables: set[str],
        supported_variables: set[str],
    ) -> str:
        return (
            f"Unsupported dataset feature check failed for {factor_name}: expression uses "
            f"{sorted(unsupported_variables)}, but only these raw input features are available: "
            f"{sorted(supported_variables)}. Rewrite the expression using only available dataset columns."
        )

    @staticmethod
    def validate_runtime_symbols(
        expression: str,
        *,
        allowed_input_features: set[str],
        allowed_runtime_functions: set[str],
    ) -> tuple[bool, str]:
        validation = validate_expression_symbols(
            expression,
            allowed_features=allowed_input_features,
            allowed_functions=allowed_runtime_functions,
        )
        if validation.passed:
            return True, ""
        return False, validation.to_feedback()

    def evaluate_factor_values(
        self,
        *,
        expression: str,
        factor_name: str,
        panel_data: pd.DataFrame | None,
        panel_data_path: str = "",
    ) -> tuple[str, bool, dict[str, object]]:
        if panel_data is None:
            return "Value generation skipped in factor_coder stage because no panel data was provided.", False, {}
        try:
            factor_values = calculate_factor_expression(expression, panel_data, factor_name)
        except Exception as exc:
            return f"Factor value generation failed with runtime error: {exc}", False, {}
        if factor_values is None:
            return "Factor value generation returned None.", False, {}
        non_null_count = int(factor_values.notna().sum())
        if non_null_count <= 0:
            return "Factor value generation completed but produced zero non-null values.", False, {}
        null_ratio = float(factor_values.isna().mean())
        cache_info: dict[str, object] = {}
        try:
            cache_key = hashlib.md5(f"{panel_data_path}|{expression}".encode("utf-8")).hexdigest()
            cache_dir = Path(__file__).resolve().parents[3] / "factor_library" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{cache_key}.pkl"
            factor_values.to_pickle(cache_path)
            cache_info = {
                "signal_cache_key": cache_key,
                "source_pickle_path": str(cache_path),
                "signal_cache_status": "written",
            }
        except Exception as exc:
            cache_info = {
                "signal_cache_status": "write_failed",
                "signal_cache_error": str(exc),
            }
        return (
            f"Factor value generation succeeded with {non_null_count} non-null values; null_ratio={null_ratio:.6f}."
            + (f" cache_path={cache_info.get('source_pickle_path')}." if cache_info.get("source_pickle_path") else ""),
            True,
            cache_info,
        )

    def local_evaluate(
        self,
        *,
        factor: dict[str, object],
        implementation: dict[str, object],
        reference_expressions: list[str],
        panel_data: pd.DataFrame | None,
        supported_feature_variables: set[str],
        panel_data_path: str,
        allowed_input_features: set[str] | None,
        allowed_runtime_functions: set[str] | None,
    ) -> dict[str, object]:
        expression = normalize_window_function_aliases(str(implementation.get("expression", "")).strip())
        if expression != str(implementation.get("expression", "")).strip():
            implementation["expression"] = expression
            implementation["code"] = self.implementation_builder.build_code(expression, str(factor.get("factor_name", "")))

        if not expression:
            feedback = FactorCoderFeedback(
                execution_feedback="No factor expression generated.",
                code_feedback="No factor expression generated.",
                value_feedback="Factor value generation skipped because expression is empty.",
                final_decision=False,
                final_feedback="Expression is empty.",
                value_generated_flag=False,
                final_decision_based_on_gt=False,
            )
            return {
                "feedback": feedback.__dict__,
                "acceptable": False,
                "quality_gate": {
                    "enabled": True,
                    "stage": "missing_expression",
                    "passed": False,
                    "feedback": "Expression is empty.",
                    "metrics": {},
                },
                "cache_info": {},
            }

        expression_variables = self.extract_expression_variables(expression)
        if supported_feature_variables:
            unsupported_variables = expression_variables - supported_feature_variables
            if unsupported_variables:
                factor_name = str(factor.get("factor_name", ""))
                unsupported_feedback = self.unsupported_variable_feedback(
                    factor_name=factor_name,
                    unsupported_variables=unsupported_variables,
                    supported_variables=supported_feature_variables,
                )
                feedback = FactorCoderFeedback(
                    execution_feedback=unsupported_feedback,
                    code_feedback=unsupported_feedback,
                    value_feedback="Factor value generation skipped because expression uses unsupported dataset features.",
                    final_decision=False,
                    final_feedback=unsupported_feedback,
                    value_generated_flag=False,
                    final_decision_based_on_gt=False,
                )
                return {
                    "feedback": feedback.__dict__,
                    "acceptable": False,
                    "quality_gate": {
                        "enabled": True,
                        "stage": "dataset_feature_whitelist",
                        "passed": False,
                        "feedback": unsupported_feedback,
                        "metrics": {
                            "used_variables": sorted(expression_variables),
                            "unsupported_variables": sorted(unsupported_variables),
                            "supported_variables": sorted(supported_feature_variables),
                        },
                    },
                    "cache_info": {},
                }

        parsable, parse_feedback = is_parsable(expression)
        parsable_soft = False
        if not parsable:
            parsable_soft = ("$" in expression) and bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expression))
        if not parsable and not parsable_soft:
            feedback = FactorCoderFeedback(
                execution_feedback=parse_feedback,
                code_feedback=parse_feedback,
                value_feedback="Factor value generation skipped because expression parsing failed.",
                final_decision=False,
                final_feedback=f"Parse failed: {parse_feedback}",
                value_generated_flag=False,
                final_decision_based_on_gt=False,
            )
            return {
                "feedback": feedback.__dict__,
                "acceptable": False,
                "quality_gate": {
                    "enabled": True,
                    "stage": "parse",
                    "passed": False,
                    "feedback": parse_feedback,
                    "metrics": {},
                },
                "cache_info": {},
            }

        if allowed_input_features is not None and allowed_runtime_functions is not None:
            symbols_ok, symbols_feedback = self.validate_runtime_symbols(
                expression,
                allowed_input_features=allowed_input_features,
                allowed_runtime_functions=allowed_runtime_functions,
            )
            if not symbols_ok:
                feedback = FactorCoderFeedback(
                    execution_feedback=symbols_feedback,
                    code_feedback=symbols_feedback,
                    value_feedback="Factor value generation skipped because expression uses unsupported variables or functions.",
                    final_decision=False,
                    final_feedback=f"Unsupported symbols: {symbols_feedback}",
                    value_generated_flag=False,
                    final_decision_based_on_gt=False,
                )
                return {
                    "feedback": feedback.__dict__,
                    "acceptable": False,
                    "quality_gate": {
                        "enabled": True,
                        "stage": "runtime_symbol_whitelist",
                        "passed": False,
                        "feedback": symbols_feedback,
                        "metrics": {
                            "used_variables": sorted(expression_variables),
                            "allowed_input_features": sorted(allowed_input_features),
                        },
                    },
                    "cache_info": {},
                }

        eval_dict: dict[str, object] = {
            "duplicated_subtree_size": 0,
            "num_free_args": 0,
            "num_unique_vars": 0,
            "num_all_nodes": max(1, len(expression.split())),
            "symbol_length": len(expression),
            "num_base_features": expression.count("$"),
        }
        try:
            complexity_result = evaluate_expression_complexity(expression, budget=self.complexity_budget)
            complexity_gate = complexity_result.as_dict()
        except Exception as exc:
            complexity_gate = {
                "expression": expression,
                "passed": False,
                "violations": [f"expression_parse_or_complexity_check_failed: {exc}"],
            }
        if not bool(complexity_gate.get("passed", False)):
            violations = list(complexity_gate.get("violations", []) or [])
            feedback_text = (
                "Factor expression is too complex or invalid. "
                "Rewrite it as a shorter, parsable expression with length <= 300, "
                "input features <= 5, nesting depth <= 5, functions <= 12, operators <= 12. "
                f"Violations: {'; '.join(str(item) for item in violations)}"
            )
            feedback = FactorCoderFeedback(
                execution_feedback=feedback_text,
                code_feedback=feedback_text,
                value_feedback="Factor value generation skipped because expression failed hard complexity constraints.",
                final_decision=False,
                final_feedback=feedback_text,
                value_generated_flag=False,
                final_decision_based_on_gt=False,
            )
            return {
                "feedback": feedback.__dict__,
                "acceptable": False,
                "quality_gate": {
                    "enabled": True,
                    "stage": "expression_complexity_hard_gate",
                    "mode": "hard_reject",
                    "passed": False,
                    "feedback": feedback_text,
                    "metrics": dict(eval_dict),
                    "complexity": complexity_gate,
                },
                "cache_info": {},
            }

        try:
            eval_dict = evaluate_expression(expression, reference_expressions=reference_expressions)
        except Exception:
            pass
        threshold_meta = {
            "duplication_threshold": 6,
            "symbol_length_threshold": _env_int("FACTOR_MAX_EXPRESSION_LENGTH", 300),
            "base_features_threshold": _env_int("FACTOR_MAX_INPUT_FEATURE_COUNT", 5),
            "free_args_ratio_threshold": 1.0,
            "unique_vars_ratio_threshold": 1.0,
        }
        advisory_metric_passed = is_expression_acceptable(eval_dict, prompt_meta=threshold_meta)
        quality_feedback = (
            "Expression metrics are within advisory FactorCoder quality ranges."
            if advisory_metric_passed
            else self.tools.render_expression_feedback(eval_dict, threshold_meta, expression)
        )
        quality_gate = {
            "enabled": True,
            "stage": "expression_quality_local_gate",
            "mode": "local_accept_with_metric_advisory",
            "passed": True,
            "advisory_metric_passed": bool(advisory_metric_passed),
            "feedback": quality_feedback,
            "metrics": dict(eval_dict),
            "complexity": complexity_gate,
        }
        acceptable = True
        execution_feedback = "Template render successful."
        if not parsable and parsable_soft:
            execution_feedback = f"Parser fallback accepted expression: {parse_feedback}"
        value_feedback, value_generated_flag, cache_info = self.evaluate_factor_values(
            expression=expression,
            factor_name=str(factor.get("factor_name", "")),
            panel_data=panel_data,
            panel_data_path=panel_data_path,
        )
        if panel_data is not None and not value_generated_flag:
            acceptable = False
            quality_gate["passed"] = False
            quality_gate["feedback"] = f"{quality_gate['feedback']} Value generation failed: {value_feedback}"
        feedback = FactorCoderFeedback(
            execution_feedback=execution_feedback,
            code_feedback=(
                "Local gate: expression passed parse, symbol, complexity, and value checks."
                if acceptable
                else str(quality_gate["feedback"])
            ),
            value_feedback=value_feedback,
            final_decision=acceptable,
            final_feedback=(
                "Local gate accepted legal expression without LLM evaluation."
                if acceptable
                else "Expression rejected by local execution checks."
            ),
            value_generated_flag=value_generated_flag,
            final_decision_based_on_gt=False,
        )
        quality_gate["passed"] = bool(acceptable)
        quality_gate["local_final_decision"] = bool(acceptable)
        quality_gate["final_feedback"] = feedback.final_feedback
        return {
            "feedback": feedback.__dict__,
            "acceptable": acceptable,
            "quality_gate": quality_gate,
            "cache_info": cache_info,
            "advisory_metric_passed": advisory_metric_passed,
        }


class ImplementationSelector:
    def select(
        self,
        *,
        local_result: dict[str, object],
        config: FactorCoderEvaluationConfig,
    ) -> dict[str, object] | None:
        if not bool(local_result.get("acceptable", False)):
            return local_result
        if config.fast_mode or config.fast_path_enabled or not config.llm_eval_enabled:
            return local_result
        return None


class ExpressionRepairAgent:
    def __init__(self, *, implementation_builder: ExpressionImplementationBuilder) -> None:
        self.implementation_builder = implementation_builder


class FactorCoderAgent(BaseAgent):
    """第三步因子实现代理：复刻演化评估闭环并输出实现结果。"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_file = Path(__file__).resolve().parent.parent / "prompts" / "factor_coder_agent.yaml"
        self.tools = FactorExpressionToolkit()
        self.skill_loader = ExperimentDesignerSkillLoader()
        self.implementation_builder = ExpressionImplementationBuilder()
        self.expression_complexity_budget = ExpressionComplexityBudget(
            max_expression_length=_env_int("FACTOR_MAX_EXPRESSION_LENGTH", 300),
            max_function_count=_env_int("FACTOR_MAX_FUNCTION_COUNT", 12),
            max_nesting_depth=_env_int("FACTOR_MAX_NESTING_DEPTH", 5),
            max_input_feature_count=_env_int("FACTOR_MAX_INPUT_FEATURE_COUNT", 5),
            max_operator_count=_env_int("FACTOR_MAX_OPERATOR_COUNT", 12),
            max_complexity_score=float(_env_int("FACTOR_MAX_COMPLEXITY_SCORE", 45)),
        )
        self.runtime_evaluator = ExpressionRuntimeEvaluator(
            implementation_builder=self.implementation_builder,
            complexity_budget=self.expression_complexity_budget,
            toolkit=self.tools,
        )
        self.implementation_selector = ImplementationSelector()
        self.repair_agent = ExpressionRepairAgent(implementation_builder=self.implementation_builder)

    def _load_prompts(self) -> dict[str, Any]:
        with self.prompt_file.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        return loaded if isinstance(loaded, dict) else {}

    def _resolve_int_setting(
        self,
        *,
        shared_context: SharedContext,
        private_context: AgentContext,
        keys: list[str],
        default: int,
    ) -> int:
        for key in keys:
            for source in (shared_context.payload, private_context.payload):
                if key in source:
                    try:
                        return max(1, int(source.get(key)))
                    except Exception:
                        pass
            env_key = key.upper()
            raw_env = os.getenv(env_key, "").strip()
            if raw_env:
                try:
                    return max(1, int(raw_env))
                except ValueError:
                    pass
        return default

    def _resolve_bool_setting(
        self,
        *,
        shared_context: SharedContext,
        private_context: AgentContext,
        keys: list[str],
        default: bool,
    ) -> bool:
        truthy = {"1", "true", "yes", "y", "on"}
        falsy = {"0", "false", "no", "n", "off"}
        for key in keys:
            for source in (shared_context.payload, private_context.payload):
                if key in source:
                    raw = str(source.get(key, "")).strip().lower()
                    if raw in truthy:
                        return True
                    if raw in falsy:
                        return False
            env_key = key.upper()
            raw_env = os.getenv(env_key, "").strip().lower()
            if raw_env in truthy:
                return True
            if raw_env in falsy:
                return False
        return default

    def _render_prompt(self, template: str, context: dict[str, object]) -> str:
        return Environment(undefined=StrictUndefined).from_string(template).render(**context)

    def _build_skill_rule_context(
        self,
        *,
        mode: str,
        domain_dataset_summary: dict[str, object],
        expressions: list[str] | None = None,
        feedback_items: list[str] | None = None,
    ) -> tuple[str, dict[str, object]]:
        skill_context = self.skill_loader.build_skill_context(
            mode=mode,
            domain_dataset_summary=domain_dataset_summary,
            expressions=expressions or [],
            feedback_items=feedback_items or [],
        )
        return skill_context.text, dict(skill_context.metadata)

    def _append_rule_context(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        skill_rule_context: str,
        label: str,
    ) -> tuple[str, str]:
        appended_user_prompt = (
            f"{user_prompt}\n\nProject-local {label} rule context:\n{skill_rule_context}"
        ).strip()
        return system_prompt, appended_user_prompt

    def _normalize_factors(self, experiment_spec: dict[str, Any]) -> list[dict[str, object]]:
        factors: list[dict[str, object]] = []
        factors_block = experiment_spec.get("factors", [])
        if isinstance(factors_block, list):
            for item in factors_block:
                if not isinstance(item, dict):
                    continue
                factor_name = str(item.get("factor_name", "") or item.get("name", "")).strip()
                if not factor_name:
                    continue
                factors.append(
                    {
                        "factor_name": factor_name,
                        "description": str(item.get("description", "")),
                        "formulation": normalize_window_function_aliases(str(item.get("formulation", ""))),
                        "expression": normalize_window_function_aliases(str(item.get("expression", ""))),
                        "variables": item.get("variables", {}),
                        "validation_warnings": list(item.get("validation_warnings", []))
                        if isinstance(item.get("validation_warnings", []), list)
                        else [],
                        "retained_for_backtest": bool(item.get("retained_for_backtest", True)),
                        "theory_alignment": str(item.get("theory_alignment", "mainline") or "mainline"),
                        "counterfactual_reason": str(item.get("counterfactual_reason", "")),
                        "counterfactual_thesis": str(item.get("counterfactual_thesis", "")),
                        "counterfactual_mechanism": str(item.get("counterfactual_mechanism", "")),
                        "counterfactual_critique": str(item.get("counterfactual_critique", "")),
                        "counterfactual_anchor": dict(item.get("counterfactual_anchor", {}))
                        if isinstance(item.get("counterfactual_anchor", {}), dict)
                        else {},
                        "counterfactual_expected_direction": str(item.get("counterfactual_expected_direction", "")),
                        "matched_anchor_features": list(item.get("matched_anchor_features", []))
                        if isinstance(item.get("matched_anchor_features", []), list)
                        else [],
                        "matched_anchor_pattern": bool(item.get("matched_anchor_pattern", False)),
                        "expected_anchor_features": list(item.get("expected_anchor_features", []))
                        if isinstance(item.get("expected_anchor_features", []), list)
                        else [],
                        "expected_anchor_patterns": list(item.get("expected_anchor_patterns", []))
                        if isinstance(item.get("expected_anchor_patterns", []), list)
                        else [],
                        "anchor_fallback_reason": str(item.get("anchor_fallback_reason", "")),
                    }
                )
        if factors:
            return factors
        for key, value in experiment_spec.items():
            if not isinstance(value, dict):
                continue
            factor_name = str(key).strip()
            if not factor_name:
                continue
            if {"description", "formulation", "expression"} & set(value.keys()):
                factors.append(
                    {
                        "factor_name": factor_name,
                        "description": str(value.get("description", "")),
                        "formulation": normalize_window_function_aliases(str(value.get("formulation", ""))),
                        "expression": normalize_window_function_aliases(str(value.get("expression", ""))),
                        "variables": value.get("variables", {}),
                        "validation_warnings": list(value.get("validation_warnings", []))
                        if isinstance(value.get("validation_warnings", []), list)
                        else [],
                        "retained_for_backtest": bool(value.get("retained_for_backtest", True)),
                        "theory_alignment": str(value.get("theory_alignment", "mainline") or "mainline"),
                        "counterfactual_reason": str(value.get("counterfactual_reason", "")),
                        "counterfactual_thesis": str(value.get("counterfactual_thesis", "")),
                        "counterfactual_mechanism": str(value.get("counterfactual_mechanism", "")),
                        "counterfactual_critique": str(value.get("counterfactual_critique", "")),
                        "counterfactual_anchor": dict(value.get("counterfactual_anchor", {}))
                        if isinstance(value.get("counterfactual_anchor", {}), dict)
                        else {},
                        "counterfactual_expected_direction": str(value.get("counterfactual_expected_direction", "")),
                        "matched_anchor_features": list(value.get("matched_anchor_features", []))
                        if isinstance(value.get("matched_anchor_features", []), list)
                        else [],
                        "matched_anchor_pattern": bool(value.get("matched_anchor_pattern", False)),
                        "expected_anchor_features": list(value.get("expected_anchor_features", []))
                        if isinstance(value.get("expected_anchor_features", []), list)
                        else [],
                        "expected_anchor_patterns": list(value.get("expected_anchor_patterns", []))
                        if isinstance(value.get("expected_anchor_patterns", []), list)
                        else [],
                        "anchor_fallback_reason": str(value.get("anchor_fallback_reason", "")),
                    }
                )
        return factors

    def _build_code(self, expression: str, factor_name: str) -> str:
        return self.implementation_builder.build_code(expression, factor_name)

    def _extract_expr_from_code(self, code: str) -> str:
        return self.implementation_builder.extract_expr_from_code(code)

    def _extract_expression_variables(self, expression: str) -> set[str]:
        return self.runtime_evaluator.extract_expression_variables(expression)

    @staticmethod
    def _supported_expression_variables(
        *,
        domain_dataset_summary: dict[str, object],
        panel_data: pd.DataFrame | None,
    ) -> set[str]:
        feature_columns = domain_dataset_summary.get("feature_columns", [])
        if isinstance(feature_columns, list) and feature_columns:
            return {f"${str(item).strip().lstrip('$')}" for item in feature_columns if str(item).strip()}
        return ExpressionRuntimeEvaluator.supported_expression_variables(
            domain_dataset_summary=domain_dataset_summary,
            panel_data=panel_data,
        )

    @staticmethod
    def _unsupported_variable_feedback(
        *,
        factor_name: str,
        unsupported_variables: set[str],
        supported_variables: set[str],
    ) -> str:
        return ExpressionRuntimeEvaluator.unsupported_variable_feedback(
            factor_name=factor_name,
            unsupported_variables=unsupported_variables,
            supported_variables=supported_variables,
        )

    def _factor_information(self, factor: dict[str, object]) -> str:
        return (
            f"Factor name: {str(factor.get('factor_name', ''))}\n"
            f"Factor description: {str(factor.get('description', ''))}\n"
            f"Factor formulation: {normalize_window_function_aliases(str(factor.get('formulation', '')))}\n"
            f"Factor expression: {normalize_window_function_aliases(str(factor.get('expression', '')))}"
        )

    def _validate_runtime_symbols(
        self,
        expression: str,
        *,
        allowed_input_features: set[str],
        allowed_runtime_functions: set[str],
    ) -> tuple[bool, str]:
        return self.runtime_evaluator.validate_runtime_symbols(
            expression,
            allowed_input_features=allowed_input_features,
            allowed_runtime_functions=allowed_runtime_functions,
        )

    def _extract_previous_calculation_failures(self, recent_trace: list[dict[str, object]]) -> list[str]:
        failures: list[str] = []
        for record in recent_trace:
            if str(record.get("agent_name", "")) != self.name:
                continue
            output_snapshot = record.get("output_snapshot", {})
            if not isinstance(output_snapshot, dict):
                continue
            shared_updates = output_snapshot.get("shared_updates", {})
            if not isinstance(shared_updates, dict):
                continue
            calculation_report = shared_updates.get("calculation_report", {})
            if not isinstance(calculation_report, dict):
                continue
            failed = calculation_report.get("failed_factors", [])
            if isinstance(failed, list):
                failures.extend(str(item) for item in failed if str(item))
        return list(dict.fromkeys(failures))

    def _build_failure_summary(self, attempts: list[dict[str, object]]) -> str:
        if not attempts:
            return ""
        lines: list[str] = []
        for idx, item in enumerate(attempts, start=1):
            impl = item.get("implementation", {})
            fb = item.get("feedback", {})
            code = str(impl.get("code", ""))
            expr = self._extract_expr_from_code(code)
            lines.append(
                "\n".join(
                    [
                        f"Attempt {idx}",
                        f"Expression: {expr}",
                        f"Execution feedback: {str(fb.get('execution_feedback', ''))}",
                        f"Code feedback: {str(fb.get('code_feedback', ''))}",
                        f"Final feedback: {str(fb.get('final_feedback', ''))}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def _evaluate_factor_values(
        self,
        *,
        expression: str,
        factor_name: str,
        panel_data: pd.DataFrame | None,
        panel_data_path: str = "",
    ) -> tuple[str, bool, dict[str, object]]:
        return self.runtime_evaluator.evaluate_factor_values(
            expression=expression,
            factor_name=factor_name,
            panel_data=panel_data,
            panel_data_path=panel_data_path,
        )

    def _evaluate_one_factor(
        self,
        *,
        factor: dict[str, object],
        implementation: dict[str, object],
        reference_expressions: list[str],
        prompts: dict[str, Any],
        model_client: ModelClient,
        scenario: str,
        panel_data: pd.DataFrame | None,
        supported_feature_variables: set[str],
        panel_data_path: str = "",
        allowed_input_features: set[str] | None = None,
        allowed_runtime_functions: set[str] | None = None,
        fast_path_enabled: bool = False,
        llm_eval_enabled: bool = False,
        domain_dataset_summary: dict[str, object] | None = None,
        private_context: AgentContext | None = None,
    ) -> dict[str, object]:
        """Evaluate one implementation. Local gates are the default final decision path."""
        fast_mode = os.getenv("FAST_MODE", "false").lower() == "true"
        config = FactorCoderEvaluationConfig(
            fast_mode=fast_mode,
            fast_path_enabled=fast_path_enabled,
            llm_eval_enabled=llm_eval_enabled,
        )
        local_result = self.runtime_evaluator.local_evaluate(
            factor=factor,
            implementation=implementation,
            reference_expressions=reference_expressions,
            panel_data=panel_data,
            supported_feature_variables=supported_feature_variables,
            panel_data_path=panel_data_path,
            allowed_input_features=allowed_input_features,
            allowed_runtime_functions=allowed_runtime_functions,
        )
        selected = self.implementation_selector.select(local_result=local_result, config=config)
        if selected is not None:
            selected["llm_eval_call_count"] = 0
            selected["local_gate_count"] = 1
            selected["local_accept_count"] = 1 if bool(selected.get("acceptable", False)) else 0
            return selected

        expression = normalize_window_function_aliases(str(implementation.get("expression", "")).strip())
        code = str(implementation.get("code", "")).strip()
        quality_gate = dict(local_result.get("quality_gate", {}))
        local_feedback = dict(local_result.get("feedback", {}))
        execution_feedback = str(local_feedback.get("execution_feedback", "Template render successful."))
        value_feedback = str(local_feedback.get("value_feedback", ""))
        value_generated_flag = bool(local_feedback.get("value_generated_flag", False))
        cache_info = dict(local_result.get("cache_info", {}))
        acceptable = bool(local_result.get("acceptable", False))
        advisory_metric_passed = bool(local_result.get("advisory_metric_passed", False))
        code_feedback = ""
        llm_eval_call_count = 0
        code_feedback_system = str(prompts.get("evaluator_code_feedback_v1_system", "")).strip()
        code_feedback_user = str(prompts.get("evaluator_code_feedback_v1_user", "")).strip()
        eval_skill_rule_context, eval_skill_rule_metadata = self._build_skill_rule_context(
            mode="retry",
            domain_dataset_summary=domain_dataset_summary or {},
            expressions=[expression],
            feedback_items=[execution_feedback, value_feedback, str(quality_gate.get("feedback", ""))],
        )
        if code_feedback_system and code_feedback_user:
            try:
                system_prompt = self._render_prompt(
                    code_feedback_system,
                    {"scenario": scenario},
                )
                user_prompt = self._render_prompt(
                    code_feedback_user,
                    {
                        "factor_information": self._factor_information(factor),
                        "code": code,
                        "execution_feedback": execution_feedback,
                        "value_feedback": value_feedback,
                        "gt_code": None,
                    },
                )
                system_prompt, user_prompt = self._append_rule_context(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    skill_rule_context=eval_skill_rule_context,
                    label="FactorCoder",
                )
                if private_context is not None:
                    private_context.payload["last_factor_coder_skill_context_summary"] = {
                        "stage": "evaluator_code_feedback",
                        "mode": eval_skill_rule_metadata.get("mode", "retry"),
                        "feature_schema_count": eval_skill_rule_metadata.get("feature_schema_count", 0),
                        "supported_function_count": eval_skill_rule_metadata.get("supported_function_count", 0),
                        "function_rule_names": list(eval_skill_rule_metadata.get("function_rule_names", [])),
                    }
                    private_context.payload["factor_coder_skill_rule_files_used"] = list(
                        eval_skill_rule_metadata.get("function_rule_files", [])
                    )
                code_feedback = str(model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=False)).strip()
                llm_eval_call_count += 1
            except Exception:
                code_feedback = ""
        if not code_feedback:
            if acceptable:
                code_feedback = "No comment found" if advisory_metric_passed else str(quality_gate["feedback"])
            else:
                code_feedback = str(quality_gate["feedback"])
        final_decision = acceptable
        final_feedback = "Expression accepted by execution checks; metric signals are advisory." if acceptable else "Expression rejected by execution checks."
        decision_system = str(prompts.get("evaluator_final_decision_v1_system", "")).strip()
        decision_user = str(prompts.get("evaluator_final_decision_v1_user", "")).strip()
        if decision_system and decision_user and acceptable:
            try:
                system_prompt = self._render_prompt(decision_system, {"scenario": scenario})
                user_prompt = self._render_prompt(
                    decision_user,
                    {
                        "factor_information": self._factor_information(factor),
                        "execution_feedback": execution_feedback,
                        "code_feedback": code_feedback,
                        "value_feedback": value_feedback,
                    },
                )
                system_prompt, user_prompt = self._append_rule_context(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    skill_rule_context=eval_skill_rule_context,
                    label="FactorCoder",
                )
                if private_context is not None:
                    private_context.payload["last_factor_coder_skill_context_summary"] = {
                        "stage": "evaluator_final_decision",
                        "mode": eval_skill_rule_metadata.get("mode", "retry"),
                        "feature_schema_count": eval_skill_rule_metadata.get("feature_schema_count", 0),
                        "supported_function_count": eval_skill_rule_metadata.get("supported_function_count", 0),
                        "function_rule_names": list(eval_skill_rule_metadata.get("function_rule_names", [])),
                    }
                decision_raw = model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
                llm_eval_call_count += 1
                decision_payload = parse_json_object(decision_raw)
                parsed_decision = decision_payload.get("final_decision")
                if isinstance(parsed_decision, bool):
                    final_decision = parsed_decision
                parsed_feedback = str(decision_payload.get("final_feedback", "")).strip()
                if parsed_feedback:
                    final_feedback = parsed_feedback
            except Exception:
                pass
        feedback = FactorCoderFeedback(
            execution_feedback=execution_feedback,
            code_feedback=code_feedback,
            value_feedback=value_feedback,
            final_decision=final_decision,
            final_feedback=final_feedback,
            value_generated_flag=value_generated_flag,
            final_decision_based_on_gt=False,
        )
        quality_gate["passed"] = bool(final_decision)
        quality_gate["llm_final_decision"] = bool(final_decision)
        quality_gate["llm_eval_enabled"] = True
        quality_gate["final_feedback"] = final_feedback
        return {
            "feedback": feedback.__dict__,
            "acceptable": final_decision,
            "quality_gate": quality_gate,
            "cache_info": cache_info,
            "llm_eval_call_count": llm_eval_call_count,
            "local_gate_count": 1,
            "local_accept_count": 0,
        }

    def _evolve_one_factor(
        self,
        *,
        factor: dict[str, object],
        attempts: list[dict[str, object]],
        successful_examples: list[dict[str, str]],
        scenario: str,
        prompts: dict[str, Any],
        model_client: ModelClient,
        allowed_input_features: set[str] | None = None,
        allowed_runtime_functions: set[str] | None = None,
        domain_dataset_summary: dict[str, object] | None = None,
        private_context: AgentContext | None = None,
    ) -> dict[str, object]:
        factor_name = str(factor.get("factor_name", "")).strip()
        base_expression = normalize_window_function_aliases(str(factor.get("expression", "")).strip())
        if not attempts:
            return {
                "factor_name": factor_name,
                "expression": base_expression,
                "code": self._build_code(base_expression, factor_name),
                "status": "generated",
                "source": "template_first_pass",
            }
        former_attempt = attempts[-1]
        former_impl = former_attempt.get("implementation", {})
        former_feedback = former_attempt.get("feedback", {})
        former_expression = normalize_window_function_aliases(
            self._extract_expr_from_code(str(former_impl.get("code", ""))) or str(former_impl.get("expression", "")).strip()
        )
        system_template = str(prompts.get("evolving_strategy_factor_implementation_v1_system", "")).strip()
        user_template = str(prompts.get("evolving_strategy_factor_implementation_v2_user", "")).strip()
        if not system_template or not user_template:
            fallback_expression = former_expression or base_expression
            return {
                "factor_name": factor_name,
                "expression": fallback_expression,
                "code": self._build_code(fallback_expression, factor_name),
                "status": "generated",
                "source": "template_fallback",
            }
        similar_successful_factor_description = ""
        similar_successful_expression = ""
        if successful_examples:
            similar_successful_factor_description = successful_examples[-1]["description"]
            similar_successful_expression = successful_examples[-1]["expression"]
        latest_attempt_to_latest_successful_execution = {
            "implementation": {"code": str(former_impl.get("code", ""))},
            "feedback": str(former_feedback.get("final_feedback", "")),
        }
        evolve_skill_rule_context, evolve_skill_rule_metadata = self._build_skill_rule_context(
            mode="retry",
            domain_dataset_summary=domain_dataset_summary or {},
            expressions=[former_expression or base_expression],
            feedback_items=[str(former_feedback.get("final_feedback", ""))],
        )
        system_prompt = self._render_prompt(
            system_template,
            {
                "scenario": scenario,
            },
        )
        user_prompt = self._render_prompt(
            user_template,
            {
                "factor_information_str": str(factor.get("description", "")) or self._factor_information(factor),
                "former_expression": former_expression,
                "former_feedback": str(former_feedback.get("final_feedback", "")),
                "queried_similar_error_knowledge": [],
                "error_summary_critics": None,
                "similar_successful_factor_description": similar_successful_factor_description,
                "similar_successful_expression": similar_successful_expression,
                "latest_attempt_to_latest_successful_execution": latest_attempt_to_latest_successful_execution,
            },
        )
        system_prompt, user_prompt = self._append_rule_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            skill_rule_context=evolve_skill_rule_context,
            label="FactorCoder",
        )
        if private_context is not None:
            private_context.payload["last_factor_coder_skill_context_summary"] = {
                "stage": "evolve_retry",
                "mode": evolve_skill_rule_metadata.get("mode", "retry"),
                "feature_schema_count": evolve_skill_rule_metadata.get("feature_schema_count", 0),
                "supported_function_count": evolve_skill_rule_metadata.get("supported_function_count", 0),
                "function_rule_names": list(evolve_skill_rule_metadata.get("function_rule_names", [])),
            }
            private_context.payload["factor_coder_skill_rule_files_used"] = list(
                evolve_skill_rule_metadata.get("function_rule_files", [])
            )
        evolved_expression = former_expression or base_expression
        try:
            raw = model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
            payload = parse_json_object(raw)
            candidate = str(payload.get("expr", "")).strip()
            if candidate:
                evolved_expression = candidate
        except Exception:
            pass
        if allowed_input_features is not None and allowed_runtime_functions is not None:
            symbols_ok, _symbols_feedback = self._validate_runtime_symbols(
                evolved_expression,
                allowed_input_features=allowed_input_features,
                allowed_runtime_functions=allowed_runtime_functions,
            )
            if not symbols_ok:
                evolved_expression = base_expression or former_expression
                return {
                    "factor_name": factor_name,
                    "expression": evolved_expression,
                    "code": self._build_code(evolved_expression, factor_name),
                    "status": "generated",
                    "source": "evolve_retry_invalid_symbols_fallback_to_spec",
                }
        return {
            "factor_name": factor_name,
            "expression": evolved_expression,
            "code": self._build_code(evolved_expression, factor_name),
            "status": "generated",
            "source": "evolve_retry",
        }

    def _judge_should_stop(self, states: list[dict[str, object]], loop_index: int, max_loop: int) -> bool:
        if loop_index + 1 >= max_loop:
            return True
        pending = [state for state in states if not bool(state.get("final_decision")) and not bool(state.get("dropped"))]
        return len(pending) == 0

    def _judge_should_drop(self, state: dict[str, object], fail_task_trial_limit: int) -> bool:
        failed_attempts = int(state.get("failed_attempts", 0))
        return failed_attempts >= fail_task_trial_limit

    def _build_results(
        self,
        *,
        states: list[dict[str, object]],
        experiment_spec: dict[str, object],
        task_plan: list[object],
        max_loop: int,
        loop_used: int,
        fast_path_enabled: bool,
        llm_eval_enabled: bool,
        diagnostics: dict[str, int],
    ) -> tuple[dict[str, object], dict[str, object]]:
        normalized_implementations: list[dict[str, object]] = []
        failed_factors: list[str] = []
        for state in states:
            factor = state["factor"]
            factor_name = str(factor.get("factor_name", ""))
            implementation = state.get("final_implementation") or state.get("latest_implementation") or {}
            quality_gate = dict(state.get("last_quality_gate", {}))
            expression = normalize_window_function_aliases(str(implementation.get("expression", "")).strip())
            code = str(implementation.get("code", "")).strip()
            dropped = bool(state.get("dropped"))
            final_decision = bool(state.get("final_decision"))
            status = "generated" if final_decision and code and expression and not dropped else "failed"
            if dropped:
                status = "filtered_out"
            if not expression:
                status = "missing_expression"
            if expression and not code:
                status = "missing_code"
            if status != "generated":
                failed_factors.append(factor_name)
            normalized_implementations.append(
                {
                    "factor_name": factor_name,
                    "description": str(factor.get("description", "")),
                    "formulation": str(factor.get("formulation", "")),
                    "expression": expression,
                    "variables": factor.get("variables", {}),
                    "validation_warnings": list(factor.get("validation_warnings", []))
                    if isinstance(factor.get("validation_warnings", []), list)
                    else [],
                    "retained_for_backtest": bool(factor.get("retained_for_backtest", True)),
                    "theory_alignment": str(factor.get("theory_alignment", "mainline") or "mainline"),
                    "counterfactual_reason": str(factor.get("counterfactual_reason", "")),
                    "counterfactual_thesis": str(factor.get("counterfactual_thesis", "")),
                    "counterfactual_mechanism": str(factor.get("counterfactual_mechanism", "")),
                    "counterfactual_critique": str(factor.get("counterfactual_critique", "")),
                    "counterfactual_anchor": dict(factor.get("counterfactual_anchor", {}))
                    if isinstance(factor.get("counterfactual_anchor", {}), dict)
                    else {},
                    "counterfactual_expected_direction": str(factor.get("counterfactual_expected_direction", "")),
                    "matched_anchor_features": list(factor.get("matched_anchor_features", []))
                    if isinstance(factor.get("matched_anchor_features", []), list)
                    else [],
                    "matched_anchor_pattern": bool(factor.get("matched_anchor_pattern", False)),
                    "expected_anchor_features": list(factor.get("expected_anchor_features", []))
                    if isinstance(factor.get("expected_anchor_features", []), list)
                    else [],
                    "expected_anchor_patterns": list(factor.get("expected_anchor_patterns", []))
                    if isinstance(factor.get("expected_anchor_patterns", []), list)
                    else [],
                    "anchor_fallback_reason": str(factor.get("anchor_fallback_reason", "")),
                    "code": code,
                    "status": status,
                    "attempts": int(len(state.get("attempts", []))),
                    "feedback": dict(state.get("last_feedback", {})),
                    "quality_gate": quality_gate,
                    "source_pickle_path": str(implementation.get("source_pickle_path", "")),
                    "signal_cache_key": str(implementation.get("signal_cache_key", "")),
                    "signal_cache_status": str(implementation.get("signal_cache_status", "")),
                }
            )
        total = len(normalized_implementations)
        success_count = len([item for item in normalized_implementations if str(item.get("status", "")) == "generated"])
        if total == 0:
            status = "empty"
        elif success_count == total:
            status = "ok"
        else:
            status = "partial"
        factor_implementation = {
            "source": "factor_costeer_multiloop",
            "code": str(normalized_implementations[0].get("code", "")) if normalized_implementations else "",
            "implementations": normalized_implementations,
            "implementation_count": total,
            "spec": experiment_spec,
            "task_plan": task_plan,
            "max_loop": max_loop,
            "loop_used": loop_used,
            "quality_gate": {
                "enabled": True,
                "status": "passed" if not failed_factors else "partial",
                "records": [dict(item.get("quality_gate", {})) for item in normalized_implementations],
            },
            "diagnostics": dict(diagnostics),
        }
        calculation_report = {
            "status": status,
            "total_factors": total,
            "generated_factors": success_count,
            "failed_factors": failed_factors,
            "execution_mode": "evolve_evaluate_loop",
            "max_loop": max_loop,
            "loop_used": loop_used,
            "quality_gate": factor_implementation["quality_gate"],
            "fast_path_enabled": fast_path_enabled,
            "llm_eval_enabled": llm_eval_enabled,
            "diagnostics": dict(diagnostics),
            "signal_cache_mode": "pickle_per_expression_panel",
            "signal_cache_hits": len(
                [
                    item
                    for item in normalized_implementations
                    if str(item.get("status", "")) == "generated" and str(item.get("source_pickle_path", "")).strip()
                ]
            ),
        }
        return factor_implementation, calculation_report

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        raw_experiment_spec = shared_context.payload.get("experiment_spec", {})
        experiment_spec = dict(raw_experiment_spec) if isinstance(raw_experiment_spec, dict) else experiment.to_experiment_spec()
        if not experiment_spec.get("factors"):
            experiment_spec.update(experiment.to_experiment_spec())
        task_plan = list(experiment.task_plan)
        scenario = str(shared_context.payload.get("scenario", ""))
        domain_dataset_summary = dict(shared_context.payload.get("domain_dataset_summary", {}))
        panel_data_path = str(domain_dataset_summary.get("path", "")).strip()
        allowed_input_features = {
            f"${str(item).strip()}"
            for item in list(domain_dataset_summary.get("feature_columns", []))
            if str(item).strip()
        }
        allowed_runtime_functions = get_supported_runtime_functions()
        panel_data: pd.DataFrame | None = None
        if panel_data_path:
            try:
                panel_data = pd.read_parquet(panel_data_path)
            except Exception:
                panel_data = None
        supported_feature_variables = self._supported_expression_variables(
            domain_dataset_summary=domain_dataset_summary,
            panel_data=panel_data,
        )
        factors = self._normalize_factors(experiment_spec)
        generation_skill_rule_context, generation_skill_rule_metadata = self._build_skill_rule_context(
            mode="generation",
            domain_dataset_summary=domain_dataset_summary,
            expressions=[str(item.get("expression", "")) for item in factors],
            feedback_items=[],
        )
        private_context.payload["factor_coder_generation_skill_context_summary"] = {
            "stage": "generation",
            "mode": generation_skill_rule_metadata.get("mode", "generation"),
            "feature_schema_count": generation_skill_rule_metadata.get("feature_schema_count", 0),
            "supported_function_count": generation_skill_rule_metadata.get("supported_function_count", 0),
            "function_rule_names": list(generation_skill_rule_metadata.get("function_rule_names", [])),
        }
        private_context.payload["factor_coder_generation_skill_rule_files_used"] = list(
            generation_skill_rule_metadata.get("function_rule_files", [])
        )
        recent_trace = list(private_context.payload.get("recent_trace", []))
        prompts = self._load_prompts()
        max_loop = self._resolve_int_setting(
            shared_context=shared_context,
            private_context=private_context,
            keys=["factor_coder_max_loop"],
            default=10,
        )
        fail_task_trial_limit = self._resolve_int_setting(
            shared_context=shared_context,
            private_context=private_context,
            keys=["factor_coder_fail_task_trial_limit", "fail_task_trial_limit"],
            default=20,
        )
        fast_path_enabled = self._resolve_bool_setting(
            shared_context=shared_context,
            private_context=private_context,
            keys=["factor_coder_fast_path_enabled"],
            default=False,
        )
        llm_eval_enabled = self._resolve_bool_setting(
            shared_context=shared_context,
            private_context=private_context,
            keys=["factor_coder_llm_eval_enabled"],
            default=False,
        )
        previous_failed_names = self._extract_previous_calculation_failures(recent_trace)
        states: list[dict[str, object]] = []
        for factor in factors:
            states.append(
                {
                    "factor": factor,
                    "attempts": [],
                    "failed_attempts": 0,
                    "final_decision": False,
                    "dropped": False,
                    "latest_implementation": {},
                    "final_implementation": {},
                    "last_feedback": {},
                    "last_quality_gate": {},
                }
            )
        successful_examples: list[dict[str, str]] = []
        loop_trace: list[dict[str, object]] = []
        diagnostics = {
            "llm_call_count": 0,
            "repair_call_count": 0,
            "llm_eval_call_count": 0,
            "local_gate_count": 0,
            "local_accept_count": 0,
        }
        loop_used = 0
        for loop_index in range(max_loop):
            loop_used = loop_index + 1
            loop_record: dict[str, object] = {
                "loop_index": loop_index,
                "attempted_factors": [],
                "accepted_factors": [],
                "failed_factors": [],
                "dropped_factors": [],
                "llm_call_count": 0,
                "repair_call_count": 0,
                "llm_eval_call_count": 0,
                "local_gate_count": 0,
                "local_accept_count": 0,
            }
            for state in states:
                if state["final_decision"] or state["dropped"]:
                    continue
                factor = state["factor"]
                implementation = self._evolve_one_factor(
                    factor=factor,
                    attempts=list(state["attempts"]),
                    successful_examples=successful_examples,
                    scenario=scenario,
                    prompts=prompts,
                    model_client=model_client,
                    allowed_input_features=allowed_input_features,
                    allowed_runtime_functions=allowed_runtime_functions,
                    domain_dataset_summary=domain_dataset_summary,
                    private_context=private_context,
                )
                if state["attempts"]:
                    diagnostics["repair_call_count"] += 1
                    diagnostics["llm_call_count"] += 1
                    loop_record["repair_call_count"] = int(loop_record["repair_call_count"]) + 1
                    loop_record["llm_call_count"] = int(loop_record["llm_call_count"]) + 1
                state["latest_implementation"] = implementation
                evaluate_result = self._evaluate_one_factor(
                    factor=factor,
                    implementation=implementation,
                    reference_expressions=[item["expression"] for item in successful_examples],
                    prompts=prompts,
                    model_client=model_client,
                    scenario=scenario,
                    panel_data=panel_data,
                    supported_feature_variables=supported_feature_variables,
                    panel_data_path=panel_data_path,
                    allowed_input_features=allowed_input_features,
                    allowed_runtime_functions=allowed_runtime_functions,
                    fast_path_enabled=fast_path_enabled,
                    llm_eval_enabled=llm_eval_enabled,
                    domain_dataset_summary=domain_dataset_summary,
                    private_context=private_context,
                )
                llm_eval_call_count = int(evaluate_result.get("llm_eval_call_count", 0) or 0)
                local_gate_count = int(evaluate_result.get("local_gate_count", 0) or 0)
                local_accept_count = int(evaluate_result.get("local_accept_count", 0) or 0)
                diagnostics["llm_eval_call_count"] += llm_eval_call_count
                diagnostics["llm_call_count"] += llm_eval_call_count
                diagnostics["local_gate_count"] += local_gate_count
                diagnostics["local_accept_count"] += local_accept_count
                loop_record["llm_eval_call_count"] = int(loop_record["llm_eval_call_count"]) + llm_eval_call_count
                loop_record["llm_call_count"] = int(loop_record["llm_call_count"]) + llm_eval_call_count
                loop_record["local_gate_count"] = int(loop_record["local_gate_count"]) + local_gate_count
                loop_record["local_accept_count"] = int(loop_record["local_accept_count"]) + local_accept_count
                feedback = dict(evaluate_result["feedback"])
                quality_gate = dict(evaluate_result.get("quality_gate", {}))
                acceptable = bool(evaluate_result["acceptable"])
                cache_info = dict(evaluate_result.get("cache_info", {}))
                if cache_info:
                    implementation.update(cache_info)
                attempt = {
                    "implementation": implementation,
                    "feedback": feedback,
                    "quality_gate": quality_gate,
                }
                state["attempts"].append(attempt)
                state["last_feedback"] = feedback
                state["last_quality_gate"] = quality_gate
                loop_record["attempted_factors"].append(str(factor.get("factor_name", "")))
                if acceptable:
                    state["final_decision"] = True
                    state["final_implementation"] = implementation
                    loop_record["accepted_factors"].append(str(factor.get("factor_name", "")))
                    successful_examples.append(
                        {
                            "description": str(factor.get("description", "")),
                            "expression": str(implementation.get("expression", "")),
                        }
                    )
                else:
                    state["failed_attempts"] = int(state["failed_attempts"]) + 1
                    loop_record["failed_factors"].append(str(factor.get("factor_name", "")))
                    if self._judge_should_drop(state, fail_task_trial_limit):
                        state["dropped"] = True
                        loop_record["dropped_factors"].append(str(factor.get("factor_name", "")))
            loop_trace.append(loop_record)
            if self._judge_should_stop(states, loop_index, max_loop):
                break
        factor_implementation, calculation_report = self._build_results(
            states=states,
            experiment_spec=experiment_spec,
            task_plan=task_plan,
            max_loop=max_loop,
            loop_used=loop_used,
            fast_path_enabled=fast_path_enabled,
            llm_eval_enabled=llm_eval_enabled,
            diagnostics=diagnostics,
        )
        experiment.factor_implementation = dict(factor_implementation)
        experiment.calculation_report = dict(calculation_report)
        experiment.extra["factor_coder_quality_gate"] = dict(factor_implementation.get("quality_gate", {}))
        experiment.update_timestamps()
        private_context.payload["latest_factor_implementation"] = factor_implementation
        private_context.payload["latest_experiment"] = experiment.to_dict()
        private_context.payload["factor_coder_loop_trace"] = loop_trace
        private_context.payload["factor_coder_diagnostics"] = dict(diagnostics)
        private_context.payload["factor_coder_previous_failed_names"] = previous_failed_names
        private_context.payload["factor_coder_state_snapshot"] = [
            {
                "factor_name": str(state["factor"].get("factor_name", "")),
                "final_decision": bool(state["final_decision"]),
                "dropped": bool(state["dropped"]),
                "failed_attempts": int(state["failed_attempts"]),
                "attempt_count": len(state["attempts"]),
                "last_feedback": dict(state["last_feedback"]),
                "failure_summary": self._build_failure_summary(list(state["attempts"])),
            }
            for state in states
        ]
        return AgentResult(
            shared_updates={
                "factor_implementation": factor_implementation,
                "calculation_report": calculation_report,
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name, "loop_trace_length": len(loop_trace)},
        )
