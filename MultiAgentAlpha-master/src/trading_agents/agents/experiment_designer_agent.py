from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from trading_agents.agents.tools import FactorExpressionToolkit
from trading_agents.agents.tools import ExperimentDesignerSkillLoader
from trading_agents.agents.tools import GenerationGate
from trading_agents.agents.tools import get_supported_runtime_functions
from trading_agents.agents.tools import normalize_window_function_aliases
from trading_agents.agents.tools import validate_expression_symbols
from trading_agents.agents.tools.factor_expression_quality import evaluate_expression
from trading_agents.agents.tools.factor_expression_quality import is_expression_acceptable
from trading_agents.agents.tools.factor_expression_quality import is_parsable
from trading_agents.core import BaseAgent, ModelClient
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.schemas import AgentContext, AgentResult, ExperimentFactorTask, QlibFactorExperiment, SharedContext


@dataclass(frozen=True)
class ExperimentDesignerPromptBundle:
    hypothesis_and_feedback: str
    function_lib_description: str
    factor_experiment_output_format: str
    experiment_designer_system_prompt: str
    experiment_designer_user_prompt: str
    expression_duplication: str


class ExperimentContextBuilder:
    def compact_function_lib_description(self, raw_description: str) -> str:
        return (
            "Use only runtime-supported functions and dataset features. "
            "Detailed function semantics, alias-confusion rules, and retry-targeted function guidance "
            "are provided by the project-local ExperimentDesigner rule context."
        )

    def compact_output_format(self) -> str:
        return (
            "Return only one strict JSON object. "
            "Use `factors` as a JSON array and follow the project-local output contract for the exact schema."
        )


class FactorProposalAgent:
    def generate(
        self,
        *,
        model_client: ModelClient,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)


class FactorSchemaNormalizer:
    def __init__(self, toolkit: FactorExpressionToolkit) -> None:
        self.toolkit = toolkit

    def extract_candidates(self, payload: dict[str, Any]) -> list[dict[str, object]]:
        candidates = self.toolkit.extract_candidates(payload)
        normalized: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for item in candidates:
            factor_name = str(item.get("factor_name", "") or item.get("name", "")).strip()
            if not factor_name:
                continue
            expression = normalize_window_function_aliases(str(item.get("expression", "")).strip())
            formulation = normalize_window_function_aliases(str(item.get("formulation", "") or expression).strip())
            key = (factor_name.lower(), "".join(expression.split()).lower())
            if key in seen:
                continue
            seen.add(key)
            theory_alignment = str(item.get("theory_alignment", "mainline") or "mainline").strip().lower()
            if theory_alignment not in {"mainline", "counterfactual", "adjacent"}:
                theory_alignment = "mainline"
            normalized.append(
                {
                    "factor_name": factor_name,
                    "description": str(item.get("description", "")),
                    "formulation": formulation,
                    "expression": expression,
                    "variables": item.get("variables", {}) if isinstance(item.get("variables", {}), dict) else {},
                    "theory_alignment": theory_alignment,
                    "counterfactual_reason": str(item.get("counterfactual_reason", "")),
                    "counterfactual_thesis": str(item.get("counterfactual_thesis", "")),
                    "counterfactual_mechanism": str(item.get("counterfactual_mechanism", "")),
                    "counterfactual_critique": str(item.get("counterfactual_critique", "")),
                    "counterfactual_expected_direction": str(item.get("counterfactual_expected_direction", "")),
                }
            )
        return normalized


class ExpressionQualityGate:
    def validate_symbols(
        self,
        expression: str,
        *,
        allowed_features: set[str],
        allowed_functions: set[str] | None = None,
        allowed_runtime_functions: set[str] | None = None,
    ):
        return validate_expression_symbols(
            expression,
            allowed_features=allowed_features,
            allowed_functions=allowed_functions or allowed_runtime_functions,
        )


class EvolutionNoveltyGate:
    @staticmethod
    def normalized_expression_key(expression: str) -> str:
        return "".join(str(expression or "").split()).lower()

    @staticmethod
    def expression_function_tokens(expression: str) -> list[str]:
        return [match.group(1).upper() for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)]

    @staticmethod
    def expression_numeric_tokens(expression: str) -> list[str]:
        return re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", expression, flags=re.IGNORECASE)

    @staticmethod
    def extract_expression_variables(expression: str) -> set[str]:
        return {match.group(0) for match in re.finditer(r"\$[A-Za-z_][A-Za-z0-9_]*", expression)}

    def check(
        self,
        *,
        phase: str,
        mutation_type: str,
        expression: str,
        expression_variables: set[str],
        parent_expressions: list[str],
        evolution_plan_summaries: list[dict[str, object]],
    ) -> tuple[str, dict[str, object]]:
        normalized_phase = str(phase or "").strip().lower()
        normalized_mutation_type = str(mutation_type or "").strip().lower()
        planned_phases = {str(item.get("phase", "")).strip().lower() for item in evolution_plan_summaries if isinstance(item, dict)}
        effective_phase = normalized_phase
        if normalized_phase == "evolved" and planned_phases:
            effective_phase = "crossover" if "crossover" in planned_phases else "mutation"
        if effective_phase not in {"mutation", "crossover"} or not parent_expressions:
            return "", {}

        expression_key = self.normalized_expression_key(expression)
        expression_functions = self.expression_function_tokens(expression)
        expression_numbers = self.expression_numeric_tokens(expression)
        best_similarity = 0.0
        closest_parent = ""
        closest_parent_functions: list[str] = []
        closest_parent_numbers: list[str] = []
        for parent_expression in parent_expressions:
            parent_key = self.normalized_expression_key(parent_expression)
            if expression_key == parent_key:
                return (
                    "Evolution novelty violation: child expression is identical to a parent expression.",
                    {
                        "closest_parent_expression": parent_expression,
                        "expression_similarity": 1.0,
                    },
                )
            parent_variables = self.extract_expression_variables(parent_expression)
            union = expression_variables | parent_variables
            variable_similarity = len(expression_variables & parent_variables) / len(union) if union else 0.0
            parent_functions = self.expression_function_tokens(parent_expression)
            function_similarity = (
                len(set(expression_functions) & set(parent_functions)) / len(set(expression_functions) | set(parent_functions))
                if expression_functions or parent_functions
                else 0.0
            )
            similarity = (variable_similarity + function_similarity) / 2.0
            if similarity > best_similarity:
                best_similarity = similarity
                closest_parent = parent_expression
                closest_parent_functions = parent_functions
                closest_parent_numbers = self.expression_numeric_tokens(parent_expression)

        metrics = {
            "closest_parent_expression": closest_parent,
            "expression_similarity": round(best_similarity, 6),
            "effective_phase": effective_phase,
            "planned_phases": sorted(planned_phases),
            "child_functions": expression_functions,
            "parent_functions": closest_parent_functions,
            "child_numeric_tokens": expression_numbers,
            "parent_numeric_tokens": closest_parent_numbers,
        }
        if normalized_mutation_type == "parameter_mutation":
            if expression_numbers == closest_parent_numbers:
                return (
                    "Evolution novelty violation: parameter_mutation must change at least one numeric window, threshold, or smoothing parameter.",
                    metrics,
                )
            if expression_functions != closest_parent_functions:
                return (
                    "Evolution novelty violation: parameter_mutation changed operator structure; keep the parent structure and tune parameters only.",
                    metrics,
                )
        elif normalized_mutation_type == "expression_mutation":
            if expression_functions == closest_parent_functions and best_similarity >= 0.85:
                return (
                    "Evolution novelty violation: expression_mutation must alter the operator composition, not only rename or lightly retune the parent.",
                    metrics,
                )
        elif normalized_mutation_type == "risk_mutation":
            risk_markers = {"TS_MEAN", "TS_MEDIAN", "TS_STD", "TS_ZSCORE", "RANK", "EMA", "WMA", "DECAYLINEAR", "REGRESI", "REGBETA"}
            added_markers = set(expression_functions) - set(closest_parent_functions)
            if not (added_markers & risk_markers):
                return (
                    "Evolution novelty violation: risk_mutation must add a new stability, smoothing, volatility, or residualization operator.",
                    metrics,
                )

        if effective_phase == "crossover" or "crossover" in planned_phases:
            parent_hits = 0
            for parent_expression in parent_expressions:
                parent_variables = self.extract_expression_variables(parent_expression)
                if expression_variables & parent_variables:
                    parent_hits += 1
            metrics["parent_hit_count"] = parent_hits
            if parent_hits < min(2, len(parent_expressions)):
                return (
                    "Evolution novelty violation: crossover-derived expression must preserve identifiable anchors from at least two parents.",
                    metrics,
                )
            if sum(expression.count(op) for op in ("+", "*")) > 5:
                return (
                    "Evolution novelty violation: crossover expression looks like mechanical stacking; keep one coherent simple signal.",
                    metrics,
                )
        elif normalized_mutation_type != "parameter_mutation" and best_similarity >= 0.98:
            return (
                "Evolution novelty violation: child expression is too similar to its closest parent; change structure or parameters materially.",
                metrics,
            )
        return "", metrics


class FactorRepairAgent:
    pass


class ExperimentDesignerAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        """初始化实验设计代理，仅保留提示词路径与工具集。

        约束（可用函数/变量）由 prompt 文本定义，不在代码中再维护默认白名单。
        """
        super().__init__(name)
        self.prompt_file = Path(__file__).resolve().parent.parent / "prompts" / "experiment_designer_agent.yaml"
        self.tools = FactorExpressionToolkit()
        self.skill_loader = ExperimentDesignerSkillLoader()
        self.context_builder = ExperimentContextBuilder()
        self.proposal_agent = FactorProposalAgent()
        self.schema_normalizer = FactorSchemaNormalizer(self.tools)
        self.expression_quality_gate = ExpressionQualityGate()
        self.novelty_gate = EvolutionNoveltyGate()
        self.repair_agent = FactorRepairAgent()

    def _load_prompt_bundle(self) -> ExperimentDesignerPromptBundle:
        """从提示词配置文件加载实验设计代理所需片段。"""
        with self.prompt_file.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"invalid experiment_designer prompts: {self.prompt_file}")
        experiment_designer_prompt = data.get("experiment_designer", {})
        if not isinstance(experiment_designer_prompt, dict):
            raise ValueError("missing experiment_designer section")
        return ExperimentDesignerPromptBundle(
            hypothesis_and_feedback=str(data.get("hypothesis_and_feedback", "")),
            function_lib_description=self._compact_function_lib_description(str(data.get("function_lib_description", ""))),
            factor_experiment_output_format=self._compact_output_format(),
            experiment_designer_system_prompt=str(experiment_designer_prompt.get("system_prompt", "")),
            experiment_designer_user_prompt=str(experiment_designer_prompt.get("user_prompt", "")),
            expression_duplication=str(data.get("expression_duplication", "")),
        )

    def _compact_function_lib_description(self, raw_description: str) -> str:
        return self.context_builder.compact_function_lib_description(raw_description)

    def _compact_output_format(self) -> str:
        return self.context_builder.compact_output_format()

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

    def _render_template(self, template: str, context: dict[str, object]) -> str:
        """按严格变量约束渲染模板，缺字段即抛错。"""
        return Environment(undefined=StrictUndefined).from_string(template).render(**context)

    def _truncate_text(self, value: object, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _extract_expression_variables(self, expression: str) -> set[str]:
        return {match.group(0) for match in re.finditer(r"\$[A-Za-z_][A-Za-z0-9_]*", expression)}

    @staticmethod
    def _supported_expression_variables(domain_dataset_summary: dict[str, object]) -> set[str]:
        feature_columns = domain_dataset_summary.get("feature_columns", [])
        if not isinstance(feature_columns, list):
            return set()
        return {f"${str(item).strip().lstrip('$')}" for item in feature_columns if str(item).strip()}

    @staticmethod
    def _unsupported_variable_feedback(
        *,
        factor_name: str,
        unsupported_variables: set[str],
        supported_variables: set[str],
    ) -> str:
        return (
            f"Unsupported dataset feature check failed for {factor_name}: expression uses "
            f"{sorted(unsupported_variables)}, but only these raw input features are available: "
            f"{sorted(supported_variables)}. Rewrite the expression using only available dataset columns. "
            "Do not invent horizon-specific fields such as $volume_change_4h unless they are listed."
        )

    def _normalize_expression(self, expression: str) -> str:
        return re.sub(r"\s+", "", expression)

    def _normalize_factor_expression_literals(self, expression: str) -> str:
        normalized = re.sub(r"\bepsilon\b", "1e-8", str(expression or ""), flags=re.IGNORECASE)
        return normalize_window_function_aliases(normalized)

    def _canonical_expression_key(self, expression: str) -> str:
        return self._normalize_expression(self._normalize_factor_expression_literals(expression)).lower()

    def _materialize_expression_pattern(self, pattern: str) -> str:
        return re.sub(r"\*", "1", self._normalize_factor_expression_literals(pattern))

    def _opposite_expected_direction(self, expected_direction: object) -> str:
        text = str(expected_direction or "").strip().lower()
        if "negative" in text and "positive" not in text:
            return "positive_or_non_negative_forward_return"
        if "positive" in text and "negative" not in text:
            return "negative_or_non_positive_forward_return"
        return "opposite_to_current_hypothesis_direction"

    def _build_counterfactual_critique_text(
        self,
        *,
        current_hypothesis: object,
        expected_direction: object,
    ) -> dict[str, str]:
        hypothesis_text = self._truncate_text(current_hypothesis, 360) or "the current hypothesis"
        opposite_direction = self._opposite_expected_direction(expected_direction)
        return {
            "thesis": (
                "Critique the current hypothesis using the same anchors: the observed anchor pattern may support "
                f"{opposite_direction}, not the current claim. Current hypothesis under critique: {hypothesis_text}"
            ),
            "mechanism": (
                "The rival mechanism must explain why the same observable anchors could be misread by the mainline "
                "thesis, such as temporary digestion, delayed confirmation, selection effects, or risk-premium "
                "compensation, without switching to unrelated variables."
            ),
            "expected_direction": opposite_direction,
        }

    def _filter_supported_anchor_patterns(
        self,
        patterns: list[str],
        *,
        allowed_features: set[str],
        allowed_functions: set[str],
    ) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        for item in patterns:
            token = self._normalize_expression(self._normalize_factor_expression_literals(str(item).strip()))
            lowered = token.lower()
            if not token or lowered in seen:
                continue
            materialized = self._materialize_expression_pattern(token)
            parsable, _ = is_parsable(materialized)
            if not parsable:
                continue
            symbol_validation = validate_expression_symbols(
                materialized,
                allowed_features=allowed_features,
                allowed_functions=allowed_functions,
            )
            if not symbol_validation.passed:
                continue
            seen.add(lowered)
            filtered.append(token)
        return filtered

    def _pattern_to_regex(self, pattern: str) -> re.Pattern[str]:
        escaped = re.escape(self._normalize_expression(pattern)).replace(r"\*", r"\d+")
        return re.compile(escaped, re.IGNORECASE)

    def _expression_matches_anchor_patterns(self, expression: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        normalized = self._normalize_expression(expression)
        parenless_normalized = normalized.replace("(", "").replace(")", "")
        for pattern in patterns:
            if not pattern:
                continue
            if self._pattern_to_regex(pattern).search(normalized):
                return True
            parenless_pattern = self._normalize_expression(pattern).replace("(", "").replace(")", "")
            parenless_regex = re.compile(re.escape(parenless_pattern).replace(r"\*", r"\d+"), re.IGNORECASE)
            if parenless_regex.search(parenless_normalized):
                return True
        return False

    def _run_consistency_gate(
        self,
        *,
        enabled: bool,
        model_client: ModelClient,
        scenario: str,
        hypothesis: str,
        phase: str,
        mutation_type: str,
        evolution_guidance: str,
        parent_trajectory_summaries: list[dict[str, object]],
        factors: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        """批量调用大模型校验器，统一判断整批表达式的语义一致性和可测试性。"""
        def default_result(
            *,
            factor_name: str,
            expression: str,
            feedback: str,
            passed: bool,
            verifier: str,
        ) -> dict[str, object]:
            return {
                "factor_name": factor_name,
                "passed": bool(passed),
                "feedback": feedback,
                "corrected_expression": expression,
                "verifier": verifier,
                "raw_result": {},
            }

        results: dict[str, dict[str, object]] = {}
        if not enabled:
            for factor in factors:
                factor_name = str(factor.get("factor_name", "")).strip()
                expression = str(factor.get("factor_expression", "")).strip()
                if factor_name:
                    results[factor_name] = default_result(
                        factor_name=factor_name,
                        expression=expression,
                        feedback="LLM verifier disabled",
                        passed=True,
                        verifier="disabled",
                    )
            return results
        system_prompt = (
            "You are a batch LLM verifier for alpha-factor experiment design. "
            "Judge semantic consistency, evolution-guidance compliance, and practical factor simplicity across a batch of factors. "
            "Hard metrics such as symbol length, base-feature count, or duplicate-subtree size are advisory signals only; "
            "do not reject solely because a numeric threshold is exceeded. "
            "Reject only if the expression is semantically inconsistent, ignores required evolution guidance, uses unsupported logic, "
            "or is too incoherent to test. "
            "Return ONLY JSON."
        )
        user_payload = {
            "scenario": scenario,
            "hypothesis": hypothesis,
            "phase": phase,
            "mutation_type": mutation_type,
            "evolution_guidance": evolution_guidance,
            "parent_trajectory_summaries": parent_trajectory_summaries,
            "factors": [
                {
                    "factor_name": str(factor.get("factor_name", "")).strip(),
                    "description": str(factor.get("factor_description", "")),
                    "formulation": str(factor.get("factor_formulation", "")),
                    "expression": str(factor.get("factor_expression", "")),
                    "variables": dict(factor.get("variables", {}))
                    if isinstance(factor.get("variables", {}), dict)
                    else {},
                    "advisory_quality_records": list(factor.get("advisory_records", [])),
                }
                for factor in factors
            ],
            "required_output_schema": {
                "results": [
                    {
                        "factor_name": "string",
                        "passed": "boolean",
                        "feedback": "single concise sentence",
                        "corrected_expression": "optional expression string; empty if no correction",
                    }
                ],
            },
        }
        try:
            raw = model_client.generate(
                system_prompt=system_prompt,
                user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
                json_mode=True,
            )
            payload = parse_json_object(raw)
        except Exception as exc:
            for factor in factors:
                factor_name = str(factor.get("factor_name", "")).strip()
                expression = str(factor.get("factor_expression", "")).strip()
                if factor_name:
                    results[factor_name] = default_result(
                        factor_name=factor_name,
                        expression=expression,
                        feedback=f"LLM verifier failed: {exc}",
                        passed=False,
                        verifier="failed",
                    )
            return results

        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        for factor in factors:
            factor_name = str(factor.get("factor_name", "")).strip()
            expression = str(factor.get("factor_expression", "")).strip()
            if factor_name:
                results[factor_name] = default_result(
                    factor_name=factor_name,
                    expression=expression,
                    feedback="LLM verifier returned no decision for this factor.",
                    passed=False,
                    verifier="llm_missing",
                )

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            factor_name = str(item.get("factor_name", "")).strip()
            if not factor_name or factor_name not in results:
                continue
            passed = bool(item.get("passed", False))
            feedback = str(item.get("feedback", "")).strip() or (
                "LLM verifier passed." if passed else "LLM verifier rejected."
            )
            corrected_expression = str(item.get("corrected_expression", "")).strip() or str(
                results[factor_name].get("corrected_expression", "")
            )
            results[factor_name] = {
                "factor_name": factor_name,
                "passed": passed,
                "feedback": feedback,
                "corrected_expression": corrected_expression,
                "verifier": "llm",
                "raw_result": item,
            }
        return results

    def _build_quality_gate_record(
        self,
        *,
        factor_name: str,
        expression: str,
        stage: str,
        passed: bool,
        feedback: str,
        eval_dict: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """构造写入实验附加信息的质量闸门记录。"""
        return {
            "factor_name": factor_name,
            "expression": expression,
            "stage": stage,
            "passed": bool(passed),
            "feedback": feedback,
            "metrics": dict(eval_dict or {}),
        }

    def _evolution_guidance_violation(
        self,
        *,
        phase: str,
        mutation_type: str,
        expression: str,
        expression_variables: set[str],
        anchor_input_features: list[str],
        anchor_expression_patterns: list[str],
        evolution_plan_summaries: list[dict[str, object]] | None = None,
    ) -> str:
        """检查变异或交叉阶段疑似违反引导信息的表达式模式。"""
        normalized_phase = str(phase or "").strip().lower()
        normalized_mutation_type = str(mutation_type or "").strip().lower()
        planned_phases = {
            str(item.get("phase", "")).strip().lower()
            for item in list(evolution_plan_summaries or [])
            if isinstance(item, dict)
        }
        effective_phase = normalized_phase
        if normalized_phase == "evolved" and planned_phases:
            effective_phase = "crossover" if "crossover" in planned_phases else "mutation"
        if effective_phase not in {"mutation", "crossover"}:
            return ""

        anchor_feature_set = set(anchor_input_features)
        has_anchor_feature = bool(expression_variables & anchor_feature_set) if anchor_feature_set else True
        has_anchor_pattern = self._expression_matches_anchor_patterns(expression, anchor_expression_patterns)
        if normalized_mutation_type == "parameter_mutation":
            new_variables = expression_variables - anchor_feature_set
            if anchor_feature_set and new_variables:
                return (
                    "Evolution guidance violation: parameter_mutation must not switch variable families; "
                    f"new variables introduced: {sorted(new_variables)}."
                )
        if normalized_mutation_type == "expression_mutation":
            if (anchor_feature_set or anchor_expression_patterns) and not (has_anchor_feature or has_anchor_pattern):
                return "Evolution guidance violation: expression_mutation must preserve at least one parent anchor feature or signal pattern."
        if normalized_mutation_type == "risk_mutation":
            stability_markers = ("TS_MEAN", "TS_MEDIAN", "TS_STD", "TS_ZSCORE", "RANK", "SIGN", "EMA", "WMA", "DECAYLINEAR")
            if not any(marker in expression.upper() for marker in stability_markers):
                return "Evolution guidance violation: risk_mutation should add smoothing, ranking, volatility, or stability control."
        if effective_phase == "crossover":
            if int(expression.count("*") + expression.count("+")) > 5:
                return "Evolution guidance violation: crossover expression looks like mechanical stacking; keep one coherent simple signal."
        return ""

    @staticmethod
    def _normalized_expression_key(expression: str) -> str:
        return "".join(str(expression or "").split()).lower()

    @staticmethod
    def _expression_function_tokens(expression: str) -> list[str]:
        return [match.group(1).upper() for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)]

    @staticmethod
    def _expression_numeric_tokens(expression: str) -> list[str]:
        return re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", expression, flags=re.IGNORECASE)

    @staticmethod
    def _parent_expressions(parent_trajectory_summaries: list[dict[str, object]]) -> list[str]:
        expressions: list[str] = []
        for summary in parent_trajectory_summaries:
            factors = summary.get("factors", [])
            if not isinstance(factors, list):
                continue
            for factor in factors:
                if not isinstance(factor, dict):
                    continue
                expression = str(factor.get("expression", factor.get("factor_expression", ""))).strip()
                if expression and expression not in expressions:
                    expressions.append(expression)
        return expressions

    def _evolution_novelty_violation(
        self,
        *,
        phase: str,
        mutation_type: str,
        expression: str,
        expression_variables: set[str],
        parent_expressions: list[str],
        evolution_plan_summaries: list[dict[str, object]],
    ) -> tuple[str, dict[str, object]]:
        """Hard gate for evolved expressions that merely repeat parent formulas."""
        return self.novelty_gate.check(
            phase=phase,
            mutation_type=mutation_type,
            expression=expression,
            expression_variables=expression_variables,
            parent_expressions=parent_expressions,
            evolution_plan_summaries=evolution_plan_summaries,
        )

    def _resolve_factor_count_bounds(self, *, fast_mode: bool) -> tuple[int, int]:
        default_min = 1 if fast_mode else 2
        default_max = 3
        raw_target = os.getenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "").strip()
        if raw_target:
            try:
                max_factors = max(1, int(raw_target))
                raw_min = os.getenv("EXPERIMENT_DESIGNER_MIN_FACTORS", "").strip()
                try:
                    min_factors = max(1, int(raw_min)) if raw_min else min(default_min, max_factors)
                except ValueError:
                    min_factors = min(default_min, max_factors)
                return min(min_factors, max_factors), max_factors
            except ValueError:
                pass

        raw_min = os.getenv("EXPERIMENT_DESIGNER_MIN_FACTORS", "").strip()
        raw_max = os.getenv("EXPERIMENT_DESIGNER_MAX_FACTORS", "").strip()
        try:
            min_factors = max(1, int(raw_min)) if raw_min else default_min
        except ValueError:
            min_factors = default_min
        try:
            max_factors = max(1, int(raw_max)) if raw_max else default_max
        except ValueError:
            max_factors = default_max
        if min_factors > max_factors:
            min_factors, max_factors = max_factors, min_factors
        return min_factors, max_factors

    def _resolve_counterfactual_factor_count(self) -> int:
        raw = os.getenv("EXPERIMENT_DESIGNER_COUNTERFACTUAL_FACTOR_COUNT", "").strip()
        if not raw:
            raw = os.getenv("COUNTERFACTUAL_FACTOR_COUNT", "").strip()
        if not raw:
            return 2
        try:
            return max(0, int(raw))
        except ValueError:
            return 2

    def _resolve_adjacent_factor_count(self) -> int:
        raw = os.getenv("EXPERIMENT_DESIGNER_ADJACENT_FACTOR_COUNT", "").strip()
        if not raw:
            raw = os.getenv("ADJACENT_FACTOR_COUNT", "").strip()
        if not raw:
            return 1
        try:
            return max(0, int(raw))
        except ValueError:
            return 1

    def _resolve_compact_history_limit(self) -> int:
        raw = os.getenv("EXPERIMENT_DESIGNER_HISTORY_LIMIT", "").strip()
        if not raw:
            raw = os.getenv("EXPERIMENT_DESIGNER_COMPACT_HISTORY_LIMIT", "").strip()
        if not raw:
            return 2
        try:
            return min(6, max(1, int(raw)))
        except ValueError:
            return 2

    def _resolve_feedback_item_limit(self) -> int:
        raw = os.getenv("EXPERIMENT_DESIGNER_FEEDBACK_ITEM_LIMIT", "").strip()
        if not raw:
            return 6
        try:
            return min(12, max(2, int(raw)))
        except ValueError:
            return 6

    def _resolve_feedback_char_limit(self) -> int:
        raw = os.getenv("EXPERIMENT_DESIGNER_FEEDBACK_CHAR_LIMIT", "").strip()
        if not raw:
            return 1800
        try:
            return min(4000, max(600, int(raw)))
        except ValueError:
            return 1800

    def _compact_duplication_feedback(self, items: list[str]) -> tuple[str, str]:
        if not items:
            return "", ""
        trimmed = [self._truncate_text(item, 320) for item in items[-self._resolve_feedback_item_limit() :]]
        expression_feedback = "\n\n".join(trimmed)
        combined = "\n\n".join(trimmed)
        if len(combined) > self._resolve_feedback_char_limit():
            combined = combined[: self._resolve_feedback_char_limit()].rstrip() + "..."
        return expression_feedback, combined

    def _compact_proxy_reconstruction_plan(self, plan: dict[str, object]) -> str:
        if not isinstance(plan, dict) or not plan:
            return "None"

        def compact_list(key: str, limit: int = 5) -> list[str]:
            values = plan.get(key, [])
            if not isinstance(values, list):
                return []
            return [self._truncate_text(item, 56) for item in values[:limit] if str(item).strip()]

        compacted = {
            "core": compact_list("core_proxy_family", 3),
            "supporting": compact_list("supporting_proxy_family", 3),
            "constraints": compact_list("implementation_constraints", 2),
            "mutations": compact_list("preferred_mutations", 2),
        }
        parts = []
        for key, values in compacted.items():
            if values:
                parts.append(f"{key}={'; '.join(values)}")
        return " | ".join(parts) if parts else "None"

    def _build_compact_report_context(
        self,
        *,
        research_report: dict[str, object],
        hypothesis_structured: dict[str, object],
        research_stage: str,
        proxy_reconstruction_plan: dict[str, object],
    ) -> str:
        thesis = self._truncate_text(hypothesis_structured.get("thesis", ""), 180)
        mechanism = self._truncate_text(hypothesis_structured.get("mechanism", ""), 180)

        source_reports = research_report.get("source_reports", [])
        if not isinstance(source_reports, list):
            source_reports = []

        caveats: list[str] = []
        proxy_families: list[str] = []
        for item in source_reports:
            if not isinstance(item, dict):
                continue
            if not thesis:
                thesis = self._truncate_text(item.get("thesis_summary", ""), 180)
            if not mechanism:
                mechanism = self._truncate_text(item.get("mechanism_summary", ""), 180)
            for caveat in list(item.get("caveats_or_conflicts", []))[:2]:
                text = self._truncate_text(caveat, 120)
                if text and text not in caveats:
                    caveats.append(text)
            for proxy in list(item.get("proxy_candidates", []))[:4]:
                text = self._truncate_text(proxy, 40)
                if text and text not in proxy_families:
                    proxy_families.append(text)

        for key in ("core_proxy_family", "supporting_proxy_family"):
            for proxy in list(proxy_reconstruction_plan.get(key, []))[:4]:
                text = self._truncate_text(proxy, 40)
                if text and text not in proxy_families:
                    proxy_families.append(text)

        compact_plan = self._compact_proxy_reconstruction_plan(proxy_reconstruction_plan)
        caveat_text = "; ".join(caveats[:3]) if caveats else "None"
        proxy_text = ", ".join(proxy_families[:6]) if proxy_families else "None"

        lines = [
            f"- Current stage: {research_stage}",
            f"- Thesis: {thesis or 'None'}",
            f"- Mechanism: {mechanism or 'None'}",
            f"- Caveats: {caveat_text}",
            f"- Proxy families: {proxy_text}",
            f"- Proxy reconstruction plan: {compact_plan}",
        ]
        return "\n".join(lines)

    def _format_signal_factor_summary(self, signal_diagnostics: dict[str, object]) -> str:
        items = signal_diagnostics.get("factor_signal_diagnostics", [])
        if not isinstance(items, list):
            return "no per-factor signal diagnostics"
        rows: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                abs_rank_ic = float(item.get("abs_rank_ic"))
            except Exception:
                abs_rank_ic = -1.0
            rows.append({**item, "_abs_rank_ic_sort": abs_rank_ic})
        rows.sort(key=lambda item: float(item.get("_abs_rank_ic_sort", -1.0)), reverse=True)
        if not rows:
            return "no evaluated factors"
        parts: list[str] = []
        for item in rows[:2]:
            parts.append(
                "{name}:ric={rank_ic},sig={strength},type={align}".format(
                    name=self._truncate_text(item.get("factor_name", "unknown"), 34),
                    rank_ic=self._truncate_text(item.get("rank_ic", "N/A"), 12),
                    strength=self._truncate_text(item.get("signal_strength", "unknown"), 10),
                    align=self._truncate_text(item.get("theory_alignment", "mainline"), 14),
                )
            )
        return "; ".join(parts)

    def _build_compact_hypothesis_feedback_text(self, history: list[dict[str, object]]) -> str:
        if not history:
            return "No previous hypothesis and feedback available."
        limit = self._resolve_compact_history_limit()
        lines: list[str] = []
        start_index = max(1, len(history) - limit + 1)
        for offset, entry in enumerate(history[-limit:], start=start_index):
            if not isinstance(entry, dict):
                continue
            signal_diagnostics = entry.get("signal_diagnostics", {})
            if not isinstance(signal_diagnostics, dict):
                signal_diagnostics = {}
            feedback_next_stage = entry.get("feedback_next_research_stage", "")
            recommended_action = signal_diagnostics.get("recommended_action", "")
            revision_reason = signal_diagnostics.get("revision_reason", "")
            lines.append(
                "\n".join(
                    [
                        f"Loop {offset}: H={self._truncate_text(entry.get('hypothesis', ''), 160)}",
                        f"Feedback={self._truncate_text(entry.get('feedback_summary', ''), 90)}",
                        f"Next={self._truncate_text(feedback_next_stage, 28)}; Action={self._truncate_text(recommended_action, 28)}",
                        f"KeyFactors={self._format_signal_factor_summary(signal_diagnostics)}",
                        f"Reason={self._truncate_text(revision_reason, 100)}",
                    ]
                )
            )
        return "\n\n".join(lines) if lines else "No previous hypothesis and feedback available."

    def _materialize_counterfactual_anchor_pattern(
        self,
        pattern: str,
        *,
        allowed_input_features: set[str],
        allowed_functions: set[str],
    ) -> str | None:
        normalized = self._normalize_expression(self._normalize_factor_expression_literals(pattern))
        if not normalized:
            return None
        for window in (6, 12, 24, 4, 1):
            expression = re.sub(r"\*", str(window), normalized)
            parsable, _ = is_parsable(expression)
            if not parsable:
                continue
            symbol_validation = validate_expression_symbols(
                expression,
                allowed_features=allowed_input_features,
                allowed_functions=allowed_functions,
            )
            if symbol_validation.passed:
                return expression
        return None

    def _select_anchor_feature(
        self,
        *,
        anchor_input_features: list[str],
        allowed_input_features: set[str],
        candidates: tuple[str, ...],
        exclude: set[str] | None = None,
        allow_fallback: bool = False,
    ) -> str | None:
        excluded = exclude or set()
        anchor_set = set(anchor_input_features)
        for item in candidates:
            if item in anchor_set and item in allowed_input_features and item not in excluded:
                return item
        if allow_fallback:
            for item in candidates:
                if item in allowed_input_features and item not in excluded:
                    return item
        return None

    def _build_counterfactual_item(
        self,
        *,
        factor_name: str,
        description: str,
        expression: str,
        allowed_input_features: set[str],
        anchor_input_features: list[str],
        anchor_expression_patterns: list[str],
        source: str,
        mainline_expression_keys: set[str] | None = None,
        counterfactual_thesis: str = "",
        counterfactual_mechanism: str = "",
        counterfactual_critique: str = "",
        counterfactual_anchor: dict[str, object] | None = None,
        counterfactual_expected_direction: str = "",
        unanchored_fallback: bool = False,
    ) -> dict[str, object] | None:
        expression = self._normalize_factor_expression_literals(expression)
        expression_key = self._canonical_expression_key(expression)
        if mainline_expression_keys and expression_key in mainline_expression_keys:
            return None
        parsable, _ = is_parsable(expression)
        if not parsable:
            return None
        symbol_validation = validate_expression_symbols(
            expression,
            allowed_features=allowed_input_features,
            allowed_functions=get_supported_runtime_functions(),
        )
        if not symbol_validation.passed:
            return None
        expression_variables = self._extract_expression_variables(expression)
        anchor_set = set(anchor_input_features)
        matched_anchor_features = sorted(expression_variables & anchor_set)
        matched_anchor_pattern = self._expression_matches_anchor_patterns(expression, anchor_expression_patterns)
        if anchor_set and not matched_anchor_pattern and len(matched_anchor_features) < 2:
            return None
        if len(expression_variables) < 2 and not matched_anchor_pattern:
            return None
        single_rank = re.fullmatch(r"RANK\(\s*RANK\(\s*\$[A-Za-z_][A-Za-z0-9_]*\s*\)\s*\)", expression, flags=re.IGNORECASE)
        if single_rank:
            return None
        if unanchored_fallback and anchor_set:
            return None
        warnings = ["counterfactual_economic_critique", "counterfactual_opposite_direction"]
        if unanchored_fallback:
            warnings.append("counterfactual_unanchored_fallback")
        counterfactual_anchor = counterfactual_anchor or {
            "input_features": matched_anchor_features,
            "expression_patterns": list(anchor_expression_patterns),
            "source": source,
        }
        return {
            "factor_name": factor_name,
            "description": description,
            "formulation": expression,
            "expression": expression,
            "variables": {feature: "current hypothesis anchor feature" for feature in sorted(expression_variables)},
            "validation_warnings": warnings,
            "retained_for_backtest": True,
            "theory_alignment": "counterfactual",
            "counterfactual_reason": (
                "Critiques the current hypothesis with the same anchors and tests a rival economic interpretation "
                f"({source}); retained to compare theory-in vs theory-out directions."
            ),
            "counterfactual_thesis": counterfactual_thesis
            or "The same anchor pattern is interpreted as evidence for the rival direction rather than the current thesis.",
            "counterfactual_mechanism": counterfactual_mechanism
            or "A rival economic mechanism explains why the current thesis may have misread the same anchors.",
            "counterfactual_critique": counterfactual_critique
            or "This factor is an adversarial test of the current mechanism, not a synonym of the mainline thesis.",
            "counterfactual_anchor": counterfactual_anchor,
            "counterfactual_expected_direction": counterfactual_expected_direction
            or "opposite_to_current_hypothesis_direction",
            "matched_anchor_features": matched_anchor_features,
            "matched_anchor_pattern": matched_anchor_pattern,
            "expected_anchor_features": sorted(anchor_set),
            "expected_anchor_patterns": list(anchor_expression_patterns),
            "anchor_fallback_reason": source if unanchored_fallback else "",
        }

    def _build_adjacent_item(
        self,
        *,
        factor_name: str,
        description: str,
        expression: str,
        allowed_input_features: set[str],
        anchor_input_features: list[str],
        anchor_expression_patterns: list[str],
        source: str,
    ) -> dict[str, object] | None:
        expression = self._normalize_factor_expression_literals(expression)
        parsable, _ = is_parsable(expression)
        if not parsable:
            return None
        symbol_validation = validate_expression_symbols(
            expression,
            allowed_features=allowed_input_features,
            allowed_functions=get_supported_runtime_functions(),
        )
        if not symbol_validation.passed:
            return None
        expression_variables = self._extract_expression_variables(expression)
        anchor_set = set(anchor_input_features)
        return {
            "factor_name": factor_name,
            "description": description,
            "formulation": expression,
            "expression": expression,
            "variables": {feature: "adjacent proxy expansion feature" for feature in sorted(expression_variables)},
            "validation_warnings": ["adjacent_proxy_expansion"],
            "retained_for_backtest": True,
            "theory_alignment": "adjacent",
            "counterfactual_reason": "",
            "adjacent_reason": (
                "Tests a nearby proxy family around the same mechanism without treating the opposite thesis as true "
                f"({source}); retained to compare mainline vs adjacent proxy expansion."
            ),
            "matched_anchor_features": sorted(expression_variables & anchor_set),
            "matched_anchor_pattern": self._expression_matches_anchor_patterns(expression, anchor_expression_patterns),
            "expected_anchor_features": sorted(anchor_set),
            "expected_anchor_patterns": list(anchor_expression_patterns),
            "anchor_fallback_reason": source,
        }

    def _build_unanchored_counterfactual_templates(
        self,
        *,
        round_idx: int,
        allowed_input_features: set[str],
        anchor_input_features: list[str],
        reason: str,
    ) -> list[dict[str, str]]:
        _ = (round_idx, allowed_input_features, anchor_input_features, reason)
        return []

    def _build_counterfactual_factor_candidates(
        self,
        *,
        count: int,
        round_idx: int,
        allowed_input_features: set[str],
        anchor_input_features: list[str],
        anchor_expression_patterns: list[str],
        mainline_expressions: list[str] | None = None,
        previous_counterfactual_state: dict[str, object] | None = None,
        current_hypothesis: str = "",
        current_expected_direction: str = "",
    ) -> list[dict[str, object]]:
        if count <= 0:
            return []

        allowed_functions = get_supported_runtime_functions()
        anchor_set = set(anchor_input_features)
        mainline_expression_keys = {
            self._canonical_expression_key(item)
            for item in (mainline_expressions or [])
            if str(item).strip()
        }
        previous_counterfactual_state = previous_counterfactual_state or {}
        anchor_expression_patterns = [
            self._normalize_expression(self._normalize_factor_expression_literals(pattern))
            for pattern in anchor_expression_patterns
            if self._normalize_expression(self._normalize_factor_expression_literals(pattern))
        ]
        critique_text = self._build_counterfactual_critique_text(
            current_hypothesis=current_hypothesis,
            expected_direction=current_expected_direction,
        )
        counterfactual_thesis = str(previous_counterfactual_state.get("counterfactual_thesis") or critique_text["thesis"])
        counterfactual_mechanism = str(
            previous_counterfactual_state.get("counterfactual_mechanism") or critique_text["mechanism"]
        )
        counterfactual_critique = str(
            previous_counterfactual_state.get("counterfactual_critique")
            or (
                "Economic criticism: the same anchors may be compatible with the opposite return direction, "
                "so the mainline mechanism must beat this rival explanation at group level."
            )
        )
        counterfactual_expected_direction = str(
            previous_counterfactual_state.get("counterfactual_expected_direction")
            or critique_text["expected_direction"]
        )
        counterfactual_anchor = {
            "input_features": list(anchor_input_features),
            "expression_patterns": list(anchor_expression_patterns),
            "continuity_source": "previous_counterfactual_state" if previous_counterfactual_state else "current_hypothesis_anchors",
        }
        templates: list[dict[str, str]] = []
        for pattern_index, pattern in enumerate(anchor_expression_patterns, start=1):
            base_expression = self._materialize_counterfactual_anchor_pattern(
                pattern,
                allowed_input_features=allowed_input_features,
                allowed_functions=allowed_functions,
            )
            if not base_expression:
                continue
            support_feature = self._select_anchor_feature(
                anchor_input_features=anchor_input_features,
                allowed_input_features=allowed_input_features,
                candidates=tuple(anchor_input_features),
                exclude=self._extract_expression_variables(base_expression),
            )
            templates.append(
                {
                    "factor_name": f"counterfactual_anchor_pattern_rival_{round_idx}_{pattern_index}",
                    "description": "Critical rival test: the current anchor pattern is interpreted against the mainline mechanism.",
                    "expression": f"RANK({base_expression})",
                    "source": f"anchor pattern {pattern}",
                }
            )
            if support_feature:
                templates.append(
                    {
                        "factor_name": f"counterfactual_anchor_pattern_confirmed_rival_{round_idx}_{pattern_index}",
                        "description": "Critical rival test: the anchor pattern plus another current anchor challenges the mainline mechanism.",
                        "expression": f"RANK({base_expression}) * RANK({support_feature})",
                        "source": f"anchor pattern {pattern} with support anchor {support_feature}",
                    }
                )
            else:
                templates.append(
                    {
                        "factor_name": f"counterfactual_inverse_anchor_pattern_rival_{round_idx}_{pattern_index}",
                        "description": "Critical rival test: the inverse anchor pattern checks whether the mainline direction is mis-specified.",
                        "expression": f"-RANK({base_expression})",
                        "source": f"inverse anchor pattern {pattern}",
                    }
                )

        if not templates:
            return []

        valid: list[dict[str, object]] = []
        seen_expressions: set[str] = set()
        for item in templates:
            expression = str(item.get("expression", ""))
            normalized_expression = self._normalize_expression(expression).lower()
            if normalized_expression in seen_expressions:
                continue
            candidate = self._build_counterfactual_item(
                factor_name=str(item.get("factor_name", "")),
                description=str(item.get("description", "")),
                expression=expression,
                allowed_input_features=allowed_input_features,
                anchor_input_features=anchor_input_features,
                anchor_expression_patterns=anchor_expression_patterns,
                source=str(item.get("source", "current hypothesis anchors")),
                mainline_expression_keys=mainline_expression_keys,
                counterfactual_thesis=counterfactual_thesis,
                counterfactual_mechanism=counterfactual_mechanism,
                counterfactual_critique=counterfactual_critique,
                counterfactual_anchor=counterfactual_anchor,
                counterfactual_expected_direction=counterfactual_expected_direction,
                unanchored_fallback=not bool(anchor_set),
            )
            if candidate is None:
                continue
            if anchor_set and not candidate.get("matched_anchor_features") and not candidate.get("matched_anchor_pattern"):
                continue
            seen_expressions.add(normalized_expression)
            valid.append(candidate)
            if len(valid) >= count:
                break
        return valid

    def _build_adjacent_factor_candidates(
        self,
        *,
        count: int,
        round_idx: int,
        allowed_input_features: set[str],
        anchor_input_features: list[str],
        anchor_expression_patterns: list[str],
    ) -> list[dict[str, object]]:
        if count <= 0:
            return []

        return_feature = self._select_anchor_feature(
            anchor_input_features=anchor_input_features,
            allowed_input_features=allowed_input_features,
            candidates=("$returns_24h", "$returns_12h", "$returns_4h", "$returns_1h", "$open_close_return"),
            allow_fallback=True,
        )
        volume_feature = self._select_anchor_feature(
            anchor_input_features=anchor_input_features,
            allowed_input_features=allowed_input_features,
            candidates=("$volume_change_24h", "$volume_change_1h", "$volume_quote", "$volume"),
            allow_fallback=True,
        )
        spread_feature = self._select_anchor_feature(
            anchor_input_features=anchor_input_features,
            allowed_input_features=allowed_input_features,
            candidates=("$high_low_spread", "$open_close_return", "$vwap"),
            allow_fallback=True,
        )

        templates: list[dict[str, str]] = []
        if return_feature and spread_feature and return_feature != spread_feature:
            templates.append(
                {
                    "factor_name": f"adjacent_proxy_spread_regime_divergence_{round_idx}",
                    "description": "Adjacent proxy expansion: keeps the divergence mechanism but tests whether spread regime explains weak price-volume proxies.",
                    "expression": f"RANK({return_feature}) * RANK({spread_feature})",
                    "source": f"adjacent spread-regime proxy using {return_feature}, {spread_feature}",
                }
            )
        if return_feature and volume_feature:
            templates.append(
                {
                    "factor_name": f"adjacent_proxy_volume_confirmation_gap_{round_idx}",
                    "description": "Adjacent proxy expansion: keeps price-volume divergence but tests a simpler confirmation gap family.",
                    "expression": f"RANK({return_feature}) - RANK({volume_feature})",
                    "source": f"adjacent volume-confirmation gap using {return_feature}, {volume_feature}",
                }
            )
        if volume_feature and spread_feature and volume_feature != spread_feature:
            templates.append(
                {
                    "factor_name": f"adjacent_proxy_liquidity_stress_{round_idx}",
                    "description": "Adjacent proxy expansion: tests liquidity/spread stress as a nearby mechanism around weak conviction.",
                    "expression": f"RANK({spread_feature}) / (RANK({volume_feature}) + 1e-8)",
                    "source": f"adjacent liquidity-stress proxy using {spread_feature}, {volume_feature}",
                }
            )

        valid: list[dict[str, object]] = []
        seen_expressions: set[str] = set()
        for item in templates:
            expression = str(item.get("expression", ""))
            normalized_expression = self._normalize_expression(expression).lower()
            if normalized_expression in seen_expressions:
                continue
            candidate = self._build_adjacent_item(
                factor_name=str(item.get("factor_name", "")),
                description=str(item.get("description", "")),
                expression=expression,
                allowed_input_features=allowed_input_features,
                anchor_input_features=anchor_input_features,
                anchor_expression_patterns=anchor_expression_patterns,
                source=str(item.get("source", "adjacent proxy expansion")),
            )
            if candidate is None:
                continue
            seen_expressions.add(normalized_expression)
            valid.append(candidate)
            if len(valid) >= count:
                break
        return valid

    def _determine_research_stage(
        self,
        *,
        has_research_report: bool,
        history_length: int,
        hypothesis_structured: dict[str, object],
        explicit_stage: str,
    ) -> str:
        normalized_stage = str(explicit_stage or "").strip()
        if normalized_stage:
            return normalized_stage
        if not has_research_report:
            return "free_exploration"
        proxy_plan = hypothesis_structured.get("proxy_reconstruction_plan", {})
        has_proxy_plan = isinstance(proxy_plan, dict) and bool(proxy_plan)
        if history_length <= 0 and has_proxy_plan:
            return "base_reconstruction"
        return "parameter_tuning"

    def _build_research_mode_task_plan(
        self,
        *,
        stage: str,
        target_hypothesis: str,
        proxy_plan: dict[str, object],
    ) -> list[str]:
        core_proxy_family = [str(item).strip() for item in list(proxy_plan.get("core_proxy_family", [])) if str(item).strip()]
        supporting_proxy_family = [
            str(item).strip() for item in list(proxy_plan.get("supporting_proxy_family", [])) if str(item).strip()
        ]
        if stage == "base_reconstruction":
            return [
                "Step 1: Reconstruct the report thesis with simple observable crypto proxies rather than free exploration.",
                f"Step 2: Prioritize core proxy families first: {', '.join(core_proxy_family) if core_proxy_family else 'core thesis proxies'}.",
                f"Step 3: Add only lightweight confirmation proxies if needed: {', '.join(supporting_proxy_family) if supporting_proxy_family else 'supporting confirmation signals'}.",
                "Step 4: Keep expressions simple, interpretable, and directly tied to the report thesis.",
            ]
        if stage == "proxy_reconstruction":
            return [
                "Step 1: Preserve the selected mechanism anchor and rebuild only the observable proxy expression.",
                f"Step 2: Keep core proxy families visible: {', '.join(core_proxy_family) if core_proxy_family else 'current core thesis proxies'}.",
                f"Step 3: Use supporting families only as lightweight filters: {', '.join(supporting_proxy_family) if supporting_proxy_family else 'supporting confirmation signals'}.",
                "Step 4: Do not reset thesis direction unless a hard reset was explicitly allowed.",
            ]
        if stage == "resonance_filtering":
            return [
                "Step 1: Stay anchored to the same report thesis and keep the previously validated proxy family.",
                "Step 2: Add lightweight confirmation signals, regime filters, or resonance structure rather than introducing unrelated variables.",
                f"Step 3: Prefer supporting proxy families for resonance/filter testing: {', '.join(supporting_proxy_family) if supporting_proxy_family else 'supporting confirmation signals'}.",
            ]
        if stage == "thesis_revision":
            return [
                "Step 1: Revise the report-driven thesis conservatively instead of resetting to unrelated exploration.",
                "Step 2: Rebuild a simple proxy set that remains close to the original transferable report mechanism.",
                "Step 3: Avoid broad variable expansion unless the previous proxy family was clearly exhausted.",
            ]
        return [
            "Step 1: Stay anchored to the same report thesis and refine the first-round proxies rather than changing direction.",
            "Step 2: Focus on parameter tuning, normalization changes, and lightweight signal combination.",
            f"Step 3: Continue refining the same proxy family for hypothesis: {target_hypothesis}",
        ]

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        """实验设计主流程：生成 -> 检验 -> 失败反馈回灌 -> 重试，直到得到合格批次。"""
        prompt_bundle = self._load_prompt_bundle()
        scenario = str(shared_context.payload.get("scenario", ""))
        hypothesis = str(shared_context.payload.get("hypothesis", ""))
        phase = str(shared_context.payload.get("phase", "original"))
        round_idx = int(shared_context.payload.get("round_idx", 0) or 0)
        direction_id = int(shared_context.payload.get("direction_id", 0) or 0)
        trajectory_id = str(shared_context.payload.get("trajectory_id", ""))
        parent_ids = [str(item) for item in list(shared_context.payload.get("parent_ids", []))]
        evolution_guidance = str(shared_context.payload.get("evolution_guidance", "")).strip()
        parent_trajectory_summaries = [
            dict(item)
            for item in list(shared_context.payload.get("parent_trajectory_summaries", []))
            if isinstance(item, dict)
        ]
        evolution_plan_summaries = [
            dict(item)
            for item in list(shared_context.payload.get("evolution_plan_summaries", []))
            if isinstance(item, dict)
        ]
        mutation_type = str(shared_context.payload.get("mutation_type", "")).strip()
        evolution_profile = str(shared_context.payload.get("evolution_profile", "quantaalpha")).strip() or "quantaalpha"
        rag_text = str(shared_context.payload.get("rag_text", ""))
        domain_dataset_summary = dict(shared_context.payload.get("domain_dataset_summary", {}))
        supported_feature_variables = self._supported_expression_variables(domain_dataset_summary)
        research_report = dict(shared_context.payload.get("research_report", {}))
        explicit_research_stage = str(shared_context.payload.get("research_stage", "")).strip()
        hypothesis_structured = dict(shared_context.payload.get("hypothesis_structured", {}))
        proxy_reconstruction_plan = dict(hypothesis_structured.get("proxy_reconstruction_plan", {}))
        evolution_state = dict(hypothesis_structured.get("hypothesis_evolution_state", {}))
        history = list(shared_context.payload.get("hypothesis_feedback_history", []))
        recent_trace = list(private_context.payload.get("recent_trace", []))
        domain_dataset_summary = dict(shared_context.payload.get("domain_dataset_summary", {}))
        allowed_input_features = {
            f"${str(item).strip()}"
            for item in list(domain_dataset_summary.get("feature_columns", []))
            if str(item).strip()
        }
        allowed_runtime_functions = get_supported_runtime_functions()

        force_fail_on_loop = os.getenv("FORCE_EXPERIMENT_DESIGNER_FAIL_ON_LOOP", "").strip()
        if force_fail_on_loop:
            try:
                if int(force_fail_on_loop) == round_idx:
                    raise RuntimeError(
                        f"Forced experiment_designer failure for testing on loop {round_idx}."
                    )
            except ValueError:
                pass

        hypothesis_and_feedback = self._build_compact_hypothesis_feedback_text(history)
        previous_factor_names = self.tools.extract_previous_factor_names(recent_trace, self.name)
        previous_factor_expressions = self.tools.extract_previous_factor_expressions(recent_trace, self.name)
        parent_expressions = self._parent_expressions(parent_trajectory_summaries)
        try:
            previous_factor_name_limit = max(
                1,
                int(os.getenv("EXPERIMENT_DESIGNER_PREVIOUS_FACTOR_NAME_LIMIT", "24").strip() or "24"),
            )
        except ValueError:
            previous_factor_name_limit = 24
        previous_factor_names_for_prompt = previous_factor_names[-previous_factor_name_limit:]
        previous_name_set = {name.lower() for name in previous_factor_names}
        core_variables = [str(item).strip() for item in list(evolution_state.get("core_variables", [])) if str(item).strip()]
        verifiable_anchors = evolution_state.get("verifiable_anchors", {})
        if not isinstance(verifiable_anchors, dict):
            verifiable_anchors = {}
        anchor_input_features = [
            str(item).strip()
            for item in list(verifiable_anchors.get("input_features", []))
            if str(item).strip()
        ]
        anchor_expression_patterns = [
            self._normalize_expression(str(item).strip())
            for item in list(verifiable_anchors.get("expression_patterns", []))
            if str(item).strip()
        ]
        anchor_expression_patterns = self._filter_supported_anchor_patterns(
            anchor_expression_patterns,
            allowed_features=allowed_input_features,
            allowed_functions=allowed_runtime_functions,
        )
        # 论文 Eq.7 多父代 segment 拼接：交叉任务的冻结前缀会携带来自其余父代的
        # 已验证因子表达式锚点（crossover_inherited_factor_anchors）。把这些表达式
        # 既并入 anchor_expression_patterns（让锚点/新颖性门要求新因子在其之上融合），
        # 也并入 parent_expressions（让交叉新颖性门正确统计多父代命中）。
        crossover_inherited_anchors = [
            dict(item)
            for item in list(shared_context.payload.get("crossover_inherited_factor_anchors", []))
            if isinstance(item, dict)
        ]
        if crossover_inherited_anchors:
            inherited_expressions = [
                self._normalize_expression(str(item.get("expression", "")).strip())
                for item in crossover_inherited_anchors
                if str(item.get("expression", "")).strip()
            ]
            inherited_expressions = self._filter_supported_anchor_patterns(
                inherited_expressions,
                allowed_features=allowed_input_features,
                allowed_functions=allowed_runtime_functions,
            )
            for expression in inherited_expressions:
                if expression and expression not in anchor_expression_patterns:
                    anchor_expression_patterns.append(expression)
            for item in crossover_inherited_anchors:
                raw_expression = str(item.get("expression", "")).strip()
                if raw_expression and raw_expression not in parent_expressions:
                    parent_expressions.append(raw_expression)
        refinement_stage = str(evolution_state.get("refinement_stage", "initial_exploration")).strip() or "initial_exploration"
        allowed_mutation_types = [
            str(item).strip() for item in list(evolution_state.get("allowed_mutation_types", [])) if str(item).strip()
        ]
        previous_counterfactual_state: dict[str, object] = {}
        for history_item in reversed(history):
            if not isinstance(history_item, dict):
                continue
            signal_diagnostics = history_item.get("signal_diagnostics", {})
            if not isinstance(signal_diagnostics, dict):
                continue
            state = signal_diagnostics.get("counterfactual_state", {})
            if isinstance(state, dict) and state:
                previous_counterfactual_state = dict(state)
                break
        consistency_enabled = os.getenv("LLM_VERIFIER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        fast_mode = os.getenv("FAST_MODE", "false").lower() == "true"
        min_factors, max_factors = self._resolve_factor_count_bounds(fast_mode=fast_mode)
        requested_factor_count = shared_context.payload.get("factor_expressions_per_hypothesis")
        if requested_factor_count not in (None, "") and not os.getenv("EXPERIMENT_DESIGNER_TARGET_FACTOR_COUNT", "").strip():
            try:
                target_count = max(1, int(requested_factor_count))
                min_factors = target_count
                max_factors = target_count
            except (TypeError, ValueError):
                pass
        counterfactual_factor_count = self._resolve_counterfactual_factor_count()
        adjacent_factor_count = self._resolve_adjacent_factor_count()
        if fast_mode:
            consistency_enabled = False
        has_research_report = bool(research_report.get("report_text"))
        research_stage = self._determine_research_stage(
            has_research_report=has_research_report,
            history_length=len(history),
            hypothesis_structured=hypothesis_structured,
            explicit_stage=explicit_research_stage,
        )
        compact_report_context = self._build_compact_report_context(
            research_report=research_report,
            hypothesis_structured=hypothesis_structured,
            research_stage=research_stage,
            proxy_reconstruction_plan=proxy_reconstruction_plan,
        )
        generation_gate = GenerationGate.from_profile(evolution_profile)
        prompt_meta = {
            "duplication_threshold": 8,
            "symbol_length_threshold": generation_gate.config.max_symbol_length,
            "base_features_threshold": generation_gate.config.max_base_features,
            "free_args_ratio_threshold": generation_gate.config.max_free_args_ratio,
            "unique_vars_ratio_threshold": 0.5,
            "expression_duplication": prompt_bundle.expression_duplication,
        }
        context_base = {
            "targets": "factors",
            "scenario": scenario,
            "target_hypothesis": self._truncate_text(hypothesis, 520),
            "phase": phase,
            "round_idx": round_idx,
            "direction_id": direction_id,
            "trajectory_id": trajectory_id,
            "parent_ids_text": ", ".join(parent_ids) if parent_ids else "None",
            "evolution_guidance": evolution_guidance,
            "parent_trajectory_summaries_json": json.dumps(parent_trajectory_summaries, ensure_ascii=False, indent=2),
            "evolution_plan_summaries_json": json.dumps(evolution_plan_summaries, ensure_ascii=False, indent=2),
            "mutation_type": mutation_type,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": prompt_bundle.function_lib_description,
            "experiment_output_format": prompt_bundle.factor_experiment_output_format,
            "target_list": ", ".join(previous_factor_names_for_prompt) if previous_factor_names_for_prompt else "None",
            "RAG": rag_text if rag_text else "None",
            "hypothesis_evolution_state_json": {
                "core_variables": core_variables[:8],
                "verifiable_anchors": {
                    "input_features": anchor_input_features[:12],
                    "expression_patterns": anchor_expression_patterns[:4],
                },
                "refinement_stage": refinement_stage,
                "allowed_mutation_types": allowed_mutation_types,
            },
            "core_variables_text": ", ".join(core_variables[:8]) if core_variables else "None",
            "anchor_input_features_text": ", ".join(anchor_input_features[:12]) if anchor_input_features else "None",
            "anchor_expression_patterns_text": (
                ", ".join(anchor_expression_patterns[:4]) if anchor_expression_patterns else "None"
            ),
            "supported_feature_variables_text": (
                ", ".join(sorted(supported_feature_variables)) if supported_feature_variables else "Not provided"
            ),
            "refinement_stage": refinement_stage,
            "allowed_mutation_types_text": ", ".join(allowed_mutation_types) if allowed_mutation_types else "None",
            "allowed_input_features_text": ", ".join(sorted(allowed_input_features)) if allowed_input_features else "None",
            "duplication_feedback_base": (
                f"Existing factors: {', '.join(previous_factor_names)}. Avoid reusing these names."
                if previous_factor_names
                else "No prior factor names detected."
            ),
            "research_stage": research_stage,
            "has_research_report": has_research_report,
            "report_context_summary": compact_report_context,
            "base_reconstruction_task_plan_text": "\n".join(
                self._build_research_mode_task_plan(
                    stage=research_stage,
                    target_hypothesis=hypothesis,
                    proxy_plan=proxy_reconstruction_plan,
                )
            ),
            "target_factor_count_text": (
                f"{min_factors}" if min_factors == max_factors else f"{min_factors}-{max_factors}"
            ),
            "local_comparison_factor_count_text": (
                f"{counterfactual_factor_count} model-written counterfactual + "
                f"{adjacent_factor_count} local adjacent/proxy-expansion"
            ),
        }
        private_context.payload["experiment_designer_prompt_compression"] = {
            "history_entries_input": len(history),
            "history_entries_used": min(len(history), self._resolve_compact_history_limit()),
            "hypothesis_feedback_chars": len(hypothesis_and_feedback),
            "report_context_chars": len(compact_report_context),
            "previous_factor_names_input": len(previous_factor_names),
            "previous_factor_names_used": len(previous_factor_names_for_prompt),
        }
        duplication_feedback_items: list[str] = []
        quality_gate_records: list[dict[str, object]] = []
        deduplicated: list[dict[str, object]] = []
        task_plan: list[str] = []
        retry_hint_expressions: list[str] = []
        diagnostics = {
            "llm_call_count": 0,
            "proposal_call_count": 0,
            "repair_call_count": 0,
            "local_gate_count": 0,
            "local_accept_count": 0,
        }
        attempt_count = 0
        max_attempts = int(os.getenv("EXPERIMENT_DESIGNER_MAX_ATTEMPTS", "5"))
        experiment_spec: dict[str, Any] = {}  # 初始化，防止未定义错误
        while attempt_count < max_attempts:
            attempt_count += 1
            reference_expressions_current = list(previous_factor_expressions)
            context = dict(context_base)
            skill_mode = "generation" if attempt_count == 1 else "retry"
            skill_rule_context, skill_rule_metadata = self._build_skill_rule_context(
                mode=skill_mode,
                domain_dataset_summary=domain_dataset_summary,
                expressions=retry_hint_expressions,
                feedback_items=duplication_feedback_items,
            )
            compact_expression_duplication, compact_duplication_feedback = self._compact_duplication_feedback(
                duplication_feedback_items
            )
            context["attempt_count"] = attempt_count
            context["expression_duplication"] = compact_expression_duplication
            context["duplication_feedback"] = "\n\n".join(
                filter(
                    None,
                    [
                        str(context_base.get("duplication_feedback_base", "")).strip(),
                        compact_duplication_feedback,
                    ],
                )
            ).strip()
            context["function_lib_description"] = (
                f"{prompt_bundle.function_lib_description}\n\n"
                f"Project-local ExperimentDesigner rule context:\n{skill_rule_context}"
            ).strip()
            context["experiment_output_format"] = (
                f"{prompt_bundle.factor_experiment_output_format}\n\n"
                "Also follow the project-local output contract included in the rule context."
            ).strip()
            system_prompt = self._render_template(prompt_bundle.experiment_designer_system_prompt, context)
            user_prompt = self._render_template(prompt_bundle.experiment_designer_user_prompt, context)
            private_context.payload["last_experiment_skill_context_summary"] = {
                "attempt_count": attempt_count,
                "mode": skill_rule_metadata.get("mode", skill_mode),
                "feature_schema_count": skill_rule_metadata.get("feature_schema_count", 0),
                "supported_function_count": skill_rule_metadata.get("supported_function_count", 0),
                "function_rule_names": list(skill_rule_metadata.get("function_rule_names", [])),
            }
            private_context.payload["skill_rule_files_used"] = list(skill_rule_metadata.get("function_rule_files", []))
            private_context.payload["feature_schema_count"] = skill_rule_metadata.get("feature_schema_count", 0)
            private_context.payload["function_rule_files_used"] = list(
                skill_rule_metadata.get("function_rule_files", [])
            )
            private_context.payload["retry_guidance_reason"] = (
                "base_generation_rules"
                if attempt_count == 1
                else "targeted_retry_rules_from_previous_candidates_and_validation_feedback"
            )
            raw_response = self.proposal_agent.generate(
                model_client=model_client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            diagnostics["llm_call_count"] += 1
            diagnostics["proposal_call_count"] += 1
            private_context.payload["last_experiment_prompt_context"] = context
            private_context.payload["last_experiment_raw_response"] = raw_response
            try:
                payload = parse_json_object(raw_response)
            except Exception as exc:
                print("\n" + "=" * 60)
                print(f"[ERROR] JSON parse failed: {exc}")
                print("=" * 60 + "\n")
                duplication_feedback_items.append(
                    f"JSON parse failed: {exc}. Your response was not valid JSON. "
                    f"Please return ONLY a valid JSON object with the following structure: "
                    f'{{"factors": [{{"factor_name": "...", "description": "...", "formulation": "...", "expression": "...", "variables": {{...}}}}]}}. '
                    f"Do NOT include LaTeX formulas like {{Z(...)}} outside of string values. "
                    f"All content must be inside properly quoted JSON strings."
                )
                continue
            candidates = self.schema_normalizer.extract_candidates(payload)
            retry_hint_expressions = [
                self._normalize_factor_expression_literals(str(item.get("expression", "")).strip())
                for item in candidates
                if str(item.get("expression", "")).strip()
            ]
            default_task_plan = self._build_research_mode_task_plan(
                stage=research_stage,
                target_hypothesis=hypothesis,
                proxy_plan=proxy_reconstruction_plan,
            )
            task_plan_raw = payload.get("task_plan", [])
            if isinstance(task_plan_raw, list):
                task_plan = [str(item).strip() for item in task_plan_raw if str(item).strip()]
            else:
                task_plan = []
            if not task_plan:
                task_plan = default_task_plan
            if len(candidates) < min_factors:
                duplication_feedback_items.append(
                    f"Factor count constraint failed: got {len(candidates)}, required at least {min_factors}."
                )
                continue
            deduplicated = []
            consistency_events: list[dict[str, object]] = []
            pending_consistency_inputs: list[dict[str, object]] = []
            seen_name_set: set[str] = set()
            anchor_feature_set = set(anchor_input_features)
            total_candidate_limit = max_factors + counterfactual_factor_count + adjacent_factor_count
            for candidate in candidates[:total_candidate_limit]:
                diagnostics["local_gate_count"] += 1
                factor_name = str(candidate.get("factor_name", "")).strip()
                expression = self._normalize_factor_expression_literals(str(candidate.get("expression", "")).strip())
                description = str(candidate.get("description", ""))
                formulation = self._normalize_factor_expression_literals(str(candidate.get("formulation", "")))
                variables = candidate.get("variables", {})
                validation_warnings: list[str] = []
                requested_alignment = str(candidate.get("theory_alignment", "mainline") or "mainline").strip().lower()
                if requested_alignment not in {"mainline", "counterfactual", "adjacent"}:
                    requested_alignment = "mainline"
                critique_text = self._build_counterfactual_critique_text(
                    current_hypothesis=hypothesis,
                    expected_direction=str(hypothesis_structured.get("expected_direction", "")),
                )
                if not factor_name:
                    duplication_feedback_items.append("Factor name is empty.")
                    continue
                lowered = factor_name.lower()
                if lowered in previous_name_set or lowered in seen_name_set:
                    duplication_feedback_items.append(f"Duplicate factor name detected: {factor_name}")
                    continue
                expression_variables = self._extract_expression_variables(expression)
                if supported_feature_variables:
                    unsupported_variables = expression_variables - supported_feature_variables
                    if unsupported_variables:
                        unsupported_feedback = self._unsupported_variable_feedback(
                            factor_name=factor_name,
                            unsupported_variables=unsupported_variables,
                            supported_variables=supported_feature_variables,
                        )
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=expression,
                                stage="dataset_feature_whitelist",
                                passed=False,
                                feedback=unsupported_feedback,
                            )
                        )
                        duplication_feedback_items.append(unsupported_feedback)
                        continue
                has_anchor_feature = bool(expression_variables & anchor_feature_set)
                has_anchor_pattern = self._expression_matches_anchor_patterns(expression, anchor_expression_patterns)
                if (anchor_feature_set or anchor_expression_patterns) and refinement_stage in {"parameter_tuning", "signal_enhancement"}:
                    if not has_anchor_feature and not has_anchor_pattern:
                        expected_anchor_text = []
                        if anchor_input_features:
                            expected_anchor_text.append(f"input features {anchor_input_features}")
                        if anchor_expression_patterns:
                            expected_anchor_text.append(f"expression patterns {anchor_expression_patterns}")
                        anchor_feedback = (
                            f"Evolution state violation for {factor_name}: expression must retain at least one "
                            f"verifiable anchor from {'; '.join(expected_anchor_text)} during {refinement_stage}."
                        )
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=expression,
                                stage="evolution_anchor",
                                passed=True,
                                feedback=anchor_feedback,
                            )
                        )
                        duplication_feedback_items.append(f"Advisory for LLM verifier: {anchor_feedback}")
                        validation_warnings.append("continuity_anchor_not_matched")
                    if "variable_expansion" not in allowed_mutation_types:
                        new_variables = expression_variables - anchor_feature_set
                        if len(new_variables) > 1:
                            variable_feedback = (
                                f"Evolution state violation for {factor_name}: too many new variables introduced "
                                f"{sorted(new_variables)}. Allowed mutation types: {allowed_mutation_types}."
                            )
                            quality_gate_records.append(
                                self._build_quality_gate_record(
                                    factor_name=factor_name,
                                    expression=expression,
                                    stage="evolution_anchor",
                                    passed=True,
                                    feedback=variable_feedback,
                                )
                            )
                            duplication_feedback_items.append(f"Advisory for LLM verifier: {variable_feedback}")
                            validation_warnings.append("new_variable_expansion_warning")
                guidance_violation = self._evolution_guidance_violation(
                    phase=phase,
                    mutation_type=mutation_type,
                    expression=expression,
                    expression_variables=expression_variables,
                    anchor_input_features=anchor_input_features,
                    anchor_expression_patterns=anchor_expression_patterns,
                    evolution_plan_summaries=evolution_plan_summaries,
                )
                if guidance_violation:
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="evolution_guidance",
                            passed=True,
                            feedback=guidance_violation,
                        )
                    )
                    duplication_feedback_items.append(f"Advisory for LLM verifier: {guidance_violation} Factor: {factor_name}.")
                novelty_violation, novelty_metrics = self._evolution_novelty_violation(
                    phase=phase,
                    mutation_type=mutation_type,
                    expression=expression,
                    expression_variables=expression_variables,
                    parent_expressions=parent_expressions,
                    evolution_plan_summaries=evolution_plan_summaries,
                )
                if novelty_violation:
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="evolution_novelty",
                            passed=False,
                            feedback=novelty_violation,
                            eval_dict=novelty_metrics,
                        )
                    )
                    duplication_feedback_items.append(f"{novelty_violation} Factor: {factor_name}.")
                    continue
                if fast_mode:
                    symbol_validation = self.expression_quality_gate.validate_symbols(
                        expression,
                        allowed_features=allowed_input_features,
                        allowed_functions=allowed_runtime_functions,
                    )
                    if not symbol_validation.passed:
                        duplication_feedback_items.append(
                            f"Unsupported expression symbols for {factor_name}: {symbol_validation.to_feedback()}. "
                            f"Allowed raw features: {sorted(allowed_input_features)}."
                        )
                        continue
                    gate_result = generation_gate.evaluate(expression)
                    if not gate_result.passed:
                        feedback = "; ".join(gate_result.violations)
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=expression,
                                stage="generation_gate",
                                passed=False,
                                feedback=feedback,
                                eval_dict=gate_result.as_dict(),
                            )
                        )
                        duplication_feedback_items.append(f"Generation gate failed for {factor_name}: {feedback}")
                        continue
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="fast_mode",
                            passed=True,
                            feedback="Fast mode: expression quality gate skipped after evolution-guidance checks.",
                        )
                    )
                    seen_name_set.add(lowered)
                    reference_expressions_current.append(expression)
                    diagnostics["local_accept_count"] += 1
                    pending_consistency_inputs.append(
                        {
                            "factor_name": factor_name,
                            "factor_description": description,
                            "factor_formulation": formulation,
                            "description": description,
                            "formulation": formulation,
                            "expression": expression,
                            "factor_expression": expression,
                            "variables": variables,
                            "validation_warnings": validation_warnings,
                            "retained_for_backtest": True,
                            "theory_alignment": requested_alignment,
                            "counterfactual_reason": str(candidate.get("counterfactual_reason", ""))
                            if requested_alignment == "counterfactual"
                            else "",
                            "counterfactual_thesis": str(candidate.get("counterfactual_thesis", ""))
                            or (critique_text["thesis"] if requested_alignment == "counterfactual" else ""),
                            "counterfactual_mechanism": str(candidate.get("counterfactual_mechanism", ""))
                            or (critique_text["mechanism"] if requested_alignment == "counterfactual" else ""),
                            "counterfactual_critique": str(candidate.get("counterfactual_critique", ""))
                            or (
                                "Model-written counterfactual expression critiques the current thesis."
                                if requested_alignment == "counterfactual"
                                else ""
                            ),
                            "counterfactual_expected_direction": str(candidate.get("counterfactual_expected_direction", ""))
                            or (critique_text["expected_direction"] if requested_alignment == "counterfactual" else ""),
                            "matched_anchor_features": sorted(expression_variables & anchor_feature_set),
                            "matched_anchor_pattern": has_anchor_pattern,
                            "expected_anchor_features": sorted(anchor_feature_set),
                            "expected_anchor_patterns": list(anchor_expression_patterns),
                            "advisory_records": [],
                        }
                    )
                    continue
                parsable, parse_feedback = is_parsable(
                    expression,
                )
                if not parsable:
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="parse",
                            passed=False,
                            feedback=parse_feedback,
                        )
                    )
                    duplication_feedback_items.append(
                        f"Parse check failed for {factor_name}: {parse_feedback}. Expression: {expression}"
                    )
                    continue
                symbol_validation = self.expression_quality_gate.validate_symbols(
                    expression,
                    allowed_features=allowed_input_features,
                    allowed_functions=allowed_runtime_functions,
                )
                if not symbol_validation.passed:
                    duplication_feedback_items.append(
                        f"Unsupported expression symbols for {factor_name}: {symbol_validation.to_feedback()}. "
                        f"Allowed raw features: {sorted(allowed_input_features)}."
                    )
                    continue
                gate_result = generation_gate.evaluate(expression)
                if not gate_result.passed:
                    feedback = "; ".join(gate_result.violations)
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="generation_gate",
                            passed=False,
                            feedback=feedback,
                            eval_dict=gate_result.as_dict(),
                        )
                    )
                    duplication_feedback_items.append(f"Generation gate failed for {factor_name}: {feedback}")
                    continue
                eval_dict = evaluate_expression(
                    expression,
                    reference_expressions=reference_expressions_current,
                )
                hard_metric_advisory_passed = is_expression_acceptable(eval_dict, prompt_meta=prompt_meta)
                if not hard_metric_advisory_passed:
                    quality_feedback = self.tools.render_expression_feedback(eval_dict, prompt_meta, expression)
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="expression_quality_advisory",
                            passed=True,
                            feedback=quality_feedback,
                            eval_dict=eval_dict,
                        )
                    )
                    duplication_feedback_items.append(f"Advisory for LLM verifier: {quality_feedback}")
                else:
                    quality_gate_records.append(
                        self._build_quality_gate_record(
                            factor_name=factor_name,
                            expression=expression,
                            stage="expression_quality_advisory",
                            passed=True,
                            feedback="Expression metrics are within advisory quality ranges.",
                            eval_dict=eval_dict,
                        )
                    )
                    validation_warnings.append("quality_gate_warning")
                seen_name_set.add(lowered)
                reference_expressions_current.append(expression)
                diagnostics["local_accept_count"] += 1
                pending_consistency_inputs.append(
                    {
                        "factor_name": factor_name,
                        "factor_description": description,
                        "factor_formulation": formulation,
                        "description": description,
                        "formulation": formulation,
                        "expression": expression,
                        "factor_expression": expression,
                        "variables": variables,
                        "validation_warnings": validation_warnings,
                        "retained_for_backtest": True,
                        "theory_alignment": requested_alignment,
                        "counterfactual_reason": str(candidate.get("counterfactual_reason", ""))
                        if requested_alignment == "counterfactual"
                        else "",
                        "counterfactual_thesis": str(candidate.get("counterfactual_thesis", ""))
                        or (critique_text["thesis"] if requested_alignment == "counterfactual" else ""),
                        "counterfactual_mechanism": str(candidate.get("counterfactual_mechanism", ""))
                        or (critique_text["mechanism"] if requested_alignment == "counterfactual" else ""),
                        "counterfactual_critique": str(candidate.get("counterfactual_critique", ""))
                        or (
                            "Model-written counterfactual expression critiques the current thesis."
                            if requested_alignment == "counterfactual"
                            else ""
                        ),
                        "counterfactual_expected_direction": str(candidate.get("counterfactual_expected_direction", ""))
                        or (critique_text["expected_direction"] if requested_alignment == "counterfactual" else ""),
                        "matched_anchor_features": sorted(expression_variables & anchor_feature_set),
                        "matched_anchor_pattern": has_anchor_pattern,
                        "expected_anchor_features": sorted(anchor_feature_set),
                        "expected_anchor_patterns": list(anchor_expression_patterns),
                        "advisory_records": [
                            record for record in quality_gate_records if record.get("factor_name") == factor_name
                        ],
                    }
                )
            consistency_call_count = 1 if consistency_enabled and pending_consistency_inputs else 0
            if consistency_call_count:
                diagnostics["llm_call_count"] += consistency_call_count
                diagnostics["repair_call_count"] += consistency_call_count
            consistency_results = self._run_consistency_gate(
                enabled=consistency_enabled,
                model_client=model_client,
                scenario=scenario,
                hypothesis=hypothesis,
                phase=phase,
                mutation_type=mutation_type,
                evolution_guidance=evolution_guidance,
                parent_trajectory_summaries=parent_trajectory_summaries,
                factors=pending_consistency_inputs,
            )
            deduplicated = []
            for pending in pending_consistency_inputs:
                factor_name = str(pending.get("factor_name", "")).strip()
                expression = str(pending.get("expression", ""))
                variables = pending.get("variables", {})
                validation_warnings = list(pending.get("validation_warnings", []))
                requested_alignment = str(pending.get("theory_alignment", "mainline"))
                consistency_result = consistency_results.get(
                    factor_name,
                    {
                        "passed": False,
                        "feedback": "LLM verifier returned no decision for this factor.",
                        "corrected_expression": expression,
                        "verifier": "llm_missing",
                    },
                )
                consistency_passed = bool(consistency_result.get("passed", False))
                consistency_feedback = str(consistency_result.get("feedback", "")).strip() or (
                    "LLM verifier passed." if consistency_passed else "LLM verifier rejected."
                )
                quality_gate_records.append(
                    self._build_quality_gate_record(
                        factor_name=factor_name,
                        expression=expression,
                        stage="llm_verifier",
                        passed=consistency_passed,
                        feedback=consistency_feedback,
                        eval_dict=consistency_result,
                    )
                )
                corrected_expression = self._normalize_factor_expression_literals(
                    str(consistency_result.get("corrected_expression", expression))
                )
                expression_variables = self._extract_expression_variables(expression)
                if corrected_expression != expression:
                    corrected_variables = self._extract_expression_variables(corrected_expression)
                    if supported_feature_variables:
                        unsupported_corrected_variables = corrected_variables - supported_feature_variables
                        if unsupported_corrected_variables:
                            unsupported_feedback = self._unsupported_variable_feedback(
                                factor_name=factor_name,
                                unsupported_variables=unsupported_corrected_variables,
                                supported_variables=supported_feature_variables,
                            )
                            quality_gate_records.append(
                                self._build_quality_gate_record(
                                    factor_name=factor_name,
                                    expression=corrected_expression,
                                    stage="corrected_dataset_feature_whitelist",
                                    passed=False,
                                    feedback=unsupported_feedback,
                                )
                            )
                            duplication_feedback_items.append(unsupported_feedback)
                            continue
                    parsable, parse_feedback = is_parsable(corrected_expression)
                    if not parsable:
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=corrected_expression,
                                stage="corrected_parse",
                                passed=False,
                                feedback=parse_feedback,
                            )
                        )
                        duplication_feedback_items.append(
                            f"Corrected expression parse failed for {factor_name}: {parse_feedback}. Expression: {corrected_expression}"
                        )
                        continue
                    symbol_validation = self.expression_quality_gate.validate_symbols(
                        corrected_expression,
                        allowed_features=allowed_input_features,
                        allowed_functions=allowed_runtime_functions,
                    )
                    if not symbol_validation.passed:
                        duplication_feedback_items.append(
                            f"Unsupported expression symbols for corrected {factor_name}: {symbol_validation.to_feedback()}. "
                            f"Allowed raw features: {sorted(allowed_input_features)}."
                        )
                        continue
                    gate_result = generation_gate.evaluate(corrected_expression)
                    if not gate_result.passed:
                        feedback = "; ".join(gate_result.violations)
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=corrected_expression,
                                stage="corrected_generation_gate",
                                passed=False,
                                feedback=feedback,
                                eval_dict=gate_result.as_dict(),
                            )
                        )
                        duplication_feedback_items.append(f"Corrected generation gate failed for {factor_name}: {feedback}")
                        continue
                    eval_dict = evaluate_expression(
                        corrected_expression,
                        reference_expressions=reference_expressions_current,
                    )
                    corrected_advisory_passed = is_expression_acceptable(eval_dict, prompt_meta=prompt_meta)
                    if not corrected_advisory_passed:
                        quality_feedback = self.tools.render_expression_feedback(eval_dict, prompt_meta, corrected_expression)
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=corrected_expression,
                                stage="corrected_expression_quality_advisory",
                                passed=True,
                                feedback=quality_feedback,
                                eval_dict=eval_dict,
                            )
                        )
                        duplication_feedback_items.append(f"Advisory for LLM verifier: {quality_feedback}")
                        validation_warnings.append("corrected_quality_gate_warning")
                    expression = corrected_expression
                    expression_variables = corrected_variables
                    novelty_violation, novelty_metrics = self._evolution_novelty_violation(
                        phase=phase,
                        mutation_type=mutation_type,
                        expression=expression,
                        expression_variables=expression_variables,
                        parent_expressions=parent_expressions,
                        evolution_plan_summaries=evolution_plan_summaries,
                    )
                    if novelty_violation:
                        quality_gate_records.append(
                            self._build_quality_gate_record(
                                factor_name=factor_name,
                                expression=expression,
                                stage="corrected_evolution_novelty",
                                passed=False,
                                feedback=novelty_violation,
                                eval_dict=novelty_metrics,
                            )
                        )
                        duplication_feedback_items.append(f"{novelty_violation} Corrected factor: {factor_name}.")
                        continue
                has_anchor_pattern = self._expression_matches_anchor_patterns(
                    expression,
                    anchor_expression_patterns,
                )
                if not consistency_passed:
                    duplication_feedback_items.append(
                        f"LLM verifier rejected {factor_name}: {consistency_feedback}"
                    )
                    validation_warnings.append("consistency_gate_warning")
                consistency_events.append(
                    {
                        "attempt": attempt_count,
                        "factor_name": factor_name,
                        "checked": bool(consistency_enabled),
                        "passed": bool(consistency_passed),
                        "corrected_expression": expression,
                        "violations": list(validation_warnings),
                        "matched_anchor_features": sorted(expression_variables & anchor_feature_set),
                        "matched_anchor_pattern": has_anchor_pattern,
                        "research_stage": research_stage,
                        "refinement_stage": refinement_stage,
                        "allowed_mutation_types": allowed_mutation_types,
                        "theory_alignment": requested_alignment,
                    }
                )
                pending["expression"] = expression
                pending["factor_expression"] = expression
                pending["validation_warnings"] = validation_warnings
                pending["matched_anchor_features"] = sorted(expression_variables & anchor_feature_set)
                pending["matched_anchor_pattern"] = has_anchor_pattern
                deduplicated.append(pending)
            mainline_count = sum(1 for item in deduplicated if item.get("theory_alignment", "mainline") == "mainline")
            if mainline_count < min_factors and attempt_count < max_attempts:
                duplication_feedback_items.append(
                    f"Only {mainline_count} legal mainline factors survived validation; required at least {min_factors}. "
                    "Keep legal factors and repair the failed mainline ones."
                )
                continue
            model_counterfactual_count = sum(
                1 for item in deduplicated if item.get("theory_alignment") == "counterfactual"
            )
            group_shortfalls: list[dict[str, object]] = []
            if mainline_count < min_factors:
                group_shortfalls.append(
                    {
                        "group": "mainline",
                        "actual": mainline_count,
                        "required": min_factors,
                        "severity": "hard",
                    }
                )
            if model_counterfactual_count < counterfactual_factor_count:
                group_shortfalls.append(
                    {
                        "group": "counterfactual",
                        "actual": model_counterfactual_count,
                        "required": counterfactual_factor_count,
                        "severity": "hard",
                    }
                )
            if group_shortfalls and attempt_count < max_attempts:
                shortfall_details = "; ".join(
                    f"{item['group']} actual={item['actual']} required={item['required']}"
                    for item in group_shortfalls
                )
                cf_hint = ""
                if any(item["group"] == "counterfactual" for item in group_shortfalls):
                    cf_hint = (
                        " COUNTERFACTUAL REPAIR GUIDE: Each counterfactual factor must have a DIFFERENT "
                        "economic mechanism from the mainline (e.g., if mainline predicts reversal via "
                        "exhaustion, counterfactual argues continuation via informed demand, OR argues "
                        "that price alone drives reversal without volume). Required fields: "
                        "'counterfactual_thesis', 'counterfactual_mechanism', 'counterfactual_critique', "
                        "'counterfactual_expected_direction'. The expression must be built from the rival "
                        "thesis — do NOT copy the mainline expression or use a single-variable proxy. "
                        "STRICT: use ONLY variables from the runtime whitelist (supported_feature_variables_text); "
                        "do NOT invent fields like $volume_change_4h — only $volume_change_1h and "
                        "$volume_change_24h exist."
                    )
                duplication_feedback_items.append(
                    f"Factor group shortfall: {shortfall_details}.{cf_hint}"
                )
                continue
            adjacent_candidates = self._build_adjacent_factor_candidates(
                count=adjacent_factor_count,
                round_idx=round_idx,
                allowed_input_features=allowed_input_features,
                anchor_input_features=anchor_input_features,
                anchor_expression_patterns=anchor_expression_patterns,
            )
            for adjacent in adjacent_candidates:
                lowered = str(adjacent.get("factor_name", "")).strip().lower()
                if not lowered or lowered in previous_name_set or lowered in seen_name_set:
                    continue
                seen_name_set.add(lowered)
                deduplicated.append(adjacent)
            group_counts = {
                "mainline": sum(1 for item in deduplicated if item.get("theory_alignment") == "mainline"),
                "counterfactual": sum(1 for item in deduplicated if item.get("theory_alignment") == "counterfactual"),
                "adjacent": sum(1 for item in deduplicated if item.get("theory_alignment") == "adjacent"),
            }
            group_shortfall_diagnostics = {
                "status": "shortfall" if group_shortfalls else "ok",
                "shortfalls": group_shortfalls,
                "message": (
                    "MISSING REQUIRED FACTOR GROUPS: "
                    + "; ".join(
                        f"{item['group']} actual={item['actual']} required={item['required']}"
                        for item in group_shortfalls
                    )
                    if group_shortfalls
                    else ""
                ),
            }
            experiment_spec = {
                "target_hypothesis": hypothesis,
                "research_stage": research_stage,
                "factors": deduplicated,
                "factor_names": [str(item.get("factor_name", "")) for item in deduplicated],
                "deduplicated_against": previous_factor_names,
                "attempt_count": attempt_count,
                "counterfactual_factor_count": sum(
                    1 for item in deduplicated if item.get("theory_alignment") == "counterfactual"
                ),
                "counterfactual_factor_target_count": counterfactual_factor_count,
                "adjacent_factor_count": sum(
                    1 for item in deduplicated if item.get("theory_alignment") == "adjacent"
                ),
                "adjacent_factor_target_count": adjacent_factor_count,
                "factor_group_counts": group_counts,
                "factor_group_shortfall_diagnostics": group_shortfall_diagnostics,
                "factor_group_targets": {
                    "mainline_min": min_factors,
                    "mainline_max": max_factors,
                    "counterfactual": counterfactual_factor_count,
                    "adjacent": adjacent_factor_count,
                    "total": max_factors + counterfactual_factor_count + adjacent_factor_count,
                },
                "requested_factor_range": {
                    "min_factors": min_factors,
                    "max_factors": max_factors + counterfactual_factor_count + adjacent_factor_count,
                    "mainline_min_factors": min_factors,
                    "mainline_max_factors": max_factors,
                    "valid_after_filtering": len(deduplicated),
                    "accepted_below_minimum": bool(group_shortfalls),
                    "retained_with_warnings_count": sum(1 for item in deduplicated if item.get("validation_warnings")),
                },
                "consistency_gate": {
                    "enabled": consistency_enabled,
                    "implemented": True,
                    "status": (
                        "partial"
                        if group_shortfalls
                        else ("llm_verifier_passed" if consistency_enabled else "disabled")
                    ),
                    "group_shortfall_diagnostics": group_shortfall_diagnostics,
                    "checked_factor_count": len(consistency_events),
                    "passed_factor_count": sum(1 for item in consistency_events if item.get("passed")),
                    "retained_with_warnings_count": sum(1 for item in deduplicated if item.get("validation_warnings")),
                    "event_count": len(consistency_events),
                    "llm_call_count": consistency_call_count,
                    "diagnostics": dict(diagnostics),
                    "events_tail": consistency_events[-5:],
                    "quality_gate": {
                        "enabled": True,
                        "mode": "llm_batch_verifier_with_metric_advisory",
                        "status": "passed",
                        "records": list(quality_gate_records),
                        "rejection_feedback": list(duplication_feedback_items),
                    },
                },
            }
            break
        if not deduplicated:
            private_context.payload["last_experiment_validation_feedback"] = duplication_feedback_items
            private_context.payload["last_experiment_quality_gate"] = {
                "enabled": True,
                "status": "failed",
                "records": list(quality_gate_records),
                "rejection_feedback": list(duplication_feedback_items),
            }
            raise RuntimeError(
                f"Experiment designer failed to generate valid factors after {max_attempts} attempts. "
                f"Feedback: {'; '.join(duplication_feedback_items[-5:])}"
            )
        final_shortfall = {}
        if isinstance(experiment_spec, dict):
            final_shortfall = dict(experiment_spec.get("factor_group_shortfall_diagnostics", {}))
        if final_shortfall.get("status") == "shortfall":
            raise RuntimeError(
                "Experiment designer failed to satisfy required factor groups after "
                f"{max_attempts} attempts: {final_shortfall.get('message', '')}. "
                "Retry the designer rather than accepting an incomplete mainline/counterfactual group."
            )
        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        experiment.target_hypothesis = hypothesis
        experiment.factors = [ExperimentFactorTask.from_dict(item) for item in deduplicated]
        experiment.factor_names = [str(item.get("factor_name", "")) for item in deduplicated]
        experiment.deduplicated_against = list(previous_factor_names)
        experiment.task_plan = list(task_plan)
        experiment.attempt_count = int(attempt_count)
        experiment.consistency_gate = dict(experiment_spec.get("consistency_gate", {}))
        experiment.extra = dict(experiment.extra)
        experiment.extra["quality_gate"] = dict(experiment.consistency_gate.get("quality_gate", {}))
        experiment.extra["factor_group_counts"] = dict(experiment_spec.get("factor_group_counts", {}))
        experiment.extra["factor_group_targets"] = dict(experiment_spec.get("factor_group_targets", {}))
        experiment.extra["factor_group_shortfall_diagnostics"] = dict(
            experiment_spec.get("factor_group_shortfall_diagnostics", {})
        )
        experiment.phase = phase
        experiment.round_idx = round_idx
        experiment.trajectory_id = trajectory_id
        experiment.parent_ids = parent_ids
        if not experiment.experiment_id:
            experiment.experiment_id = QlibFactorExperiment.generate_id(
                round_idx=round_idx,
                phase=phase,
                seed=trajectory_id or hypothesis[:24],
            )
        experiment.update_timestamps()
        private_context.payload["latest_experiment_spec"] = experiment_spec
        private_context.payload["latest_experiment"] = experiment.to_dict()
        private_context.payload["last_experiment_attempt_count"] = attempt_count
        private_context.payload["last_experiment_validation_feedback"] = duplication_feedback_items
        private_context.payload["last_experiment_research_stage"] = research_stage
        private_context.payload["experiment_designer_diagnostics"] = dict(diagnostics)
        return AgentResult(
            shared_updates={
                "experiment_spec": experiment_spec,
                "task_plan": task_plan,
                "experiment_id": experiment.experiment_id,
                "research_stage": research_stage,
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name},
        )
