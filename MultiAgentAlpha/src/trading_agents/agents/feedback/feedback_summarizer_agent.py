from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from trading_agents.agents.tools.factor_expression_ast import calculate_symbol_length
from trading_agents.agents.tools.factor_expression_ast import count_base_features
from trading_agents.core import BaseAgent, ModelClient
from trading_agents.infra.factor_signal import build_loop_signal_diagnostics
from trading_agents.infra.factor_signal import WEAK_RANK_IC_THRESHOLD
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.prompts import PromptStore, render_prompt
from trading_agents.schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


class FeedbackSummarizerAgent(BaseAgent):
    _DISALLOWED_LANGUAGE_SCRIPT_RE = re.compile(
        r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF\uD7B0-\uD7FF\u3040-\u30FF\u31F0-\u31FF]"
    )

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parent.parent.parent / "prompts")

    def _find_disallowed_language_scripts(self, value: object, path: str = "") -> list[str]:
        if isinstance(value, dict):
            findings: list[str] = []
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                findings.extend(self._find_disallowed_language_scripts(item, child_path))
            return findings
        if isinstance(value, list):
            findings = []
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]"
                findings.extend(self._find_disallowed_language_scripts(item, child_path))
            return findings
        text = str(value or "")
        match = self._DISALLOWED_LANGUAGE_SCRIPT_RE.search(text)
        return [path or "<root>"] if match else []

    def _validate_no_disallowed_language_scripts(self, payload: dict[str, object]) -> None:
        bad_fields = self._find_disallowed_language_scripts(payload)
        if bad_fields:
            raise ValueError(f"disallowed non-Chinese language script detected in feedback output: {bad_fields}")

    def _to_bool(self, value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return default

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _metrics_score_tuple(self, metrics: dict[str, float]) -> tuple[float, ...]:
        return (
            self._to_float(metrics.get("Rank ICIR"), self._to_float(metrics.get("ICIR"), 0.0)),
            self._to_float(metrics.get("Rank IC"), self._to_float(metrics.get("IC"), 0.0)),
            self._to_float(metrics.get("sharpe"), 0.0),
            -self._to_float(metrics.get("drawdown"), 0.0),
            self._to_float(metrics.get("coverage"), 0.0),
            -self._to_float(metrics.get("turnover"), 0.0),
        )

    def _extract_sota_entry(self, history: list[dict[str, object]]) -> dict[str, object] | None:
        best_item: dict[str, object] | None = None
        best_score: tuple[float, ...] | None = None
        for item in history:
            metrics = item.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            normalized_metrics = {str(k): self._to_float(v) for k, v in metrics.items()}
            score = self._metrics_score_tuple(normalized_metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_item = dict(item)
                best_item["metrics"] = normalized_metrics
        return best_item

    def _process_results(
        self,
        current_metrics: dict[str, object],
        sota_metrics: dict[str, float] | None,
    ) -> str:
        important_metrics = [
            "annualized_return",
            "information_ratio",
            "max_drawdown",
            "IC",
            "Rank IC",
            "Rank ICIR",
            "sharpe",
            "drawdown",
        ]
        rows: list[dict[str, object]] = []
        for metric in important_metrics:
            if metric not in current_metrics and (not sota_metrics or metric not in sota_metrics):
                continue
            current_value = self._to_float(current_metrics.get(metric)) if metric in current_metrics else None
            sota_value = self._to_float(sota_metrics.get(metric)) if sota_metrics and metric in sota_metrics else None
            bigger = None
            if current_value is not None and sota_value is not None:
                bigger = "当前结果" if current_value > sota_value else "历史最优结果"
            elif current_value is not None:
                bigger = "当前结果"
            elif sota_value is not None:
                bigger = "历史最优结果"
            rows.append(
                {
                    "metric": metric,
                    "当前结果": current_value,
                    "历史最优结果": sota_value,
                    "较优列名（未考虑方向）": bigger,
                }
            )
        if not rows:
            rows.append(
                {
                    "metric": "兜底",
                    "当前结果": {str(k): current_metrics.get(k) for k in current_metrics},
                    "历史最优结果": sota_metrics or {},
                    "较优列名（未考虑方向）": "不适用",
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _build_task_details(
        self,
        factor_implementation: dict[str, object],
        experiment_spec: dict[str, object],
    ) -> list[dict[str, object]]:
        implementation_items = factor_implementation.get("implementations", [])
        implementation_map: dict[str, dict[str, object]] = {}
        if isinstance(implementation_items, list):
            for item in implementation_items:
                if not isinstance(item, dict):
                    continue
                factor_name = str(item.get("factor_name", "")).strip()
                if factor_name:
                    implementation_map[factor_name.lower()] = item
        details: list[dict[str, object]] = []
        factors_block = experiment_spec.get("factors", [])
        if isinstance(factors_block, list):
            for factor in factors_block:
                if not isinstance(factor, dict):
                    continue
                factor_name = str(factor.get("factor_name", "") or factor.get("name", "")).strip()
                if not factor_name:
                    continue
                implementation = implementation_map.get(factor_name.lower(), {})
                status = str(implementation.get("status", "")).strip().lower()
                expression = str(factor.get("expression", "") or implementation.get("expression", ""))
                row = {
                    "factor_name": factor_name,
                    "factor_description": str(factor.get("description", "")),
                    "factor_formulation": str(factor.get("formulation", "")),
                    "variables": factor.get("variables", {}),
                    "factor_expression": expression,
                    "factor_implementation": str(status == "generated"),
                }
                complexity_feedback = self._build_complexity_feedback(expression)
                if complexity_feedback:
                    row["complexity_feedback"] = complexity_feedback
                details.append(row)
        if details:
            return details
        for _factor_key, implementation in implementation_map.items():
            status = str(implementation.get("status", "")).strip().lower()
            expression = str(implementation.get("expression", ""))
            row = {
                "factor_name": str(implementation.get("factor_name", "")),
                "factor_description": "",
                "factor_formulation": "",
                "variables": implementation.get("variables", {}),
                "factor_expression": expression,
                "factor_implementation": str(status == "generated"),
            }
            complexity_feedback = self._build_complexity_feedback(expression)
            if complexity_feedback:
                row["complexity_feedback"] = complexity_feedback
            details.append(row)
        return details

    def _build_complexity_feedback(self, expression: str) -> str:
        expr = str(expression).strip()
        if not expr:
            return ""
        warnings: list[str] = []
        symbol_length = calculate_symbol_length(expr)
        if symbol_length > 300:
            warnings.append(
                f"Symbol Length (SL) Check Failed: Symbol length ({symbol_length}) exceeds threshold (300)."
            )
        base_features = count_base_features(expr)
        if base_features > 6:
            warnings.append(
                f"Base Features Count (ER) Check Failed: Number of base features ({base_features}) exceeds threshold (6)."
            )
        return "\n".join(warnings)

    def _render_task_details_text(self, task_details: list[dict[str, object]]) -> str:
        if not task_details:
            return "None"
        rows: list[str] = []
        for task in task_details:
            lines = [
                f"- {task.get('factor_name', '')}: {task.get('factor_description', '')}",
                f"  - Factor Formulation: {task.get('factor_formulation', '')}",
                f"  - Variables: {task.get('variables', {})}",
                f"  - Factor Implementation: {task.get('factor_implementation', '')}",
            ]
            complexity_feedback = str(task.get("complexity_feedback", "")).strip()
            if complexity_feedback:
                lines.append(f"  - Complexity Feedback: {complexity_feedback}")
            rows.append("\n".join(lines))
        return "\n".join(rows)

    def _build_feedback_from_payload(
        self,
        payload: dict[str, Any],
        *,
        fallback_reason: str,
    ) -> tuple[dict[str, object], str, str]:
        observations = str(payload.get("Observations", "No observations provided")).strip()
        hypothesis_evaluation = str(payload.get("Feedback for Hypothesis", payload.get("hypothesis_evaluation", "No feedback provided"))).strip()
        new_hypothesis = str(payload.get("New Hypothesis", payload.get("new_hypothesis", ""))).strip()
        reason = str(payload.get("Reasoning", payload.get("reason", fallback_reason))).strip()
        hypothesis_supported = self._to_bool(
            payload.get("Support Hypothesis", payload.get("hypothesis_supported", False)),
            False,
        )
        feedback = {
            "hypothesis_supported": hypothesis_supported,
            "summary": hypothesis_evaluation or "No feedback provided",
            "observations": observations or "No observations provided",
            "reason": reason or fallback_reason,
            "new_hypothesis": new_hypothesis,
            "target_status": str(payload.get("Target Status", payload.get("target_status", ""))).strip(),
            "metric_gaps": str(payload.get("Metric Gaps", payload.get("metric_gaps", ""))).strip(),
            "preserve_segment": str(payload.get("Preserve Segment", payload.get("preserve_segment", ""))).strip(),
            "rewrite_instruction": str(payload.get("Rewrite Instruction", payload.get("rewrite_instruction", ""))).strip(),
            "next_action": str(payload.get("Next Action", payload.get("next_action", ""))).strip(),
            "semantic_diagnosis": str(payload.get("Semantic Diagnosis", payload.get("semantic_diagnosis", ""))).strip(),
            "semantic_diagnosis_reason": str(
                payload.get("Semantic Diagnosis Reason", payload.get("semantic_diagnosis_reason", ""))
            ).strip(),
        }
        next_hypothesis_hint = new_hypothesis or "未提供新假说"
        distilled_knowledge = f"{feedback['observations']}; {feedback['reason']}"
        return feedback, next_hypothesis_hint, distilled_knowledge

    def _is_higher_better(self, metric_name: str) -> bool:
        return metric_name not in {"max_drawdown", "drawdown", "turnover"}

    def _compare_metric(self, metric_name: str, current_value: float | None, sota_value: float | None) -> bool:
        if current_value is None or sota_value is None:
            return False
        if self._is_higher_better(metric_name):
            return current_value > sota_value
        return current_value < sota_value

    def _infer_semantic_diagnosis(
        self,
        *,
        hypothesis_text: str,
        metrics: dict[str, float],
        backtest_report: dict[str, object],
        history: list[dict[str, object]],
        signal_diagnostics: dict[str, object] | None = None,
    ) -> tuple[str, str]:
        backtest_status = str(backtest_report.get("status", "")).strip().lower()
        if backtest_status in {"failed", "partial"}:
            return (
                "implementation_unstable",
                "The current loop cannot cleanly validate the report thesis because factor execution/backtest was unstable or only partially evaluated.",
            )
        signal_diagnostics = signal_diagnostics or {}
        counterfactual_level = str(signal_diagnostics.get("counterfactual_challenge_level", "")).strip().lower()
        if counterfactual_level == "strong":
            return (
                "mechanism_refuted",
                "Counterfactual factors show a strong group-level challenge, so the current mechanism may need thesis-level revision.",
            )
        if counterfactual_level == "moderate":
            return (
                "mechanism_unclear",
                "Counterfactual factors are moderately stronger than mainline, so broaden proxy reconstruction before flipping the thesis.",
            )

        ic = self._to_float(metrics.get("IC"), 0.0)
        rank_ic = self._to_float(metrics.get("Rank IC"), 0.0)
        abs_ic = abs(ic)
        abs_rank_ic = abs(rank_ic)
        weak_signal = abs_rank_ic < WEAK_RANK_IC_THRESHOLD

        lower_hypothesis = str(hypothesis_text or "").lower()
        expects_reversal = any(token in lower_hypothesis for token in {"reversal", "mean reversion", "exhaustion"})
        expects_continuation = any(token in lower_hypothesis for token in {"continuation", "trend following", "momentum continuation"})

        if expects_reversal and ic > 0.0 and abs_ic >= 0.01:
            return (
                "direction_mismatch",
                "The measured factor direction conflicts with the stated reversal thesis, so the issue is more about directional interpretation than raw proxy strength.",
            )
        if expects_continuation and ic < 0.0 and abs_ic >= 0.01:
            return (
                "direction_mismatch",
                "The measured factor direction conflicts with the stated continuation thesis, so the issue is more about directional interpretation than raw proxy strength.",
            )

        recent_finished = [
            item
            for item in history[-3:]
            if str((item.get("backtest_report") or {}).get("status", "")).strip().lower() == "finished"
        ]
        recent_weak_finished = 0
        recent_mechanism_supported = 0
        for item in recent_finished:
            item_metrics = item.get("metrics", {})
            if not isinstance(item_metrics, dict):
                continue
            item_ic = abs(self._to_float(item_metrics.get("IC"), 0.0))
            item_rank_ic = abs(self._to_float(item_metrics.get("Rank IC"), 0.0))
            if item_rank_ic < WEAK_RANK_IC_THRESHOLD:
                recent_weak_finished += 1
            diagnosis = str(item.get("feedback_semantic_diagnosis", "")).strip().lower()
            if diagnosis.startswith("mechanism_supported"):
                recent_mechanism_supported += 1

        if recent_weak_finished >= 3 and weak_signal and not (expects_reversal or expects_continuation):
            return (
                "mechanism_unclear",
                "Multiple completed loops produced only weak signal and the expected mechanism direction is not clear enough to judge transferability.",
            )
        if recent_weak_finished >= 3 and weak_signal:
            return (
                "mechanism_supported_proxy_exhausted" if recent_mechanism_supported >= 1 else "mechanism_unclear",
                "The mechanism has not been cleanly refuted, but repeated weak results suggest the current proxy family may be exhausted.",
            )
        if weak_signal:
            return (
                "mechanism_supported_proxy_weak",
                "The report mechanism remains directionally plausible, but the current proxy expression is too weak or noisy to validate it strongly.",
            )
        return (
            "mechanism_supported_proxy_weak",
            "The mechanism is still usable, but the proxy design should be refined before drawing a thesis-level conclusion.",
        )

    def _build_group_level_summary(
        self,
        *,
        signal_diagnostics: dict[str, object],
        fallback_summary: str,
    ) -> str:
        group_performance = signal_diagnostics.get("group_performance", {})
        if not isinstance(group_performance, dict):
            return fallback_summary
        challenge_level = str(signal_diagnostics.get("counterfactual_challenge_level", "none")).strip() or "none"
        recommended_action = str(signal_diagnostics.get("recommended_action", "continue")).strip() or "continue"
        challenge_labels = {
            "none": "无",
            "weak": "弱",
            "moderate": "中等",
            "strong": "强",
        }
        action_labels = {
            "continue": "继续当前方向",
            "proxy_reconstruction": "重构代理变量",
            "parameter_tuning": "参数调优",
            "hypothesis_revision": "修订假说",
            "hypothesis_replacement": "替换假说",
            "hypothesis_reset": "重置假说",
        }
        group_labels = {
            "mainline": "主线组",
            "counterfactual": "反事实组",
            "adjacent": "相邻代理组",
        }
        parts = [
            (
                "组别诊断："
                f"反事实挑战={challenge_labels.get(challenge_level, challenge_level)}，"
                f"建议动作={action_labels.get(recommended_action, recommended_action)}。"
            )
        ]
        shortfall = signal_diagnostics.get("factor_group_shortfall_diagnostics", {})
        if isinstance(shortfall, dict) and shortfall.get("status") == "shortfall":
            parts.append(f"因子分组不足：{shortfall.get('message')}")
        for group_name in ("mainline", "counterfactual", "adjacent"):
            group = group_performance.get(group_name, {})
            if not isinstance(group, dict) or not group.get("count"):
                continue
            parts.append(
                (
                    f"{group_labels.get(group_name, group_name)}："
                    f"数量={group.get('count')}，"
                    f"最佳因子={group.get('best_factor')}，"
                    f"最佳 |Rank IC|={group.get('best_abs_rank_ic')}，"
                    f"中位 |Rank IC|={group.get('median_abs_rank_ic')}。"
                )
            )
        if fallback_summary:
            parts.append(str(fallback_summary))
        return " ".join(parts)

    def _count_recent_unsupported_loops(self, history: list[dict[str, object]], *, window: int = 3) -> int:
        unsupported = 0
        for item in reversed(history[-window:]):
            if self._to_bool(item.get("feedback_hypothesis_supported", False), False):
                break
            unsupported += 1
        return unsupported

    def _allow_base_reconstruction_reset(
        self,
        *,
        completed_feedback_loops: int,
        recommended_action: str,
        revision_recommended: bool,
        signal_diagnostics: dict[str, object],
    ) -> bool:
        if completed_feedback_loops <= 0:
            return True
        explicit_reset = str(signal_diagnostics.get("explicit_hard_reset", "")).strip().lower()
        if explicit_reset in {"true", "1", "yes", "hard_reset"}:
            return True
        challenge_level = str(signal_diagnostics.get("counterfactual_challenge_level", "")).strip().lower()
        return bool(
            revision_recommended
            and recommended_action in {"hypothesis_replacement", "hypothesis_reset"}
            and challenge_level == "strong"
        )

    def _guard_base_reconstruction_stage(
        self,
        *,
        candidate_stage: str,
        candidate_reason: str,
        completed_feedback_loops: int,
        recommended_action: str,
        revision_recommended: bool,
        signal_diagnostics: dict[str, object],
    ) -> tuple[str, str]:
        if candidate_stage != "base_reconstruction" or completed_feedback_loops <= 0:
            return candidate_stage, candidate_reason
        if self._allow_base_reconstruction_reset(
            completed_feedback_loops=completed_feedback_loops,
            recommended_action=recommended_action,
            revision_recommended=revision_recommended,
            signal_diagnostics=signal_diagnostics,
        ):
            return candidate_stage, candidate_reason
        return (
            "proxy_reconstruction",
            (
                "Blocked base_reconstruction after loop 1 because no explicit hard reset was allowed. "
                "Continue proxy reconstruction within the selected mechanism anchor."
            ),
        )

    def _compact_history_entry(self, entry: dict[str, object], *, keep_full: bool) -> dict[str, object]:
        compact = dict(entry)
        if keep_full:
            compact["history_entry_full"] = True
            return compact
        compact["history_entry_full"] = False
        hypothesis_structured = compact.get("hypothesis_structured", {})
        if isinstance(hypothesis_structured, dict):
            compact["hypothesis_structured"] = {
                "hypothesis": hypothesis_structured.get("hypothesis", compact.get("hypothesis", "")),
                "thesis": hypothesis_structured.get("thesis", ""),
                "expected_direction": hypothesis_structured.get("expected_direction", ""),
                "mechanism_anchor": hypothesis_structured.get("mechanism_anchor", {}),
                "proxy_reconstruction_plan": hypothesis_structured.get("proxy_reconstruction_plan", {}),
                "hypothesis_evolution_state": hypothesis_structured.get("hypothesis_evolution_state", {}),
                "continuity_diagnostics": hypothesis_structured.get("continuity_diagnostics", {}),
            }
        signal_diagnostics = compact.get("signal_diagnostics", {})
        if isinstance(signal_diagnostics, dict):
            compact["signal_diagnostics"] = {
                "counterfactual_challenge_level": signal_diagnostics.get("counterfactual_challenge_level", "none"),
                "recommended_action": signal_diagnostics.get("recommended_action", ""),
                "revision_reason": signal_diagnostics.get("revision_reason", ""),
                "group_performance": signal_diagnostics.get("group_performance", {}),
                "factor_group_shortfall_diagnostics": signal_diagnostics.get("factor_group_shortfall_diagnostics", {}),
            }
        metrics = compact.get("metrics", {})
        if isinstance(metrics, dict):
            compact["metrics"] = {
                key: metrics.get(key)
                for key in ("IC", "Rank IC", "ICIR", "Rank ICIR", "sharpe", "coverage", "turnover")
                if key in metrics
            }
        compact.pop("backtest_report", None)
        compact.pop("sota_metrics", None)
        return compact

    def _compact_feedback_history(self, history: list[dict[str, object]], history_limit: int) -> list[dict[str, object]]:
        try:
            full_tail_count = max(1, int(os.getenv("HYPOTHESIS_HISTORY_FULL_TAIL_COUNT", "1").strip() or "1"))
        except ValueError:
            full_tail_count = 1
        try:
            compact_limit_after_loop2 = max(
                2,
                int(os.getenv("HYPOTHESIS_HISTORY_COMPACT_LIMIT_AFTER_LOOP2", "4").strip() or "4"),
            )
        except ValueError:
            compact_limit_after_loop2 = 4
        effective_limit = history_limit if len(history) <= 2 else min(history_limit, compact_limit_after_loop2)
        trimmed = history[-effective_limit:]
        full_start = max(0, len(trimmed) - full_tail_count)
        return [
            self._compact_history_entry(item, keep_full=index >= full_start)
            for index, item in enumerate(trimmed)
            if isinstance(item, dict)
        ]

    def _determine_next_research_stage(
        self,
        *,
        has_research_report: bool,
        current_stage: str,
        history: list[dict[str, object]],
        feedback: dict[str, object],
        metrics: dict[str, float],
        backtest_report: dict[str, object],
        signal_diagnostics: dict[str, object] | None = None,
    ) -> tuple[str, str]:
        if not has_research_report:
            return "free_exploration", "No research report provided; keep free exploration mode."

        stage = str(current_stage or "").strip() or "base_reconstruction"
        hypothesis_supported = self._to_bool(feedback.get("hypothesis_supported", False), False)
        backtest_status = str(backtest_report.get("status", "")).strip().lower()
        ic = abs(self._to_float(metrics.get("IC"), 0.0))
        rank_ic = abs(self._to_float(metrics.get("Rank IC"), 0.0))
        weak_signal = rank_ic < WEAK_RANK_IC_THRESHOLD
        improving_signal = rank_ic >= WEAK_RANK_IC_THRESHOLD or hypothesis_supported
        recent_unsupported = self._count_recent_unsupported_loops(history)
        completed_feedback_loops = len(history)
        signal_diagnostics = signal_diagnostics or {}
        recommended_action = str(signal_diagnostics.get("recommended_action", "")).strip()
        revision_reason = str(signal_diagnostics.get("revision_reason", "")).strip()
        revision_recommended = self._to_bool(signal_diagnostics.get("hypothesis_revision_recommended", False), False)

        if backtest_status in {"failed", "partial"}:
            return stage, "Backtest was partial/failed, so keep the current stage and stabilize implementation first."

        if completed_feedback_loops < 3 and recommended_action in {"hypothesis_replacement", "hypothesis_reset"}:
            if revision_recommended:
                return self._guard_base_reconstruction_stage(
                    candidate_stage="base_reconstruction",
                    candidate_reason=(
                        "Early-stage weak signals should trigger proxy reconstruction first; defer thesis replacement until at least three completed feedback loops."
                    ),
                    completed_feedback_loops=completed_feedback_loops,
                    recommended_action=recommended_action,
                    revision_recommended=revision_recommended,
                    signal_diagnostics=signal_diagnostics,
                )
            return (
                "parameter_tuning",
                "Early-stage run should continue tuning before considering thesis replacement.",
            )

        if recommended_action == "proxy_reconstruction":
            return (
                "proxy_reconstruction",
                revision_reason or "Broaden or repair proxy construction while preserving the selected mechanism anchor.",
            )

        if revision_recommended and recommended_action in {"hypothesis_replacement", "hypothesis_reset"}:
            return self._guard_base_reconstruction_stage(
                candidate_stage="base_reconstruction",
                candidate_reason=revision_reason or "Most factors have weak Rank IC and the proxy family should be reconsidered.",
                completed_feedback_loops=completed_feedback_loops,
                recommended_action=recommended_action,
                revision_recommended=revision_recommended,
                signal_diagnostics=signal_diagnostics,
            )
        if revision_recommended and recommended_action == "hypothesis_revision":
            return "thesis_revision", revision_reason or "A stronger counterfactual path suggests revising the thesis."
        if recommended_action == "parameter_tuning" and signal_diagnostics.get("majority_weak_signal") is True:
            return "parameter_tuning", revision_reason or "Signals are weak but directionally stable enough to tune parameters first."

        if stage == "base_reconstruction":
            return (
                "parameter_tuning",
                "Base reconstruction has been completed once; next loop should refine parameters and normalization while staying on the same thesis.",
            )

        if stage == "parameter_tuning":
            diagnosis = str(feedback.get("semantic_diagnosis", "")).strip().lower()
            if diagnosis in {"mechanism_refuted", "direction_mismatch"} and recent_unsupported >= 2 and weak_signal and not hypothesis_supported:
                return (
                    "thesis_revision",
                    "Multiple recent tuning loops remained weak, so the next loop should reconsider or tighten the thesis rather than keep tuning the same expression family.",
                )
            if improving_signal:
                return (
                    "resonance_filtering",
                    "The current thesis shows some signal, so the next loop should test confirmation signals, resonance structure, and regime filters.",
                )
            return (
                "parameter_tuning",
                "Signal remains weak but not yet exhausted, so continue parameter tuning before revising the thesis.",
            )

        if stage == "resonance_filtering":
            diagnosis = str(feedback.get("semantic_diagnosis", "")).strip().lower()
            if diagnosis in {"mechanism_refuted", "direction_mismatch"} and recent_unsupported >= 2 and weak_signal and not hypothesis_supported:
                return (
                    "thesis_revision",
                    "Resonance and filter refinements still failed to support the thesis, so the next loop should revise the thesis.",
                )
            return (
                "resonance_filtering" if improving_signal else "parameter_tuning",
                "Stay within the same report-driven framework and continue testing resonance/filter refinements unless simpler parameter tuning is needed.",
            )

        if stage == "thesis_revision":
            return (
                "base_reconstruction",
                "After thesis revision, return to a clean reconstruction pass for the revised report-driven idea.",
            )

        return "parameter_tuning", "Default to parameter tuning within the current report-driven thesis."

    @staticmethod
    def _qualification_gate_status(per_factor_metrics: dict[str, object]) -> dict[str, object]:
        """资格线判定：任一 mainline 因子训练窗 Sharpe > 1.0 或 |IC| > 0.02。

        阈值与 EvolutionConfig.bootstrap_min_sharpe / bootstrap_min_abs_ic 默认值保持同步。
        只统计 mainline 因子（mainline_factor=1，缺字段按 mainline 兼容旧数据），
        adjacent/counterfactual 对照因子不能替假说过资格线。
        """
        min_sharpe = 1.0
        min_abs_ic = 0.02
        best_factor: dict[str, object] = {}
        best_score = float("-inf")
        qualified = False
        checked = 0
        for factor_name, metrics in (per_factor_metrics or {}).items():
            if not isinstance(metrics, dict):
                continue
            try:
                if float(metrics.get("mainline_factor", 1.0)) < 0.5:
                    continue
            except (TypeError, ValueError):
                pass
            checked += 1
            try:
                sharpe = float(metrics.get("sharpe", metrics.get("information_ratio", 0.0)) or 0.0)
            except (TypeError, ValueError):
                sharpe = 0.0
            try:
                abs_ic = abs(float(metrics.get("IC", 0.0) or 0.0))
            except (TypeError, ValueError):
                abs_ic = 0.0
            passes = sharpe > min_sharpe or abs_ic > min_abs_ic
            score = max(sharpe / min_sharpe, abs_ic / min_abs_ic)
            if score > best_score:
                best_score = score
                best_factor = {"factor_name": str(factor_name), "sharpe": sharpe, "abs_ic": abs_ic, "passes": passes}
            if passes:
                qualified = True
        return {
            "qualified": qualified,
            "checked_mainline_factor_count": checked,
            "min_sharpe": min_sharpe,
            "min_abs_ic": min_abs_ic,
            "best_factor": best_factor,
        }

    def _build_local_fallback_feedback(
        self,
        *,
        metrics: dict[str, float],
        sota_metrics: dict[str, float] | None,
        backtest_report: dict[str, object],
        calculation_report: dict[str, object],
        last_error: Exception | None,
    ) -> tuple[dict[str, object], str, str]:
        status = str(backtest_report.get("status", "")).strip().lower() or "unknown"
        generated_factors = int(calculation_report.get("generated_factors", 0) or 0)
        total_factors = int(calculation_report.get("total_factors", 0) or 0)
        ic = metrics.get("IC")
        rank_ic = metrics.get("Rank IC")
        sharpe = metrics.get("sharpe")
        metric_priority = ("IC", "Rank IC", "sharpe", "coverage")
        improved_metrics = [
            metric_name
            for metric_name in metric_priority
            if self._compare_metric(metric_name, metrics.get(metric_name), None if sota_metrics is None else sota_metrics.get(metric_name))
        ]
        hypothesis_supported = bool(improved_metrics)
        observations = (
            f"LLM 反馈生成不可用，因此使用本地兜底评估。"
            f"回测状态={status}，已生成因子={generated_factors}/{total_factors}，"
            f"IC={ic if ic is not None else '无'}，Rank IC={rank_ic if rank_ic is not None else '无'}，"
            f"sharpe={sharpe if sharpe is not None else '无'}。"
        )
        if improved_metrics:
            summary = f"当前结果在 {', '.join(improved_metrics)} 上优于历史最优。"
            reason = (
                f"本地兜底评估选择替换最佳结果，因为当前运行在 {', '.join(improved_metrics)} 上改善，"
                f"并且回测成功完成。"
            )
            next_hypothesis_hint = ""
        else:
            summary = "当前结果没有明确超过历史最佳结果。"
            reason = (
                "本地兜底评估保留当前假说方向，但不替换最佳结果，因为观察到的指标没有相对历史最优形成明确改善。"
            )
            next_hypothesis_hint = "调优现有信号参数"
        if status in {"failed", "partial"}:
            summary = "回测状态为部分完成或失败；需要继续优化实现和信号稳定性。"
            reason = (
                f"本地兜底评估观察到回测状态={status}，说明因子生成或评估链路仍需稳定后再替换最佳结果。"
            )
            hypothesis_supported = False
            next_hypothesis_hint = "稳定因子生成并降低失败率"
        if last_error is not None:
            reason = f"{reason} LLM 反馈错误：{last_error}"
        feedback = {
            "hypothesis_supported": hypothesis_supported,
            "summary": summary,
            "observations": observations,
            "reason": reason,
            "new_hypothesis": next_hypothesis_hint,
            "semantic_diagnosis": "",
            "semantic_diagnosis_reason": "",
        }
        distilled_knowledge = f"{observations}; {reason}"
        return feedback, next_hypothesis_hint, distilled_knowledge

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        hypothesis = str(shared_context.payload.get("hypothesis", ""))
        hypothesis_structured = dict(shared_context.payload.get("hypothesis_structured", {}))
        research_report = dict(shared_context.payload.get("research_report", {}))
        current_research_stage = str(shared_context.payload.get("research_stage", "")).strip()
        scenario = str(shared_context.payload.get("scenario", ""))
        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        experiment_spec = experiment.to_experiment_spec()
        factor_implementation = dict(experiment.factor_implementation)
        calculation_report = dict(experiment.calculation_report)
        backtest_report = dict(experiment.backtest_report)
        metrics = {str(k): float(v) for k, v in experiment.metrics.items()}
        history = list(shared_context.payload.get("hypothesis_feedback_history", []))
        prompt_bundle = self.prompt_store.load("feedback_summarizer_agent")
        task_details = self._build_task_details(factor_implementation, experiment_spec)
        sota_entry = self._extract_sota_entry(history)
        sota_metrics = None if sota_entry is None else dict(sota_entry.get("metrics", {}))
        result_semantics = str(backtest_report.get("metrics_semantics", "best_single_factor_not_combined"))
        combined_result_text = (
            f"Result semantics: {result_semantics}. "
            "Top-level metrics are not a combined-factor backtest; use per_factor_metrics for factor-level feedback. "
            + self._process_results(metrics, sota_metrics)
        )
        per_factor_metrics = dict(experiment.per_factor_metrics)
        prior_signal_diagnostics = [
            dict(item.get("signal_diagnostics", {}))
            for item in history
            if isinstance(item, dict) and isinstance(item.get("signal_diagnostics", {}), dict)
        ]
        signal_diagnostics = build_loop_signal_diagnostics(
            factors=[dict(item) for item in experiment_spec.get("factors", []) if isinstance(item, dict)],
            per_factor_metrics={
                str(name): dict(value)
                for name, value in per_factor_metrics.items()
                if isinstance(value, dict)
            },
            previous_loop_diagnostics=prior_signal_diagnostics,
        )
        group_shortfall_diagnostics = (
            dict(experiment.extra.get("factor_group_shortfall_diagnostics", {}))
            if isinstance(experiment.extra.get("factor_group_shortfall_diagnostics", {}), dict)
            else {}
        )
        if group_shortfall_diagnostics:
            signal_diagnostics["factor_group_shortfall_diagnostics"] = group_shortfall_diagnostics
        context = {
            "scenario": scenario,
            "hypothesis_text": hypothesis,
            "task_details_text": self._render_task_details_text(task_details),
            "combined_result_text": combined_result_text,
            "per_factor_metrics": json.dumps(per_factor_metrics, ensure_ascii=False, indent=2),
            "backtest_report": json.dumps(backtest_report, ensure_ascii=False, indent=2),
            "calculation_report": json.dumps(calculation_report, ensure_ascii=False, indent=2),
            "signal_diagnostics": json.dumps(signal_diagnostics, ensure_ascii=False, indent=2),
        }
        system_prompt = render_prompt(prompt_bundle.system_prompt, context)
        user_prompt = render_prompt(prompt_bundle.user_prompt, context)
        max_retries = int(prompt_bundle.metadata.get("max_json_parse_retries", 3))
        response_json: dict[str, Any] | None = None
        raw_response = ""
        last_error: Exception | None = None
        for _attempt in range(max_retries):
            try:
                raw_response = model_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
                response_json = parse_json_object(raw_response)
                self._validate_no_disallowed_language_scripts(response_json)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue
        fallback_reason = str(prompt_bundle.metadata.get("fallback_reason", "json_parse_failed_or_missing_feedback_fields"))
        if response_json is None:
            feedback, next_hypothesis_hint, distilled_knowledge = self._build_local_fallback_feedback(
                metrics=metrics,
                sota_metrics=sota_metrics,
                backtest_report=backtest_report,
                calculation_report=calculation_report,
                last_error=last_error,
            )
        else:
            feedback, next_hypothesis_hint, distilled_knowledge = self._build_feedback_from_payload(
                response_json,
                fallback_reason=fallback_reason,
            )
        semantic_diagnosis, semantic_diagnosis_reason = self._infer_semantic_diagnosis(
            hypothesis_text=hypothesis,
            metrics=metrics,
            backtest_report=backtest_report,
            history=history,
            signal_diagnostics=signal_diagnostics,
        )
        feedback["llm_semantic_diagnosis_raw"] = str(feedback.get("semantic_diagnosis", "")).strip()
        feedback["llm_semantic_diagnosis_reason_raw"] = str(feedback.get("semantic_diagnosis_reason", "")).strip()
        feedback["semantic_diagnosis"] = semantic_diagnosis
        feedback["semantic_diagnosis_reason"] = semantic_diagnosis_reason
        feedback["signal_diagnostics"] = signal_diagnostics
        feedback["summary"] = self._build_group_level_summary(
            signal_diagnostics=signal_diagnostics,
            fallback_summary=str(feedback.get("summary", "")),
        )
        feedback["hypothesis_revision_recommended"] = signal_diagnostics.get("hypothesis_revision_recommended", False)
        feedback["revision_reason"] = signal_diagnostics.get("revision_reason", "")
        feedback["recommended_action"] = signal_diagnostics.get("recommended_action", "continue")

        # ── 资格线对齐（与 bootstrap 资格赛同口径）──
        # 任一 mainline 因子训练窗 Sharpe > 1.0 或 |IC| > 0.02 → 假说不得被判"不成立"。
        # LLM/组对比仍可建议 proxy_reconstruction 等动作，但 hypothesis_supported 必须反映
        # "假说已产出过线因子"这一事实，避免 UI 出现"指标过线却说假说不成立"的矛盾。
        qualification = self._qualification_gate_status(per_factor_metrics)
        feedback["qualification_gate"] = qualification
        if qualification.get("qualified") and not self._to_bool(feedback.get("hypothesis_supported", False), False):
            feedback["hypothesis_supported"] = True
            feedback["hypothesis_supported_override"] = True
            best = qualification.get("best_factor", {})
            override_note = (
                f"资格线覆盖：mainline 因子 {best.get('factor_name', '')} 训练窗 "
                f"Sharpe={best.get('sharpe', 0):.4f}、|IC|={best.get('abs_ic', 0):.4f}，"
                f"已满足资格线（Sharpe>{qualification['min_sharpe']} 或 |IC|>{qualification['min_abs_ic']}），假说判定为成立。"
            )
            feedback["reason"] = f"{str(feedback.get('reason', '')).strip()} {override_note}".strip()

        backtest_status = str(backtest_report.get("status", "")).strip().lower()
        if backtest_status in {"failed", "partial"}:
            replace_best_result = False
        elif sota_metrics is None:
            replace_best_result = True
        else:
            replace_best_result = self._metrics_score_tuple(metrics) > self._metrics_score_tuple(sota_metrics)
        feedback["replace_best_result"] = replace_best_result
        next_research_stage, research_stage_reason = self._determine_next_research_stage(
            has_research_report=bool(research_report.get("report_text")),
            current_stage=current_research_stage,
            history=history,
            feedback=feedback,
            metrics=metrics,
            backtest_report=backtest_report,
            signal_diagnostics=signal_diagnostics,
        )
        feedback["next_research_stage"] = next_research_stage
        feedback["research_stage_reason"] = research_stage_reason
        history_limit = int(prompt_bundle.metadata.get("history_limit", 10))
        history.append(
            {
                "hypothesis": hypothesis,
                "feedback_summary": feedback["summary"],
                "feedback_reason": feedback["reason"],
                "feedback_hypothesis_supported": feedback["hypothesis_supported"],
                "feedback_replace_best_result": replace_best_result,
                "feedback_new_hypothesis": next_hypothesis_hint,
                "feedback_semantic_diagnosis": feedback.get("semantic_diagnosis", ""),
                "feedback_research_stage": current_research_stage,
                "feedback_next_research_stage": next_research_stage,
                "signal_diagnostics": signal_diagnostics,
                "hypothesis_structured": hypothesis_structured,
                "metrics": metrics,
                "backtest_report": backtest_report,
                "sota_metrics": sota_metrics or str(prompt_bundle.metadata.get("sota_empty_fallback", "no_prior_sota_result")),
            }
        )
        private_context.payload["last_feedback_prompt_context"] = context
        private_context.payload["last_feedback_raw_response"] = raw_response
        private_context.payload["latest_feedback"] = feedback
        experiment.feedback = dict(feedback)
        experiment.update_timestamps()
        private_context.payload["latest_experiment"] = experiment.to_dict()
        return AgentResult(
            shared_updates={
                "feedback": feedback,
                "next_hypothesis_hint": next_hypothesis_hint,
                "distilled_knowledge": distilled_knowledge,
                "research_stage": next_research_stage,
                "research_stage_reason": research_stage_reason,
                "hypothesis_feedback_history": self._compact_feedback_history(history, history_limit),
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name},
        )
