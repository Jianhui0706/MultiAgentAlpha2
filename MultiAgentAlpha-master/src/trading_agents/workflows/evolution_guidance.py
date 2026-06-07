from __future__ import annotations

from typing import Any

from trading_agents.schemas.models import RoundPhase, StrategyTrajectory


class EvolutionGuidanceBuilder:
    """基于规则生成轨迹级遗传迭代引导信息。

    核心职责：
    1. summarize_trajectory: 将 StrategyTrajectory 浓缩为可注入 LLM 提示词的摘要
    2. infer_mutation_type: 根据指标和反馈推断最合适的变异类型
    3. build_mutation_guidance: 为变异任务生成引导信息（诊断、保留项、允许变更项）
    4. build_crossover_guidance: 为交叉任务生成引导信息（父代优缺点、融合指导）
    """

    # 四种变异类型
    MUTATION_TYPES = {
        "hypothesis_mutation",   # 假说变异：替换假说和代理族
        "expression_mutation",   # 表达式变异：重写归一化、算子组合
        "parameter_mutation",    # 参数变异：调整窗口、阈值、平滑
        "risk_mutation",         # 风控变异：添加稳定性/回撤/换手率控制
    }

    def summarize_trajectory(self, trajectory: StrategyTrajectory) -> dict[str, object]:
        """生成可安全注入提示词的精简父轨迹摘要。

        提取 trajectory_id、phase、round_idx、direction_id、假说、前 3 个因子、
        关键指标（RankIC/ICIR/IC/Sharpe/drawdown/coverage/turnover）、
        反馈文本和奖励分数。去除敏感/冗余信息以控制 token 消耗。
        """
        return {
            "trajectory_id": trajectory.trajectory_id,
            "phase": trajectory.phase.value if isinstance(trajectory.phase, RoundPhase) else str(trajectory.phase),
            "round_idx": int(trajectory.round_idx),
            "direction_id": int(trajectory.direction_id),
            "hypothesis": trajectory.hypothesis,
            "factors": [self._summarize_factor(item) for item in trajectory.factors[:3] if isinstance(item, dict)],
            "metrics": self._summarize_metrics(trajectory.backtest_metrics),
            "feedback": self._summarize_feedback(trajectory),
            "reward_score": round(float(trajectory.get_reward_score()), 6),
            "segments": dict(trajectory.extra_info.get("segments", {}))
            if isinstance(trajectory.extra_info.get("segments", {}), dict)
            else {},
        }

    def infer_mutation_type(self, trajectory: StrategyTrajectory) -> str:
        """根据指标和反馈推断最合适的变异类型。

        推断优先级（从高到低）：
        1. hypothesis_mutation — 假说被推翻(direction_mismatch)、IC 极弱(<0.005)且无信号
        2. expression_mutation — 代理弱(proxy_weak)
        3. risk_mutation — 实现不稳定(implementation_unstable)、高回撤(>0.2)、高换手(>0.5)
        4. parameter_mutation — 参数未调(parameter_not_tuned)、IC 可以但 ICIR 低(<0.1)
        5. expression_mutation — 默认 fallback
        """
        metrics = trajectory.backtest_metrics
        feedback_text = self._summarize_feedback(trajectory).lower()
        diagnosis = self._feedback_diagnosis(trajectory).lower()
        ic = self._metric_value(metrics, "IC")
        rank_ic = self._metric_value(metrics, "Rank IC", "RankIC")
        icir = self._metric_value(metrics, "Rank ICIR", "RankICIR", "ICIR")
        drawdown = abs(self._metric_value(metrics, "drawdown", "max_drawdown"))
        turnover = self._metric_value(metrics, "turnover")

        if diagnosis in {"thesis_invalid", "direction_mismatch"}:
            return "hypothesis_mutation"
        if any(marker in feedback_text for marker in ("thesis invalid", "hypothesis reset", "direction mismatch")):
            return "hypothesis_mutation"
        if abs(ic) < 0.005 and abs(rank_ic) < 0.005 and (
            "no signal" in feedback_text or "invalid" in feedback_text or "failed" in feedback_text
        ):
            return "hypothesis_mutation"

        if diagnosis == "proxy_weak" or any(marker in feedback_text for marker in ("proxy weak", "weak proxy", "expression")):
            return "expression_mutation"

        if diagnosis == "implementation_unstable":
            return "risk_mutation"
        if drawdown > 0.2 or turnover > 0.5 or any(
            marker in feedback_text for marker in ("drawdown", "turnover", "unstable", "risk")
        ):
            return "risk_mutation"

        if diagnosis == "parameter_not_tuned" or any(
            marker in feedback_text for marker in ("parameter", "window", "threshold", "not tuned")
        ):
            return "parameter_mutation"
        if max(abs(ic), abs(rank_ic)) >= 0.005 and abs(icir) < 0.1:
            return "parameter_mutation"

        return "expression_mutation"

    def build_mutation_guidance(self, parent: StrategyTrajectory) -> tuple[str, str, list[dict[str, object]]]:
        """为单条父轨迹生成变异引导信息。

        Returns:
            (mutation_type, guidance_text, [parent_summary])
            - mutation_type: 推断的变异类型
            - guidance_text: 包含变异类型、诊断、保留项、允许变更项、边界规则的引导文本
            - parent_summary: 父轨迹摘要列表（单元素）

        边界规则：
        - hypothesis_mutation 允许完全重置假说，但要求解释哪些可迁移观察仍有用
        - 其他类型要求保持血统连续性，只变更允许的部分
        """
        mutation_type = self.infer_mutation_type(parent)
        summary = self.summarize_trajectory(parent)
        diagnosis = self._mutation_diagnosis_text(parent, mutation_type)
        preserve_text = self._mutation_preserve_text(mutation_type)
        allowed_text = self._mutation_allowed_text(mutation_type)
        restart_rule = (
            "A full hypothesis reset is allowed, but explain which transferable observation from the parent remains useful."
            if mutation_type == "hypothesis_mutation"
            else "Do not perform a fully random restart; keep lineage continuity and only change the allowed parts."
        )
        guidance = "\n".join(
            [
                f"Evolution phase: mutation from parent trajectory {parent.trajectory_id}.",
                f"Parent origin: phase={summary['phase']}, round_idx={summary['round_idx']}, direction_id={summary['direction_id']}.",
                f"Mutation type: {mutation_type}.",
                f"Current diagnosis: {diagnosis}",
                f"Preserve: {preserve_text}",
                f"Allowed changes: {allowed_text}",
                f"Boundary: {restart_rule}",
                "Keep the child factor expression simple, testable, and grounded in supported raw input features.",
            ]
        )
        return mutation_type, guidance, [summary]

    def build_crossover_guidance(self, parents: list[StrategyTrajectory]) -> tuple[str, list[dict[str, object]]]:
        """生成融合父代优点且避免机械拼接表达式的交叉引导信息。

        对每个父代分析 strengths（正信号、稳定 IC、覆盖率等）和 weaknesses（弱指标、高回撤等），
        指导子代进行"概念融合而非机械拼接"，保持表达式简短可解析。

        Returns:
            (guidance_text, [parent_summaries])
        """
        if not parents:
            return (
                "Evolution phase: crossover requested without parent trajectories. No crossover guidance can be generated.",
                [],
            )

        summaries = [self.summarize_trajectory(parent) for parent in parents]
        parent_blocks = []
        for parent, summary in zip(parents, summaries, strict=False):
            strengths = self._crossover_strengths(parent)
            weaknesses = self._crossover_weaknesses(parent)
            parent_blocks.append(
                "\n".join(
                    [
                        f"Parent {parent.trajectory_id}:",
                        f"  Inherit: {strengths}",
                        f"  Avoid: {weaknesses}",
                    ]
                )
            )

        guidance = "\n".join(
            [
                "Evolution phase: crossover from multiple parent trajectories.",
                *parent_blocks,
                "Child guidance: combine the strongest economic mechanism, proxy family, or stabilization idea from the parents.",
                "Do not mechanically concatenate, multiply, or stack parent expressions.",
                "The child expression must stay short, parsable, and simpler than a direct formula merge.",
                "Prefer one coherent signal with at most one stabilizing or risk-control modifier.",
            ]
        )
        return guidance, summaries

    @staticmethod
    def _summarize_factor(factor: dict[str, Any]) -> dict[str, object]:
        """提取单个因子的名称、表达式、描述，用于轨迹摘要。"""
        return {
            "factor_name": str(factor.get("factor_name", factor.get("name", ""))).strip(),
            "expression": str(factor.get("expression", factor.get("factor_expression", ""))).strip(),
            "description": str(factor.get("description", factor.get("factor_description", ""))).strip(),
        }

    @staticmethod
    def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        """从完整指标中提取关键指标子集（RankIC/ICIR/IC/Sharpe/drawdown/coverage/turnover）。"""
        aliases = (
            "Rank IC",
            "RankIC",
            "Rank ICIR",
            "RankICIR",
            "IC",
            "ICIR",
            "sharpe",
            "information_ratio",
            "drawdown",
            "max_drawdown",
            "coverage",
            "turnover",
        )
        summary: dict[str, float] = {}
        for key in aliases:
            if key not in metrics:
                continue
            try:
                summary[key] = float(metrics.get(key) or 0.0)
            except (TypeError, ValueError):
                summary[key] = 0.0
        return summary

    @staticmethod
    def _summarize_feedback(trajectory: StrategyTrajectory) -> str:
        """提取反馈文本：优先取 feedback 字段，其次从 feedback_details 中取 summary/Feedback for Hypothesis 等。"""
        if trajectory.feedback:
            return trajectory.feedback.strip()
        details = trajectory.feedback_details
        for key in ("summary", "Feedback for Hypothesis", "Reasoning", "Observations"):
            value = details.get(key) if isinstance(details, dict) else None
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _feedback_diagnosis(trajectory: StrategyTrajectory) -> str:
        """从 feedback_details 中提取语义诊断文本（Semantic Diagnosis / semantic_diagnosis / diagnosis）。"""
        details = trajectory.feedback_details
        if not isinstance(details, dict):
            return ""
        for key in ("Semantic Diagnosis", "semantic_diagnosis", "diagnosis"):
            value = details.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _metric_value(metrics: dict[str, Any], *aliases: str) -> float:
        """按别名列表查找第一个可用的指标值，缺失或非法时返回 0.0。"""
        for alias in aliases:
            if alias not in metrics:
                continue
            try:
                return float(metrics.get(alias) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _mutation_diagnosis_text(self, trajectory: StrategyTrajectory, mutation_type: str) -> str:
        """生成变异诊断文本：将关键指标和反馈拼接为一段描述，解释为何选择该变异类型。"""
        metrics = self._summarize_metrics(trajectory.backtest_metrics)
        feedback = self._summarize_feedback(trajectory) or "No explicit feedback was recorded."
        metric_text = ", ".join(f"{key}={value:g}" for key, value in metrics.items()) or "no usable metrics"
        return f"{mutation_type} selected from metrics ({metric_text}) and feedback: {feedback}"

    @staticmethod
    def _mutation_preserve_text(mutation_type: str) -> str:
        """根据变异类型返回"应保留什么"的说明文本。

        hypothesis_mutation: 只保留可迁移观察和领域约束
        expression_mutation: 保留假说、核心变量族和预测方向
        parameter_mutation: 保留假说、表达式结构、变量族和经济机制
        risk_mutation: 保留主 alpha 信号和经验有用的代理族
        """
        if mutation_type == "hypothesis_mutation":
            return "preserve only transferable observations, domain constraints, and lessons from failed evidence."
        if mutation_type == "expression_mutation":
            return "preserve the parent hypothesis, core variable family, and intended prediction direction."
        if mutation_type == "parameter_mutation":
            return "preserve the hypothesis, expression structure, variable family, and economic mechanism."
        if mutation_type == "risk_mutation":
            return "preserve the main alpha signal and any empirically useful proxy family."
        return "preserve the parent lineage and any supported evidence."

    @staticmethod
    def _mutation_allowed_text(mutation_type: str) -> str:
        """根据变异类型返回"允许变更什么"的说明文本。

        hypothesis_mutation: 可替换假说和代理族
        expression_mutation: 可重写归一化、排名、滞后或算子组合
        parameter_mutation: 可调整窗口、阈值、平滑、排名等参数
        risk_mutation: 可添加稳定性、换手率、回撤等风控约束
        """
        if mutation_type == "hypothesis_mutation":
            return "replace the hypothesis and proxy family if the original mechanism is unsupported."
        if mutation_type == "expression_mutation":
            return "rewrite normalization, ranking, lagging, or operator composition while staying in the same variable family."
        if mutation_type == "parameter_mutation":
            return "adjust windows, thresholds, smoothing, lag, rank, or zscore settings without switching variable family."
        if mutation_type == "risk_mutation":
            return "add stability, turnover, drawdown, volatility-regime, or smoothing controls around the main signal."
        return "make a local, evidence-driven change only."

    def _crossover_strengths(self, trajectory: StrategyTrajectory) -> str:
        """评估父代轨迹的优势：正信号、稳定 IC、合理覆盖率、适度换手、经济机制。"""
        metrics = trajectory.backtest_metrics
        strengths: list[str] = []
        if self._metric_value(metrics, "Rank IC", "RankIC", "IC") > 0:
            strengths.append("positive predictive signal")
        if self._metric_value(metrics, "Rank ICIR", "RankICIR", "ICIR") > 0:
            strengths.append("stable IC profile")
        if self._metric_value(metrics, "coverage") >= 0.5:
            strengths.append("reasonable data coverage")
        if self._metric_value(metrics, "turnover") and self._metric_value(metrics, "turnover") <= 0.3:
            strengths.append("moderate turnover")
        if trajectory.hypothesis:
            strengths.append("economic mechanism or hypothesis")
        return ", ".join(strengths) if strengths else "the clearest transferable idea from the parent summary"

    def _crossover_weaknesses(self, trajectory: StrategyTrajectory) -> str:
        """评估父代轨迹的劣势：弱/负预测指标、低稳定性、高回撤、高换手、缺失因子表达式。"""
        metrics = trajectory.backtest_metrics
        weaknesses: list[str] = []
        if self._metric_value(metrics, "Rank IC", "RankIC", "IC") <= 0:
            weaknesses.append("weak or negative predictive metric")
        if abs(self._metric_value(metrics, "Rank ICIR", "RankICIR", "ICIR")) < 0.1:
            weaknesses.append("low stability")
        if self._metric_value(metrics, "drawdown", "max_drawdown") > 0.2:
            weaknesses.append("high drawdown")
        if self._metric_value(metrics, "turnover") > 0.5:
            weaknesses.append("high turnover")
        if not trajectory.factors:
            weaknesses.append("missing factor expression evidence")
        return ", ".join(weaknesses) if weaknesses else "avoid copying implementation details without revalidation"
