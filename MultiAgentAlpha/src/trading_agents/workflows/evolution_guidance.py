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

    @staticmethod
    def _clip_text(value: object, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

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
            "hypothesis": self._clip_text(trajectory.hypothesis, 700),
            "factors": [self._summarize_factor(item) for item in trajectory.factors[:2] if isinstance(item, dict)],
            "metrics": self._summarize_metrics(trajectory.backtest_metrics),
            "target_status": dict(trajectory.get_target_status()),
            "feedback": self._summarize_feedback(trajectory),
            "reward_score": round(float(trajectory.get_reward_score()), 6),
            "segments": self._summarize_segments(trajectory),
        }

    def infer_mutation_type(self, trajectory: StrategyTrajectory) -> str:
        """根据指标和反馈推断最合适的变异类型。

        推断优先级（从高到低）：
        1. risk_mutation — IC/RankIC 达标但 Sharpe 不达标，或回撤/换手过高
        2. hypothesis_mutation — 假说被推翻(direction_mismatch)、IC 极弱(<0.005)且无信号
        3. expression_mutation — IC/RankIC 缺口或代理弱(proxy_weak)
        4. parameter_mutation — 参数未调(parameter_not_tuned)、IC 可以但 ICIR 低(<0.1)
        5. expression_mutation — 默认 fallback
        """
        metrics = trajectory.backtest_metrics
        feedback_text = self._summarize_feedback(trajectory).lower()
        diagnosis = self._feedback_diagnosis(trajectory).lower()
        ic = self._metric_value(metrics, "IC")
        rank_ic = self._metric_value(metrics, "Rank IC", "RankIC")
        icir = self._metric_value(metrics, "Rank ICIR", "RankICIR", "ICIR")
        sharpe = self._metric_value(metrics, "sharpe", "information_ratio")
        drawdown = abs(self._metric_value(metrics, "drawdown", "max_drawdown"))
        turnover = self._metric_value(metrics, "turnover")

        if max(abs(ic), abs(rank_ic)) > 0.03 and sharpe <= 1.5:
            return "risk_mutation"

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

    def infer_target_gap(self, trajectory: StrategyTrajectory) -> str:
        """按当前硬目标缺口选择本轮应优先修的指标。"""
        status = trajectory.get_target_status()
        metrics = trajectory.backtest_metrics
        turnover = self._metric_value(metrics, "turnover")
        complexity = trajectory.get_reward_components().get("complexity_penalty", 0.0)
        if float(status.get("sharpe_gap", 0.0) or 0.0) > 0.0:
            if turnover > 0.5:
                return "reduce_turnover"
            return "improve_sharpe"
        if float(status.get("rank_ic_gap", 0.0) or 0.0) > 0.0:
            return "improve_rank_ic"
        if float(status.get("ic_gap", 0.0) or 0.0) > 0.0:
            return "improve_ic"
        if complexity > 0.5:
            return "reduce_complexity"
        if turnover > 0.5:
            return "reduce_turnover"
        return "improve_sharpe"

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
        target_gap = self.infer_target_gap(parent)
        summary = self.summarize_trajectory(parent)
        diagnosis = self._mutation_diagnosis_text(parent, mutation_type)
        preserve_text = self._mutation_preserve_text(mutation_type)
        allowed_text = self._mutation_allowed_text(mutation_type)
        restart_rule = (
            "允许完整重置假设，但必须说明父代中仍然可迁移的有效观察。"
            if mutation_type == "hypothesis_mutation"
            else "不要完全随机重启；保持谱系连续性，只修改允许变动的部分。"
        )
        guidance = "\n".join(
            [
                f"进化阶段：从父代轨迹 {parent.trajectory_id} 进行变异。",
                f"父代来源：phase={summary['phase']}，round_idx={summary['round_idx']}，direction_id={summary['direction_id']}。",
                f"变异类型：{mutation_type}。",
                f"目标缺口：{target_gap}。",
                "目标指标规则：通过关闭净 Sharpe > 1.5、abs(IC) > 0.03、abs(Rank IC) > 0.03 的缺口来最大化 reward。",
                f"当前诊断：{diagnosis}",
                f"需要保留：{preserve_text}",
                f"允许修改：{allowed_text}",
                f"边界条件：{restart_rule}",
                "子代因子表达式必须简单、可检验，并基于受支持的原始输入字段。",
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
                "进化阶段：请求交叉但没有可用父代轨迹，因此无法生成交叉引导。",
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
                        f"父代 {parent.trajectory_id}：",
                        f"  可继承优势：{strengths}",
                        f"  需要避开弱点：{weaknesses}",
                    ]
                )
            )

        guidance = "\n".join(
            [
                "进化阶段：基于多条父代轨迹进行交叉。",
                *parent_blocks,
                "子代引导：组合父代中最强的经济机制、代理变量族或稳定化思路。",
                "不要机械拼接、相乘或堆叠父代表达式。",
                "子代表达式必须短小、可解析，并且比直接合并公式更简单。",
                "优先形成一个连贯信号，最多加入一个稳定化或风险控制修饰项。",
            ]
        )
        return guidance, summaries

    @staticmethod
    def _summarize_factor(factor: dict[str, Any]) -> dict[str, object]:
        """提取单个因子的名称、表达式、描述，用于轨迹摘要。"""
        return {
            "factor_name": str(factor.get("factor_name", factor.get("name", ""))).strip(),
            "expression": EvolutionGuidanceBuilder._clip_text(
                factor.get("expression", factor.get("factor_expression", "")),
                260,
            ),
            "description": EvolutionGuidanceBuilder._clip_text(
                factor.get("description", factor.get("factor_description", "")),
                240,
            ),
        }

    @classmethod
    def _summarize_segments(cls, trajectory: StrategyTrajectory) -> dict[str, object]:
        """只暴露可复用的分段标签，避免把完整轨迹快照递归塞进 EvolutionAgent prompt。"""
        raw_segments = trajectory.extra_info.get("segments", {})
        if not isinstance(raw_segments, dict):
            return {}
        compact: dict[str, object] = {}
        for key in ("hypothesis_generation", "factor_expression", "implementation", "evaluation_feedback", "task_metadata"):
            segment = raw_segments.get(key)
            if not isinstance(segment, dict):
                continue
            item: dict[str, object] = {}
            for field, limit in (
                ("summary", 260),
                ("failure_label", 120),
                ("selected_factor_name", 120),
                ("target_segment", 80),
            ):
                if field in segment:
                    text = cls._clip_text(segment.get(field), limit)
                    if text:
                        item[field] = text
            reusable = segment.get("reusable_anchors")
            if isinstance(reusable, list):
                item["reusable_anchors"] = [cls._clip_text(value, 120) for value in reusable[:3] if str(value).strip()]
            source_segments = segment.get("source_segments")
            if isinstance(source_segments, list):
                item["source_segments"] = source_segments[:3]
            if item:
                compact[key] = item
        return compact

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
            return EvolutionGuidanceBuilder._clip_text(trajectory.feedback, 700)
        details = trajectory.feedback_details
        for key in ("summary", "Feedback for Hypothesis", "Reasoning", "Observations"):
            value = details.get(key) if isinstance(details, dict) else None
            if value:
                return EvolutionGuidanceBuilder._clip_text(value, 700)
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
        feedback = self._summarize_feedback(trajectory) or "未记录明确反馈。"
        metric_text = ", ".join(f"{key}={value:g}" for key, value in metrics.items()) or "没有可用指标"
        return f"根据指标（{metric_text}）和反馈选择 {mutation_type}：{feedback}"

    @staticmethod
    def _mutation_preserve_text(mutation_type: str) -> str:
        """根据变异类型返回"应保留什么"的说明文本。

        hypothesis_mutation: 只保留可迁移观察和领域约束
        expression_mutation: 保留假说、核心变量族和预测方向
        parameter_mutation: 保留假说、表达式结构、变量族和经济机制
        risk_mutation: 保留主 alpha 信号和经验有用的代理族
        """
        if mutation_type == "hypothesis_mutation":
            return "只保留可迁移观察、领域约束，以及失败证据带来的教训。"
        if mutation_type == "expression_mutation":
            return "保留父代假说、核心变量族和预期预测方向。"
        if mutation_type == "parameter_mutation":
            return "保留假说、表达式结构、变量族和经济机制。"
        if mutation_type == "risk_mutation":
            return "保留主 alpha 信号和经验上有用的代理变量族。"
        return "保留父代谱系和任何已被支持的证据。"

    @staticmethod
    def _mutation_allowed_text(mutation_type: str) -> str:
        """根据变异类型返回"允许变更什么"的说明文本。

        hypothesis_mutation: 可替换假说和代理族
        expression_mutation: 可重写归一化、排名、滞后或算子组合
        parameter_mutation: 可调整窗口、阈值、平滑、排名等参数
        risk_mutation: 可添加稳定性、换手率、回撤等风控约束
        """
        if mutation_type == "hypothesis_mutation":
            return "如果原机制缺乏支持，可以替换假说和代理变量族。"
        if mutation_type == "expression_mutation":
            return "在同一变量族内重写归一化、排序、滞后或算子组合。"
        if mutation_type == "parameter_mutation":
            return "在不切换变量族的前提下调整窗口、阈值、平滑、滞后、排序或 zscore 设置。"
        if mutation_type == "risk_mutation":
            return "围绕主信号加入稳定性、换手、回撤、波动状态或平滑控制。"
        return "只做局部且由证据驱动的修改。"

    def _crossover_strengths(self, trajectory: StrategyTrajectory) -> str:
        """评估父代轨迹的优势：正信号、稳定 IC、合理覆盖率、适度换手、经济机制。"""
        metrics = trajectory.backtest_metrics
        strengths: list[str] = []
        if self._metric_value(metrics, "Rank IC", "RankIC", "IC") > 0:
            strengths.append("预测信号为正")
        if self._metric_value(metrics, "Rank ICIR", "RankICIR", "ICIR") > 0:
            strengths.append("IC 稳定性较好")
        if self._metric_value(metrics, "coverage") >= 0.5:
            strengths.append("数据覆盖率合理")
        if self._metric_value(metrics, "turnover") and self._metric_value(metrics, "turnover") <= 0.3:
            strengths.append("换手适中")
        if trajectory.hypothesis:
            strengths.append("具备经济机制或假说基础")
        return ", ".join(strengths) if strengths else "父代摘要中最清晰的可迁移思路"

    def _crossover_weaknesses(self, trajectory: StrategyTrajectory) -> str:
        """评估父代轨迹的劣势：弱/负预测指标、低稳定性、高回撤、高换手、缺失因子表达式。"""
        metrics = trajectory.backtest_metrics
        weaknesses: list[str] = []
        if self._metric_value(metrics, "Rank IC", "RankIC", "IC") <= 0:
            weaknesses.append("预测指标较弱或为负")
        if abs(self._metric_value(metrics, "Rank ICIR", "RankICIR", "ICIR")) < 0.1:
            weaknesses.append("稳定性较低")
        if self._metric_value(metrics, "drawdown", "max_drawdown") > 0.2:
            weaknesses.append("回撤较高")
        if self._metric_value(metrics, "turnover") > 0.5:
            weaknesses.append("换手较高")
        if not trajectory.factors:
            weaknesses.append("缺少因子表达式证据")
        return ", ".join(weaknesses) if weaknesses else "避免未经重新验证就复制实现细节"
