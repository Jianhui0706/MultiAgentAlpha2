from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class AgentContext:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedContext:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    statement: str
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: list[str]
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Execution:
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    summary: str
    score: float
    actions: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    shared_updates: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    agent_name: str
    started_at: datetime
    ended_at: datetime
    input_snapshot: dict[str, Any]
    output_snapshot: dict[str, Any]
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class LoopTrace:
    records: list[TraceRecord] = field(default_factory=list)
    final_shared_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentFactorTask:
    factor_name: str
    description: str = ""
    formulation: str = ""
    expression: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    validation_warnings: list[str] = field(default_factory=list)
    retained_for_backtest: bool = True
    theory_alignment: str = "mainline"
    counterfactual_reason: str = ""
    counterfactual_thesis: str = ""
    counterfactual_mechanism: str = ""
    counterfactual_critique: str = ""
    counterfactual_anchor: dict[str, Any] = field(default_factory=dict)
    counterfactual_expected_direction: str = ""
    matched_anchor_features: list[str] = field(default_factory=list)
    matched_anchor_pattern: bool = False
    expected_anchor_features: list[str] = field(default_factory=list)
    expected_anchor_patterns: list[str] = field(default_factory=list)
    anchor_fallback_reason: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentFactorTask":
        return cls(
            factor_name=str(payload.get("factor_name", payload.get("name", ""))).strip(),
            description=str(payload.get("description", payload.get("factor_description", ""))),
            formulation=str(payload.get("formulation", payload.get("factor_formulation", ""))),
            expression=str(payload.get("expression", payload.get("factor_expression", ""))),
            variables=dict(payload.get("variables", {})) if isinstance(payload.get("variables", {}), dict) else {},
            validation_warnings=[
                str(item)
                for item in list(payload.get("validation_warnings", []))
                if str(item).strip()
            ],
            retained_for_backtest=bool(payload.get("retained_for_backtest", True)),
            theory_alignment=str(payload.get("theory_alignment", "mainline") or "mainline"),
            counterfactual_reason=str(payload.get("counterfactual_reason", "")),
            counterfactual_thesis=str(payload.get("counterfactual_thesis", "")),
            counterfactual_mechanism=str(payload.get("counterfactual_mechanism", "")),
            counterfactual_critique=str(payload.get("counterfactual_critique", "")),
            counterfactual_anchor=dict(payload.get("counterfactual_anchor", {}))
            if isinstance(payload.get("counterfactual_anchor", {}), dict)
            else {},
            counterfactual_expected_direction=str(payload.get("counterfactual_expected_direction", "")),
            matched_anchor_features=[
                str(item)
                for item in list(payload.get("matched_anchor_features", []))
                if str(item).strip()
            ],
            matched_anchor_pattern=bool(payload.get("matched_anchor_pattern", False)),
            expected_anchor_features=[
                str(item)
                for item in list(payload.get("expected_anchor_features", []))
                if str(item).strip()
            ],
            expected_anchor_patterns=[
                str(item)
                for item in list(payload.get("expected_anchor_patterns", []))
                if str(item).strip()
            ],
            anchor_fallback_reason=str(payload.get("anchor_fallback_reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "description": self.description,
            "formulation": self.formulation,
            "expression": self.expression,
            "variables": dict(self.variables),
            "validation_warnings": list(self.validation_warnings),
            "retained_for_backtest": bool(self.retained_for_backtest),
            "theory_alignment": self.theory_alignment,
            "counterfactual_reason": self.counterfactual_reason,
            "counterfactual_thesis": self.counterfactual_thesis,
            "counterfactual_mechanism": self.counterfactual_mechanism,
            "counterfactual_critique": self.counterfactual_critique,
            "counterfactual_anchor": dict(self.counterfactual_anchor),
            "counterfactual_expected_direction": self.counterfactual_expected_direction,
            "matched_anchor_features": list(self.matched_anchor_features),
            "matched_anchor_pattern": bool(self.matched_anchor_pattern),
            "expected_anchor_features": list(self.expected_anchor_features),
            "expected_anchor_patterns": list(self.expected_anchor_patterns),
            "anchor_fallback_reason": self.anchor_fallback_reason,
        }


@dataclass
class QlibFactorExperiment:
    experiment_id: str
    target_hypothesis: str = ""
    factors: list[ExperimentFactorTask] = field(default_factory=list)
    factor_names: list[str] = field(default_factory=list)
    deduplicated_against: list[str] = field(default_factory=list)
    task_plan: list[str] = field(default_factory=list)
    attempt_count: int = 0
    consistency_gate: dict[str, Any] = field(default_factory=dict)
    factor_implementation: dict[str, Any] = field(default_factory=dict)
    calculation_report: dict[str, Any] = field(default_factory=dict)
    backtest_report: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    per_factor_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    phase: str = "original"
    round_idx: int = 0
    trajectory_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(*, round_idx: int, phase: str, seed: str = "") -> str:
        safe_seed = "".join(ch for ch in str(seed) if ch.isalnum() or ch in {"_", "-"}).strip("_-")
        suffix = safe_seed or "exp"
        return f"{phase.lower()}_{int(round_idx):03d}_{suffix}"

    @classmethod
    def from_shared_payload(cls, payload: dict[str, Any]) -> "QlibFactorExperiment":
        raw = payload.get("qlib_factor_experiment")
        if isinstance(raw, dict):
            return cls.from_dict(raw)
        factor_rows = payload.get("experiment_spec", {}).get("factors", []) if isinstance(payload.get("experiment_spec", {}), dict) else []
        factors = [ExperimentFactorTask.from_dict(item) for item in factor_rows if isinstance(item, dict)]
        experiment_spec = payload.get("experiment_spec", {})
        experiment_id = str(payload.get("experiment_id", "")).strip() or cls.generate_id(
            round_idx=int(payload.get("round_idx", 0) or 0),
            phase=str(payload.get("phase", "original")),
            seed=str(payload.get("trajectory_id", "")),
        )
        return cls(
            experiment_id=experiment_id,
            target_hypothesis=str(experiment_spec.get("target_hypothesis", "")) if isinstance(experiment_spec, dict) else "",
            factors=factors,
            factor_names=[task.factor_name for task in factors],
            deduplicated_against=list(experiment_spec.get("deduplicated_against", [])) if isinstance(experiment_spec, dict) else [],
            task_plan=list(payload.get("task_plan", [])),
            attempt_count=int(experiment_spec.get("attempt_count", 0)) if isinstance(experiment_spec, dict) else 0,
            consistency_gate=dict(experiment_spec.get("consistency_gate", {})) if isinstance(experiment_spec, dict) else {},
            factor_implementation=dict(payload.get("factor_implementation", {})),
            calculation_report=dict(payload.get("calculation_report", {})),
            backtest_report=dict(payload.get("backtest_report", {})),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items() if isinstance(v, (int, float))},
            per_factor_metrics=dict(payload.get("per_factor_metrics", {})),
            feedback=dict(payload.get("feedback", {})),
            phase=str(payload.get("phase", "original")),
            round_idx=int(payload.get("round_idx", 0) or 0),
            trajectory_id=str(payload.get("trajectory_id", "")),
            parent_ids=[str(item) for item in list(payload.get("parent_ids", []))],
            extra={},
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QlibFactorExperiment":
        factors = payload.get("factors", [])
        return cls(
            experiment_id=str(payload.get("experiment_id", "")),
            target_hypothesis=str(payload.get("target_hypothesis", "")),
            factors=[ExperimentFactorTask.from_dict(item) for item in factors if isinstance(item, dict)],
            factor_names=[str(item) for item in list(payload.get("factor_names", []))],
            deduplicated_against=[str(item) for item in list(payload.get("deduplicated_against", []))],
            task_plan=[str(item) for item in list(payload.get("task_plan", []))],
            attempt_count=int(payload.get("attempt_count", 0) or 0),
            consistency_gate=dict(payload.get("consistency_gate", {})),
            factor_implementation=dict(payload.get("factor_implementation", {})),
            calculation_report=dict(payload.get("calculation_report", {})),
            backtest_report=dict(payload.get("backtest_report", {})),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items() if isinstance(v, (int, float))},
            per_factor_metrics=dict(payload.get("per_factor_metrics", {})),
            feedback=dict(payload.get("feedback", {})),
            phase=str(payload.get("phase", "original")),
            round_idx=int(payload.get("round_idx", 0) or 0),
            trajectory_id=str(payload.get("trajectory_id", "")),
            parent_ids=[str(item) for item in list(payload.get("parent_ids", []))],
            created_at=str(payload.get("created_at", datetime.now().isoformat())),
            updated_at=str(payload.get("updated_at", datetime.now().isoformat())),
            extra=dict(payload.get("extra", {})),
        )

    def to_experiment_spec(self) -> dict[str, Any]:
        return {
            "target_hypothesis": self.target_hypothesis,
            "factors": [task.to_dict() for task in self.factors],
            "factor_names": list(self.factor_names or [task.factor_name for task in self.factors]),
            "deduplicated_against": list(self.deduplicated_against),
            "attempt_count": int(self.attempt_count),
            "consistency_gate": dict(self.consistency_gate),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "target_hypothesis": self.target_hypothesis,
            "factors": [task.to_dict() for task in self.factors],
            "factor_names": list(self.factor_names or [task.factor_name for task in self.factors]),
            "deduplicated_against": list(self.deduplicated_against),
            "task_plan": list(self.task_plan),
            "attempt_count": int(self.attempt_count),
            "consistency_gate": dict(self.consistency_gate),
            "factor_implementation": dict(self.factor_implementation),
            "calculation_report": dict(self.calculation_report),
            "backtest_report": dict(self.backtest_report),
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "per_factor_metrics": dict(self.per_factor_metrics),
            "feedback": dict(self.feedback),
            "phase": self.phase,
            "round_idx": int(self.round_idx),
            "trajectory_id": self.trajectory_id,
            "parent_ids": list(self.parent_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "extra": dict(self.extra),
        }

    def update_timestamps(self) -> None:
        self.updated_at = datetime.now().isoformat()


class RoundPhase(str, Enum):
    """遗传进化阶段枚举。"""
    ORIGINAL = "original"      # 原始探索阶段（第 0 轮）
    MUTATION = "mutation"      # 变异阶段（第 1+ 轮）
    CROSSOVER = "crossover"    # 交叉阶段（第 1+ 轮）
    EVOLVED = "evolved"        # mutation/crossover 规划综合后的下一代执行分支


@dataclass
class StrategyTrajectory:
    """策略轨迹：记录一次完整因子挖掘循环的输入、输出和遗传元数据。

    是遗传进化系统的核心数据单元，每完成一个 EvolutionTask 就会生成一条轨迹。
    轨迹被存储在 TrajectoryPool 中，供后续轮次的父代选择使用。

    关键字段说明：
    - trajectory_id: 唯一标识，格式为 {phase}_{round_idx:03d}_{direction_id:03d}
    - direction_id: 方向编号，用于多样性评分
    - round_idx: 所属进化轮次
    - phase: 遗传阶段（ORIGINAL/MUTATION/CROSSOVER/EVOLVED）
    - parent_ids: 父代轨迹 ID，构成进化血缘关系
    - backtest_metrics: 回测指标（RankIC/ICIR/Sharpe/drawdown/coverage/turnover）
    """

    trajectory_id: str                     # 轨迹唯一标识
    direction_id: int                      # 方向编号
    round_idx: int                         # 所属进化轮次
    phase: RoundPhase                      # 遗传阶段
    hypothesis: str = ""                   # 假说文本
    hypothesis_details: dict[str, Any] = field(default_factory=dict)  # 假说详情
    factors: list[dict[str, Any]] = field(default_factory=list)        # 因子列表（每个因子含 expression、description 等）
    backtest_result: Any = None            # 完整回测结果
    backtest_metrics: dict[str, float | None] = field(default_factory=dict)  # 回测指标
    feedback: str = ""                     # 反馈摘要文本
    feedback_details: dict[str, Any] = field(default_factory=dict)    # 完整反馈详情
    parent_ids: list[str] = field(default_factory=list)               # 父代轨迹 ID（血缘关系）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())  # 创建时间
    extra_info: dict[str, Any] = field(default_factory=dict)          # 扩展信息（experiment_id、mutation_type 等）

    REWARD_WEIGHTS: ClassVar[dict[str, float]] = {
        "norm_rank_ic": 0.35,
        "norm_rank_icir": 0.15,
        "norm_sharpe": 0.10,
        "norm_coverage": 0.10,
        "norm_drawdown": -0.10,
        "norm_turnover": -0.05,
        "norm_complexity": -0.08,
        "norm_redundancy": -0.06,
        "norm_corr": -0.06,
    }

    @staticmethod
    def generate_id(direction_id: int, round_idx: int, phase: RoundPhase) -> str:
        """生成轨迹 ID，格式: {phase}_{round_idx:03d}_{direction_id:03d}。"""
        return f"{phase.value}_{int(round_idx):03d}_{int(direction_id):03d}"

    def get_primary_metric(self) -> float | None:
        """获取主评估指标值，按优先级查找：RankIC > IC > information_ratio > annualized_return。"""
        for key in ("RankIC", "Rank IC", "IC", "information_ratio", "annualized_return"):
            value = self.backtest_metrics.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    continue
        return None

    def get_reward_score(self) -> float:
        """计算用于遗传父代选择的标准化加权奖励分数。

        先把每个指标按固定阈值压缩到 0~1，再做加权求和：

        reward = Σ weight_i * norm_i

        RankIC 是预测能力主项；ICIR/Sharpe/Coverage 是质量辅助项；
        drawdown/turnover/complexity/redundancy/corr 是风险、交易成本与论文式正则惩罚。
        缺失指标按 0 处理，兼容旧轨迹。
        """
        components = self.get_reward_components()
        return sum(weight * components.get(name, 0.0) for name, weight in self.REWARD_WEIGHTS.items())

    def get_reward_components(self) -> dict[str, float]:
        """返回标准化后的 reward 组件，保持每个指标基本落在 0~1。

        这个组件表对应讨论中的标准化设计，确保不同量纲的指标在加权前可比较。
        """
        norm_redundancy = self._clip(
            self._penalty_value(
                "structural_similarity",
                "structural_redundancy",
                "structural_redundancy_score",
                "redundancy_score",
                "max_structural_similarity",
            ),
            0.0,
            1.0,
        )
        return {
            # Fix1: 去掉 abs()，负 IC 贡献为 0（clip 到下界），只有正向预测能力才加分
            "norm_rank_ic": self._clip(self._metric_value("Rank IC", "RankIC") / 0.05, 0.0, 1.0),
            "norm_rank_icir": self._clip(self._metric_value("Rank ICIR", "RankICIR", "ICIR") / 0.5, 0.0, 1.0),
            # Fix2: 下界从 0 改为 -1，负 Sharpe 产生实际惩罚
            "norm_sharpe": self._clip(self._metric_value("sharpe", "information_ratio") / 3.0, -1.0, 1.0),
            "norm_coverage": self._clip(self._metric_value("coverage"), 0.0, 1.0),
            "norm_drawdown": self._clip(abs(self._metric_value("drawdown", "max_drawdown")) / 0.30, 0.0, 1.0),
            "norm_turnover": self._clip(self._metric_value("turnover") / 1.0, 0.0, 1.0),
            "norm_complexity": self._normalized_complexity_penalty(),
            "norm_redundancy": norm_redundancy,
            "norm_corr": self._clip(
                self._penalty_value("max_abs_corr", "max_abs_correlation", "correlation_penalty", "max_corr"),
                0.0,
                1.0,
            ),
        }

    def _metric_value(self, *aliases: str) -> float:
        """按别名列表查找第一个可用的回测指标值，缺失或非法时返回 0.0。"""
        for alias in aliases:
            if alias not in self.backtest_metrics:
                continue
            value = self.backtest_metrics.get(alias)
            try:
                return float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _finite_float(value: object) -> float | None:
        try:
            parsed = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        if parsed is None or parsed != parsed or parsed in {float("inf"), float("-inf")}:
            return None
        return parsed

    @classmethod
    def _clip(cls, value: object, lower: float, upper: float) -> float:
        parsed = cls._finite_float(value)
        if parsed is None:
            return lower
        return max(lower, min(upper, parsed))

    def _penalty_value(self, *aliases: str) -> float:
        """从指标、扩展信息或因子元数据里读取第一个可用惩罚项。"""
        containers: list[dict[str, Any]] = [self.backtest_metrics, self.extra_info]
        for key in ("reward_penalties", "reward_components", "penalties"):
            value = self.extra_info.get(key)
            if isinstance(value, dict):
                containers.append(value)
        for factor in self.factors:
            if isinstance(factor, dict):
                containers.append(factor)
                for key in ("complexity", "generation_gate", "novelty_gate", "redundancy_filter_decision"):
                    nested = factor.get(key)
                    if isinstance(nested, dict):
                        containers.append(nested)
        if isinstance(self.backtest_result, dict):
            for key in ("reward_penalties", "redundancy_filter", "generation_gate"):
                value = self.backtest_result.get(key)
                if isinstance(value, dict):
                    containers.append(value)
        for container in containers:
            for alias in aliases:
                parsed = self._finite_float(container.get(alias))
                if parsed is not None:
                    return parsed
        return 0.0

    def _normalized_complexity_penalty(self) -> float:
        """按论文复杂度约束归一化复杂度惩罚，优先使用已有 complexity_score。"""
        complexity_score = self._penalty_value("complexity_score")
        complexity_threshold = self._penalty_value("complexity_threshold", "max_complexity_score")
        if complexity_score > 0.0 and complexity_threshold > 0.0:
            return self._clip(complexity_score / complexity_threshold, 0.0, 1.0)

        symbol_length = self._penalty_value("symbol_length", "expression_length")
        base_feature_count = self._penalty_value("base_feature_count", "input_feature_count")
        free_args_ratio = self._penalty_value("free_args_ratio")
        if symbol_length <= 0.0 and base_feature_count <= 0.0 and free_args_ratio <= 0.0:
            return 0.0
        norm_symbol_length = self._clip(symbol_length / 250.0, 0.0, 1.0)
        norm_base_features = self._clip(base_feature_count / 6.0, 0.0, 1.0)
        norm_free_args = self._clip(free_args_ratio / 0.5, 0.0, 1.0)
        return 0.4 * norm_symbol_length + 0.3 * norm_free_args + 0.3 * norm_base_features

    def is_successful(self) -> bool:
        """判断轨迹是否成功：有主指标值 AND 通过质量门槛（train_filter）。

        质量门槛：若回测启用了 train_filter（sample_windows_enabled），
        则最佳因子必须通过 Sharpe 阈值检验，否则视为失败轨迹，进入
        evolution_runner 的 failed_trajectories 桶，LLM 专门学习失败原因。
        """
        metric = self.get_primary_metric()
        if metric is None:
            return False
        # 检查质量门槛（train_filter 决策）
        backtest_result = self.backtest_result if isinstance(self.backtest_result, dict) else {}
        train_filter = backtest_result.get("train_filter", {})
        if isinstance(train_filter, dict) and bool(train_filter.get("enabled", False)):
            selected_name = str(backtest_result.get("selected_factor_name", "")).strip()
            decisions = train_filter.get("decisions", {})
            if isinstance(decisions, dict) and selected_name and selected_name in decisions:
                decision = decisions[selected_name]
                if isinstance(decision, dict) and "passed" in decision:
                    return bool(decision["passed"])
            # 兜底：检查是否有任何因子通过
            retained = train_filter.get("retained_factor_names", None)
            if isinstance(retained, list):
                return len(retained) > 0
        return True

    def to_dict(self) -> dict[str, Any]:
        """将轨迹序列化为字典（用于持久化和 JSON 输出）。"""
        payload = {
            "trajectory_id": self.trajectory_id,
            "direction_id": self.direction_id,
            "round_idx": self.round_idx,
            "phase": self.phase.value,
            "hypothesis": self.hypothesis,
            "hypothesis_details": self.hypothesis_details,
            "factors": self.factors,
            "backtest_result": self.backtest_result,
            "backtest_metrics": self.backtest_metrics,
            "feedback": self.feedback,
            "feedback_details": self.feedback_details,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
            "extra_info": self.extra_info,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyTrajectory":
        """从字典反序列化轨迹（用于从 JSON 加载持久化的轨迹池）。"""
        return cls(
            trajectory_id=str(payload.get("trajectory_id", "")),
            direction_id=int(payload.get("direction_id", 0) or 0),
            round_idx=int(payload.get("round_idx", 0) or 0),
            phase=RoundPhase(str(payload.get("phase", RoundPhase.ORIGINAL.value))),
            hypothesis=str(payload.get("hypothesis", "")),
            hypothesis_details=dict(payload.get("hypothesis_details", {})),
            factors=list(payload.get("factors", [])),
            backtest_result=payload.get("backtest_result"),
            backtest_metrics=dict(payload.get("backtest_metrics", {})),
            feedback=str(payload.get("feedback", "")),
            feedback_details=dict(payload.get("feedback_details", {})),
            parent_ids=[str(item) for item in list(payload.get("parent_ids", []))],
            created_at=str(payload.get("created_at", datetime.now().isoformat())),
            extra_info=dict(payload.get("extra_info", {})),
        )


@dataclass
class TrajectoryPool:
    """轨迹池：存储所有进化轮次中产生的 StrategyTrajectory。

    核心功能：
    1. 持久化到 artifacts/trajectory_pool.jsonl（支持跨会话恢复）
    2. 按阶段（ORIGINAL/MUTATION/CROSSOVER/EVOLVED）筛选轨迹
    3. 按奖励分数排序选择最佳轨迹
    4. 为变异选择 top-N 父代（select_parents_for_mutation）
    5. 为交叉选择最优父代组合（select_groups_for_crossover）
    6. 多样性评分（direction_id/phase/expression/hypothesis 去重程度）
    """

    trajectories: list[StrategyTrajectory] = field(default_factory=list)  # 所有轨迹列表
    save_path: str | None = None  # 持久化路径（相对于项目根目录）

    def add(self, trajectory: StrategyTrajectory) -> None:
        """添加或更新轨迹，并自动持久化到磁盘。"""
        for idx, existing in enumerate(self.trajectories):
            if existing.trajectory_id == trajectory.trajectory_id:
                self.trajectories[idx] = trajectory
                self._append_or_save(trajectory)
                return
        self.trajectories.append(trajectory)
        self._append_or_save(trajectory)

    def get_all(self) -> list[StrategyTrajectory]:
        """返回所有轨迹的副本。"""
        return list(self.trajectories)

    def get_by_phase(self, phase: RoundPhase) -> list[StrategyTrajectory]:
        """按阶段筛选轨迹（ORIGINAL/MUTATION/CROSSOVER）。"""
        return [item for item in self.trajectories if item.phase == phase]

    def get_by_id(self, trajectory_id: str) -> StrategyTrajectory | None:
        """按 trajectory_id 精确查找轨迹。"""
        for item in self.trajectories:
            if item.trajectory_id == trajectory_id:
                return item
        return None

    def get_statistics(self) -> dict[str, Any]:
        """返回轨迹池统计信息（总数和各阶段数量）。"""
        phase_counts = {
            RoundPhase.ORIGINAL.value: len(self.get_by_phase(RoundPhase.ORIGINAL)),
            RoundPhase.MUTATION.value: len(self.get_by_phase(RoundPhase.MUTATION)),
            RoundPhase.CROSSOVER.value: len(self.get_by_phase(RoundPhase.CROSSOVER)),
            RoundPhase.EVOLVED.value: len(self.get_by_phase(RoundPhase.EVOLVED)),
        }
        return {
            "total": len(self.trajectories),
            "by_phase": phase_counts,
        }

    def get_best_trajectories(self, top_n: int = 5) -> list[StrategyTrajectory]:
        """按奖励分数降序返回 top-N 成功轨迹（用于提供给 EvolutionAgent 的 best_trajectories 上下文）。"""
        valid = [item for item in self.trajectories if item.is_successful()]
        valid.sort(key=lambda item: self._selection_score(item), reverse=True)
        return valid[: max(0, int(top_n))]

    def select_parents_for_mutation(self, top_n: int = 3) -> list[StrategyTrajectory]:
        """按奖励分数+多样性选择变异阶段的父轨迹（贪心多样性感知）。

        先按奖励排序取最高分轨迹做种子，之后每一步在剩余候选中选择
        ``reward_score + 0.05 * diversity_score(已选 + 候选)`` 最大者，
        使新加入的父代既高分又与已选父代尽量不同，避免变异集中在高度相似的轨迹上。

        说明：旧实现用 ``diversity_score([单条轨迹])`` 作为 tiebreaker，但单条轨迹的
        多样性恒为 1.0，等于没有 tiebreaker；此处改为相对“已选集合”的增量多样性。
        """
        limit = max(0, int(top_n))
        if limit <= 0:
            return []
        ranked = sorted(self.trajectories, key=lambda item: self._selection_score(item), reverse=True)
        if not ranked:
            return []
        selected: list[StrategyTrajectory] = [ranked[0]]
        remaining = ranked[1:]
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda item: self._selection_score(item)
                + 0.05 * self.calculate_diversity_score(selected + [item]),
            )
            selected.append(best)
            remaining = [item for item in remaining if item is not best]
        return selected

    def select_groups_for_crossover(self, group_size: int = 2, n_groups: int = 3) -> list[list[StrategyTrajectory]]:
        """结合奖励和多样性生成不重复的交叉父代组合。

        流程：
        1. 按奖励分数排序，取前 50 个候选轨迹
        2. 枚举所有 group_size 大小的组合（去重）
        3. 对每个组合计算综合分 = 平均奖励 + 0.05 * 多样性分数
        4. 按综合分降序返回 top n_groups 个组合
        """
        size = max(0, int(group_size))
        limit = max(0, int(n_groups))
        if size <= 0 or limit <= 0 or len(self.trajectories) < size:
            return []

        ranked_candidates = sorted(self.trajectories, key=lambda item: self._selection_score(item), reverse=True)
        candidates = ranked_candidates[:50]
        scored_groups: list[tuple[float, tuple[str, ...], list[StrategyTrajectory]]] = []
        seen_parent_sets: set[tuple[str, ...]] = set()
        for group_items in combinations(candidates, size):
            parent_key = tuple(sorted(item.trajectory_id for item in group_items))
            if len(set(parent_key)) != size:
                continue
            if parent_key in seen_parent_sets:
                continue
            seen_parent_sets.add(parent_key)
            reward_score = sum(self._selection_score(item) for item in group_items) / float(size)
            diversity_score = self.calculate_diversity_score(list(group_items))
            scored_groups.append((reward_score + 0.05 * diversity_score, parent_key, list(group_items)))

        scored_groups.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [group for _, _, group in scored_groups[:limit]]

    @staticmethod
    def _selection_score(trajectory: StrategyTrajectory) -> float:
        """论文式父代选择分数：统一使用标准化 reward。"""
        return trajectory.get_reward_score()

    @staticmethod
    def calculate_diversity_score(trajectories: list[StrategyTrajectory]) -> float:
        """计算一组轨迹的多样性分数（0.0 ~ 1.0）。

        综合考虑四个维度的去重程度：
        - direction_score: 不同 direction_id 的占比
        - phase_score: 不同 phase 的占比
        - expression_score: 不同因子表达式签名的占比
        - hypothesis_score: 不同假说文本的占比

        最终分数为四维平均。
        """
        if not trajectories:
            return 0.0
        count = float(len(trajectories))
        direction_score = len({item.direction_id for item in trajectories}) / count
        phase_score = len({item.phase for item in trajectories}) / count
        expression_score = len({TrajectoryPool._factor_expression_signature(item) for item in trajectories}) / count
        hypothesis_score = len({item.hypothesis.strip() for item in trajectories}) / count
        return (direction_score + phase_score + expression_score + hypothesis_score) / 4.0

    @staticmethod
    def _factor_expression_signature(trajectory: StrategyTrajectory) -> tuple[str, ...]:
        """提取轨迹中所有因子的表达式，作为有序元组用于多样性去重比较。"""
        expressions: list[str] = []
        for factor in trajectory.factors:
            if not isinstance(factor, dict):
                continue
            expression = str(factor.get("expression", factor.get("factor_expression", ""))).strip()
            if expression:
                expressions.append(expression)
        return tuple(expressions)

    def to_dict(self) -> dict[str, Any]:
        """将轨迹池序列化为字典（路径转为相对路径以保证可移植性）。"""
        return {
            "trajectories": [item.to_dict() for item in self.trajectories],
            "save_path": self._to_portable_path(self.save_path),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryPool":
        """从字典反序列化轨迹池。"""
        items = payload.get("trajectories", [])
        return cls(
            trajectories=[StrategyTrajectory.from_dict(item) for item in items if isinstance(item, dict)],
            save_path=str(payload.get("save_path", "")) or None,
        )

    def load(self) -> None:
        """从磁盘加载轨迹池（如果文件不存在则静默跳过）。"""
        if not self.save_path:
            return
        path = self._resolve_save_path()
        if path.suffix.lower() == ".jsonl":
            self._load_jsonl(path)
            return
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = self.from_dict(data)
        self.trajectories = loaded.trajectories

    def _load_jsonl(self, path: Path) -> None:
        """从 append-only JSONL 日志加载轨迹，按 trajectory_id 做最后写入胜出。"""
        by_id: dict[str, StrategyTrajectory] = {}
        ordered_ids: list[str] = []
        candidate_paths = [path]
        legacy_path = path.with_suffix(".json")
        if not path.exists() and legacy_path.exists():
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
            loaded = self.from_dict(data)
            self.trajectories = loaded.trajectories
            return
        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                trajectory = StrategyTrajectory.from_dict(item)
                if not trajectory.trajectory_id:
                    continue
                if trajectory.trajectory_id not in by_id:
                    ordered_ids.append(trajectory.trajectory_id)
                by_id[trajectory.trajectory_id] = trajectory
        self.trajectories = [by_id[item_id] for item_id in ordered_ids if item_id in by_id]

    def _append_or_save(self, trajectory: StrategyTrajectory) -> None:
        if not self.save_path:
            return
        path = self._resolve_save_path()
        if path.suffix.lower() == ".jsonl":
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, separators=(",", ":"), default=str))
                file.write("\n")
            return
        self._save()

    def _save(self) -> None:
        """将轨迹池持久化到磁盘（自动创建目录）。"""
        if not self.save_path:
            return
        path = self._resolve_save_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".jsonl":
            with path.open("w", encoding="utf-8") as file:
                for trajectory in self.trajectories:
                    file.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, separators=(",", ":"), default=str))
                    file.write("\n")
            return
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _project_root() -> Path:
        """获取项目根目录（从当前文件向上 3 级）。"""
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _to_portable_path(cls, raw_path: str | None) -> str:
        """将绝对路径转为相对于项目根目录的路径，确保跨机器可移植。"""
        text = str(raw_path or "").strip()
        if not text:
            return ""
        path = Path(text)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(cls._project_root().resolve()).as_posix()
        except Exception:
            return path.as_posix()

    def _resolve_save_path(self) -> Path:
        """将相对路径解析为绝对路径（相对于项目根目录）。"""
        text = str(self.save_path or "").strip()
        path = Path(text)
        if path.is_absolute():
            return path
        return self._project_root() / path
