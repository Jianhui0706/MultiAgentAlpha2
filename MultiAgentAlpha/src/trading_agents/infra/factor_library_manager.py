from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .factor_direction import build_direction_diagnostics, expected_sign_for_factor, expected_sign_from_hypothesis
from .factor_signal import build_factor_signal_diagnostics
from trading_agents.schemas import QlibFactorExperiment, RoundPhase, StrategyTrajectory


class FactorLibraryManager:
    def __init__(self, library_path: str | Path) -> None:
        self.library_path = Path(library_path)
        self.data = self._load()

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def to_portable_path(cls, raw_path: str | Path) -> str:
        text = str(raw_path).strip()
        if not text:
            return ""
        path = Path(text)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(cls.project_root().resolve()).as_posix()
        except Exception:
            return path.as_posix()

    @staticmethod
    def default_library_path() -> Path:
        explicit = os.environ.get("FACTOR_LIBRARY_PATH", "").strip()
        if explicit:
            return Path(explicit)
        suffix = os.environ.get("FACTOR_LIBRARY_SUFFIX", "").strip()
        filename = f"all_factors_library_{suffix}.json" if suffix else "all_factors_library.json"
        return FactorLibraryManager.project_root() / "factor_library" / filename

    @staticmethod
    def default_cache_dir() -> Path:
        return Path(os.environ.get("FACTOR_CACHE_DIR", str(FactorLibraryManager.project_root() / "factor_library" / "cache")))

    def _load(self) -> dict[str, Any]:
        if self.library_path.exists():
            try:
                return json.loads(self.library_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        now = datetime.now().isoformat()
        return {
            "metadata": {
                "created_at": now,
                "last_updated": now,
                "total_factors": 0,
                "version": "1.0",
                "cache_reuse_mode": "realtime_compute",
                "cache_reuse_todo": "source_pickle_path_persistence",
            },
            "factors": {},
        }

    def _save(self) -> None:
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.data["metadata"]["total_factors"] = len(self.data.get("factors", {}))
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.library_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _factor_id(factor_name: str, factor_expression: str) -> str:
        return hashlib.md5(f"{factor_name}_{factor_expression}".encode()).hexdigest()[:16]

    @staticmethod
    def _extract_backtest_period(backtest_report: dict[str, Any]) -> dict[str, str | None]:
        """从回测报告中提取回测区间（start/end），用于给入库因子打标记。

        优先使用回测引擎写入的 ``backtest_period`` 字段；缺失时退回到
        ``sample_windows`` 的 train.start ~ validation.end 推断整体区间，
        以兼容旧回测报告。
        """
        if not isinstance(backtest_report, dict):
            return {"start": None, "end": None}
        period = backtest_report.get("backtest_period")
        if isinstance(period, dict):
            start = period.get("start")
            end = period.get("end")
            return {
                "start": str(start) if start else None,
                "end": str(end) if end else None,
            }
        sample_windows = backtest_report.get("sample_windows", {})
        if isinstance(sample_windows, dict):
            train = sample_windows.get("train", {}) if isinstance(sample_windows.get("train", {}), dict) else {}
            validation = sample_windows.get("validation", {}) if isinstance(sample_windows.get("validation", {}), dict) else {}
            start = train.get("start")
            end = validation.get("end") or validation.get("start") or train.get("end")
            return {
                "start": str(start) if start else None,
                "end": str(end) if end else None,
            }
        return {"start": None, "end": None}

    def get_factor_expressions(self) -> list[str]:
        """Return unique non-empty factor expressions stored in the library."""
        expressions: list[str] = []
        for item in self.data.get("factors", {}).values():
            if not isinstance(item, dict):
                continue
            expression = str(item.get("factor_expression", "") or item.get("expression", "")).strip()
            if expression:
                expressions.append(expression)
        return list(dict.fromkeys(expressions))

    @staticmethod
    def _find_implementation(experiment: QlibFactorExperiment, factor_name: str) -> dict[str, Any]:
        implementation = experiment.factor_implementation.get("implementations", [])
        if not isinstance(implementation, list):
            return {}
        lowered = factor_name.strip().lower()
        for item in implementation:
            if not isinstance(item, dict):
                continue
            if str(item.get("factor_name", "")).strip().lower() == lowered:
                return item
        return {}

    def add_experiment(self, experiment: QlibFactorExperiment, panel_data_path: str | None = None) -> None:
        factors = experiment.factors
        if not factors:
            return
        factor_bucket = self.data.setdefault("factors", {})
        run_id = os.environ.get("FACTOR_LIBRARY_RUN_ID", "").strip()
        # 回测区间标记：本次回测覆盖的时间范围，写入每个入库因子。
        backtest_period = self._extract_backtest_period(experiment.backtest_report)
        panel_data_path_value = self.to_portable_path(panel_data_path or str(experiment.backtest_report.get("panel_data_path", "")))
        train_filter_report = experiment.backtest_report.get("train_filter", {})
        train_filter_enabled = (
            bool(train_filter_report.get("enabled", False))
            if isinstance(train_filter_report, dict)
            else False
        )
        retained_factor_names = {
            str(item).strip()
            for item in list(train_filter_report.get("retained_factor_names", []))
            if str(item).strip()
        } if isinstance(train_filter_report, dict) else set()
        train_filtered_factor_names = {
            str(item).strip()
            for item in list(train_filter_report.get("filtered_factor_names", []))
            if str(item).strip()
        } if isinstance(train_filter_report, dict) else set()
        train_filter_decisions = train_filter_report.get("decisions", {}) if isinstance(train_filter_report, dict) else {}
        coverage_filter_report = experiment.backtest_report.get("coverage_filter", {})
        coverage_filter_enabled = (
            bool(coverage_filter_report.get("enabled", False))
            if isinstance(coverage_filter_report, dict)
            else False
        )
        coverage_filtered_factor_names = {
            str(item).strip()
            for item in list(coverage_filter_report.get("filtered_factor_names", []))
            if str(item).strip()
        } if isinstance(coverage_filter_report, dict) else set()
        coverage_filter_decisions = (
            coverage_filter_report.get("decisions", {}) if isinstance(coverage_filter_report, dict) else {}
        )
        redundancy_filter_report = experiment.backtest_report.get("redundancy_filter", {})
        redundancy_filter_enabled = (
            bool(redundancy_filter_report.get("enabled", False))
            if isinstance(redundancy_filter_report, dict)
            else False
        )
        redundancy_filter_decisions = (
            redundancy_filter_report.get("decisions", {}) if isinstance(redundancy_filter_report, dict) else {}
        )
        # 留样验证衰减门：val 综合 reward 相对 train 衰减过大者不入库。
        validation_decay_report = experiment.backtest_report.get("validation_decay_filter", {})
        validation_decay_enabled = (
            bool(validation_decay_report.get("enabled", False))
            if isinstance(validation_decay_report, dict)
            else False
        )
        validation_decay_decisions = (
            validation_decay_report.get("decisions", {}) if isinstance(validation_decay_report, dict) else {}
        )
        per_factor_validation_metrics = (
            experiment.backtest_report.get("per_factor_validation_metrics", {})
            if isinstance(experiment.backtest_report.get("per_factor_validation_metrics", {}), dict)
            else {}
        )
        saved_factor_count = 0
        for task in factors:
            implementation = self._find_implementation(experiment, task.factor_name)
            factor_expression = str(implementation.get("expression", "")).strip() or task.expression
            factor_code = str(implementation.get("code", "")).strip()
            factor_id = self._factor_id(task.factor_name, factor_expression)
            factor_metrics = dict(experiment.per_factor_metrics.get(task.factor_name, {}))
            # 回测完全没有产出任何指标 → 无法评估（不是"缺某个数值"，而是整体没跑出结果），跳过
            if not factor_metrics:
                continue
            if train_filter_enabled:
                train_decision = (
                    dict(train_filter_decisions.get(task.factor_name, {}))
                    if isinstance(train_filter_decisions, dict)
                    and isinstance(train_filter_decisions.get(task.factor_name, {}), dict)
                    else {}
                )
                if task.factor_name in train_filtered_factor_names:
                    continue
                if retained_factor_names and task.factor_name not in retained_factor_names:
                    continue
                if train_decision and not bool(train_decision.get("passed", False)):
                    continue
            if coverage_filter_enabled:
                coverage_decision = (
                    dict(coverage_filter_decisions.get(task.factor_name, {}))
                    if isinstance(coverage_filter_decisions, dict)
                    and isinstance(coverage_filter_decisions.get(task.factor_name, {}), dict)
                    else {}
                )
                if task.factor_name in coverage_filtered_factor_names:
                    continue
                if coverage_decision and not bool(coverage_decision.get("passed", False)):
                    continue
            if validation_decay_enabled:
                validation_decay_decision = (
                    dict(validation_decay_decisions.get(task.factor_name, {}))
                    if isinstance(validation_decay_decisions, dict)
                    and isinstance(validation_decay_decisions.get(task.factor_name, {}), dict)
                    else {}
                )
                if task.factor_name in {
                    str(item).strip()
                    for item in list(validation_decay_report.get("filtered_factor_names", []))
                    if str(item).strip()
                }:
                    continue
                if validation_decay_decision and not bool(validation_decay_decision.get("passed", False)):
                    continue
            # 入库判定：仅「基础有效性」必要条件硬淘汰；缺失指标跳过该项判断，
            # 不因缺某个数据淘汰好因子。预测稳定性/策略表现/鲁棒性 交由 reward 评分排序。
            admission_gate = self.evaluate_admission_gate(factor_metrics)
            if not admission_gate["passed"]:
                continue
            source_pickle_path = self.to_portable_path(str(implementation.get("source_pickle_path", "")).strip())
            cache_location = {
                "panel_data_path": panel_data_path_value,
                "source_pickle_path": source_pickle_path,
            }
            backtest_report = dict(experiment.backtest_report)
            if "panel_data_path" in backtest_report:
                backtest_report["panel_data_path"] = self.to_portable_path(str(backtest_report.get("panel_data_path", "")))
            if "qlib_config_path" in backtest_report:
                backtest_report["qlib_config_path"] = self.to_portable_path(str(backtest_report.get("qlib_config_path", "")))
            factor_metrics = dict(experiment.per_factor_metrics.get(task.factor_name, {}))
            backtest_errors = backtest_report.get("errors", {})
            factor_failure = {}
            if isinstance(backtest_errors, dict):
                factor_failure = dict(backtest_errors.get(task.factor_name, {})) if isinstance(backtest_errors.get(task.factor_name, {}), dict) else {}
            expected_direction = expected_sign_for_factor(
                dict(experiment.extra.get("hypothesis_structured", {}))
                if isinstance(experiment.extra.get("hypothesis_structured", {}), dict)
                else {},
                factor_name=task.factor_name,
                factor_expression=factor_expression,
                factor_description=task.description,
            )
            if expected_direction == "unknown":
                expected_direction = expected_sign_from_hypothesis(
                    {"expected_direction": dict(experiment.feedback).get("expected_direction", "")}
                )
            direction_diagnostics = build_direction_diagnostics(
                expected_direction=expected_direction,
                metrics=factor_metrics,
            )
            signal_diagnostics = build_factor_signal_diagnostics(factor_metrics)
            # 留样验证：训练/验证两段指标 + 衰减判定 + train/val 综合 reward。
            validation_metrics = dict(per_factor_validation_metrics.get(task.factor_name, {})) if isinstance(
                per_factor_validation_metrics.get(task.factor_name, {}), dict
            ) else {}
            validation_decay_decision = dict(validation_decay_decisions.get(task.factor_name, {})) if isinstance(
                validation_decay_decisions, dict
            ) and isinstance(validation_decay_decisions.get(task.factor_name, {}), dict) else {}
            train_reward = self.compute_reward_score(factor_metrics, factor_entry={"factor_expression": factor_expression})
            validation_reward = (
                self.compute_reward_score(validation_metrics, factor_entry={"factor_expression": factor_expression})
                if validation_metrics
                else None
            )
            combined_reward = (
                0.75 * train_reward + 0.25 * validation_reward
                if validation_reward is not None
                else train_reward
            )
            factor_bucket[factor_id] = {
                "factor_id": factor_id,
                "factor_name": task.factor_name,
                "factor_expression": factor_expression,
                "factor_implementation_code": factor_code,
                "factor_description": task.description,
                "factor_formulation": task.formulation,
                "theory_alignment": task.theory_alignment,
                "counterfactual_reason": task.counterfactual_reason,
                "counterfactual_thesis": task.counterfactual_thesis,
                "counterfactual_mechanism": task.counterfactual_mechanism,
                "counterfactual_critique": task.counterfactual_critique,
                "counterfactual_anchor": dict(task.counterfactual_anchor),
                "counterfactual_expected_direction": task.counterfactual_expected_direction,
                "validation_warnings": list(task.validation_warnings),
                "anchor_diagnostics": {
                    "matched_anchor_features": list(task.matched_anchor_features),
                    "matched_anchor_pattern": bool(task.matched_anchor_pattern),
                    "expected_anchor_features": list(task.expected_anchor_features),
                    "expected_anchor_patterns": list(task.expected_anchor_patterns),
                    "anchor_fallback_reason": task.anchor_fallback_reason,
                },
                "implementation_status": str(implementation.get("status", "")),
                "cache_location": cache_location,
                # 回测区间标记：标识该因子指标来自哪一段历史数据的回测。
                "backtest_period": dict(backtest_period),
                "metadata": {
                    "experiment_id": experiment.experiment_id,
                    "run_id": run_id,
                    "round_number": experiment.round_idx,
                    "evolution_phase": experiment.phase,
                    "trajectory_id": experiment.trajectory_id,
                    "parent_trajectory_ids": list(experiment.parent_ids),
                    "hypothesis": experiment.target_hypothesis,
                    "backtest_period": dict(backtest_period),
                    "created_at": datetime.now().isoformat(),
                },
                "backtest_results": factor_metrics or dict(experiment.metrics),
                "validation_results": validation_metrics,
                "validation_decay": validation_decay_decision,
                "train_reward_score": round(float(train_reward), 6),
                "validation_reward_score": round(float(validation_reward), 6) if validation_reward is not None else None,
                "combined_reward_score": round(float(combined_reward), 6),
                "combined_backtest_results": dict(experiment.metrics),
                "backtest_failure": factor_failure,
                "experiment_metrics": dict(experiment.metrics),
                "direction_diagnostics": direction_diagnostics,
                "expected_direction": direction_diagnostics["expected_direction"],
                "actual_rank_ic_sign": direction_diagnostics["actual_rank_ic_sign"],
                "direction_aligned": direction_diagnostics["direction_aligned"],
                "recommended_usage": direction_diagnostics["recommended_usage"],
                "signal_diagnostics": signal_diagnostics,
                "signal_strength": signal_diagnostics["signal_strength"],
                "signal_action": signal_diagnostics["signal_action"],
                "abs_rank_ic": signal_diagnostics["abs_rank_ic"],
                "weak_signal_flag": signal_diagnostics["weak_signal_flag"],
                "admission_gate": admission_gate,
                "train_filter_decision": dict(train_filter_decisions.get(task.factor_name, {}))
                if isinstance(train_filter_decisions, dict) and isinstance(train_filter_decisions.get(task.factor_name, {}), dict)
                else {},
                "coverage_filter_decision": dict(coverage_filter_decisions.get(task.factor_name, {}))
                if isinstance(coverage_filter_decisions, dict) and isinstance(coverage_filter_decisions.get(task.factor_name, {}), dict)
                else {},
                "redundancy_filter_decision": dict(redundancy_filter_decisions.get(task.factor_name, {}))
                if isinstance(redundancy_filter_decisions, dict) and isinstance(redundancy_filter_decisions.get(task.factor_name, {}), dict)
                else {},
                "backtest_report": backtest_report,
                "feedback": dict(experiment.feedback),
            }
            saved_factor_count += 1
        if saved_factor_count > 0:
            self._save()

    @staticmethod
    def evaluate_admission_gate(metrics: dict[str, Any], *, sharpe_threshold: float | None = None) -> dict[str, Any]:
        """入库硬门——必要条件，任一【有数据且不达标】→ 淘汰（不入库）。

        关键约定：
        - **缺失的指标跳过该项判断**，绝不因为缺某个数据而淘汰一个本来很好的因子。
        - 三档硬门：①基础有效性 ②预测稳定性 ③策略表现。鲁棒性 交由 reward 评分排序。

        ① 基础有效性（截图标准）：
        1. 有效截面数均值 ≥ 30（或标的池的 50%，取较易者，适配小标的池）
        2. |IC| 均值 > 0.03
        3. IC 胜率 > 52%（逐期 IC 与均值同号的比例）

        ② 预测稳定性（核心，硬性规定）：
        4. |ICIR| > 0.05
        5. |RankICIR| > 0.05
        6. RankIC 胜率 > 55%

        ③ 策略表现（硬性规定）：
        7. 多空夏普 > 阈值（默认 1.0，TRAIN_SHARPE_THRESHOLD 可配）
        8. 最大回撤 < 30%
        9. 胜率 > 55%（多空组合正收益期占比）
        """
        import math
        import os

        def _get(*aliases: str) -> float | None:
            for alias in aliases:
                v = metrics.get(alias)
                if v is None:
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    return f
            return None  # 缺失 → 跳过该项

        checks: list[dict[str, Any]] = []
        passed = True

        def _record(tier: str, name: str, value: float | None, threshold: str, ok: bool | None) -> None:
            nonlocal passed
            if value is None or ok is None:
                checks.append({"tier": tier, "name": name, "skipped": True})
                return
            checks.append({
                "tier": tier,
                "name": name,
                "value": round(value, 4),
                "threshold": threshold,
                "passed": bool(ok),
                "skipped": False,
            })
            if not ok:
                passed = False

        # ── ① 基础有效性 ──
        # 1) 有效截面数均值 ≥ 30（或标的池 50%）
        valid_sections = _get("valid_cross_section_mean")
        universe = _get("cross_section_universe_size")
        if valid_sections is None:
            _record("基础有效性", "有效截面数均值", None, "", None)
        else:
            threshold_val = 30.0
            if universe is not None and universe > 0:
                threshold_val = min(30.0, 0.5 * universe)  # ≥30 或 ≥50%池，取较易达成者
            _record("基础有效性", "有效截面数均值", valid_sections, f">={threshold_val:g}",
                    valid_sections >= threshold_val)

        # 2) |IC| 均值 > 0.03
        ic = _get("IC")
        _record("基础有效性", "|IC|均值", abs(ic) if ic is not None else None, ">0.03",
                (abs(ic) > 0.03) if ic is not None else None)

        # 3) IC 胜率 > 52%
        ic_win = _get("ic_win_rate")
        _record("基础有效性", "IC胜率", ic_win, ">0.52", (ic_win > 0.52) if ic_win is not None else None)

        # ── ② 预测稳定性（硬性规定）──
        # 4) |ICIR| > 0.05
        icir = _get("ICIR")
        _record("预测稳定性", "|ICIR|", abs(icir) if icir is not None else None, ">0.05",
                (abs(icir) > 0.05) if icir is not None else None)

        # 5) |RankICIR| > 0.05
        rank_icir = _get("Rank ICIR", "RankICIR")
        _record("预测稳定性", "|RankICIR|", abs(rank_icir) if rank_icir is not None else None, ">0.05",
                (abs(rank_icir) > 0.05) if rank_icir is not None else None)

        # 6) RankIC 胜率 > 55%
        rank_ic_win = _get("rank_ic_win_rate")
        _record("预测稳定性", "RankIC胜率", rank_ic_win, ">0.55",
                (rank_ic_win > 0.55) if rank_ic_win is not None else None)

        # ── ③ 策略表现（硬性规定）──
        # 7) 多空夏普 > 阈值（默认 1.0，可由 TRAIN_SHARPE_THRESHOLD 覆盖）
        try:
            sharpe_thr = (
                float(sharpe_threshold)
                if sharpe_threshold is not None
                else float(os.getenv("TRAIN_SHARPE_THRESHOLD", "").strip() or 1.0)
            )
        except (TypeError, ValueError):
            sharpe_thr = 1.0
        sharpe = _get("sharpe", "information_ratio")
        _record("策略表现", "多空夏普", sharpe, f">{sharpe_thr:g}",
                (sharpe > sharpe_thr) if sharpe is not None else None)

        # 8) 最大回撤 < 30%（drawdown 以正的分数存储）
        drawdown = _get("drawdown", "max_drawdown")
        _record("策略表现", "最大回撤", abs(drawdown) if drawdown is not None else None, "<0.30",
                (abs(drawdown) < 0.30) if drawdown is not None else None)

        # 9) 胜率 > 55%（多空组合正收益期占比）
        win_rate = _get("long_short_positive_ratio", "win_rate")
        _record("策略表现", "胜率", win_rate, ">0.55", (win_rate > 0.55) if win_rate is not None else None)

        return {
            "passed": passed,
            "tiers": ["基础有效性", "预测稳定性", "策略表现"],
            "checks": checks,
            "failed_checks": [c["name"] for c in checks if not c.get("skipped") and not c.get("passed")],
            "skipped_checks": [c["name"] for c in checks if c.get("skipped")],
        }

    @staticmethod
    def compute_reward_score(
        metrics: dict[str, Any],
        factor_entry: dict[str, Any] | None = None,
    ) -> float:
        """从 backtest_results dict 计算目标感知 reward。

        口径与 StrategyTrajectory.get_reward_score() 完全一致：净 Sharpe、
        abs(IC)、abs(Rank IC) 是主目标；ICIR、回撤、换手、复杂度和冗余为
        次级质量项。负 IC/Rank IC 仍按绝对值计分，因为信号可翻转方向使用。
        """
        return FactorLibraryManager._reward_trajectory(metrics, factor_entry).get_reward_score()

    @staticmethod
    def compute_reward_breakdown(
        metrics: dict[str, Any],
        factor_entry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return reward score internals for UI/debug display."""
        trajectory = FactorLibraryManager._reward_trajectory(metrics, factor_entry)
        components = trajectory.get_reward_components()
        weights = dict(StrategyTrajectory.REWARD_WEIGHTS)

        def _finite(value: object) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return 0.0
            if parsed != parsed or parsed in {float("inf"), float("-inf")}:
                return 0.0
            return parsed

        terms: list[dict[str, Any]] = []
        for name, weight in weights.items():
            value = _finite(components.get(name, 0.0))
            terms.append(
                {
                    "name": name,
                    "value": value,
                    "weight": float(weight),
                    "contribution": float(weight) * value,
                }
            )
        return {
            "score": sum(float(term["contribution"]) for term in terms),
            "terms": terms,
            "components": {key: _finite(value) for key, value in components.items()},
            "weights": weights,
            "target_status": trajectory.get_target_status(),
        }

    @staticmethod
    def compute_target_status(metrics: dict[str, Any]) -> dict[str, Any]:
        """返回好因子硬目标的通过状态和缺口。"""
        return dict(FactorLibraryManager._reward_trajectory(metrics, None).get_target_status())

    @staticmethod
    def _reward_trajectory(
        metrics: dict[str, Any],
        factor_entry: dict[str, Any] | None,
    ) -> StrategyTrajectory:
        factors: list[dict[str, Any]] = []
        if isinstance(factor_entry, dict):
            factor = dict(factor_entry)
            expression = str(
                factor.get("factor_expression", factor.get("expression", ""))
            ).strip()
            if expression and "symbol_length" not in factor and "expression_length" not in factor:
                factor["symbol_length"] = float(len(expression))
            factors.append(factor)
        return StrategyTrajectory(
            trajectory_id="_reward_probe",
            direction_id=0,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
            factors=factors,
            backtest_metrics=dict(metrics),
        )

    def list_train_metric_candidates(
        self,
        *,
        metric: str = "sharpe",
        threshold: float = 2.0,
        experiment_ids: set[str] | None = None,
        run_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        metric_name = str(metric).strip() or "sharpe"
        try:
            min_value = float(threshold)
        except Exception:
            min_value = 2.0
        filter_by_experiment = experiment_ids is not None
        allowed_experiment_ids = {str(item) for item in experiment_ids or set() if str(item).strip()}
        filter_by_run = run_ids is not None
        allowed_run_ids = {str(item) for item in run_ids or set() if str(item).strip()}
        candidates: list[dict[str, Any]] = []
        for factor in self.data.get("factors", {}).values():
            if not isinstance(factor, dict):
                continue
            metadata = factor.get("metadata", {})
            experiment_id = str(metadata.get("experiment_id", "")) if isinstance(metadata, dict) else ""
            run_id = str(metadata.get("run_id", "")) if isinstance(metadata, dict) else ""
            if filter_by_experiment and experiment_id not in allowed_experiment_ids:
                continue
            if filter_by_run and run_id not in allowed_run_ids:
                continue
            train_metrics = dict(factor.get("backtest_results", {})) if isinstance(factor.get("backtest_results", {}), dict) else {}
            try:
                value = float(train_metrics.get(metric_name))
            except Exception:
                continue
            if value <= min_value:
                continue
            candidates.append(
                {
                    "factor_id": str(factor.get("factor_id", "")),
                    "factor_name": str(factor.get("factor_name", "")),
                    "factor_expression": str(factor.get("factor_expression", "")),
                    "train_metrics": train_metrics,
                    "reward_score": self.compute_reward_score(train_metrics, factor_entry=factor),
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                }
            )
        return sorted(candidates, key=lambda item: float(item["reward_score"]), reverse=True)

    def add_from_shared_payload(self, payload: dict[str, Any]) -> None:
        experiment = QlibFactorExperiment.from_shared_payload(payload)
        if not experiment.factors:
            return
        if isinstance(payload.get("hypothesis_structured"), dict):
            experiment.extra["hypothesis_structured"] = dict(payload.get("hypothesis_structured", {}))
        panel_data_path = str(payload.get("domain_dataset_summary", {}).get("path", "")).strip() if isinstance(payload.get("domain_dataset_summary", {}), dict) else ""
        self.add_experiment(experiment, panel_data_path=panel_data_path)

    @classmethod
    def check_cache_status(cls, library_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, Any]:
        data = json.loads(Path(library_path).read_text(encoding="utf-8"))
        factors = data.get("factors", {})
        cache_root = Path(cache_dir) if cache_dir else cls.default_cache_dir()
        total = len(factors)
        md5_cached = 0
        need_compute = 0
        details: list[dict[str, Any]] = []
        for factor_id, factor_info in factors.items():
            expression = str(factor_info.get("factor_expression", ""))
            md5_key = hashlib.md5(expression.encode()).hexdigest() if expression else ""
            pkl_path = cache_root / f"{md5_key}.pkl" if md5_key else Path("")
            if md5_key and pkl_path.exists():
                md5_cached += 1
                status = "md5_cached"
            else:
                need_compute += 1
                status = "need_compute"
            details.append(
                {
                    "factor_id": factor_id,
                    "factor_name": str(factor_info.get("factor_name", factor_id)),
                    "status": status,
                }
            )
        return {
            "total": total,
            "md5_cached": md5_cached,
            "need_compute": need_compute,
            "factors": details,
        }

    @classmethod
    def warm_cache_from_json(cls, library_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, int]:
        data = json.loads(Path(library_path).read_text(encoding="utf-8"))
        factors = data.get("factors", {})
        cache_root = Path(cache_dir) if cache_dir else cls.default_cache_dir()
        cache_root.mkdir(parents=True, exist_ok=True)
        synced = 0
        skipped = 0
        failed = 0
        for _factor_id, factor_info in factors.items():
            expression = str(factor_info.get("factor_expression", ""))
            source_pickle = str(factor_info.get("cache_location", {}).get("source_pickle_path", "")).strip()
            if not expression or not source_pickle:
                skipped += 1
                continue
            src = Path(source_pickle)
            if not src.exists():
                failed += 1
                continue
            md5_key = hashlib.md5(expression.encode()).hexdigest()
            target = cache_root / f"{md5_key}.pkl"
            if target.exists():
                skipped += 1
                continue
            try:
                shutil.copy2(src, target)
                synced += 1
            except Exception:
                failed += 1
        return {
            "total": len(factors),
            "synced": synced,
            "skipped": skipped,
            "failed": failed,
        }
