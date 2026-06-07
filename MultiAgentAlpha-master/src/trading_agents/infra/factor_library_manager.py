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
from trading_agents.schemas import QlibFactorExperiment


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
        train_filter_decisions = train_filter_report.get("decisions", {}) if isinstance(train_filter_report, dict) else {}
        redundancy_filter_report = experiment.backtest_report.get("redundancy_filter", {})
        redundancy_filter_enabled = (
            bool(redundancy_filter_report.get("enabled", False))
            if isinstance(redundancy_filter_report, dict)
            else False
        )
        redundancy_filter_decisions = (
            redundancy_filter_report.get("decisions", {}) if isinstance(redundancy_filter_report, dict) else {}
        )
        saved_factor_count = 0
        for task in factors:
            if train_filter_enabled and task.factor_name not in retained_factor_names:
                continue
            # 构造阶段冗余门：与已保留因子高度相关者不入库
            if redundancy_filter_enabled and isinstance(redundancy_filter_decisions, dict):
                redundancy_decision = redundancy_filter_decisions.get(task.factor_name, {})
                if isinstance(redundancy_decision, dict) and redundancy_decision and not bool(
                    redundancy_decision.get("passed", True)
                ):
                    continue
            implementation = self._find_implementation(experiment, task.factor_name)
            factor_expression = str(implementation.get("expression", "")).strip() or task.expression
            factor_code = str(implementation.get("code", "")).strip()
            factor_id = self._factor_id(task.factor_name, factor_expression)
            factor_metrics = dict(experiment.per_factor_metrics.get(task.factor_name, {}))
            if train_filter_enabled and not factor_metrics:
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
                "metadata": {
                    "experiment_id": experiment.experiment_id,
                    "run_id": run_id,
                    "round_number": experiment.round_idx,
                    "evolution_phase": experiment.phase,
                    "trajectory_id": experiment.trajectory_id,
                    "parent_trajectory_ids": list(experiment.parent_ids),
                    "hypothesis": experiment.target_hypothesis,
                    "created_at": datetime.now().isoformat(),
                },
                "backtest_results": factor_metrics or dict(experiment.metrics),
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
                "train_filter_decision": dict(train_filter_decisions.get(task.factor_name, {}))
                if isinstance(train_filter_decisions, dict) and isinstance(train_filter_decisions.get(task.factor_name, {}), dict)
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
    def compute_reward_score(
        metrics: dict[str, Any],
        factor_entry: dict[str, Any] | None = None,
    ) -> float:
        """从 backtest_results dict 计算与 StrategyTrajectory.get_reward_score() 等价的分数。

        各指标归一化后加权求和，缺失指标按 0 处理。
        权重与 StrategyTrajectory.REWARD_WEIGHTS 保持一致。

        修改记录：
        - Fix1: 去掉 abs()，负 IC/ICIR 贡献为 0（clip 到 0），只有正向预测能力才加分
        - Fix2: norm_sharpe 下界从 0 改为 -1，允许负 Sharpe 产生惩罚
        - Fix3: 接受可选 factor_entry，从 factor_expression 长度补算 complexity
        """
        def _m(*aliases: str) -> float:
            for alias in aliases:
                v = metrics.get(alias)
                try:
                    import math
                    f = float(v) if v is not None else 0.0
                    return f if math.isfinite(f) else 0.0
                except Exception:
                    pass
            return 0.0

        def _clip(v: float, lo: float, hi: float) -> float:
            import math
            return max(lo, min(hi, v)) if math.isfinite(v) else lo

        # Fix1: 去掉 abs()，负 IC → clip 到 0，不再虚假加分
        norm_rank_ic   = _clip(_m("Rank IC", "RankIC") / 0.05, 0.0, 1.0)
        norm_rank_icir = _clip(_m("Rank ICIR", "RankICIR", "ICIR") / 0.5, 0.0, 1.0)
        # Fix2: 下界 -1.0，负 Sharpe 会真实扣分
        norm_sharpe    = _clip(_m("sharpe", "information_ratio") / 3.0, -1.0, 1.0)
        norm_coverage  = _clip(_m("coverage"), 0.0, 1.0)
        norm_drawdown  = _clip(abs(_m("drawdown", "max_drawdown")) / 0.30, 0.0, 1.0)
        norm_turnover  = _clip(_m("turnover") / 1.0, 0.0, 1.0)
        # Fix3: metrics 里没有 complexity 字段时，从 factor_expression 长度兜底
        symbol_length  = _m("symbol_length", "expression_length")
        base_features  = _m("base_feature_count", "input_feature_count")
        free_args      = _m("free_args_ratio")
        if symbol_length <= 0.0 and base_features <= 0.0 and free_args <= 0.0 and factor_entry is not None:
            expression = str(factor_entry.get("factor_expression", "")).strip()
            symbol_length = float(len(expression)) if expression else 0.0
        if symbol_length > 0.0 or base_features > 0.0 or free_args > 0.0:
            norm_complexity = (
                0.4 * _clip(symbol_length / 250.0, 0.0, 1.0)
                + 0.3 * _clip(free_args / 0.5, 0.0, 1.0)
                + 0.3 * _clip(base_features / 6.0, 0.0, 1.0)
            )
        else:
            norm_complexity = 0.0
        norm_redundancy = _clip(_m("structural_similarity", "structural_redundancy", "redundancy_score"), 0.0, 1.0)
        norm_corr       = _clip(_m("max_abs_corr", "max_abs_correlation", "correlation_penalty", "max_corr"), 0.0, 1.0)

        return (
            +0.35 * norm_rank_ic
            + 0.15 * norm_rank_icir
            + 0.10 * norm_sharpe
            + 0.10 * norm_coverage
            - 0.10 * norm_drawdown
            - 0.05 * norm_turnover
            - 0.08 * norm_complexity
            - 0.06 * norm_redundancy
            - 0.06 * norm_corr
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
