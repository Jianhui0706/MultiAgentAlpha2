from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_agents.agents.backtest_runner_agent import QlibFactorBacktestEngine
from trading_agents.factor_runtime import calculate_factor_expression
from trading_agents.infra.factor_library_manager import FactorLibraryManager


@dataclass(frozen=True)
class FactorPoolConfig:
    enabled: bool = True
    corr_threshold: float = 0.7
    max_fraction: float = 0.5
    metric: str = "Rank IC"


class FactorPoolManager:
    """Maintain a QuantaAlpha-style global high-quality factor pool."""

    def __init__(
        self,
        *,
        library_path: str | Path,
        panel_data_path: str | Path,
        output_dir: str | Path,
        config: FactorPoolConfig | None = None,
    ) -> None:
        self.library_path = Path(library_path)
        self.panel_data_path = Path(panel_data_path)
        self.output_dir = Path(output_dir)
        self.config = config or FactorPoolConfig()
        self.pool_path = self.output_dir / "factor_pool.json"
        self.combined_factors_path = self.output_dir / "combined_factors_df.parquet"

    def build(self, *, run_final_combined_backtest: bool = False) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled"}
        if not self.library_path.exists():
            return {"status": "missing_factor_library", "factor_pool_path": str(self.pool_path)}
        if not self.panel_data_path.exists():
            return {"status": "missing_panel_data", "factor_pool_path": str(self.pool_path)}

        manager = FactorLibraryManager(self.library_path)
        candidates = self._load_candidates(manager)
        if not candidates:
            payload = self._write_payload(
                status="no_candidates",
                candidates=[],
                selected=[],
                rejected=[],
                combined_backtest={},
            )
            return payload

        engine = self._load_engine()
        prepared = engine._prepare_panel(str(self.panel_data_path))
        if prepared is None:
            return {"status": "invalid_panel_or_target_column", "factor_pool_path": str(self.pool_path)}
        full_panel, target_column = prepared
        validation_panel = self._validation_panel(engine, full_panel)
        label_series = pd.to_numeric(validation_panel[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan).rename("label")
        max_size = max(1, int(len(candidates) * max(0.0, min(1.0, float(self.config.max_fraction)))))

        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        selected_signals: dict[str, pd.Series] = {}
        for candidate in candidates:
            if len(selected) >= max_size:
                rejected.append({**candidate, "rejection_reason": "pool_capacity_reached"})
                continue
            expression = str(candidate.get("factor_expression", "")).strip()
            factor_name = str(candidate.get("factor_name", candidate.get("factor_id", "factor"))).strip()
            try:
                signal = calculate_factor_expression(expression, validation_panel, factor_name)
                if signal is None or signal.dropna().empty:
                    raise ValueError("empty signal")
            except Exception as exc:
                rejected.append({**candidate, "rejection_reason": f"signal_failed:{exc}"})
                continue
            max_abs_corr = self._max_abs_corr(signal, selected_signals.values())
            if max_abs_corr >= float(self.config.corr_threshold):
                rejected.append(
                    {
                        **candidate,
                        "rejection_reason": "correlation_threshold",
                        "max_abs_corr": round(float(max_abs_corr), 6),
                    }
                )
                continue
            selected_entry = {
                **candidate,
                "pool_rank": len(selected) + 1,
                "max_abs_corr": round(float(max_abs_corr), 6),
            }
            selected.append(selected_entry)
            selected_signals[self._signal_key(candidate)] = signal

        combined_df = self._write_combined_factors(selected, selected_signals)
        combined_backtest = (
            self._evaluate_combined_signal(engine, validation_panel, label_series, selected_signals)
            if run_final_combined_backtest and selected_signals
            else {"status": "skipped", "reason": "run_final_combined_backtest disabled or empty pool"}
        )
        payload = self._write_payload(
            status="finished" if selected else "empty_after_filter",
            candidates=candidates,
            selected=selected,
            rejected=rejected,
            combined_backtest=combined_backtest,
        )
        payload["combined_factors_path"] = str(combined_df) if combined_df else ""
        return payload

    def _load_candidates(self, manager: FactorLibraryManager) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for factor in manager.data.get("factors", {}).values():
            if not isinstance(factor, dict):
                continue
            metrics = dict(factor.get("backtest_results", {})) if isinstance(factor.get("backtest_results", {}), dict) else {}
            train_decision = dict(factor.get("train_filter_decision", {})) if isinstance(factor.get("train_filter_decision", {}), dict) else {}
            if train_decision and not bool(train_decision.get("passed", True)):
                continue
            # 保留有主指标的因子（保证有回测结果）
            metric_value = self._metric_value(metrics, self.config.metric)
            if metric_value is None:
                metric_value = self._metric_value(metrics, "RankIC")
            if metric_value is None:
                continue
            reward_score = FactorLibraryManager.compute_reward_score(metrics, factor_entry=factor)
            candidates.append(
                {
                    "factor_id": str(factor.get("factor_id", "")),
                    "factor_name": str(factor.get("factor_name", "")),
                    "factor_expression": str(factor.get("factor_expression", "")),
                    "metric": self.config.metric,
                    "metric_value": float(metric_value),
                    "abs_metric_value": abs(float(metric_value)),
                    "reward_score": reward_score,
                    "backtest_results": metrics,
                    "metadata": dict(factor.get("metadata", {})) if isinstance(factor.get("metadata", {}), dict) else {},
                }
            )
        return sorted(candidates, key=lambda item: float(item["reward_score"]), reverse=True)

    @staticmethod
    def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
        try:
            value = float(metrics.get(key))
        except Exception:
            return None
        return value if np.isfinite(value) else None

    @staticmethod
    def _max_abs_corr(signal: pd.Series, existing: Any) -> float:
        max_corr = 0.0
        for other in existing:
            frame = pd.concat([signal.rename("candidate"), other.rename("selected")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(frame) < 3 or frame["candidate"].nunique() < 2 or frame["selected"].nunique() < 2:
                continue
            corr = frame["candidate"].corr(frame["selected"])
            if corr is not None and np.isfinite(corr):
                max_corr = max(max_corr, abs(float(corr)))
        return max_corr

    def _write_combined_factors(self, selected: list[dict[str, Any]], signals: dict[str, pd.Series]) -> Path | None:
        if not selected or not signals:
            return None
        frame = pd.concat(
            [
                signals[self._signal_key(item)].rename(str(item.get("factor_name", item.get("factor_id", ""))))
                for item in selected
                if self._signal_key(item) in signals
            ],
            axis=1,
        )
        self.combined_factors_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.combined_factors_path)
        return self.combined_factors_path

    def _evaluate_combined_signal(
        self,
        engine: QlibFactorBacktestEngine,
        panel: pd.DataFrame,
        label_series: pd.Series,
        signals: dict[str, pd.Series],
    ) -> dict[str, Any]:
        try:
            signal_frame = pd.concat([signal.rename(name) for name, signal in signals.items()], axis=1)
            # 论文最终因子池采用 LGBModel + TopkDropout 组合多因子。优先尝试 LightGBM 学习式组合，
            # 不可用时回退到 mean-rank 代理（生成的 parquet 仍可供外部 Qlib 使用）。
            lgb_result = self._lgb_combined_signal(signal_frame, label_series)
            if lgb_result is not None:
                combined_signal, model_label = lgb_result
            else:
                combined_signal = signal_frame.rank(pct=True).mean(axis=1)
                model_label = "factor_pool_mean_rank_proxy"
            combined_signal = combined_signal.rename("combined_factor_pool_signal")
            metrics = engine._calculate_factor_metrics_bundle(
                factor_name="combined_factor_pool_signal",
                expression=model_label,
                signal_series=combined_signal,
                label_series=label_series,
                panel=panel,
            )
            return {
                "status": "finished",
                "model": model_label,
                "qlib_combined_config": str(Path(__file__).resolve().parent.parent / "configs" / "qlib" / "conf_combined_factors.yaml"),
                "metrics": engine._round_metrics_dict(metrics),
                "note": (
                    "LightGBM model-based combination (Qlib LGBModel-equivalent)."
                    if model_label == "factor_pool_lgbm"
                    else "LightGBM unavailable; mean-rank proxy used. Generated parquet is ready for Qlib LGBModel + TopkDropout."
                ),
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

    @staticmethod
    def _lgb_combined_signal(
        signal_frame: pd.DataFrame,
        label_series: pd.Series,
    ) -> tuple[pd.Series, str] | None:
        """用 LightGBM 学习多因子到未来收益的映射，输出组合打分（对齐论文 LGBModel）。

        LightGBM 不可用、样本不足或训练失败时返回 None，由调用方回退到 mean-rank 代理。
        """
        try:
            import lightgbm as lgb  # type: ignore
        except Exception:
            return None
        try:
            frame = signal_frame.copy()
            label = pd.to_numeric(label_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            joined = frame.join(label.rename("__label__"), how="inner").replace([np.inf, -np.inf], np.nan).dropna()
            if len(joined) < 200 or joined["__label__"].nunique() < 5:
                return None
            feature_columns = list(signal_frame.columns)
            x = joined[feature_columns].astype(float)
            y = joined["__label__"].astype(float)
            # 按时间顺序切分训练/验证，避免前视偏差（前 70% 训练）
            split = int(len(joined) * 0.7)
            if split < 50 or (len(joined) - split) < 20:
                return None
            train_x, train_y = x.iloc[:split], y.iloc[:split]
            model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
            )
            model.fit(train_x, train_y)
            predictions = pd.Series(model.predict(x.values), index=joined.index, name="combined")
            # 还原到完整信号索引（缺失处用截面排名均值兜底）
            combined = pd.Series(np.nan, index=signal_frame.index)
            combined.loc[predictions.index] = predictions.values
            fallback = signal_frame.rank(pct=True).mean(axis=1)
            combined = combined.fillna(fallback)
            return combined, "factor_pool_lgbm"
        except Exception:
            return None

    @staticmethod
    def _validation_panel(engine: QlibFactorBacktestEngine, panel: pd.DataFrame) -> pd.DataFrame:
        if engine.sample_windows_enabled:
            engine._ensure_sample_windows_resolved(panel)
            mask = engine._build_sample_window_mask(panel, engine.validation_window)
            window = panel.iloc[np.flatnonzero(mask)].copy()
            if not window.empty:
                return window
        return panel

    @staticmethod
    def _load_engine() -> QlibFactorBacktestEngine:
        config_path = FactorLibraryManager.project_root() / "src" / "trading_agents" / "configs" / "qlib" / "backtest_engine.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return QlibFactorBacktestEngine(config, config_path)

    @staticmethod
    def _signal_key(candidate: dict[str, Any]) -> str:
        factor_id = str(candidate.get("factor_id", "")).strip()
        if factor_id:
            return factor_id
        return str(candidate.get("factor_name", "factor")).strip() or "factor"

    def _write_payload(
        self,
        *,
        status: str,
        candidates: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        combined_backtest: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "created_at": datetime.now().isoformat(),
            "library_path": FactorLibraryManager.to_portable_path(self.library_path),
            "panel_data_path": FactorLibraryManager.to_portable_path(self.panel_data_path),
            "factor_pool_path": str(self.pool_path),
            "combined_factors_path": str(self.combined_factors_path),
            "config": {
                "corr_threshold": float(self.config.corr_threshold),
                "max_fraction": float(self.config.max_fraction),
                "metric": self.config.metric,
            },
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "rejected_count": len(rejected),
            "selected_factors": selected,
            "rejected_factors": rejected,
            "combined_backtest": combined_backtest,
        }
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return payload
