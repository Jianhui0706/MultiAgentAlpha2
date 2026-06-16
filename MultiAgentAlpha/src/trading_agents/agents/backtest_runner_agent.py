from __future__ import annotations

import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.factor_runtime import build_matrix_panel, calculate_factor_expression, downcast_float_columns
from trading_agents.schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


BACKTEST_DEBUG = os.getenv("BACKTEST_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

# Thread-local storage: runner.py sets target_column + annualization_factor per run thread
# so concurrent runs with different prediction horizons don't interfere with each other.
_BACKTEST_TLS = threading.local()


def _resolve_factor_eval_concurrency() -> int:
    """全进程内可同时进行的"重型因子矩阵评估"上限。

    进化层并行(N 个任务) × 回测层并行(M 个因子线程) 会叠乘出 N×M 个并发，
    每个都在构建 (T,N) 稠密矩阵 → 峰值内存随嵌套深度爆炸。用一个进程级信号量
    把同时计算的因子数钳制在 ~CPU 核数，无论嵌套多深内存都有上界。
    可用环境变量 MAX_FACTOR_EVAL_CONCURRENCY 覆盖。
    """
    raw = os.getenv("MAX_FACTOR_EVAL_CONCURRENCY", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


# 进程级信号量：钳制并发的重型因子评估数，防止嵌套并行把 (T,N) 矩阵内存叠乘。
_FACTOR_EVAL_SEMAPHORE = threading.BoundedSemaphore(_resolve_factor_eval_concurrency())


def _install_pandas_frequency_compat() -> None:
    """Keep legacy month-end aliases working on newer pandas versions."""
    original = getattr(pd, "_multiagent_original_date_range", None)
    if original is not None:
        return
    pd._multiagent_original_date_range = pd.date_range  # type: ignore[attr-defined]

    def _date_range_compat(*args: object, **kwargs: object) -> pd.DatetimeIndex:
        if kwargs.get("freq") == "M":
            kwargs["freq"] = "ME"
        return pd._multiagent_original_date_range(*args, **kwargs)  # type: ignore[attr-defined]

    pd.date_range = _date_range_compat  # type: ignore[assignment]


_install_pandas_frequency_compat()


def set_thread_backtest_config(*, target_column: str, annualization_factor: float | None = None) -> None:
    """Called by runner.py inside the run thread before the workflow starts."""
    _BACKTEST_TLS.target_column = target_column
    if annualization_factor is not None:
        _BACKTEST_TLS.annualization_factor = annualization_factor


def _debug_print(message: str) -> None:
    if BACKTEST_DEBUG:
        print(message)


class QlibFactorBacktestEngine:
    """因子回测引擎。

    职责：
    1. 加载面板数据（parquet 格式的多资产时间序列）
    2. 对每个 generated 状态的因子，计算其信号值并评估回测指标
    3. 计算截面 IC/ICIR/RankIC/RankICIR/Coverage/Turnover
    4. 执行分层回测（5 层），计算多空收益、Sharpe、最大回撤、单调性

    注意：本引擎不调用 LLM，纯数值计算。
    """

    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self.config_path = config_path
        self.project_root = self._resolve_project_root(config_path)
        # ── 解析器配置（表达式别名与黑名单）──
        resolver = config.get("resolver", {})
        dataset = config.get("dataset", {})
        metrics = config.get("metrics", {})
        engine = config.get("engine", {})
        self.alias_map = dict(resolver.get("alias_map", {})) if isinstance(resolver, dict) else {}
        blacklist = resolver.get("function_token_blacklist", []) if isinstance(resolver, dict) else []
        self.function_token_blacklist = {str(item).strip().lower() for item in blacklist if str(item).strip()}
        # ── 数据集配置 ──
        # TLS wins over yaml so runner.py can set the user-selected prediction horizon
        # per run thread without touching the shared yaml file.
        self.target_column = (
            getattr(_BACKTEST_TLS, "target_column", None)
            or (str(dataset.get("target_column", "fwd_returns_4h")) if isinstance(dataset, dict) else "fwd_returns_4h")
        )
        index_levels = dataset.get("required_index_levels", ["datetime", "symbol"]) if isinstance(dataset, dict) else ["datetime", "symbol"]
        self.required_index_levels = {str(item).strip() for item in index_levels if str(item).strip()}
        exclude_prefixes = dataset.get("signal_exclude_prefixes", ["fwd_"]) if isinstance(dataset, dict) else ["fwd_"]
        self.signal_exclude_prefixes = tuple(str(item) for item in exclude_prefixes)
        # ── 指标计算配置 ──
        self.annualization_factor = (
            getattr(_BACKTEST_TLS, "annualization_factor", None)
            or (float(metrics.get("annualization_factor", 2190)) if isinstance(metrics, dict) else 2190.0)
        )
        # Legacy config field. Current metric semantics are fixed:
        # IC = Pearson cross-sectional correlation; Rank IC = rank/Spearman-style correlation.
        self.corr_method = str(metrics.get("corr_method", "pearson")) if isinstance(metrics, dict) else "pearson"
        transaction_cost = metrics.get("transaction_cost", {}) if isinstance(metrics, dict) else {}
        self.one_way_fee_rate = (
            max(0.0, float(transaction_cost.get("one_way_fee_rate", 0.0)))
            if isinstance(transaction_cost, dict)
            else 0.0
        )
        self.time_per_step = str(dataset.get("time_per_step", "60min")) if isinstance(dataset, dict) else "60min"
        self.portfolio_key = str(dataset.get("portfolio_key", "60min")) if isinstance(dataset, dict) else "60min"
        sample_windows = config.get("sample_windows", {})
        robustness_filter = config.get("robustness_filter", {})
        train_filter = sample_windows.get("train_filter", {}) if isinstance(sample_windows, dict) else {}
        if not train_filter and isinstance(sample_windows, dict):
            train_filter = sample_windows.get("filter", {})
        coverage_filter = sample_windows.get("coverage_filter", {}) if isinstance(sample_windows, dict) else {}
        self.sample_windows_enabled = bool(sample_windows.get("enabled", False)) if isinstance(sample_windows, dict) else False
        self.sample_window_split_mode = (
            str(sample_windows.get("split_mode", "")).strip().lower()
            if isinstance(sample_windows, dict)
            else ""
        )
        self.sample_window_train_ratio = self._finite_float(
            sample_windows.get("train_ratio", 0.75) if isinstance(sample_windows, dict) else 0.75,
            0.75,
        )
        self.train_window = self._normalize_sample_window(
            sample_windows.get("train", sample_windows.get("in_sample", {})) if isinstance(sample_windows, dict) else {}
        )
        self.validation_window = self._normalize_sample_window(
            sample_windows.get(
                "validation",
                sample_windows.get("test", sample_windows.get("out_of_sample", {})),
            ) if isinstance(sample_windows, dict) else {}
        )
        self._sample_window_resolution: dict[str, object] = {"mode": "explicit"}
        # 回测区间：实际参与回测的面板数据时间范围（start/end），用于给入库因子打标记。
        self._backtest_period: dict[str, str | None] = {"start": None, "end": None}
        self.train_filter_metric = (
            str(train_filter.get("metric", "sharpe")).strip() if isinstance(train_filter, dict) else "sharpe"
        ) or "sharpe"
        # 默认净 Sharpe 硬目标 1.5。新建运行可通过
        # TRAIN_SHARPE_THRESHOLD 覆盖；未设置时回退到此默认。
        config_train_filter_min_value = (
            self._finite_float(train_filter.get("min_value", 1.5), 1.5)
            if isinstance(train_filter, dict)
            else 1.5
        )
        threshold_override = os.getenv("TRAIN_SHARPE_THRESHOLD", "").strip()
        self.train_filter_min_value = (
            self._finite_float(threshold_override, config_train_filter_min_value)
            if threshold_override
            else config_train_filter_min_value
        )
        self.coverage_filter_enabled = (
            bool(coverage_filter.get("enabled", True)) if isinstance(coverage_filter, dict) else True
        )
        self.coverage_filter_metric = (
            str(coverage_filter.get("metric", "coverage")).strip() if isinstance(coverage_filter, dict) else "coverage"
        ) or "coverage"
        config_coverage_filter_min_value = (
            self._finite_float(coverage_filter.get("min_value", 0.9), 0.9)
            if isinstance(coverage_filter, dict)
            else 0.9
        )
        coverage_threshold_override = os.getenv("TRAIN_COVERAGE_THRESHOLD", "").strip()
        self.coverage_filter_min_value = (
            self._finite_float(coverage_threshold_override, config_coverage_filter_min_value)
            if coverage_threshold_override
            else config_coverage_filter_min_value
        )
        # ── 留样验证衰减门（validation decay gate）──────────────────────────────
        # 入库门槛主要基于训练集表现（train_filter），但若因子在 validation 上的
        # 综合 reward 相对 train 衰减过大（或正负翻转），说明过拟合，不应入库。
        # 衡量指标用与因子质量一致的 composite reward score。
        validation_decay = sample_windows.get("validation_decay", {}) if isinstance(sample_windows, dict) else {}
        self.validation_decay_enabled = (
            bool(validation_decay.get("enabled", True)) if isinstance(validation_decay, dict) else True
        )
        config_decay_min_retention = (
            self._finite_float(validation_decay.get("min_retention", 0.5), 0.5)
            if isinstance(validation_decay, dict)
            else 0.5
        )
        decay_retention_override = os.getenv("VALIDATION_DECAY_MIN_RETENTION", "").strip()
        self.validation_decay_min_retention = (
            self._finite_float(decay_retention_override, config_decay_min_retention)
            if decay_retention_override
            else config_decay_min_retention
        )
        self.robustness_filter_enabled = (
            bool(robustness_filter.get("enabled", False)) if isinstance(robustness_filter, dict) else False
        )
        self.robustness_market_beta_abs_max = (
            self._finite_float(robustness_filter.get("market_beta_abs_max", 0.20), 0.20)
            if isinstance(robustness_filter, dict) else 0.20
        )
        self.robustness_residual_sharpe_min = (
            self._finite_float(robustness_filter.get("residual_sharpe_min", 0.50), 0.50)
            if isinstance(robustness_filter, dict) else 0.50
        )
        self.robustness_yearly_positive_ratio_min = (
            self._finite_float(robustness_filter.get("yearly_positive_ratio_min", 0.50), 0.50)
            if isinstance(robustness_filter, dict) else 0.50
        )
        self.robustness_max_year_positive_contribution_share = (
            self._finite_float(robustness_filter.get("max_year_positive_contribution_share", 0.65), 0.65)
            if isinstance(robustness_filter, dict) else 0.65
        )
        self.robustness_regime_positive_sharpe_count_min = int(
            robustness_filter.get("regime_positive_sharpe_count_min", 2)
            if isinstance(robustness_filter, dict) else 2
        )
        # ── 冗余门（构造阶段相关性过滤，对齐论文 Component B 的 redundancy gate）──
        # 在因子评估阶段即按信号相关性剔除与已保留因子高度冗余的新因子，
        # 而非仅在最终因子池阶段过滤。
        redundancy_filter = config.get("redundancy_filter", {})
        self.redundancy_filter_enabled = (
            bool(redundancy_filter.get("enabled", True)) if isinstance(redundancy_filter, dict) else True
        )
        self.redundancy_corr_threshold = (
            self._finite_float(redundancy_filter.get("corr_threshold", 0.7), 0.7)
            if isinstance(redundancy_filter, dict) else 0.7
        )
        # ── Qlib 初始化配置 ──
        config_dir_raw = str(engine.get("qlib_config_dir", "../configs/qlib")) if isinstance(engine, dict) else "../configs/qlib"
        config_dir_path = Path(config_dir_raw)
        if not config_dir_path.is_absolute():
            config_dir_path = (config_path.parent / config_dir_path).resolve()
        self.qlib_config_dir = config_dir_path
        self.baseline_config_name = str(engine.get("baseline_config_name", "conf_baseline.yaml")) if isinstance(engine, dict) else "conf_baseline.yaml"
        self.combined_config_name = str(engine.get("combined_config_name", "conf_combined_factors.yaml")) if isinstance(engine, dict) else "conf_combined_factors.yaml"
        self.last_error = ""
        self._qlib_initialized = False
        self._current_sharpe_sampling_step = 1
        self._current_sharpe_annualization_factor = self.annualization_factor
        self._current_panel_step = ""
        self._current_target_horizon: pd.Timedelta | None = None
        self.metric_round_digits = 6
        self.high_precision_metric_round_digits = 12
        self.high_precision_metric_keys = {
            "sharpe",
            "sharpe_abs",
            "raw_sharpe",
            "gross_sharpe",
            "gross_raw_sharpe",
            "net_sharpe",
            "net_raw_sharpe",
            "long_short_mean_return",
            "raw_long_short_mean_return",
            "gross_long_short_mean_return",
            "gross_raw_long_short_mean_return",
            "transaction_cost_mean",
            "raw_transaction_cost_mean",
            "long_short_turnover",
            "market_beta",
            "residual_sharpe",
            "market_neutral_mean_return",
            "yearly_positive_ratio",
            "max_year_positive_contribution_share",
            "long_short_positive_ratio",
            "raw_long_short_positive_ratio",
            "drawdown",
            "raw_drawdown",
            "layer_1_mean_return",
            "layer_2_mean_return",
            "layer_3_mean_return",
            "layer_4_mean_return",
            "layer_5_mean_return",
        }

    def _finite_float(self, value: object, default: float = 0.0) -> float:
        """安全地将值转为有限浮点数，NaN/Inf 返回默认值。"""
        try:
            number = float(value)
        except Exception:
            return default
        return number if np.isfinite(number) else default

    def _finite_metrics(self, metrics: dict[str, object]) -> dict[str, float]:
        """批量将指标字典中的值转为有限浮点数。"""
        return {str(key): self._finite_float(value) for key, value in metrics.items()}

    def _set_error(self, message: str) -> None:
        """记录最近的错误信息。"""
        self.last_error = message

    def _round_metric_value(self, key: str, value: object) -> float:
        digits = self.high_precision_metric_round_digits if key in self.high_precision_metric_keys else self.metric_round_digits
        return round(self._finite_float(value), digits)

    def _round_metrics_dict(self, metrics: dict[str, object]) -> dict[str, float]:
        return {str(key): self._round_metric_value(str(key), value) for key, value in metrics.items()}

    @staticmethod
    def _coerce_optional_timestamp(value: object) -> pd.Timestamp | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        timestamp = pd.Timestamp(text)
        return None if pd.isna(timestamp) else timestamp

    def _normalize_sample_window(self, raw_window: object) -> dict[str, pd.Timestamp | None]:
        window = raw_window if isinstance(raw_window, dict) else {}
        return {
            "start": self._coerce_optional_timestamp(window.get("start")),
            "end": self._coerce_optional_timestamp(window.get("end")),
        }

    @staticmethod
    def _format_sample_window(window: dict[str, pd.Timestamp | None]) -> dict[str, str | None]:
        return {
            "start": window.get("start").isoformat() if isinstance(window.get("start"), pd.Timestamp) else None,
            "end": window.get("end").isoformat() if isinstance(window.get("end"), pd.Timestamp) else None,
        }

    def _build_sample_window_mask(
        self,
        panel: pd.DataFrame,
        window: dict[str, pd.Timestamp | None],
    ) -> np.ndarray:
        if "datetime" not in panel.index.names:
            return np.zeros(len(panel), dtype=bool)
        datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
        mask = np.ones(len(panel), dtype=bool)
        start = window.get("start")
        end = window.get("end")
        if isinstance(start, pd.Timestamp):
            mask &= datetimes >= start
        if isinstance(end, pd.Timestamp):
            mask &= datetimes < end
        mask &= ~pd.isna(datetimes)
        return np.asarray(mask, dtype=bool)

    def _resolve_ratio_sample_windows(self, panel: pd.DataFrame) -> bool:
        """按时间顺序把完整面板切成前 train_ratio 与后续验证窗口。"""
        if "datetime" not in panel.index.names:
            return False
        datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
        unique_datetimes = pd.Index(datetimes[~pd.isna(datetimes)]).unique().sort_values()
        if len(unique_datetimes) < 2:
            return False
        ratio = self._finite_float(self.sample_window_train_ratio, 0.75)
        ratio = max(0.01, min(0.99, ratio))
        split_idx = int(len(unique_datetimes) * ratio)
        split_idx = max(1, min(len(unique_datetimes) - 1, split_idx))
        first_time = pd.Timestamp(unique_datetimes[0])
        split_time = pd.Timestamp(unique_datetimes[split_idx])
        self.train_window = {"start": first_time, "end": split_time}
        self.validation_window = {"start": split_time, "end": None}
        self._sample_window_resolution = {
            "mode": "chronological_ratio",
            "train_ratio": ratio,
            "validation_ratio": round(1.0 - ratio, 12),
            "unique_datetime_count": int(len(unique_datetimes)),
            "train_datetime_count": int(split_idx),
            "validation_datetime_count": int(len(unique_datetimes) - split_idx),
            "split_at": split_time.isoformat(),
        }
        return True

    def _capture_backtest_period(self, panel: pd.DataFrame) -> None:
        """记录实际参与回测的面板时间范围（回测区间），作为入库因子的标记来源。"""
        if "datetime" not in panel.index.names:
            self._backtest_period = {"start": None, "end": None}
            return
        datetimes = pd.to_datetime(panel.index.get_level_values("datetime"), errors="coerce")
        valid = datetimes[~pd.isna(datetimes)]
        if len(valid) == 0:
            self._backtest_period = {"start": None, "end": None}
            return
        self._backtest_period = {
            "start": pd.Timestamp(valid.min()).isoformat(),
            "end": pd.Timestamp(valid.max()).isoformat(),
        }

    def _ensure_sample_windows_resolved(self, panel: pd.DataFrame) -> None:
        self._capture_backtest_period(panel)
        if not self.sample_windows_enabled:
            self._sample_window_resolution = {"mode": "disabled"}
            return
        if self.sample_window_split_mode in {"chronological_ratio", "ratio", "auto_ratio"}:
            if self._resolve_ratio_sample_windows(panel):
                return
            self._sample_window_resolution = {
                "mode": "chronological_ratio",
                "status": "failed",
                "reason": "datetime_index_missing_or_too_short",
            }
            return
        self._sample_window_resolution = {"mode": "explicit"}

    @staticmethod
    def _metric_value(value: object) -> float | None:
        try:
            number = float(value)
        except Exception:
            return None
        return number if np.isfinite(number) else None

    def _build_train_filter_decision(self, train_metrics: dict[str, object]) -> dict[str, object]:
        metric = self.train_filter_metric
        threshold = float(self.train_filter_min_value)
        value = self._metric_value(dict(train_metrics).get(metric))
        if not self.sample_windows_enabled:
            return {
                "enabled": False,
                "metric": metric,
                "threshold": threshold,
                "comparison": ">",
                "window": "train",
                "value": value,
                "passed": True,
            }
        return {
            "enabled": True,
            "metric": metric,
            "threshold": threshold,
            "comparison": ">",
            "window": "train",
            "value": value,
            "passed": bool(value is not None and value > threshold),
        }

    def _build_coverage_filter_decision(self, metrics: dict[str, object]) -> dict[str, object]:
        metric = self.coverage_filter_metric
        threshold = float(self.coverage_filter_min_value)
        value = self._metric_value(dict(metrics).get(metric))
        window = "train" if self.sample_windows_enabled else "evaluation"
        if not self.coverage_filter_enabled:
            return {
                "enabled": False,
                "metric": metric,
                "threshold": threshold,
                "comparison": ">=",
                "window": window,
                "value": value,
                "passed": True,
            }
        return {
            "enabled": True,
            "metric": metric,
            "threshold": threshold,
            "comparison": ">=",
            "window": window,
            "value": value,
            "passed": bool(value is not None and value >= threshold),
        }

    @staticmethod
    def _reward_score(metrics: dict[str, object]) -> float:
        """因子综合 reward（与 FactorLibraryManager 入库口径一致）。"""
        from trading_agents.infra.factor_library_manager import FactorLibraryManager

        return FactorLibraryManager.compute_reward_score(dict(metrics))

    @staticmethod
    def _target_status(metrics: dict[str, object]) -> dict[str, object]:
        """好因子硬目标状态：净 Sharpe、abs(IC)、abs(Rank IC)。"""
        from trading_agents.infra.factor_library_manager import FactorLibraryManager

        return FactorLibraryManager.compute_target_status(dict(metrics))

    def _build_validation_decay_decision(
        self,
        train_metrics: dict[str, object],
        validation_metrics: dict[str, object] | None,
        *,
        validation_available: bool,
    ) -> dict[str, object]:
        """留样验证衰减判定：基于 train/val 的 composite reward。

        - 入库主要看 train，因此本门只在 sample_windows 启用时生效。
        - val reward 相对 train reward 保留比例 >= min_retention 且不正负翻转才通过。
        - train reward <= 0 时 train_filter 已会拦截弱因子，这里不重复惩罚，直接放行。
        - validation 数据不足（区间太短/为空）时无法判定，放行但标注 status，便于排查。
        """
        min_retention = float(self.validation_decay_min_retention)
        base = {
            "enabled": bool(self.sample_windows_enabled and self.validation_decay_enabled),
            "metric": "reward_score",
            "comparison": ">=",
            "min_retention": round(min_retention, 6),
            "window": "validation",
        }
        if not self.sample_windows_enabled or not self.validation_decay_enabled:
            return {**base, "applicable": False, "status": "disabled", "passed": True,
                    "train_reward": None, "validation_reward": None, "retention_ratio": None, "sign_flip": False}
        if not validation_available or not validation_metrics:
            return {**base, "applicable": False, "status": "insufficient_validation_data", "passed": True,
                    "train_reward": None, "validation_reward": None, "retention_ratio": None, "sign_flip": False}
        train_reward = self._reward_score(train_metrics)
        val_reward = self._reward_score(validation_metrics)
        sign_flip = bool(train_reward > 0.0 and val_reward <= 0.0)
        if train_reward <= 0.0:
            retention = 1.0
            passed = True
        else:
            retention = val_reward / train_reward
            passed = bool((not sign_flip) and retention >= min_retention)
        return {
            **base,
            "applicable": True,
            "status": "evaluated",
            "passed": passed,
            "train_reward": round(float(train_reward), 6),
            "validation_reward": round(float(val_reward), 6),
            "retention_ratio": round(float(retention), 6),
            "sign_flip": sign_flip,
        }

    def _build_robustness_filter_decision(self, metrics: dict[str, object]) -> dict[str, object]:
        reasons: list[str] = []
        market_beta = self._metric_value(metrics.get("market_beta"))
        residual_sharpe = self._metric_value(metrics.get("residual_sharpe"))
        yearly_positive_ratio = self._metric_value(metrics.get("yearly_positive_ratio"))
        max_year_share = self._metric_value(metrics.get("max_year_positive_contribution_share"))
        year_count = self._metric_value(metrics.get("year_count"))
        regime_positive_count = self._metric_value(metrics.get("regime_positive_sharpe_count"))
        if not self.robustness_filter_enabled:
            return {
                "enabled": False,
                "passed": True,
                "reasons": [],
            }
        if market_beta is None or abs(market_beta) > self.robustness_market_beta_abs_max:
            reasons.append(
                f"abs(market_beta)={market_beta} exceeds {self.robustness_market_beta_abs_max:g}"
            )
        if residual_sharpe is None or residual_sharpe < self.robustness_residual_sharpe_min:
            reasons.append(
                f"residual_sharpe={residual_sharpe} below {self.robustness_residual_sharpe_min:g}"
            )
        if yearly_positive_ratio is None or yearly_positive_ratio < self.robustness_yearly_positive_ratio_min:
            reasons.append(
                f"yearly_positive_ratio={yearly_positive_ratio} below {self.robustness_yearly_positive_ratio_min:g}"
            )
        if (
            year_count is not None
            and year_count >= 2
            and (max_year_share is None or max_year_share > self.robustness_max_year_positive_contribution_share)
        ):
            reasons.append(
                f"max_year_positive_contribution_share={max_year_share} exceeds {self.robustness_max_year_positive_contribution_share:g}"
            )
        if regime_positive_count is None or regime_positive_count < self.robustness_regime_positive_sharpe_count_min:
            reasons.append(
                f"regime_positive_sharpe_count={regime_positive_count} below {self.robustness_regime_positive_sharpe_count_min}"
            )
        return {
            "enabled": True,
            "passed": not reasons,
            "reasons": reasons,
            "thresholds": {
                "market_beta_abs_max": self.robustness_market_beta_abs_max,
                "residual_sharpe_min": self.robustness_residual_sharpe_min,
                "yearly_positive_ratio_min": self.robustness_yearly_positive_ratio_min,
                "max_year_positive_contribution_share": self.robustness_max_year_positive_contribution_share,
                "regime_positive_sharpe_count_min": self.robustness_regime_positive_sharpe_count_min,
            },
            "values": {
                "market_beta": market_beta,
                "residual_sharpe": residual_sharpe,
                "yearly_positive_ratio": yearly_positive_ratio,
                "max_year_positive_contribution_share": max_year_share,
                "year_count": year_count,
                "regime_positive_sharpe_count": regime_positive_count,
            },
        }

    def _slice_sample_window_inputs(
        self,
        *,
        panel: pd.DataFrame,
        label_series: pd.Series,
        signal_series: pd.Series,
        window: dict[str, pd.Timestamp | None],
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        mask = self._build_sample_window_mask(panel, window)
        window_panel = panel.iloc[np.flatnonzero(mask)].copy()
        window_label = label_series.reindex(window_panel.index)
        window_signal = signal_series.reindex(window_panel.index)
        return window_panel, window_label, window_signal

    def _target_horizon_timedelta(self) -> pd.Timedelta | None:
        # Support minute / hour / day horizons (fwd_returns_5min, _4h, _1d). "min"
        # must precede "m" in the alternation so it matches greedily.
        match = re.search(r"fwd_returns_(\d+)(min|m|h|d)", self.target_column)
        if not match:
            return None
        value = int(match.group(1))
        unit = match.group(2)
        if value <= 0:
            return None
        if unit in ("min", "m"):
            return pd.Timedelta(minutes=value)
        if unit == "h":
            return pd.Timedelta(hours=value)
        return pd.Timedelta(days=value)

    def _infer_panel_step_timedelta(self, panel: pd.DataFrame) -> pd.Timedelta | None:
        if "datetime" not in panel.index.names:
            return None
        datetimes = panel.index.get_level_values("datetime").unique().sort_values()
        if len(datetimes) < 2:
            return None
        diffs = datetimes.to_series().diff().dropna()
        if diffs.empty:
            return None
        return pd.Timedelta(diffs.mode().iloc[0])

    def _configure_sharpe_sampling(self, panel: pd.DataFrame) -> None:
        self._current_sharpe_sampling_step = 1
        self._current_sharpe_annualization_factor = self.annualization_factor
        self._current_panel_step = ""
        horizon = self._target_horizon_timedelta()
        self._current_target_horizon = horizon
        panel_step = self._infer_panel_step_timedelta(panel)
        if panel_step is None or horizon is None or panel_step <= pd.Timedelta(0) or horizon <= pd.Timedelta(0):
            return
        holding_steps = max(1, int(round(horizon / panel_step)))
        effective_horizon = panel_step * holding_steps
        self._current_sharpe_sampling_step = holding_steps
        self._current_sharpe_annualization_factor = float(pd.Timedelta(days=365) / effective_horizon)
        self._current_panel_step = str(panel_step)

    def _subsample_by_time_horizon(self, series: pd.Series, horizon: pd.Timedelta | None) -> pd.Series:
        """Sample one point per non-overlapping target horizon using real timestamps."""
        if horizon is None or horizon <= pd.Timedelta(0) or len(series) <= 1:
            return series
        ordered = series.sort_index()
        try:
            timestamps = pd.DatetimeIndex(pd.to_datetime(ordered.index, errors="coerce"))
        except Exception:
            return series
        if timestamps.isna().any():
            return series

        keep_positions: list[int] = []
        next_allowed: pd.Timestamp | None = None
        for pos, timestamp in enumerate(timestamps):
            if next_allowed is None or timestamp >= next_allowed:
                keep_positions.append(pos)
                next_allowed = timestamp + horizon
        return ordered.iloc[keep_positions]

    def _subsample_ic_to_horizon(self, ic_series: pd.Series) -> pd.Series:
        """将逐 bar 的截面 IC 序列按预测窗口降采样为非重叠样本。

        IC 在每个 bar 上都以「未来 horizon 收益」为标签计算，相邻 bar 的前向
        窗口高度重叠（如 5min 步长 + 24h 窗口，相邻 IC 重叠约 99.7%）。直接对
        全量逐 bar 序列求 mean/std 会因强自相关让 ICIR 失真且极不稳定。这里取每隔
        sharpe_sampling_step（= horizon / bar，由前端所选预测窗口决定）一个点，
        使相邻 IC 的前向窗口不再重叠，与 Sharpe/分层收益的非重叠采样口径一致。

        采样在「丢 NaN 之前」、在完整且按时间排序的网格上进行，保证窗口真正
        非重叠；sampling_step==1（bar 即预测窗口）时退化为原逐 bar 行为。
        """
        horizon = getattr(self, "_current_target_horizon", None)
        if horizon is not None:
            ic_series = self._subsample_by_time_horizon(ic_series, horizon)
        else:
            sampling_step = max(1, int(self._current_sharpe_sampling_step))
            if sampling_step > 1 and len(ic_series) > 1:
                ic_series = ic_series.sort_index().iloc[::sampling_step]
        return ic_series.dropna()

    def _subsample_returns_to_horizon(self, series: pd.Series) -> pd.Series:
        """Sample return-like series to the same non-overlapping horizon as IC."""
        horizon = getattr(self, "_current_target_horizon", None)
        if horizon is not None:
            return self._subsample_by_time_horizon(series, horizon).dropna()
        sampling_step = max(1, int(self._current_sharpe_sampling_step))
        if sampling_step > 1 and len(series) > 1:
            return series.sort_index().iloc[::sampling_step].dropna()
        return series.dropna()

    def _subsample_frame_to_horizon(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sample a datetime-indexed return frame to non-overlapping target windows."""
        horizon = getattr(self, "_current_target_horizon", None)
        if horizon is not None and len(frame) > 1:
            sampled_index = self._subsample_by_time_horizon(
                pd.Series(np.arange(len(frame), dtype=float), index=frame.index),
                horizon,
            ).index
            return frame.sort_index().loc[sampled_index]
        sampling_step = max(1, int(self._current_sharpe_sampling_step))
        if sampling_step > 1 and len(frame) > 1:
            return frame.sort_index().iloc[::sampling_step]
        return frame

    @staticmethod
    def _sample_std(values: np.ndarray) -> float:
        """Sample standard deviation (ddof=1), matching pandas Series.std()."""
        clean = np.asarray(values, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size <= 1:
            return 0.0
        std = float(np.std(clean, ddof=1))
        return std if np.isfinite(std) else 0.0

    def _series_sharpe(self, values: pd.Series) -> float:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return 0.0
        std = float(clean.std())
        if abs(std) <= 1e-12:
            return 0.0
        return self._finite_float(float(clean.mean()) / std * np.sqrt(self._current_sharpe_annualization_factor))

    @staticmethod
    def _compute_yearly_returns(returns_series: pd.Series) -> dict[str, float]:
        """把多空收益序列按自然年拆分，计算每年的累计收益率（非年化，就是当年总回报）。"""
        if returns_series is None or returns_series.empty:
            return {}
        try:
            idx = pd.to_datetime(returns_series.index, errors="coerce")
            s = returns_series.copy()
            s.index = idx
            s = s[~pd.isna(s.index)].dropna()
            if s.empty:
                return {}
            result: dict[str, float] = {}
            for year, group in s.groupby(s.index.year):
                if len(group) > 0:
                    annual_ret = float((1.0 + group).prod() - 1.0)
                    result[str(int(year))] = round(annual_ret, 6)
            return result
        except Exception:
            return {}

    def _calculate_market_neutral_robustness(
        self,
        *,
        oriented_net_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict[str, float]:
        """Calculate market-neutral and regime/year robustness diagnostics."""
        portfolio = pd.to_numeric(oriented_net_returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
        benchmark = pd.to_numeric(benchmark_returns, errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame = pd.concat([portfolio.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
        if frame.empty:
            return {
                "market_beta": 0.0,
                "residual_sharpe": 0.0,
                "market_neutral_mean_return": 0.0,
                "yearly_positive_ratio": 0.0,
                "max_year_positive_contribution_share": 1.0,
                "year_count": 0.0,
                "regime_positive_sharpe_count": 0.0,
                "robustness_pass": 0.0,
                "robustness_failure_count": 1.0,
            }
        benchmark_var = float(frame["benchmark"].var())
        beta = (
            float(frame["portfolio"].cov(frame["benchmark"]) / benchmark_var)
            if abs(benchmark_var) > 1e-12 else 0.0
        )
        residual = frame["portfolio"] - beta * frame["benchmark"]
        residual_sharpe = self._series_sharpe(residual)

        years = pd.to_datetime(frame.index, errors="coerce").year
        yearly_mean = residual.groupby(years).mean()
        yearly_mean = yearly_mean[~pd.isna(yearly_mean.index)]
        year_count = len(yearly_mean)
        positive_years = yearly_mean[yearly_mean > 0]
        yearly_positive_ratio = float(len(positive_years) / max(1, len(yearly_mean))) if len(yearly_mean) else 0.0
        positive_sum = float(positive_years.sum()) if not positive_years.empty else 0.0
        max_year_share = (
            float(positive_years.max() / positive_sum)
            if positive_sum > 1e-12 and not positive_years.empty
            else 1.0
        )

        regime_positive_count = 0
        if len(frame) >= 3 and frame["benchmark"].nunique(dropna=True) >= 3:
            try:
                regimes = pd.qcut(frame["benchmark"], q=3, labels=["down", "flat", "up"], duplicates="drop")
                for _regime, group in residual.groupby(regimes, observed=False):
                    if self._series_sharpe(group) > 0:
                        regime_positive_count += 1
            except Exception:
                regime_positive_count = 0
        failures = 0
        if self.robustness_filter_enabled:
            if abs(beta) > self.robustness_market_beta_abs_max:
                failures += 1
            if residual_sharpe < self.robustness_residual_sharpe_min:
                failures += 1
            if yearly_positive_ratio < self.robustness_yearly_positive_ratio_min:
                failures += 1
            if year_count >= 2 and max_year_share > self.robustness_max_year_positive_contribution_share:
                failures += 1
            if regime_positive_count < self.robustness_regime_positive_sharpe_count_min:
                failures += 1
        return {
            "market_beta": self._finite_float(beta),
            "residual_sharpe": self._finite_float(residual_sharpe),
            "market_neutral_mean_return": self._finite_float(float(residual.mean())),
            "yearly_positive_ratio": self._finite_float(yearly_positive_ratio),
            "max_year_positive_contribution_share": self._finite_float(max_year_share),
            "year_count": self._finite_float(float(year_count)),
            "regime_positive_sharpe_count": self._finite_float(float(regime_positive_count)),
            "robustness_pass": 1.0 if failures == 0 else 0.0,
            "robustness_failure_count": self._finite_float(float(failures)),
        }

    def _load_qlib_task_config(self, config_path: Path) -> dict[str, Any]:
        """从 YAML 文件加载 Qlib 任务配置（包含 qlib_init 等初始化参数）。"""
        with config_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid qlib config: {config_path}")
        return loaded

    @staticmethod
    def _resolve_project_root(config_path: Path) -> Path:
        config_path = config_path.resolve()
        for candidate in [config_path.parent, *config_path.parents, Path.cwd().resolve()]:
            if (candidate / "pyproject.toml").exists() or (candidate / "main.py").exists():
                return candidate
        for candidate in config_path.parents:
            if candidate.name == "src":
                return candidate.parent
        return Path.cwd().resolve()

    def _resolve_qlib_provider_uri(self, provider_uri_raw: str) -> str:
        override = os.getenv("QLIB_PROVIDER_URI", "").strip()
        provider_uri = override or provider_uri_raw
        provider_uri = os.path.expandvars(os.path.expanduser(provider_uri))
        provider_path = Path(provider_uri)
        if not provider_path.is_absolute():
            provider_path = self.project_root / provider_path
        return str(provider_path.resolve())

    def _ensure_qlib_initialized(self, qlib_task_config: dict[str, Any]) -> None:
        """懒初始化 Qlib（仅执行一次）。从配置中读取 provider_uri 和 region 进行初始化。"""
        if self._qlib_initialized:
            return
        import os
        import qlib  # type: ignore

        qlib_init = qlib_task_config.get("qlib_init", {})
        if not isinstance(qlib_init, dict):
            raise ValueError("qlib_init is missing in qlib config")
        provider_uri_raw = str(qlib_init.get("provider_uri", "")).strip()
        if not provider_uri_raw and not os.getenv("QLIB_PROVIDER_URI", "").strip():
            raise ValueError("qlib_init.provider_uri is empty")
        provider_uri = self._resolve_qlib_provider_uri(provider_uri_raw)
        region = str(qlib_init.get("region", "cn")).strip() or "cn"
        qlib.init(provider_uri=provider_uri, region=region)
        self._qlib_initialized = True

    def _prepare_panel(self, panel_data_path: str) -> tuple[pd.DataFrame, str] | None:
        """加载面板数据并校验目标列和索引层级。返回 (panel_df, target_column) 或 None。

        目标列优先级：
        1. backtest_engine.yaml 中配置的 target_column（当前应为 fwd_returns_4h）
        2. 若配置列不存在，按预测窗口从短到长自动选取 fwd_returns_* 备选列，并打印警告
        3. 均不存在时返回 None（回测失败）
        """
        try:
            panel = downcast_float_columns(pd.read_parquet(panel_data_path))
        except Exception:
            return None
        if panel.empty:
            return None
        if not self.required_index_levels.issubset(set(panel.index.names)):
            return None

        # 解析目标列，按优先级：
        #   1) 每轮运行通过 TLS 设置的 target（与所选预测窗口一致）——再次读取以
        #      防引擎在 set_thread_backtest_config 之前就被构造（构造时 TLS 还没设）。
        #   2) 引擎构造时确定的 self.target_column（来自 TLS 或 yaml 默认）。
        #   3) 面板里实际存在的 fwd_returns_* 列（兜底：面板按单一窗口预处理时
        #      只有一个 fwd 列，避免因默认值不匹配而误判“无目标列”）。
        tls_target = getattr(_BACKTEST_TLS, "target_column", None)
        target_column = None
        for cand in (tls_target, self.target_column):
            if cand and cand in panel.columns:
                target_column = cand
                break
        if target_column is None:
            available = [c for c in panel.columns if str(c).startswith("fwd_returns_")]
            if not available:
                print(
                    f"[BacktestEngine] ERROR: 面板中没有任何 fwd_returns_* 目标列。\n"
                    f"  期望: {tls_target or self.target_column}；请先在「数据预处理」生成预测标签。"
                )
                return None
            desired = str(tls_target or self.target_column or "")
            target_column = next((c for c in available if c == desired), available[-1])
            print(
                f"[BacktestEngine] WARN: 目标列 '{desired}' 不在面板，回退使用 '{target_column}'。"
                f"  可用 fwd_returns_* 列: {available}"
            )

        self.target_column = str(target_column)
        return panel, target_column

    def _collect_generated_impls(self, factor_implementation: dict[str, object]) -> list[dict[str, object]]:
        """从 factor_implementation 中筛选 status=="generated" 的因子实现列表。"""
        implementation_items = factor_implementation.get("implementations", [])
        if not isinstance(implementation_items, list):
            return []
        return [
            item
            for item in implementation_items
            if isinstance(item, dict) and str(item.get("status", "")).strip().lower() == "generated"
        ]

    def _collect_failure_errors(
        self,
        *,
        factor_implementation: dict[str, object],
        calculation_report: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        """收集所有失败因子的错误信息（来自 factor_implementation 和 calculation_report 两处）。"""
        errors: dict[str, dict[str, object]] = {}
        failed_names = {
            str(item).strip()
            for item in list(calculation_report.get("failed_factors", []))
            if str(item).strip()
        } if isinstance(calculation_report.get("failed_factors", []), list) else set()
        implementation_items = factor_implementation.get("implementations", [])
        if not isinstance(implementation_items, list):
            implementation_items = []
        for item in implementation_items:
            if not isinstance(item, dict):
                continue
            factor_name = str(item.get("factor_name", item.get("name", ""))).strip()
            if not factor_name:
                continue
            status = str(item.get("status", "")).strip().lower()
            if status == "generated" and factor_name not in failed_names:
                continue
            feedback = dict(item.get("feedback", {})) if isinstance(item.get("feedback", {}), dict) else {}
            errors[factor_name] = {
                "stage": "calculation",
                "status": status or "unknown",
                "expression": str(item.get("expression", "")),
                "execution_feedback": str(feedback.get("execution_feedback", "")),
                "code_feedback": str(feedback.get("code_feedback", "")),
                "value_feedback": str(feedback.get("value_feedback", "")),
                "final_feedback": str(feedback.get("final_feedback", "")),
                "final_decision": feedback.get("final_decision"),
            }
        for factor_name in failed_names:
            errors.setdefault(
                factor_name,
                {
                    "stage": "calculation",
                    "status": "failed",
                    "message": "Listed in calculation_report.failed_factors but no implementation detail was found.",
                },
            )
        return errors

    def _normalize_expression_key(self, expression: str) -> str:
        """将表达式文本规范化为无空白小写形式，用于去重分组。"""
        return "".join(str(expression or "").split()).lower()

    def _resolve_parallel_workers(self, task_count: int) -> int:
        """确定并行工作线程数。优先读环境变量 BACKTEST_PARALLEL_WORKERS，默认 min(4, cpu_count)。"""
        raw = os.getenv("BACKTEST_PARALLEL_WORKERS", "").strip()
        if raw:
            try:
                workers = int(raw)
            except ValueError:
                workers = 1
        else:
            workers = min(4, os.cpu_count() or 1)
        return max(1, min(workers, max(1, task_count)))

    def _load_signal_from_cache(self, impl: dict[str, object], panel: pd.DataFrame) -> pd.Series | None:
        """从 pickle 缓存加载因子信号（由 FactorCoderAgent 预计算）。加载后对齐到面板索引。"""
        raw_path = str(impl.get("source_pickle_path", "")).strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return None
        try:
            loaded = pd.read_pickle(path)
        except Exception:
            return None
        if not isinstance(loaded, pd.Series) or loaded.empty:
            return None
        if not loaded.index.equals(panel.index):
            loaded = loaded.reindex(panel.index)
        return loaded

    def _calculate_single_factor_metrics(
        self,
        signal_series: pd.Series,
        label_series: pd.Series,
        panel: pd.DataFrame,
    ) -> dict[str, float]:
        """计算单个因子的核心截面指标。

        包含：
        - IC / ICIR：因子信号与未来收益的 Pearson 截面相关系数及其 IR
        - Rank IC / Rank ICIR：先排名再计算的 Spearman-style 截面相关系数及其 IR
        - coverage：有效信号占面板总行数的比例
        - turnover：因子排名的日间变化率（换手率代理指标）

        注意：将 index 的 "symbol" 层重命名为 "instrument" 以兼容 Qlib 接口。
        """
        metrics: dict[str, float] = {}
        factor_frame = (
            pd.concat([signal_series.rename("signal"), label_series.rename("label")], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if factor_frame.empty:
            return {"IC": 0.0, "ICIR": 0.0, "Rank IC": 0.0, "Rank ICIR": 0.0, "coverage": 0.0}
        
        eval_frame = factor_frame.copy()
        # Safely rename index levels
        if isinstance(eval_frame.index, pd.MultiIndex):
            index_names = list(eval_frame.index.names)
            if "symbol" in index_names:
                index_names[index_names.index("symbol")] = "instrument"
                eval_frame.index = eval_frame.index.set_names(index_names)
            elif len(index_names) >= 2:
                # Assume second level is symbol if not named
                index_names[1] = "instrument"
                eval_frame.index = eval_frame.index.set_names(index_names)
        
        # ★ 中等耗时：截面 IC（Spearman 逐日 apply），~0.5-1s
        # 按预测窗口降采样为非重叠 IC，口径与矩阵路径/ Sharpe 一致。
        ic_values = self._subsample_ic_to_horizon(self._cross_sectional_corr(eval_frame, rank=False))

        if isinstance(ic_values, pd.Series) and not ic_values.empty:
            ic_mean = float(ic_values.mean())
            ic_std = float(ic_values.std())
            metrics["IC"] = ic_mean
            metrics["ICIR"] = float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else 0.0
        else:
            metrics["IC"] = 0.0
            metrics["ICIR"] = 0.0
        
        # ★ 中等耗时：截面 Rank IC（先排名再 Spearman 逐日 apply），~0.5-1s
        rank_ic_values = self._subsample_ic_to_horizon(self._cross_sectional_corr(eval_frame, rank=True))
        if isinstance(rank_ic_values, pd.Series) and not rank_ic_values.empty:
            rank_ic_mean = float(rank_ic_values.mean())
            rank_ic_std = float(rank_ic_values.std())
            metrics["Rank IC"] = rank_ic_mean
            metrics["Rank ICIR"] = float(rank_ic_mean / rank_ic_std) if abs(rank_ic_std) > 1e-12 else 0.0
        else:
            metrics["Rank IC"] = 0.0
            metrics["Rank ICIR"] = 0.0
        
        coverage = float(len(eval_frame) / max(1, len(panel)))
        metrics["coverage"] = round(coverage, 4)

        ranked_signal = eval_frame["signal"].groupby(level="datetime").rank(pct=True)
        rank_wide = ranked_signal.unstack(level="instrument")
        rank_diff = rank_wide.diff().abs().mean(axis=1).dropna()
        metrics["turnover"] = float(rank_diff.mean()) if not rank_diff.empty else 1.0

        return metrics

    def _calculate_factor_metrics_bundle(
        self,
        *,
        factor_name: str,
        expression: str,
        signal_series: pd.Series,
        label_series: pd.Series,
        panel: pd.DataFrame,
    ) -> dict[str, float]:
        """调度入口：优先矩阵加速，失败回退原始路径。"""
        try:
            return self._calculate_factor_metrics_bundle_matrix(
                factor_name=factor_name,
                expression=expression,
                signal_series=signal_series,
                label_series=label_series,
                panel=panel,
            )
        except Exception as exc:
            _debug_print(f"[WARN] Matrix metrics failed for '{factor_name}': {exc}, falling back")
            return self._calculate_factor_metrics_bundle_original(
                factor_name=factor_name,
                expression=expression,
                signal_series=signal_series,
                label_series=label_series,
                panel=panel,
            )

    def _calculate_factor_metrics_bundle_matrix(
        self,
        *,
        factor_name: str,
        expression: str,
        signal_series: pd.Series,
        label_series: pd.Series,
        panel: pd.DataFrame,
    ) -> dict[str, float]:
        """矩阵加速版因子指标计算。

        将 signal/label 从 MultiIndex Series 转为 (T, N) numpy 矩阵，
        用向量化运算替代 groupby().apply() 逐日循环，加速 IC/RankIC 和分层回测。
        """
        signal_series = pd.to_numeric(signal_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if signal_series.dropna().empty:
            raise ValueError("factor produced empty or all-null signal")
        label_series = pd.to_numeric(label_series, errors="coerce").replace([np.inf, -np.inf], np.nan)

        # ── Step 1: 对齐 signal + label，转为 (T, N) 矩阵 ──
        factor_frame = pd.concat(
            [signal_series.rename("signal"), label_series.rename("label")], axis=1,
        ).dropna()
        if factor_frame.empty:
            raise ValueError("factor produced empty or all-null signal")

        eval_frame = factor_frame.copy()
        if isinstance(eval_frame.index, pd.MultiIndex):
            names = list(eval_frame.index.names)
            if "symbol" in names:
                names[names.index("symbol")] = "instrument"
                eval_frame.index = eval_frame.index.set_names(names)

        signal_wide = eval_frame["signal"].unstack(level="instrument")   # (T, N)
        label_wide = eval_frame["label"].unstack(level="instrument")     # (T, N)
        signal_mat = signal_wide.values.astype(np.float32)
        label_mat = label_wide.values.astype(np.float32)
        T, N = signal_mat.shape
        datetimes = signal_wide.index
        valid_mask = ~(np.isnan(signal_mat) | np.isnan(label_mat))       # (T, N)
        valid_per_row = valid_mask.sum(axis=1)                            # (T,)

        # 横截面分层数。某个时间点可用标的数 < n_layers 时无法构成有效的分位
        # 分组，这类「薄」时间点从分层收益 / 多空指标中剔除（设为 NaN），而不是
        # 塞进 L1 —— 否则薄横截面会用 0 和噪声污染分层结果（典型场景：早期只有
        # 少数标的上市，或整体标的数不足）。IC 不受此约束（见下方 IC 计算）。
        n_layers = 5
        thick_row = valid_per_row >= n_layers                            # (T,) bool

        # ── Step 2: IC / Rank IC（向量化 Spearman = Pearson of ranks）──
        # float32：排名矩阵与 signal_mat/label_mat 同精度，省一半内存。N（标的数）
        # 量级小，秩相关的 nansum 在 float32 下精度足够。
        s_ranked = signal_wide.rank(axis=1, method="average").values.astype(np.float32)   # (T, N)
        l_ranked = label_wide.rank(axis=1, method="average").values.astype(np.float32)     # (T, N)

        sx = s_ranked - np.nanmean(s_ranked, axis=1, keepdims=True)
        ly = l_ranked - np.nanmean(l_ranked, axis=1, keepdims=True)
        cov = np.nansum(sx * ly, axis=1)
        std_s = np.sqrt(np.nansum(sx ** 2, axis=1))
        std_l = np.sqrt(np.nansum(ly ** 2, axis=1))

        # IC 是截面相关系数，≥2 个标的即有定义，最小样本数与分层（thick_row）不同，
        # 因此 IC 仍用 >= 2，不随分层阈值变化。
        ic_arr = np.where(
            (std_s > 1e-12) & (std_l > 1e-12) & (valid_per_row >= 2),
            cov / (std_s * std_l),
            np.nan,
        )
        # 按预测窗口降采样为非重叠 IC（口径与 Sharpe 一致），再算 ICIR/胜率/累计曲线。
        ic_series = self._subsample_ic_to_horizon(pd.Series(ic_arr, index=datetimes))

        # Pearson IC（在原始值上，不排名）——与 Spearman 的 Rank IC 区分开，
        # 供前端「累计 IC / 累计 RankIC」双线图使用。
        sxr = signal_mat - np.nanmean(signal_mat, axis=1, keepdims=True)
        lyr = label_mat - np.nanmean(label_mat, axis=1, keepdims=True)
        cov_r = np.nansum(sxr * lyr, axis=1)
        std_sr = np.sqrt(np.nansum(sxr ** 2, axis=1))
        std_lr = np.sqrt(np.nansum(lyr ** 2, axis=1))
        ic_pearson_arr = np.where(
            (std_sr > 1e-12) & (std_lr > 1e-12) & (valid_per_row >= 2),
            cov_r / (std_sr * std_lr),
            np.nan,
        )
        ic_pearson_series = self._subsample_ic_to_horizon(pd.Series(ic_pearson_arr, index=datetimes))

        metrics: dict[str, float] = {}
        # IC = Pearson（原始值上的线性相关）；Rank IC = Spearman（排名后的相关）。
        # 两者口径不同，不能相等：Pearson 对量纲/异常值敏感，Spearman 更稳健。
        if not ic_pearson_series.empty:
            ic_p_mean = float(ic_pearson_series.mean())
            ic_p_std = float(ic_pearson_series.std())
            metrics["IC"] = ic_p_mean
            metrics["ICIR"] = float(ic_p_mean / ic_p_std) if abs(ic_p_std) > 1e-12 else 0.0
        else:
            metrics["IC"] = 0.0
            metrics["ICIR"] = 0.0
        if not ic_series.empty:
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            metrics["Rank IC"] = ic_mean
            metrics["Rank ICIR"] = float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else 0.0
        else:
            metrics["Rank IC"] = 0.0
            metrics["Rank ICIR"] = 0.0

        # Coverage & Turnover
        metrics["coverage"] = round(float(len(eval_frame) / max(1, len(panel))), 4)
        rank_pct = signal_wide.rank(axis=1, pct=True)
        rank_diff = rank_pct.diff().abs().mean(axis=1).dropna()
        metrics["turnover"] = float(rank_diff.mean()) if not rank_diff.empty else 1.0

        # ── 入库判定「基础有效性」指标 ──
        # 有效截面数均值：每个时间点可用标的数(valid_per_row)的均值，仅统计非空截面。
        _rows_with_data = valid_per_row[valid_per_row > 0]
        metrics["valid_cross_section_mean"] = (
            round(float(_rows_with_data.mean()), 2) if _rows_with_data.size else 0.0
        )
        metrics["cross_section_universe_size"] = float(N)
        # IC / RankIC 胜率：逐期相关系数与其【均值同号】的比例（方向一致性）。
        # 用「同号」而非「>0」，使负相关因子（翻向后等价正相关）也能正确计胜率。
        if not ic_pearson_series.empty:
            _ic_sign = np.sign(ic_pearson_series.mean())
            metrics["ic_win_rate"] = (
                round(float((np.sign(ic_pearson_series) == _ic_sign).mean()), 4) if _ic_sign != 0 else 0.0
            )
        else:
            metrics["ic_win_rate"] = 0.0
        if not ic_series.empty:
            _ric_sign = np.sign(ic_series.mean())
            metrics["rank_ic_win_rate"] = (
                round(float((np.sign(ic_series) == _ric_sign).mean()), 4) if _ric_sign != 0 else 0.0
            )
        else:
            metrics["rank_ic_win_rate"] = 0.0

        # ── Step 3: 分层回测（向量化）──
        # 3a. 分层赋值：rank → quintile
        s_rank_first = signal_wide.rank(axis=1, method="first").values.astype(np.float32)    # (T, N)
        layer_size = np.where(
            thick_row,
            valid_per_row / n_layers,
            1.0,  # 占位，薄行下面整行置 NaN，这里只是避免除零
        )
        layer_mat = np.floor((s_rank_first - 1.0) / layer_size[:, None]).astype(np.float32) + 1.0
        layer_mat = np.clip(layer_mat, 1, n_layers)
        # 薄横截面时间点剔除：整行置 NaN（不计入任何层），而非旧逻辑的全塞 L1
        layer_mat[~thick_row] = np.nan
        layer_mat[~valid_mask] = np.nan

        # 3b. 各层平均收益 → lr_df (T, n_layers)
        lr_mat = np.full((T, n_layers), np.nan)
        for li in range(n_layers):
            mask = (layer_mat == (li + 1)) & valid_mask
            cnt = mask.sum(axis=1).clip(1)
            lr_mat[:, li] = np.where(
                mask.any(axis=1),
                np.nansum(np.where(mask, label_mat, np.nan), axis=1) / cnt,
                np.nan,
            )
        lr_df = pd.DataFrame(lr_mat, index=datetimes, columns=list(range(1, n_layers + 1)))

        # 3c. 多空收益（全时间线）
        ls_full = (lr_df[n_layers] - lr_df[1]).dropna().sort_index()
        benchmark_full = label_wide.mean(axis=1).dropna().sort_index()

        # 3d. 权重矩阵 → 向量化 turnover cost
        high_mask = (layer_mat == n_layers) & valid_mask
        low_mask = (layer_mat == 1) & valid_mask
        n_high = high_mask.sum(axis=1, keepdims=True).clip(1)
        n_low = low_mask.sum(axis=1, keepdims=True).clip(1)
        weight_mat = np.where(
            high_mask, 0.5 / n_high,
            np.where(low_mask, -0.5 / n_low, 0.0),
        )
        weight_mat[~valid_mask] = 0.0

        first_toc = np.nansum(np.abs(weight_mat[0])) * self.one_way_fee_rate
        diff_toc = np.abs(np.diff(weight_mat, axis=0)).sum(axis=1) * self.one_way_fee_rate
        full_toc = np.concatenate([[first_toc], diff_toc])
        full_toc_series = pd.Series(full_toc, index=datetimes)

        # 3e. 采样
        ls = self._subsample_returns_to_horizon(ls_full)
        lr_df = self._subsample_frame_to_horizon(lr_df)
        benchmark = self._subsample_returns_to_horizon(benchmark_full)

        # 3f. 各层平均收益指标
        for li in range(1, n_layers + 1):
            col = lr_df[li].dropna()
            metrics[f"layer_{li}_mean_return"] = self._finite_float(float(col.mean())) if not col.empty else 0.0

        # 单调性
        layer_keys = [f"layer_{i}_mean_return" for i in range(1, n_layers + 1)]
        if all(k in metrics for k in layer_keys):
            vals = [metrics[k] for k in layer_keys]
            metrics["layer_monotonicity"] = (
                sum(1 for i in range(len(vals) - 1) if vals[i] < vals[i + 1]) / (n_layers - 1)
                if n_layers > 1 else 0.0
            )

        # 对齐 turnover cost 到采样后的 ls
        toc = full_toc_series.reindex(ls.index, fill_value=0.0)

        # ── Step 4: 多空绩效指标 ──
        direction_sign = (
            1.0 if metrics.get("Rank IC", 0) > 0
            else (-1.0 if metrics.get("Rank IC", 0) < 0 else None)
        )
        orientation_mul = -1.0 if direction_sign is not None and direction_sign < 0 else 1.0

        def _empty_perf() -> dict[str, float]:
            return {
                "long_short_mean_return": 0.0, "gross_long_short_mean_return": 0.0,
                "long_short_positive_ratio": 0.0, "drawdown": 0.0, "sharpe": 0.0,
                "raw_long_short_mean_return": 0.0, "gross_raw_long_short_mean_return": 0.0,
                "raw_long_short_positive_ratio": 0.0, "raw_drawdown": 0.0, "raw_sharpe": 0.0,
                "gross_sharpe": 0.0, "gross_raw_sharpe": 0.0,
                "net_sharpe": 0.0, "net_raw_sharpe": 0.0,
                "transaction_cost_mean": 0.0, "raw_transaction_cost_mean": 0.0,
                "long_short_turnover": 0.0, "one_way_fee_rate": self._finite_float(self.one_way_fee_rate),
                "long_short_orientation": "high_minus_low", "direction_aligned_by_rank_ic": False,
                "market_beta": 0.0, "residual_sharpe": 0.0, "market_neutral_mean_return": 0.0,
                "yearly_positive_ratio": 0.0, "max_year_positive_contribution_share": 1.0,
                "regime_positive_sharpe_count": 0.0, "robustness_pass": 0.0, "robustness_failure_count": 1.0,
            }

        if ls.empty:
            metrics.update(_empty_perf())
        else:
            ls_arr = ls.values.astype(np.float32)
            raw_mean = float(np.nanmean(ls_arr))
            raw_std = self._sample_std(ls_arr)
            raw_positive = float(np.mean(ls_arr > 0))
            raw_sharpe = (
                float(raw_mean / raw_std * np.sqrt(self._current_sharpe_annualization_factor))
                if abs(raw_std) > 1e-12 else 0.0
            )
            raw_cum = np.cumprod(1.0 + ls_arr)
            raw_max_dd = float(abs(np.min(raw_cum / np.maximum.accumulate(raw_cum) - 1.0))) if len(raw_cum) > 0 else 0.0

            gross_mean = raw_mean
            gross_std = raw_std
            gross_sharpe = raw_sharpe

            oriented_arr = ls_arr * orientation_mul
            net_arr = oriented_arr - toc.values.astype(np.float32)
            net_mean = float(np.nanmean(net_arr))
            net_std = self._sample_std(net_arr)
            net_positive = float(np.mean(net_arr > 0))
            net_sharpe = (
                float(net_mean / net_std * np.sqrt(self._current_sharpe_annualization_factor))
                if abs(net_std) > 1e-12 else 0.0
            )
            net_cum = np.cumprod(1.0 + net_arr)
            max_dd = float(abs(np.min(net_cum / np.maximum.accumulate(net_cum) - 1.0))) if len(net_cum) > 0 else 0.0

            metrics.update({
                "long_short_mean_return": self._finite_float(net_mean),
                "gross_long_short_mean_return": self._finite_float(gross_mean * orientation_mul),
                "long_short_positive_ratio": self._finite_float(net_positive),
                "drawdown": self._finite_float(max_dd),
                "sharpe": self._finite_float(net_sharpe),
                "net_sharpe": self._finite_float(net_sharpe),
                "raw_long_short_mean_return": self._finite_float(raw_mean),
                "gross_raw_long_short_mean_return": self._finite_float(raw_mean),
                "raw_long_short_positive_ratio": self._finite_float(raw_positive),
                "raw_drawdown": self._finite_float(raw_max_dd),
                "raw_sharpe": self._finite_float(raw_sharpe),
                "net_raw_sharpe": self._finite_float(raw_sharpe),
                "gross_sharpe": self._finite_float(gross_sharpe * abs(orientation_mul)),
                "gross_raw_sharpe": self._finite_float(gross_sharpe),
                "transaction_cost_mean": self._finite_float(float(np.nanmean(toc.values))),
                "raw_transaction_cost_mean": self._finite_float(float(np.nanmean(toc.values))),
                "long_short_turnover": (
                    self._finite_float(float(np.nanmean(toc.values) / self.one_way_fee_rate))
                    if self.one_way_fee_rate > 0 else 0.0
                ),
                "one_way_fee_rate": self._finite_float(self.one_way_fee_rate),
                "long_short_orientation": "low_minus_high" if orientation_mul < 0 else "high_minus_low",
                "direction_aligned_by_rank_ic": direction_sign is not None,
            })
            metrics.update(
                self._calculate_market_neutral_robustness(
                    oriented_net_returns=pd.Series(net_arr, index=ls.index),
                    benchmark_returns=benchmark.reindex(ls.index),
                )
            )

        # 绝对值和符号指标
        metrics["IC_abs"] = abs(metrics.get("IC", 0.0))
        metrics["IC_sign"] = 1.0 if metrics.get("IC", 0.0) >= 0 else -1.0
        metrics["Rank IC_abs"] = abs(metrics.get("Rank IC", 0.0))
        metrics["Rank IC_sign"] = 1.0 if metrics.get("Rank IC", 0.0) >= 0 else -1.0
        metrics["ICIR_abs"] = abs(metrics.get("ICIR", 0.0))
        metrics["Rank ICIR_abs"] = abs(metrics.get("Rank ICIR", 0.0))
        metrics["sharpe_abs"] = abs(metrics.get("sharpe", 0.0))

        # 通用：先累计再采样至 ≤300 点（顺序很重要——先采样再累计会跳过收益，
        # 严重低估净值）。mode='prod' → 净值(cumprod)；'sum' → 累计 IC(cumsum)。
        def _cum_series(series: pd.Series, mode: str) -> list:
            s = series.dropna()
            if s.empty:
                return []
            if mode == "prod":
                # winsorize 单期极端收益（0.5%/99.5% 分位），避免个别离群值主导
                # 净值曲线——加密货币单期可暴涨暴跌。与离线绘图脚本口径一致。
                lo, hi = s.quantile(0.005), s.quantile(0.995)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    s = s.clip(lo, hi)
                cum = (1.0 + s).cumprod()
            else:
                cum = s.cumsum()
            step_s = max(1, len(cum) // 300)
            sampled = cum.iloc[::step_s]
            return [
                [str(t), round(float(v), 6)]
                for t, v in zip(sampled.index, sampled.values)
                if np.isfinite(v)
            ]

        # 多空累计净值曲线（与各分组同口径，供前端绘图）
        if not ls.empty:
            try:
                ls_curve = _cum_series(pd.Series(net_arr, index=ls.index), "prod")
                if ls_curve:
                    metrics["ls_cumulative_series"] = ls_curve  # type: ignore[assignment]
            except Exception:
                pass

        # 五分组各层累计净值曲线（G1..G5），供前端「五分组净值图」
        for li in range(1, n_layers + 1):
            if li in lr_df.columns:
                try:
                    series = _cum_series(lr_df[li], "prod")
                    if series:
                        metrics[f"layer_{li}_cumulative_series"] = series  # type: ignore[assignment]
                except Exception:
                    pass

        # 基准（横截面等权）累计净值曲线
        try:
            bench_series = _cum_series(benchmark, "prod")
            if bench_series:
                metrics["benchmark_cumulative_series"] = bench_series  # type: ignore[assignment]
        except Exception:
            pass

        # 累计 IC（Pearson）/ 累计 RankIC（Spearman）曲线，供前端「累计 IC/RankIC 图」
        try:
            ic_cum = _cum_series(ic_pearson_series, "sum")
            if ic_cum:
                metrics["ic_cumulative_series"] = ic_cum  # type: ignore[assignment]
            rank_ic_cum = _cum_series(ic_series, "sum")
            if rank_ic_cum:
                metrics["rank_ic_cumulative_series"] = rank_ic_cum  # type: ignore[assignment]
        except Exception:
            pass

        # 年化收益率 + 逐年收益率分解
        if not ls.empty:
            metrics["annualized_return"] = self._finite_float(
                net_mean * self._current_sharpe_annualization_factor
            )
            try:
                net_series_full = pd.Series(net_arr, index=ls.index)
                yearly = self._compute_yearly_returns(net_series_full)
                if yearly:
                    metrics["yearly_returns"] = yearly  # type: ignore[assignment]
            except Exception:
                pass

        return metrics

    def _calculate_factor_metrics_bundle_original(
        self,
        *,
        factor_name: str,
        expression: str,
        signal_series: pd.Series,
        label_series: pd.Series,
        panel: pd.DataFrame,
    ) -> dict[str, float]:
        """计算单个因子的完整指标包（原始 pandas groupby 版本，作为回退路径）。

        在 _calculate_single_factor_metrics 基础上追加：
        - 分层回测指标（_calculate_layer_returns）：各层收益、多空收益、Sharpe、最大回撤、单调性
        - 辅助指标：IC_abs、IC_sign、Rank IC_abs、ICIR_abs、Rank ICIR_abs、sharpe_abs
        """
        signal_series = pd.to_numeric(signal_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if signal_series.dropna().empty:
            raise ValueError("factor produced empty or all-null signal")
        _debug_print(f"[DEBUG] Factor '{factor_name}' calculated successfully, non-null values: {signal_series.notna().sum()}")
        single_metrics = self._calculate_single_factor_metrics(signal_series, label_series, panel)
        # ★ 中低耗时：5 层分层收益 + 多空 Sharpe/回撤/单调性，~0.5-1.5s
        rank_ic = self._finite_float(single_metrics.get("Rank IC"), 0.0)
        direction_sign = 1.0 if rank_ic > 0 else (-1.0 if rank_ic < 0 else None)
        layer_returns = self._calculate_layer_returns(
            signal_series,
            label_series,
            direction_sign=direction_sign,
        )
        single_metrics.update(layer_returns)
        single_metrics["IC_abs"] = abs(single_metrics.get("IC", 0.0))
        single_metrics["IC_sign"] = 1.0 if single_metrics.get("IC", 0.0) >= 0 else -1.0
        single_metrics["Rank IC_abs"] = abs(single_metrics.get("Rank IC", 0.0))
        single_metrics["Rank IC_sign"] = 1.0 if single_metrics.get("Rank IC", 0.0) >= 0 else -1.0
        single_metrics["ICIR_abs"] = abs(single_metrics.get("ICIR", 0.0))
        single_metrics["Rank ICIR_abs"] = abs(single_metrics.get("Rank ICIR", 0.0))
        single_metrics["sharpe_abs"] = abs(single_metrics.get("sharpe", 0.0))
        _debug_print(f"[DEBUG] Factor '{factor_name}' metrics: IC={single_metrics.get('IC', 0):.4f}, Rank IC={single_metrics.get('Rank IC', 0):.4f}")
        return single_metrics

    def _select_summary_metrics(self, per_factor_metrics: dict[str, dict[str, float]]) -> tuple[dict[str, float], str]:
        """从所有因子指标中选出最佳单因子（按目标 reward 降序），返回其指标和名称。

        注意：这是"最佳单因子"指标，不是多因子组合回测结果。
        mainline 因子绝对优先：轨迹摘要代表假说本身的成绩，
        不能让搭车的 adjacent/counterfactual 对照因子顶替（它们跨方向重复，
        会让多个假说显示出一模一样的"最佳"指标并污染父代选择）。
        """
        if not per_factor_metrics:
            return {}, ""
        selected_name, selected_metrics = max(
            per_factor_metrics.items(),
            key=lambda item: (
                self._finite_float(item[1].get("mainline_factor", 1.0)),
                self._finite_float(item[1].get("robustness_pass", 1.0)),
                self._finite_float(item[1].get("target_passed", 0.0)),
                self._reward_score(dict(item[1])),
                abs(self._finite_float(item[1].get("IC", 0.0))),
                abs(self._finite_float(item[1].get("Rank IC", item[1].get("RankIC", 0.0)))),
            ),
        )
        return self._finite_metrics(selected_metrics), selected_name

    def _cross_sectional_corr(self, eval_frame: pd.DataFrame, *, rank: bool) -> pd.Series:
        """按日期分组计算截面相关系数（IC 或 Rank IC），返回每个时间步的相关系数序列。

        耗时瓶颈：groupby().apply() 对每个交易日创建子 DataFrame 再调用 spearman corr，
        固定开销约 0.5-2s（取决于交易日数 T 和截面资产数 N）。
        优化方向：可用 pd.DataFrame.corrwith() 或手动向量化 Spearman 排名后批量计算。
        """
        return eval_frame.groupby(level="datetime").apply(
            lambda frame: self._safe_corr(frame["signal"], frame["label"], rank=rank)
        )

    def _safe_corr(self, signal: pd.Series, label: pd.Series, *, rank: bool) -> float:
        """安全计算两个序列的相关系数。样本 <2 或唯一值不足时返回 NaN。

        rank=True 时先做百分位排名再计算 Spearman 相关（即 Rank IC）。
        注意：IC 是截面相关系数，≥2 个标的即有定义，最小样本数与分层（需 ≥5）不同，
        因此这里保持 2，不随分层阈值变化。
        """
        frame = pd.concat([signal.rename("signal"), label.rename("label")], axis=1).dropna()
        if len(frame) < 2:
            return float("nan")

        signal_values = frame["signal"]
        label_values = frame["label"]
        if signal_values.nunique(dropna=True) < 2 or label_values.nunique(dropna=True) < 2:
            return float("nan")

        # IC = Pearson（原始值线性相关）；Rank IC = 排名后再 Pearson（等价 Spearman）。
        # 关键修复：此前 IC/Rank IC 都用 spearman，而 spearman 对单调变换不变，
        # 导致 spearman(x,y) == spearman(rank(x),rank(y))，两个指标恒等。
        if rank:
            signal_values = signal_values.rank(pct=True)
            label_values = label_values.rank(pct=True)
        value = signal_values.corr(label_values, method="pearson")
        if pd.isna(value):
            return float("nan")
        value = float(value)
        if np.isclose(value, 1.0, atol=1e-12):
            return 1.0
        if np.isclose(value, -1.0, atol=1e-12):
            return -1.0
        return value

    def _max_abs_signal_corr(self, signal: pd.Series, existing: Any) -> float:
        """计算 signal 与已保留因子信号的最大绝对相关系数（构造阶段冗余门用）。"""
        max_corr = 0.0
        for other in existing:
            if other is None:
                continue
            frame = (
                pd.concat([signal.rename("candidate"), other.rename("selected")], axis=1)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(frame) < 3 or frame["candidate"].nunique() < 2 or frame["selected"].nunique() < 2:
                continue
            # 先做百分位排名再取 Pearson，等价于 Spearman，且无需 scipy 依赖。
            corr = frame["candidate"].rank(pct=True).corr(frame["selected"].rank(pct=True))
            if corr is not None and np.isfinite(corr):
                max_corr = max(max_corr, abs(float(corr)))
        return max_corr

    def _build_layered_factor_frame(
        self,
        signal_series: pd.Series,
        label_series: pd.Series,
        n_layers: int = 5,
    ) -> pd.DataFrame:
        """构建分层因子框架：按日期分组，将截面内的资产按信号排名等分为 5 层。

        每个资产被标记其所属层（1=最弱信号, 5=最强信号），
        用于后续计算各层平均收益和多空对冲收益。

        耗时瓶颈：groupby().apply(assign_layer) 逐日创建子 DataFrame 做 qcut，
        固定开销约 0.3-1s。
        优化方向：可一次性 rank + 向量化分组，避免逐日 apply。
        """
        factor_frame = (
            pd.concat([signal_series.rename("signal"), label_series.rename("label")], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if factor_frame.empty:
            return factor_frame

        def assign_layer(group: pd.DataFrame) -> pd.Series:
            if len(group) < n_layers:
                # 薄横截面时间点剔除（NaN = 不计入任何层），与矩阵路径一致；
                # 旧逻辑是全塞 L1，会用退化分组污染分层收益。
                return pd.Series([np.nan] * len(group), index=group.index, dtype=float)
            ranked = group["signal"].rank(method="first")
            return pd.qcut(ranked, n_layers, labels=False, duplicates="drop") + 1

        # ★ 中低耗时：逐日 apply 分层，~0.3-1s
        layer_result = factor_frame.groupby(level="datetime", group_keys=False).apply(assign_layer)
        factor_frame["layer"] = pd.Series(layer_result, index=factor_frame.index)
        return factor_frame

    def _calculate_layer_turnover_cost(
        self,
        factor_frame: pd.DataFrame,
        *,
        low_layer: int,
        high_layer: int,
    ) -> pd.Series:
        """Estimate per-rebalance long-short transaction cost for top/bottom layers."""
        if self.one_way_fee_rate <= 0 or factor_frame.empty:
            return pd.Series(dtype=float)

        positions: list[tuple[Any, pd.Series]] = []
        datetimes: list[Any] = []
        for dt, frame in factor_frame.groupby(level="datetime", sort=True):
            symbols = frame.index.get_level_values("symbol")
            layers = pd.Series(frame["layer"].to_numpy(), index=symbols)
            low_symbols = list(layers[layers == low_layer].index)
            high_symbols = list(layers[layers == high_layer].index)
            if not low_symbols or not high_symbols:
                continue
            weights = pd.Series(0.0, index=pd.Index(sorted(set(low_symbols + high_symbols)), name="symbol"))
            weights.loc[high_symbols] = 0.5 / max(1, len(high_symbols))
            weights.loc[low_symbols] = -0.5 / max(1, len(low_symbols))
            positions.append((dt, weights))
            datetimes.append(dt)

        if not positions:
            return pd.Series(dtype=float)

        cost_values: list[float] = []
        previous = pd.Series(dtype=float)
        for _, weights in positions:
            aligned_index = previous.index.union(weights.index)
            turnover = float((weights.reindex(aligned_index, fill_value=0.0) - previous.reindex(aligned_index, fill_value=0.0)).abs().sum())
            cost_values.append(turnover * self.one_way_fee_rate)
            previous = weights
        return pd.Series(cost_values, index=pd.Index(datetimes, name="datetime"), dtype=float)

    def _calculate_long_short_performance(
        self,
        factor_frame: pd.DataFrame,
        n_layers: int = 5,
        direction_sign: float | None = None,
    ) -> tuple[dict[str, float], pd.Series]:
        """计算多空对冲绩效指标。

        策略：做多第 5 层（最强信号），做空第 1 层（最弱信号）。
        计算指标：
        - long_short_mean_return：日度多空收益均值
        - long_short_positive_ratio：正收益日占比（胜率）
        - sharpe：年化 Sharpe（均值/标准差 * sqrt(年化因子)）
        - drawdown：最大回撤（基于累积净值曲线的峰谷比）

        Returns:
            (performance_dict, long_short_returns_series)
        """
        if factor_frame.empty or "layer" not in factor_frame.columns:
            return {
                "long_short_mean_return": 0.0,
                "long_short_positive_ratio": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "raw_long_short_mean_return": 0.0,
                "raw_long_short_positive_ratio": 0.0,
                "raw_drawdown": 0.0,
                "raw_sharpe": 0.0,
                "long_short_orientation": "high_minus_low",
                "direction_aligned_by_rank_ic": False,
                "gross_sharpe": 0.0,
                "gross_raw_sharpe": 0.0,
                "net_sharpe": 0.0,
                "net_raw_sharpe": 0.0,
                "transaction_cost_mean": 0.0,
                "raw_transaction_cost_mean": 0.0,
                "long_short_turnover": 0.0,
                "one_way_fee_rate": self.one_way_fee_rate,
            }, pd.Series(dtype=float)

        layer_returns_by_dt = (
            factor_frame.groupby([factor_frame.index.get_level_values("datetime"), "layer"])["label"]
            .mean()
            .unstack("layer")
            .sort_index()
        )
        if layer_returns_by_dt.empty:
            return {
                "long_short_mean_return": 0.0,
                "long_short_positive_ratio": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "raw_long_short_mean_return": 0.0,
                "raw_long_short_positive_ratio": 0.0,
                "raw_drawdown": 0.0,
                "raw_sharpe": 0.0,
                "long_short_orientation": "high_minus_low",
                "direction_aligned_by_rank_ic": False,
                "gross_sharpe": 0.0,
                "gross_raw_sharpe": 0.0,
                "net_sharpe": 0.0,
                "net_raw_sharpe": 0.0,
                "transaction_cost_mean": 0.0,
                "raw_transaction_cost_mean": 0.0,
                "long_short_turnover": 0.0,
                "one_way_fee_rate": self.one_way_fee_rate,
            }, pd.Series(dtype=float)

        if 1 not in layer_returns_by_dt.columns or n_layers not in layer_returns_by_dt.columns:
            return {
                "long_short_mean_return": 0.0,
                "long_short_positive_ratio": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "raw_long_short_mean_return": 0.0,
                "raw_long_short_positive_ratio": 0.0,
                "raw_drawdown": 0.0,
                "raw_sharpe": 0.0,
                "long_short_orientation": "high_minus_low",
                "direction_aligned_by_rank_ic": False,
                "gross_sharpe": 0.0,
                "gross_raw_sharpe": 0.0,
                "net_sharpe": 0.0,
                "net_raw_sharpe": 0.0,
                "transaction_cost_mean": 0.0,
                "raw_transaction_cost_mean": 0.0,
                "long_short_turnover": 0.0,
                "one_way_fee_rate": self.one_way_fee_rate,
            }, pd.Series(dtype=float)

        long_short_returns = (layer_returns_by_dt[n_layers] - layer_returns_by_dt[1]).dropna().sort_index()
        turnover_cost = self._calculate_layer_turnover_cost(factor_frame, low_layer=1, high_layer=n_layers).reindex(
            long_short_returns.index,
            fill_value=0.0,
        )
        # Sample non-overlapping forward-return windows so Sharpe matches the label horizon.
        long_short_returns = self._subsample_returns_to_horizon(long_short_returns)
        turnover_cost = turnover_cost.reindex(long_short_returns.index, fill_value=0.0)
        benchmark_returns = factor_frame["label"].groupby(level="datetime").mean().dropna().sort_index()
        benchmark_returns = self._subsample_returns_to_horizon(benchmark_returns)
        if long_short_returns.empty:
            return {
                "long_short_mean_return": 0.0,
                "long_short_positive_ratio": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "raw_long_short_mean_return": 0.0,
                "raw_long_short_positive_ratio": 0.0,
                "raw_drawdown": 0.0,
                "raw_sharpe": 0.0,
                "long_short_orientation": "high_minus_low",
                "direction_aligned_by_rank_ic": False,
                "gross_sharpe": 0.0,
                "gross_raw_sharpe": 0.0,
                "net_sharpe": 0.0,
                "net_raw_sharpe": 0.0,
                "transaction_cost_mean": 0.0,
                "raw_transaction_cost_mean": 0.0,
                "long_short_turnover": 0.0,
                "one_way_fee_rate": self.one_way_fee_rate,
            }, long_short_returns

        gross_long_short_returns = long_short_returns
        net_long_short_returns = gross_long_short_returns - turnover_cost
        raw_long_short_mean_return = float(net_long_short_returns.mean())
        raw_positive_ratio = float((net_long_short_returns > 0).mean())
        raw_long_short_std = float(net_long_short_returns.std())
        raw_sharpe = (
            float(raw_long_short_mean_return / raw_long_short_std) * np.sqrt(self._current_sharpe_annualization_factor)
            if abs(raw_long_short_std) > 1e-12
            else 0.0
        )
        raw_cumulative = (1.0 + net_long_short_returns).cumprod()
        raw_rolling_peak = raw_cumulative.cummax()
        raw_drawdown_series = raw_cumulative / raw_rolling_peak - 1.0
        raw_max_drawdown = float(abs(raw_drawdown_series.min())) if not raw_drawdown_series.empty else 0.0
        gross_raw_mean_return = float(gross_long_short_returns.mean())
        gross_raw_std = float(gross_long_short_returns.std())
        gross_raw_sharpe = (
            float(gross_raw_mean_return / gross_raw_std) * np.sqrt(self._current_sharpe_annualization_factor)
            if abs(gross_raw_std) > 1e-12
            else 0.0
        )

        orientation_multiplier = -1.0 if direction_sign is not None and direction_sign < 0 else 1.0
        gross_oriented_returns = gross_long_short_returns * orientation_multiplier
        oriented_returns = gross_oriented_returns - turnover_cost
        long_short_mean_return = float(oriented_returns.mean())
        positive_ratio = float((oriented_returns > 0).mean())
        long_short_std = float(oriented_returns.std())
        sharpe = (
            float(long_short_mean_return / long_short_std) * np.sqrt(self._current_sharpe_annualization_factor)
            if abs(long_short_std) > 1e-12
            else 0.0
        )
        gross_long_short_mean_return = float(gross_oriented_returns.mean())
        gross_long_short_std = float(gross_oriented_returns.std())
        gross_sharpe = (
            float(gross_long_short_mean_return / gross_long_short_std) * np.sqrt(self._current_sharpe_annualization_factor)
            if abs(gross_long_short_std) > 1e-12
            else 0.0
        )
        cumulative = (1.0 + oriented_returns).cumprod()
        rolling_peak = cumulative.cummax()
        drawdown_series = cumulative / rolling_peak - 1.0
        max_drawdown = float(abs(drawdown_series.min())) if not drawdown_series.empty else 0.0
        performance = {
            "long_short_mean_return": self._finite_float(long_short_mean_return),
            "gross_long_short_mean_return": self._finite_float(gross_long_short_mean_return),
            "long_short_positive_ratio": self._finite_float(positive_ratio),
            "drawdown": self._finite_float(max_drawdown),
            "sharpe": self._finite_float(sharpe),
            "net_sharpe": self._finite_float(sharpe),
            "gross_sharpe": self._finite_float(gross_sharpe),
            "raw_long_short_mean_return": self._finite_float(raw_long_short_mean_return),
            "gross_raw_long_short_mean_return": self._finite_float(gross_raw_mean_return),
            "raw_long_short_positive_ratio": self._finite_float(raw_positive_ratio),
            "raw_drawdown": self._finite_float(raw_max_drawdown),
            "raw_sharpe": self._finite_float(raw_sharpe),
            "net_raw_sharpe": self._finite_float(raw_sharpe),
            "gross_raw_sharpe": self._finite_float(gross_raw_sharpe),
            "transaction_cost_mean": self._finite_float(float(turnover_cost.mean()) if not turnover_cost.empty else 0.0),
            "raw_transaction_cost_mean": self._finite_float(float(turnover_cost.mean()) if not turnover_cost.empty else 0.0),
            "long_short_turnover": self._finite_float(float(turnover_cost.mean() / self.one_way_fee_rate) if self.one_way_fee_rate > 0 and not turnover_cost.empty else 0.0),
            "one_way_fee_rate": self._finite_float(self.one_way_fee_rate),
            "long_short_orientation": "low_minus_high" if orientation_multiplier < 0 else "high_minus_low",
            "direction_aligned_by_rank_ic": direction_sign is not None,
        }
        performance.update(
            self._calculate_market_neutral_robustness(
                oriented_net_returns=oriented_returns,
                benchmark_returns=benchmark_returns.reindex(oriented_returns.index),
            )
        )
        return performance, oriented_returns

    def _calculate_layer_returns(
        self,
        signal_series: pd.Series,
        label_series: pd.Series,
        n_layers: int = 5,
        direction_sign: float | None = None,
    ) -> dict[str, float]:
        """计算分层收益指标。

        返回：
        - layer_1~5_mean_return：各层平均收益（检验因子单调性）
        - long_short_mean_return / positive_ratio / drawdown / sharpe：多空对冲指标
        - layer_monotonicity：单调性比例（0.0~1.0，越高越好）
          计算方式：统计相邻层收益递增的次数占比
        """
        factor_frame = self._build_layered_factor_frame(signal_series, label_series, n_layers=n_layers)
        if factor_frame.empty:
            return {}

        layer_returns: dict[str, float] = {}
        layer_returns_by_dt = (
            factor_frame.groupby([factor_frame.index.get_level_values("datetime"), "layer"])["label"]
            .mean()
            .unstack("layer")
            .sort_index()
        )
        if not layer_returns_by_dt.empty:
            # Keep layer-level summary statistics on the same non-overlapping horizon
            # used by Sharpe and the report plots.
            layer_returns_by_dt = self._subsample_frame_to_horizon(layer_returns_by_dt)

        for layer in range(1, n_layers + 1):
            if layer in layer_returns_by_dt.columns:
                layer_data = layer_returns_by_dt[layer].dropna()
                if layer_data.empty:
                    continue
                layer_mean_return = float(layer_data.mean())
                layer_returns[f"layer_{layer}_mean_return"] = layer_mean_return

        performance_metrics, ls_series = self._calculate_long_short_performance(
            factor_frame,
            n_layers=n_layers,
            direction_sign=direction_sign,
        )
        layer_returns.update(performance_metrics)

        layer_returns["layer_monotonicity"] = 0.0
        layer_keys = [f"layer_{i}_mean_return" for i in range(1, n_layers + 1)]
        if all(k in layer_returns for k in layer_keys):
            values = [layer_returns[k] for k in layer_keys]
            monotonic = sum(1 for i in range(len(values) - 1) if values[i] < values[i + 1])
            layer_returns["layer_monotonicity"] = monotonic / (n_layers - 1) if n_layers > 1 else 0.0

        # 把多空累计收益曲线（采样至 ≤300 点）写入 metrics，供前端绘图
        if ls_series is not None and not ls_series.empty:
            try:
                step = max(1, len(ls_series) // 300)
                ls_sampled = ls_series.iloc[::step]
                cum = (1.0 + ls_sampled).cumprod()
                layer_returns["ls_cumulative_series"] = [  # type: ignore[assignment]
                    [str(t), round(float(v), 6)]
                    for t, v in zip(cum.index, cum.values)
                    if pd.notna(v)
                ]
            except Exception:
                pass

        # 做多组（Layer 5）和做空组（Layer 1）各自的累计净值曲线
        for layer_key, series_name in [(1, "layer_1_cumulative_series"), (n_layers, "layer_5_cumulative_series")]:
            if layer_key in layer_returns_by_dt.columns:
                try:
                    col = layer_returns_by_dt[layer_key].dropna()
                    if not col.empty:
                        step_l = max(1, len(col) // 300)
                        sampled_l = col.iloc[::step_l]
                        cum_l = (1.0 + sampled_l).cumprod()
                        layer_returns[series_name] = [  # type: ignore[assignment]
                            [str(t), round(float(v), 6)]
                            for t, v in zip(cum_l.index, cum_l.values)
                            if pd.notna(v)
                        ]
                except Exception:
                    pass

        # 年化收益率 + 逐年收益率分解
        ls_mean = layer_returns.get("long_short_mean_return", 0.0)
        if isinstance(ls_mean, float) and ls_mean != 0.0:
            layer_returns["annualized_return"] = self._finite_float(
                ls_mean * self._current_sharpe_annualization_factor
            )
        if ls_series is not None and not ls_series.empty:
            try:
                yearly = self._compute_yearly_returns(ls_series)
                if yearly:
                    layer_returns["yearly_returns"] = yearly  # type: ignore[assignment]
            except Exception:
                pass

        return layer_returns

    def evaluate_with_qlib(
        self,
        *,
        factor_implementation: dict[str, object],
        calculation_report: dict[str, object],
        panel_data_path: str,
    ) -> tuple[dict[str, object], dict[str, float], dict[str, dict[str, float]]] | None:
        """因子回测主入口。

        完整流程：
        1. 初始化 Qlib（懒加载，仅首次）
        2. 加载面板数据，提取目标列（未来收益）作为 label
        3. 筛选 status=="generated" 的因子实现
        4. 按表达式去重分组（相同表达式只计算一次）
        5. 对每组因子：
           a. 优先从 pickle 缓存加载信号（FactorCoderAgent 预计算）
           b. 缓存未命中时，实时计算因子表达式
           c. 计算完整指标包（IC/ICIR/RankIC/分层收益等）
        6. 汇总所有因子指标，选出最佳单因子
        7. 构建回测报告

        Returns:
            (backtest_report, summary_metrics, per_factor_metrics) 或 None（初始化失败时）
        """
        self._set_error("")
        try:
            from qlib.contrib.eva.alpha import calc_ic  # type: ignore
        except Exception:
            self._set_error("qlib_dependency_unavailable")
            return None
        # ── 步骤 1：加载面板数据 ── 低耗时 ~0.5-2s ──
        prepared = self._prepare_panel(panel_data_path)
        if prepared is None:
            self._set_error("invalid_panel_or_target_column")
            return None
        full_panel, target_column = prepared
        self._ensure_sample_windows_resolved(full_panel)
        # 验证窗口（25% holdout）面板；信号在 full_panel 上计算以保证验证段有正确的
        # 滚动回看历史（与 FactorCoder 预计算的整段信号缓存口径一致）。
        validation_panel: pd.DataFrame | None = None
        if self.sample_windows_enabled:
            train_mask = self._build_sample_window_mask(full_panel, self.train_window)
            panel = full_panel.iloc[np.flatnonzero(train_mask)].copy()
            if panel.empty:
                self._set_error("empty_train_sample_window")
                return None
            validation_mask = self._build_sample_window_mask(full_panel, self.validation_window)
            candidate_validation = full_panel.iloc[np.flatnonzero(validation_mask)].copy()
            # 验证段至少要有 2 个时间点才有意义，否则视为数据不足、跳过 val 评估。
            if not candidate_validation.empty and "datetime" in candidate_validation.index.names:
                unique_val_dt = pd.Index(
                    pd.to_datetime(candidate_validation.index.get_level_values("datetime"), errors="coerce")
                ).dropna().unique()
                if len(unique_val_dt) >= 2:
                    validation_panel = candidate_validation
        else:
            panel = full_panel
        self._configure_sharpe_sampling(panel)
        # ── 步骤 2：筛选 generated 因子 ── 极低耗时 ──
        generated_impls = self._collect_generated_impls(factor_implementation)
        if not generated_impls:
            self._set_error("no_generated_factor_implementations")
            return None
        # ── 步骤 3：初始化 Qlib（懒加载，仅首次）── 首次 ~0.1s，后续 ~0s ──
        selected_config_name = self.baseline_config_name
        selected_config_path = self.qlib_config_dir / selected_config_name
        try:
            qlib_task_config = self._load_qlib_task_config(selected_config_path)
            self._ensure_qlib_initialized(qlib_task_config)
        except Exception as exc:
            self._set_error(f"qlib_config_or_init_failed:{exc}")
            return None

        # train / validation 两段各自的 label（fwd 收益）。指标按窗口分别计算。
        label_series = pd.to_numeric(panel[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan).rename("label")
        validation_label_series = (
            pd.to_numeric(validation_panel[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan).rename("label")
            if validation_panel is not None
            else None
        )

        # ── 安全防护：构造不含未来特征列的 feature_panel ──────────────────────────
        # signal_exclude_prefixes（默认 ["fwd_"]）过滤掉所有未来收益列，防止
        # AI 生成的因子表达式通过 $fwd_returns_4h 等变量引用未来数据（look-ahead bias）。
        # 注意：信号在整段 full_panel 上计算（与预计算缓存一致），保证验证段有正确的
        # 滚动回看历史；随后按 train/val 的 label 分别取交集计算指标。
        signal_source_panel = full_panel
        feature_cols = [c for c in signal_source_panel.columns if not c.startswith(self.signal_exclude_prefixes)]
        feature_panel = signal_source_panel[feature_cols] if feature_cols else signal_source_panel

        per_factor_metrics: dict[str, dict[str, float]] = {}
        # ── 步骤 4：收集失败因子错误信息 ── 极低耗时 ──
        factor_errors = self._collect_failure_errors(
            factor_implementation=factor_implementation,
            calculation_report=calculation_report,
        )
        # ── 步骤 5：按表达式去重分组 ── 极低耗时 ──
        expression_groups: dict[str, list[dict[str, object]]] = {}
        missing_expression_count = 0
        for idx, impl in enumerate(generated_impls):
            factor_name = str(impl.get("factor_name", impl.get("name", f"factor_{idx}")))
            expression = str(impl.get("expression", "")).strip()
            if not expression:
                missing_expression_count += 1
                factor_errors[factor_name] = {
                    "stage": "backtest",
                    "status": "missing_expression",
                    "message": "Generated implementation has no expression.",
                }
                _debug_print(f"[DEBUG] Factor '{factor_name}' has no expression")
                continue
            expression_groups.setdefault(self._normalize_expression_key(expression), []).append(impl)

        # ── 共享 MatrixPanel ───────────────────────────────────────────────
        # 同一回测内所有因子用的是同一个 feature_panel，(T,N) 稠密矩阵只构建一次，
        # 只读共享给各（并行）因子，避免逐因子重复构建——显著降低峰值内存且更快。
        # 构建失败时回退为 None，各因子自行构建（calculate_factor_expression 内部）。
        shared_matrix_panel = None
        if expression_groups:
            try:
                shared_matrix_panel = build_matrix_panel(feature_panel)
            except Exception as exc:  # pragma: no cover - defensive
                _debug_print(f"[WARN] build shared MatrixPanel failed: {exc}; per-factor build fallback")
                shared_matrix_panel = None

        def evaluate_group(
            key: str,
            group: list[dict[str, object]],
        ) -> tuple[str, dict[str, object] | None, dict[str, object] | None]:
            """评估一组共享相同表达式的因子（去重后每组只需计算一次信号）。

            Returns:
                (expression_key, result_dict, error_dict)
                - 成功时 error_dict 为 None
                - 失败时 result_dict 为 None
                """
            representative = group[0]
            factor_name = str(representative.get("factor_name", representative.get("name", "factor")))
            expression = str(representative.get("expression", "")).strip()
            try:
                # ── 表达式级别的未来变量引用检查 ──────────────────────────────────
                # 检测表达式中是否显式引用了被排除的未来列（如 $fwd_returns_4h），
                # 作为防护的第二层（即使 feature_panel 已经过滤，这里再显式报错）。
                # 同时检查 function_token_blacklist 中配置的禁用函数 token。
                referenced_vars = re.findall(r'\$([A-Za-z_][A-Za-z0-9_]*)', expression)
                lookahead_vars = [
                    v for v in referenced_vars
                    if v.startswith(tuple(p.lstrip("$") for p in self.signal_exclude_prefixes))
                ]
                if lookahead_vars:
                    raise ValueError(
                        f"Look-ahead bias detected: expression references future-data columns "
                        f"{lookahead_vars}. These columns are excluded by signal_exclude_prefixes."
                    )
                if self.function_token_blacklist:
                    expr_tokens = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', expression.lower()))
                    blacklisted_used = expr_tokens & self.function_token_blacklist
                    if blacklisted_used:
                        raise ValueError(
                            f"Expression uses blacklisted tokens: {sorted(blacklisted_used)}"
                        )
                _debug_print(f"[DEBUG] Calculating factor '{factor_name}' with expression: {expression}")
                # 进程级信号量：限制并发的重型评估（信号计算 + 指标矩阵），无论进化/
                # 回测嵌套并行多深，同时构建 (T,N) 矩阵的因子数都被钳制在 ~CPU 核数。
                with _FACTOR_EVAL_SEMAPHORE:
                    signal_series = self._load_signal_from_cache(representative, feature_panel)
                    if signal_series is not None:
                        _debug_print(f"[DEBUG] Loaded cached signal for factor '{factor_name}'")
                    else:
                        signal_series = calculate_factor_expression(
                            expression, feature_panel, factor_name, matrix_panel=shared_matrix_panel
                        )
                    if signal_series is None:
                        raise ValueError("factor calculation returned None")
                    # ── train 窗口指标（入库门槛主依据）──
                    metrics = self._calculate_factor_metrics_bundle(
                        factor_name=factor_name,
                        expression=expression,
                        signal_series=signal_series,
                        label_series=label_series,
                        panel=panel,
                    )
                    # ── validation 窗口指标（同一信号，按 val label 取交集计算）──
                    validation_metrics: dict[str, float] | None = None
                    if validation_panel is not None and validation_label_series is not None:
                        try:
                            validation_metrics = self._calculate_factor_metrics_bundle(
                                factor_name=factor_name,
                                expression=expression,
                                signal_series=signal_series,
                                label_series=validation_label_series,
                                panel=validation_panel,
                            )
                        except Exception as val_exc:
                            _debug_print(f"[WARN] Validation metrics failed for '{factor_name}': {val_exc}")
                            validation_metrics = None
                train_filter_decision = self._build_train_filter_decision(metrics)
                target_filter_decision = self._target_status(metrics)
                coverage_filter_decision = self._build_coverage_filter_decision(metrics)
                robustness_filter_decision = self._build_robustness_filter_decision(metrics)
                validation_decay_decision = self._build_validation_decay_decision(
                    metrics,
                    validation_metrics,
                    validation_available=validation_metrics is not None,
                )
                return key, {
                    "metrics": metrics,
                    "validation_metrics": validation_metrics,
                    "train_filter_decision": train_filter_decision,
                    "target_filter_decision": target_filter_decision,
                    "coverage_filter_decision": coverage_filter_decision,
                    "robustness_filter_decision": robustness_filter_decision,
                    "validation_decay_decision": validation_decay_decision,
                    "signal_series": signal_series,
                }, None
            except Exception as exc:
                return key, None, {
                    "stage": "backtest",
                    "status": "failed",
                    "factor_name": factor_name,
                    "expression": expression,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                }

        # 确定并行线程数，并根据是否为单线程选择串行或并行执行
        workers = self._resolve_parallel_workers(len(expression_groups))
        _debug_print(f"[DEBUG] Backtest unique expressions={len(expression_groups)}, parallel_workers={workers}")
        group_results: dict[str, dict[str, object]] = {}
        group_failures: dict[str, dict[str, object]] = {}
        if workers == 1:
            for key, group in expression_groups.items():
                result_key, result, error = evaluate_group(key, group)
                if result is not None:
                    group_results[result_key] = result
                elif error is not None:
                    group_failures[result_key] = error
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(evaluate_group, key, group): key
                    for key, group in expression_groups.items()
                }
                for future in as_completed(futures):
                    result_key, result, error = future.result()
                    if result is not None:
                        group_results[result_key] = result
                    elif error is not None:
                        group_failures[result_key] = error

        # 将去重后的指标结果回填到每个因子（同一表达式的多个因子共享指标）
        expression_cache_hits = 0
        per_factor_validation_metrics: dict[str, dict[str, float]] = {}
        train_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        target_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        coverage_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        robustness_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        validation_decay_decisions_by_factor: dict[str, dict[str, object]] = {}
        redundancy_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        retained_factor_names: list[str] = []
        filtered_factor_names: list[str] = []
        target_filtered_factor_names: list[str] = []
        coverage_filtered_factor_names: list[str] = []
        decay_filtered_factor_names: list[str] = []
        # 通过 train+robustness 门后的候选因子，待进入构造阶段冗余门
        redundancy_candidates: list[dict[str, object]] = []
        retained_signals_by_factor: dict[str, pd.Series] = {}
        for key, group in expression_groups.items():
            if len(group) > 1:
                expression_cache_hits += len(group) - 1
            result = group_results.get(key)
            error = group_failures.get(key)
            for impl in group:
                factor_name = str(impl.get("factor_name", impl.get("name", ""))).strip()
                if result is not None:
                    metrics = dict(result.get("metrics", {}))
                    target_filter_decision = dict(result.get("target_filter_decision", {}))
                    if target_filter_decision:
                        metrics["target_passed"] = 1.0 if bool(target_filter_decision.get("passed", False)) else 0.0
                        metrics["target_sharpe_gap"] = self._finite_float(target_filter_decision.get("sharpe_gap"))
                        metrics["target_ic_gap"] = self._finite_float(target_filter_decision.get("ic_gap"))
                        metrics["target_rank_ic_gap"] = self._finite_float(target_filter_decision.get("rank_ic_gap"))
                        metrics["target_sharpe_threshold"] = self._finite_float(target_filter_decision.get("target_sharpe"))
                        metrics["target_abs_ic_threshold"] = self._finite_float(target_filter_decision.get("target_abs_ic"))
                        metrics["target_abs_rank_ic_threshold"] = self._finite_float(target_filter_decision.get("target_abs_rank_ic"))
                    metrics["reward_score"] = self._reward_score(metrics)
                    # 因子组别（mainline=1 / adjacent、counterfactual=0）：
                    # 资格赛、轨迹摘要、反馈判定只认 mainline——对照/邻近因子不得替假说过关。
                    alignment = str(impl.get("theory_alignment", "mainline") or "mainline").strip().lower()
                    metrics["mainline_factor"] = 1.0 if alignment == "mainline" else 0.0
                    train_filter_decision = dict(result.get("train_filter_decision", {}))
                    per_factor_metrics[factor_name] = {
                        str(metric_name): self._finite_float(metric_value)
                        for metric_name, metric_value in metrics.items()
                    }
                    target_filter_decisions_by_factor[factor_name] = target_filter_decision
                    train_filter_decisions_by_factor[factor_name] = train_filter_decision
                    coverage_filter_decision = dict(result.get("coverage_filter_decision", {}))
                    coverage_filter_decisions_by_factor[factor_name] = coverage_filter_decision
                    robustness_filter_decision = dict(result.get("robustness_filter_decision", {}))
                    robustness_filter_decisions_by_factor[factor_name] = robustness_filter_decision
                    validation_metrics = result.get("validation_metrics")
                    if isinstance(validation_metrics, dict):
                        per_factor_validation_metrics[factor_name] = {
                            str(metric_name): self._finite_float(metric_value)
                            for metric_name, metric_value in validation_metrics.items()
                        }
                    validation_decay_decision = dict(result.get("validation_decay_decision", {}))
                    validation_decay_decisions_by_factor[factor_name] = validation_decay_decision
                    train_passed = not self.sample_windows_enabled or bool(train_filter_decision.get("passed", False))
                    target_passed = bool(target_filter_decision.get("passed", False))
                    coverage_passed = (
                        not self.coverage_filter_enabled
                        or bool(coverage_filter_decision.get("passed", False))
                    )
                    robustness_passed = (
                        not self.robustness_filter_enabled
                        or bool(robustness_filter_decision.get("passed", False))
                    )
                    # 留样验证衰减门：val 相对 train 衰减过大者不入库。
                    decay_passed = (
                        not (self.sample_windows_enabled and self.validation_decay_enabled)
                        or bool(validation_decay_decision.get("passed", True))
                    )
                    if target_passed and train_passed and coverage_passed and robustness_passed and decay_passed:
                        rank_ic = self._finite_float(metrics.get("Rank IC", metrics.get("RankIC")))
                        redundancy_candidates.append(
                            {
                                "factor_name": factor_name,
                                "abs_rank_ic": abs(rank_ic) if np.isfinite(rank_ic) else 0.0,
                                "reward_score": self._reward_score(metrics),
                                "signal": result.get("signal_series"),
                            }
                        )
                    else:
                        filtered_factor_names.append(factor_name)
                        if not target_passed:
                            target_filtered_factor_names.append(factor_name)
                        if not coverage_passed:
                            coverage_filtered_factor_names.append(factor_name)
                        if train_passed and coverage_passed and robustness_passed and not decay_passed:
                            decay_filtered_factor_names.append(factor_name)
                elif error is not None:
                    factor_error = dict(error)
                    factor_error["factor_name"] = factor_name
                    factor_errors[factor_name] = factor_error

        # ── 构造阶段冗余门：按 target reward 降序贪心，剔除与已保留因子高度相关者 ──
        redundancy_candidates.sort(key=lambda item: float(item.get("reward_score", 0.0)), reverse=True)
        for candidate in redundancy_candidates:
            factor_name = str(candidate["factor_name"])
            signal = candidate.get("signal")
            if not self.redundancy_filter_enabled or signal is None:
                retained_factor_names.append(factor_name)
                if signal is not None:
                    retained_signals_by_factor[factor_name] = signal
                continue
            max_abs_corr = self._max_abs_signal_corr(signal, retained_signals_by_factor.values())
            passed = max_abs_corr < float(self.redundancy_corr_threshold)
            redundancy_filter_decisions_by_factor[factor_name] = {
                "passed": bool(passed),
                "max_abs_corr": round(float(max_abs_corr), 6),
                "corr_threshold": round(float(self.redundancy_corr_threshold), 6),
            }
            if passed:
                retained_factor_names.append(factor_name)
                retained_signals_by_factor[factor_name] = signal
            else:
                filtered_factor_names.append(factor_name)

        # ── 释放全长信号 ──────────────────────────────────────────────────────
        # 冗余门是本函数内唯一消费信号的地方；此后只剩报告/汇总（仅用指标）。
        # 主动丢弃所有 N×(全panel长) 的信号引用，让它们在后续阶段前就被回收，
        # 避免 N 个因子的整段信号长期驻留内存。
        retained_signals_by_factor.clear()
        for _cand in redundancy_candidates:
            _cand.pop("signal", None)
        for _res in group_results.values():
            if isinstance(_res, dict):
                _res.pop("signal_series", None)

        # 选出最佳单因子（按 |Rank IC| 排序），作为顶层摘要指标
        evaluated_factors = len(per_factor_metrics)
        selected_metrics, selected_factor_name = self._select_summary_metrics(per_factor_metrics)

        # 把质量门槛判定结果写入每个因子的 per_factor_metrics。
        # coverage 是硬门槛；reward 只评价通过可用性门槛后的因子质量。
        if self.sample_windows_enabled or self.coverage_filter_enabled:
            for _fn, _m in per_factor_metrics.items():
                _train_decision = train_filter_decisions_by_factor.get(_fn, {})
                _coverage_decision = coverage_filter_decisions_by_factor.get(_fn, {})
                _train_passed = (
                    not self.sample_windows_enabled
                    or bool(_train_decision.get("passed", False))
                )
                _coverage_passed = (
                    not self.coverage_filter_enabled
                    or bool(_coverage_decision.get("passed", False))
                )
                if isinstance(_train_decision, dict) and _train_decision:
                    _m["quality_gate_threshold"] = self._finite_float(self.train_filter_min_value)
                if isinstance(_coverage_decision, dict) and _coverage_decision:
                    _m["coverage_filter_passed"] = 1.0 if bool(_coverage_decision.get("passed", False)) else 0.0
                    _m["coverage_filter_threshold"] = self._finite_float(self.coverage_filter_min_value)
                _m["quality_gate_passed"] = 1.0 if (_train_passed and _coverage_passed) else 0.0

        total_factors = int(calculation_report.get("total_factors", 0) or 0)
        generated_factors = int(calculation_report.get("generated_factors", 0) or 0)
        failed_items = calculation_report.get("failed_factors", [])
        failed_factor_names = [
            str(item).strip()
            for item in list(failed_items)
            if str(item).strip()
        ] if isinstance(failed_items, list) else []
        for factor_name in factor_errors:
            if factor_name not in failed_factor_names and factor_name not in per_factor_metrics:
                failed_factor_names.append(factor_name)
        failed_factors = len(failed_factor_names)
        
        status = "finished" if evaluated_factors > 0 else "failed"
        if failed_factors > 0 or missing_expression_count > 0 or evaluated_factors < generated_factors:
            status = "partial"

        rounded_summary = self._round_metrics_dict(selected_metrics)
        rounded_per_factor = {
            name: self._round_metrics_dict(metrics)
            for name, metrics in per_factor_metrics.items()
        }
        rounded_per_factor_validation = {
            name: self._round_metrics_dict(metrics)
            for name, metrics in per_factor_validation_metrics.items()
        }

        backtest_report = {
            "status": status,
            "factor_source": str(factor_implementation.get("source", "unknown")),
            "summary": f"qlib_eval_{status}_with_{evaluated_factors}_factors",
            "total_factors": total_factors,
            "generated_factors": generated_factors,
            "failed_factors": failed_factor_names,
            "evaluated_factors": evaluated_factors,
            "train_retained_factors": len(retained_factor_names),
            "train_filtered_factors": len(filtered_factor_names),
            "execution_mode": "qlib_local",
            "cache_mode": "pickle_precompute_if_available",
            "metrics_semantics": "best_single_factor_not_combined",
            "selected_factor_name": selected_factor_name,
            "signal_cache_hits": sum(1 for item in generated_impls if str(item.get("source_pickle_path", "")).strip()),
            "signal_cache_misses": max(0, generated_factors - sum(1 for item in generated_impls if str(item.get("source_pickle_path", "")).strip())),
            "expression_cache_hits": expression_cache_hits,
            "unique_expressions": len(expression_groups),
            "parallel_workers": workers,
            "errors": factor_errors,
            "target_column": target_column,
            "panel_data_path": panel_data_path,
            "qlib_config_path": str(selected_config_path),
            # 回测区间标记：本次回测覆盖的面板数据起止时间。
            "backtest_period": dict(self._backtest_period),
            "sharpe_sampling_step": int(self._current_sharpe_sampling_step),
            "sharpe_annualization_factor": round(float(self._current_sharpe_annualization_factor), 6),
            "panel_time_step": self._current_panel_step,
            "transaction_cost_model": "long_short_layer_turnover",
            "one_way_fee_rate": round(float(self.one_way_fee_rate), 8),
            "one_way_fee_description": "4 bps one-way transaction fee, i.e. 0.04% per side.",
            "sample_windows": {
                "enabled": bool(self.sample_windows_enabled),
                "split_mode": self.sample_window_split_mode or "explicit",
                "train_ratio": round(float(self.sample_window_train_ratio), 6),
                "validation_ratio": round(float(1.0 - self.sample_window_train_ratio), 6),
                "train": self._format_sample_window(self.train_window),
                "validation": self._format_sample_window(self.validation_window),
                "resolution": dict(self._sample_window_resolution),
            },
            "train_filter": {
                "enabled": bool(self.sample_windows_enabled),
                "metric": self.train_filter_metric,
                "threshold": round(float(self.train_filter_min_value), 12),
                "comparison": ">",
                "retained_factor_names": list(retained_factor_names),
                "filtered_factor_names": list(filtered_factor_names),
                "decisions": train_filter_decisions_by_factor,
            },
            "target_filter": {
                "enabled": True,
                "targets": {
                    "sharpe": ">1.5",
                    "abs_ic": ">0.03",
                    "abs_rank_ic": ">0.03",
                },
                "stage": "pre_retention",
                "retained_factor_names": list(retained_factor_names),
                "filtered_factor_names": list(target_filtered_factor_names),
                "decisions": target_filter_decisions_by_factor,
            },
            "coverage_filter": {
                "enabled": bool(self.coverage_filter_enabled),
                "metric": self.coverage_filter_metric,
                "threshold": round(float(self.coverage_filter_min_value), 12),
                "comparison": ">=",
                "stage": "pre_retention",
                "filtered_factor_names": list(coverage_filtered_factor_names),
                "decisions": coverage_filter_decisions_by_factor,
            },
            # 留样验证：train/val 各测一次，val 综合 reward 相对 train 衰减过大者不入库。
            "per_factor_validation_metrics": rounded_per_factor_validation,
            "validation_decay_filter": {
                "enabled": bool(self.sample_windows_enabled and self.validation_decay_enabled),
                "metric": "reward_score",
                "min_retention": round(float(self.validation_decay_min_retention), 6),
                "comparison": ">=",
                "stage": "pre_retention",
                "filtered_factor_names": list(decay_filtered_factor_names),
                "decisions": validation_decay_decisions_by_factor,
            },
            "robustness_filter": {
                "enabled": bool(self.robustness_filter_enabled),
                "market_beta_abs_max": round(float(self.robustness_market_beta_abs_max), 12),
                "residual_sharpe_min": round(float(self.robustness_residual_sharpe_min), 12),
                "yearly_positive_ratio_min": round(float(self.robustness_yearly_positive_ratio_min), 12),
                "max_year_positive_contribution_share": round(float(self.robustness_max_year_positive_contribution_share), 12),
                "regime_positive_sharpe_count_min": int(self.robustness_regime_positive_sharpe_count_min),
                "decisions": robustness_filter_decisions_by_factor,
            },
            "redundancy_filter": {
                "enabled": bool(self.redundancy_filter_enabled),
                "corr_threshold": round(float(self.redundancy_corr_threshold), 6),
                "comparison": "<",
                "stage": "construction",
                "decisions": redundancy_filter_decisions_by_factor,
            },
        }

        return backtest_report, rounded_summary, rounded_per_factor


class BacktestRunnerAgent(BaseAgent):
    """回测执行智能体（5-Agent 链路中的第 4 个）。

    核心职责：
    1. 从 SharedContext 读取 factor_implementation 和 calculation_report
    2. 调用 QlibFactorBacktestEngine.evaluate_with_qlib() 执行完整回测
    3. 将 backtest_report、metrics、per_factor_metrics 写回 SharedContext

    本智能体不调用 LLM，纯数值计算。

    输出指标说明：
    - metrics（顶层摘要）：最佳单因子的所有指标（非多因子组合）
    - per_factor_metrics：每个因子的完整指标
    - backtest_report：回测执行报告（状态、缓存命中率、错误信息等）
    """

    def __init__(self, name: str) -> None:
        """加载回测引擎配置并实例化 QlibFactorBacktestEngine。"""
        super().__init__(name)
        self.engine_config_path = Path(__file__).resolve().parent.parent / "configs" / "qlib" / "backtest_engine.yaml"
        self.engine_config = self._load_engine_config(self.engine_config_path)
        engine_block = self.engine_config.get("engine", {})
        metrics_block = self.engine_config.get("metrics", {})
        dataset_block = self.engine_config.get("dataset", {})
        formula_block = metrics_block.get("default_metric_formula", {}) if isinstance(metrics_block, dict) else {}
        self.qlib_enabled = bool(engine_block.get("qlib_enabled", True)) if isinstance(engine_block, dict) else True
        self.no_factor_metrics = dict(metrics_block.get("no_factor_metrics", {})) if isinstance(metrics_block, dict) else {}
        self.formula_sharpe_coverage_weight = float(formula_block.get("sharpe_coverage_weight", 0.9)) if isinstance(formula_block, dict) else 0.9
        self.formula_sharpe_failure_weight = float(formula_block.get("sharpe_failure_weight", 0.2)) if isinstance(formula_block, dict) else 0.2
        self.formula_turnover_base = float(formula_block.get("turnover_base", 0.18)) if isinstance(formula_block, dict) else 0.18
        self.formula_turnover_coverage_weight = float(formula_block.get("turnover_coverage_weight", 0.27)) if isinstance(formula_block, dict) else 0.27
        self.formula_drawdown_base = float(formula_block.get("drawdown_base", 0.1)) if isinstance(formula_block, dict) else 0.1
        self.formula_drawdown_failure_weight = float(formula_block.get("drawdown_failure_weight", 0.25)) if isinstance(formula_block, dict) else 0.25
        self.time_per_step = str(dataset_block.get("time_per_step", "60min")) if isinstance(dataset_block, dict) else "60min"
        self.portfolio_key = str(dataset_block.get("portfolio_key", "60min")) if isinstance(dataset_block, dict) else "60min"
        self.engine = QlibFactorBacktestEngine(self.engine_config, self.engine_config_path)

    def _engine_backtest_period(self) -> dict[str, str | None]:
        period = getattr(self.engine, "_backtest_period", None)
        if isinstance(period, dict):
            return {
                "start": str(period.get("start")) if period.get("start") is not None else None,
                "end": str(period.get("end")) if period.get("end") is not None else None,
            }
        return {"start": None, "end": None}

    def _load_engine_config(self, config_path: Path) -> dict[str, Any]:
        """从 YAML 加载回测引擎配置。"""
        with config_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"invalid backtest engine config: {config_path}")
        return loaded

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        """执行回测：从 SharedContext 读取因子实现，调用引擎计算指标，写回结果。

        两个分支：
        1. Qlib 启用且面板数据可用：调用 evaluate_with_qlib() 执行完整回测
        2. Qlib 不可用或面板数据缺失：构建失败报告，metrics 为空
        """
        experiment = QlibFactorExperiment.from_shared_payload(shared_context.payload)
        # 优先使用 shared_context.payload 中的最新 factor_implementation 和 calculation_report
        # 避免被 qlib_factor_experiment 中的旧值覆盖
        if "factor_implementation" in shared_context.payload:
            factor_implementation = dict(shared_context.payload["factor_implementation"])
        else:
            factor_implementation = dict(experiment.factor_implementation)
        if "calculation_report" in shared_context.payload:
            calculation_report = dict(shared_context.payload["calculation_report"])
        else:
            calculation_report = dict(experiment.calculation_report)
        domain_dataset_summary = dict(shared_context.payload.get("domain_dataset_summary", {}))
        panel_data_path = str(domain_dataset_summary.get("path", "")).strip()
        if panel_data_path and self.qlib_enabled:
            qlib_result = self.engine.evaluate_with_qlib(
                factor_implementation=factor_implementation,
                calculation_report=calculation_report,
                panel_data_path=panel_data_path,
            )
            if qlib_result is not None:
                backtest_report, metrics, per_factor_metrics = qlib_result
                experiment.backtest_report = dict(backtest_report)
                experiment.metrics = {str(k): float(v) for k, v in metrics.items()}
                experiment.per_factor_metrics = dict(per_factor_metrics)
                experiment.update_timestamps()
                private_context.payload["last_backtest_mode"] = "qlib_local"
                private_context.payload["latest_backtest_report"] = backtest_report
                private_context.payload["latest_metrics"] = metrics
                private_context.payload["latest_per_factor_metrics"] = per_factor_metrics
                private_context.payload["latest_experiment"] = experiment.to_dict()
                return AgentResult(
                    shared_updates={
                        "backtest_report": backtest_report,
                        "metrics": metrics,
                        "per_factor_metrics": per_factor_metrics,
                        "qlib_factor_experiment": experiment.to_dict(),
                    },
                    artifacts={"agent": self.name},
                )
        total_factors = int(calculation_report.get("total_factors", 0) or 0)
        generated_factors = int(calculation_report.get("generated_factors", 0) or 0)
        failed_items = calculation_report.get("failed_factors", [])
        metrics: dict[str, float] = {}
        factor_errors = self.engine._collect_failure_errors(
            factor_implementation=factor_implementation,
            calculation_report=calculation_report,
        )
        failed_factor_names = [
            str(item).strip()
            for item in list(failed_items)
            if str(item).strip()
        ] if isinstance(failed_items, list) else []
        for factor_name in factor_errors:
            if factor_name not in failed_factor_names:
                failed_factor_names.append(factor_name)
        backtest_report = {
            "status": "failed",
            "factor_source": str(factor_implementation.get("source", "unknown")),
            "summary": "qlib_backtest_failed",
            "total_factors": total_factors,
            "generated_factors": generated_factors,
            "failed_factors": failed_factor_names,
            "execution_mode": "qlib_local",
            "cache_mode": "realtime_compute",
            "panel_data_path": panel_data_path,
            "backtest_period": self._engine_backtest_period(),
            "error": self.engine.last_error or "qlib_backtest_failed",
            "errors": factor_errors,
            "metrics_semantics": "invalid_due_to_backtest_failure",
        }
        experiment.backtest_report = dict(backtest_report)
        experiment.metrics = {str(k): float(v) for k, v in metrics.items()}
        experiment.per_factor_metrics = {}
        experiment.update_timestamps()
        private_context.payload["last_backtest_mode"] = "qlib_local_failed"
        private_context.payload["latest_backtest_report"] = backtest_report
        private_context.payload["latest_metrics"] = metrics
        private_context.payload["latest_per_factor_metrics"] = {}
        private_context.payload["latest_experiment"] = experiment.to_dict()
        return AgentResult(
            shared_updates={
                "backtest_report": backtest_report,
                "metrics": metrics,
                "per_factor_metrics": {},
                "qlib_factor_experiment": experiment.to_dict(),
            },
            artifacts={"agent": self.name},
        )
