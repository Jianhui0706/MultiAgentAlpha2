from __future__ import annotations

import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.factor_runtime import calculate_factor_expression
from trading_agents.schemas import AgentContext, AgentResult, QlibFactorExperiment, SharedContext


BACKTEST_DEBUG = os.getenv("BACKTEST_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


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
        # ── 解析器配置（表达式别名与黑名单）──
        resolver = config.get("resolver", {})
        dataset = config.get("dataset", {})
        metrics = config.get("metrics", {})
        engine = config.get("engine", {})
        self.alias_map = dict(resolver.get("alias_map", {})) if isinstance(resolver, dict) else {}
        blacklist = resolver.get("function_token_blacklist", []) if isinstance(resolver, dict) else []
        self.function_token_blacklist = {str(item).strip().lower() for item in blacklist if str(item).strip()}
        # ── 数据集配置 ──
        self.target_column = str(dataset.get("target_column", "fwd_returns_4h")) if isinstance(dataset, dict) else "fwd_returns_4h"  # 因变量列名（未来收益）
        index_levels = dataset.get("required_index_levels", ["datetime", "symbol"]) if isinstance(dataset, dict) else ["datetime", "symbol"]
        self.required_index_levels = {str(item).strip() for item in index_levels if str(item).strip()}
        exclude_prefixes = dataset.get("signal_exclude_prefixes", ["fwd_"]) if isinstance(dataset, dict) else ["fwd_"]
        self.signal_exclude_prefixes = tuple(str(item) for item in exclude_prefixes)
        # ── 指标计算配置 ──
        self.annualization_factor = float(metrics.get("annualization_factor", 2190)) if isinstance(metrics, dict) else 2190.0  # 年化因子（加密市场 60min * 24h * 365d）
        self.corr_method = str(metrics.get("corr_method", "spearman")) if isinstance(metrics, dict) else "spearman"  # 相关性方法
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
        self.test_window = self.validation_window  # Backward-compatible alias; this project uses validation as the 25% holdout.
        self._sample_window_resolution: dict[str, object] = {"mode": "explicit"}
        self.train_filter_metric = (
            str(train_filter.get("metric", "sharpe")).strip() if isinstance(train_filter, dict) else "sharpe"
        ) or "sharpe"
        config_train_filter_min_value = (
            self._finite_float(train_filter.get("min_value", 2.0), 2.0)
            if isinstance(train_filter, dict)
            else 2.0
        )
        threshold_override = os.getenv("TRAIN_SHARPE_THRESHOLD", "").strip()
        self.train_filter_min_value = (
            self._finite_float(threshold_override, config_train_filter_min_value)
            if threshold_override
            else config_train_filter_min_value
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
        self.test_window = self.validation_window
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

    def _ensure_sample_windows_resolved(self, panel: pd.DataFrame) -> None:
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
        match = re.search(r"fwd_returns_(\d+)([hd])", self.target_column)
        if not match:
            return None
        value = int(match.group(1))
        unit = match.group(2)
        if value <= 0:
            return None
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
        panel_step = self._infer_panel_step_timedelta(panel)
        horizon = self._target_horizon_timedelta()
        if panel_step is None or horizon is None or panel_step <= pd.Timedelta(0) or horizon <= pd.Timedelta(0):
            return
        holding_steps = max(1, int(round(horizon / panel_step)))
        effective_horizon = panel_step * holding_steps
        self._current_sharpe_sampling_step = holding_steps
        self._current_sharpe_annualization_factor = float(pd.Timedelta(days=365) / effective_horizon)
        self._current_panel_step = str(panel_step)

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
        if not provider_uri_raw:
            raise ValueError("qlib_init.provider_uri is empty")
        provider_uri = os.path.expanduser(provider_uri_raw)
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
            panel = pd.read_parquet(panel_data_path)
        except Exception:
            return None
        if panel.empty:
            return None
        if not self.required_index_levels.issubset(set(panel.index.names)):
            return None

        if self.target_column in panel.columns:
            target_column = self.target_column
        else:
            # 配置列不在面板中 —— 从所有 fwd_returns_* 列中按预测窗口升序选最短周期
            import re as _re
            fwd_cols = [c for c in panel.columns if c.startswith("fwd_returns_")]

            def _horizon_minutes(col: str) -> int:
                m = _re.search(r"fwd_returns_(\d+)([hd])", col)
                if not m:
                    return 10 ** 9
                n, unit = int(m.group(1)), m.group(2)
                return n * (60 if unit == "h" else 1440)

            fwd_cols_sorted = sorted(fwd_cols, key=_horizon_minutes)
            if not fwd_cols_sorted:
                print(
                    f"[BacktestEngine] ERROR: target_column '{self.target_column}' 不在面板列中，"
                    f"且未找到任何 fwd_returns_* 列。请检查 backtest_engine.yaml 与 panel 数据是否匹配。"
                )
                return None
            target_column = fwd_cols_sorted[0]
            print(
                f"[BacktestEngine] WARNING: 配置的 target_column '{self.target_column}' "
                f"不在面板列中。自动回退到 '{target_column}'。\n"
                f"  可用 fwd_returns_* 列: {fwd_cols_sorted}\n"
                f"  请更新 backtest_engine.yaml 中的 dataset.target_column，"
                f"避免使用错误的预测标签。"
            )

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
        - IC / ICIR：因子信号与未来收益的 Spearman 相关系数及其 IR
        - Rank IC / Rank ICIR：先排名再计算的相关系数（更稳健）
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
        ic_values = self._cross_sectional_corr(eval_frame, rank=False).dropna()
        
        if isinstance(ic_values, pd.Series) and not ic_values.empty:
            ic_mean = float(ic_values.mean())
            ic_std = float(ic_values.std())
            metrics["IC"] = ic_mean
            metrics["ICIR"] = float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else 0.0
        else:
            metrics["IC"] = 0.0
            metrics["ICIR"] = 0.0
        
        # ★ 中等耗时：截面 Rank IC（先排名再 Spearman 逐日 apply），~0.5-1s
        rank_ic_values = self._cross_sectional_corr(eval_frame, rank=True).dropna()
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
        signal_mat = signal_wide.values.astype(np.float64)
        label_mat = label_wide.values.astype(np.float64)
        T, N = signal_mat.shape
        datetimes = signal_wide.index
        valid_mask = ~(np.isnan(signal_mat) | np.isnan(label_mat))       # (T, N)
        valid_per_row = valid_mask.sum(axis=1)                            # (T,)

        # ── Step 2: IC / Rank IC（向量化 Spearman = Pearson of ranks）──
        s_ranked = signal_wide.rank(axis=1, method="average").values     # (T, N)
        l_ranked = label_wide.rank(axis=1, method="average").values       # (T, N)

        sx = s_ranked - np.nanmean(s_ranked, axis=1, keepdims=True)
        ly = l_ranked - np.nanmean(l_ranked, axis=1, keepdims=True)
        cov = np.nansum(sx * ly, axis=1)
        std_s = np.sqrt(np.nansum(sx ** 2, axis=1))
        std_l = np.sqrt(np.nansum(ly ** 2, axis=1))

        ic_arr = np.where(
            (std_s > 1e-12) & (std_l > 1e-12) & (valid_per_row >= 2),
            cov / (std_s * std_l),
            np.nan,
        )
        ic_series = pd.Series(ic_arr, index=datetimes).dropna()

        metrics: dict[str, float] = {}
        if not ic_series.empty:
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            metrics["IC"] = ic_mean
            metrics["ICIR"] = float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else 0.0
            # 原始代码中 IC 和 Rank IC 均使用 spearman 方法，结果相同
            metrics["Rank IC"] = ic_mean
            metrics["Rank ICIR"] = metrics["ICIR"]
        else:
            metrics["IC"] = 0.0
            metrics["ICIR"] = 0.0
            metrics["Rank IC"] = 0.0
            metrics["Rank ICIR"] = 0.0

        # Coverage & Turnover
        metrics["coverage"] = round(float(len(eval_frame) / max(1, len(panel))), 4)
        rank_pct = signal_wide.rank(axis=1, pct=True)
        rank_diff = rank_pct.diff().abs().mean(axis=1).dropna()
        metrics["turnover"] = float(rank_diff.mean()) if not rank_diff.empty else 1.0

        # ── Step 3: 分层回测（向量化）──
        n_layers = 5

        # 3a. 分层赋值：rank → quintile
        s_rank_first = signal_wide.rank(axis=1, method="first").values    # (T, N)
        layer_size = np.where(
            valid_per_row >= n_layers,
            valid_per_row / n_layers,
            valid_per_row.astype(float),
        )
        layer_mat = np.floor((s_rank_first - 1.0) / layer_size[:, None]).astype(np.float64) + 1.0
        layer_mat = np.clip(layer_mat, 1, n_layers)
        layer_mat[valid_per_row < n_layers] = 1.0
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
        sampling_step = max(1, int(self._current_sharpe_sampling_step))
        if sampling_step > 1 and len(ls_full) > 1:
            ls = ls_full.iloc[::sampling_step]
            lr_df = lr_df.iloc[::sampling_step]
            benchmark = benchmark_full.iloc[::sampling_step]
        else:
            ls = ls_full
            benchmark = benchmark_full

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
            ls_arr = ls.values.astype(np.float64)
            raw_mean = float(np.nanmean(ls_arr))
            raw_std = float(np.nanstd(ls_arr))
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
            net_arr = oriented_arr - toc.values.astype(np.float64)
            net_mean = float(np.nanmean(net_arr))
            net_std = float(np.nanstd(net_arr))
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
        metrics["sharpe_abs"] = abs(metrics.get("sharpe", 0.0))

        # 把多空累计收益曲线（采样至 ≤300 点）写入 metrics，供前端绘图
        if not ls.empty:
            try:
                net_series = pd.Series(net_arr, index=ls.index)
                step = max(1, len(net_series) // 300)
                sampled = net_series.iloc[::step]
                cum = (1.0 + sampled).cumprod()
                metrics["ls_cumulative_series"] = [  # type: ignore[assignment]
                    [str(t), round(float(v), 6)]
                    for t, v in zip(cum.index, cum.values)
                    if np.isfinite(v)
                ]
            except Exception:
                pass

        # 做多组（Layer 5）和做空组（Layer 1）各自的累计净值曲线
        for layer_key, series_name in [(1, "layer_1_cumulative_series"), (n_layers, "layer_5_cumulative_series")]:
            if layer_key in lr_df.columns:
                try:
                    col = lr_df[layer_key].dropna()
                    if not col.empty:
                        step_l = max(1, len(col) // 300)
                        sampled_l = col.iloc[::step_l]
                        cum_l = (1.0 + sampled_l).cumprod()
                        metrics[series_name] = [  # type: ignore[assignment]
                            [str(t), round(float(v), 6)]
                            for t, v in zip(cum_l.index, cum_l.values)
                            if np.isfinite(v)
                        ]
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
        - 辅助指标：IC_abs、IC_sign、Rank IC_abs、ICIR_abs、sharpe_abs
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
        single_metrics["sharpe_abs"] = abs(single_metrics.get("sharpe", 0.0))
        _debug_print(f"[DEBUG] Factor '{factor_name}' metrics: IC={single_metrics.get('IC', 0):.4f}, Rank IC={single_metrics.get('Rank IC', 0):.4f}")
        return single_metrics

    def _select_summary_metrics(self, per_factor_metrics: dict[str, dict[str, float]]) -> tuple[dict[str, float], str]:
        """从所有因子指标中选出最佳单因子（按 |Rank IC| 降序），返回其指标和名称。

        注意：这是"最佳单因子"指标，不是多因子组合回测结果。
        """
        if not per_factor_metrics:
            return {}, ""
        selected_name, selected_metrics = max(
            per_factor_metrics.items(),
            key=lambda item: (
                self._finite_float(item[1].get("robustness_pass", 1.0)),
                abs(self._finite_float(item[1].get("Rank IC", item[1].get("IC", 0.0)))),
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
        """
        frame = pd.concat([signal.rename("signal"), label.rename("label")], axis=1).dropna()
        if len(frame) < 2:
            return float("nan")

        signal_values = frame["signal"]
        label_values = frame["label"]
        if signal_values.nunique(dropna=True) < 2 or label_values.nunique(dropna=True) < 2:
            return float("nan")

        if rank:
            signal_values = signal_values.rank(pct=True)
            label_values = label_values.rank(pct=True)
        value = signal_values.corr(label_values, method=self.corr_method)
        return float(value) if pd.notna(value) else float("nan")

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
                return pd.Series([1] * len(group), index=group.index, dtype=float)
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
        sampling_step = max(1, int(self._current_sharpe_sampling_step))
        if sampling_step > 1 and len(long_short_returns) > 1:
            # Sample non-overlapping forward-return windows so Sharpe matches the label horizon.
            long_short_returns = long_short_returns.iloc[::sampling_step]
            turnover_cost = turnover_cost.iloc[::sampling_step]
        benchmark_returns = factor_frame["label"].groupby(level="datetime").mean().dropna().sort_index()
        if sampling_step > 1 and len(benchmark_returns) > 1:
            benchmark_returns = benchmark_returns.iloc[::sampling_step]
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
        sampling_step = max(1, int(self._current_sharpe_sampling_step))
        if sampling_step > 1 and not layer_returns_by_dt.empty:
            # Keep layer-level summary statistics on the same non-overlapping horizon
            # used by the final 4h Sharpe and the report plots.
            layer_returns_by_dt = layer_returns_by_dt.iloc[::sampling_step]

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
        if self.sample_windows_enabled:
            train_mask = self._build_sample_window_mask(full_panel, self.train_window)
            panel = full_panel.iloc[np.flatnonzero(train_mask)].copy()
            if panel.empty:
                self._set_error("empty_train_sample_window")
                return None
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

        label_series = pd.to_numeric(panel[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan).rename("label")

        # ── 安全防护：构造不含未来特征列的 feature_panel ──────────────────────────
        # signal_exclude_prefixes（默认 ["fwd_"]）过滤掉所有未来收益列，防止
        # AI 生成的因子表达式通过 $fwd_returns_4h 等变量引用未来数据（look-ahead bias）。
        # 注意：label_series 已在上一行从 panel 中单独提取，feature_panel 仅用于因子信号计算。
        feature_cols = [c for c in panel.columns if not c.startswith(self.signal_exclude_prefixes)]
        feature_panel = panel[feature_cols] if feature_cols else panel

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
                signal_series = self._load_signal_from_cache(representative, feature_panel)
                if signal_series is not None:
                    _debug_print(f"[DEBUG] Loaded cached signal for factor '{factor_name}'")
                else:
                    signal_series = calculate_factor_expression(expression, feature_panel, factor_name)
                if signal_series is None:
                    raise ValueError("factor calculation returned None")
                metrics = self._calculate_factor_metrics_bundle(
                    factor_name=factor_name,
                    expression=expression,
                    signal_series=signal_series,
                    label_series=label_series,
                    panel=panel,
                )
                train_filter_decision = self._build_train_filter_decision(metrics)
                robustness_filter_decision = self._build_robustness_filter_decision(metrics)
                return key, {
                    "metrics": metrics,
                    "train_filter_decision": train_filter_decision,
                    "robustness_filter_decision": robustness_filter_decision,
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
        train_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        robustness_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        redundancy_filter_decisions_by_factor: dict[str, dict[str, object]] = {}
        retained_factor_names: list[str] = []
        filtered_factor_names: list[str] = []
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
                    train_filter_decision = dict(result.get("train_filter_decision", {}))
                    per_factor_metrics[factor_name] = {
                        str(metric_name): self._finite_float(metric_value)
                        for metric_name, metric_value in metrics.items()
                    }
                    train_filter_decisions_by_factor[factor_name] = train_filter_decision
                    robustness_filter_decision = dict(result.get("robustness_filter_decision", {}))
                    robustness_filter_decisions_by_factor[factor_name] = robustness_filter_decision
                    train_passed = not self.sample_windows_enabled or bool(train_filter_decision.get("passed", False))
                    robustness_passed = (
                        not self.robustness_filter_enabled
                        or bool(robustness_filter_decision.get("passed", False))
                    )
                    if train_passed and robustness_passed:
                        rank_ic = self._finite_float(metrics.get("Rank IC", metrics.get("RankIC")))
                        redundancy_candidates.append(
                            {
                                "factor_name": factor_name,
                                "abs_rank_ic": abs(rank_ic) if np.isfinite(rank_ic) else 0.0,
                                "signal": result.get("signal_series"),
                            }
                        )
                    else:
                        filtered_factor_names.append(factor_name)
                elif error is not None:
                    factor_error = dict(error)
                    factor_error["factor_name"] = factor_name
                    factor_errors[factor_name] = factor_error

        # ── 构造阶段冗余门：按 |Rank IC| 降序贪心，剔除与已保留因子高度相关者 ──
        redundancy_candidates.sort(key=lambda item: float(item.get("abs_rank_ic", 0.0)), reverse=True)
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

        # 选出最佳单因子（按 |Rank IC| 排序），作为顶层摘要指标
        evaluated_factors = len(per_factor_metrics)
        selected_metrics, selected_factor_name = self._select_summary_metrics(per_factor_metrics)

        # 把质量门槛判定结果（train_filter passed/failed）写入每个因子的 per_factor_metrics
        # 供 UI 展示"质量未达标"状态，也供 StrategyTrajectory.is_successful() 使用
        if self.sample_windows_enabled:
            for _fn, _m in per_factor_metrics.items():
                _decision = train_filter_decisions_by_factor.get(_fn, {})
                if isinstance(_decision, dict) and _decision:
                    _m["quality_gate_passed"] = 1.0 if bool(_decision.get("passed", False)) else 0.0
                    _m["quality_gate_threshold"] = self._finite_float(self.train_filter_min_value)

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
