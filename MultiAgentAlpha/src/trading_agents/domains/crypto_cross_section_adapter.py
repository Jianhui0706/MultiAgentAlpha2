from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_agents.factor_runtime import downcast_float_columns


class CryptoCrossSectionDomainAdapter:
    def __init__(self, panel_data_path: str | Path, *, target_column: str | None = None) -> None:
        self.panel_data_path = Path(panel_data_path)
        self._target_column = target_column or "fwd_returns_24h"

    def _load_panel_data(self) -> pd.DataFrame:
        panel = downcast_float_columns(pd.read_parquet(self.panel_data_path))
        # 兼容「扁平存储」的文件：若磁盘上没有 (datetime, symbol) 多级索引，但这两列
        # 以普通列存在，则在加载时重建索引。这样含 fwd 的文件可直接使用，无需事先
        # 另存一份带索引的副本。已是多级索引的文件（预处理产物）此处为无操作。
        index_names = {n for n in (getattr(panel.index, "names", []) or []) if n}
        if not {"datetime", "symbol"}.issubset(index_names) and {"datetime", "symbol"}.issubset(panel.columns):
            panel = panel.set_index(["datetime", "symbol"]).sort_index()
        return panel

    def summarize_panel_data(self) -> dict[str, Any]:
        panel = self._load_panel_data()
        datetime_values = panel.index.get_level_values("datetime")
        symbol_values = panel.index.get_level_values("symbol")
        all_columns = list(panel.columns)
        target_columns = [col for col in all_columns if col.startswith("fwd_")]
        feature_columns = [col for col in all_columns if not col.startswith("fwd_")]
        summary = {
            "path": str(self.panel_data_path),
            "rows": int(panel.shape[0]),
            "columns": int(panel.shape[1]),
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "validation_target_column": self._target_column if self._target_column in all_columns else (next((c for c in all_columns if c.startswith("fwd_")), self._target_column)),
            "symbol_count": int(symbol_values.nunique()),
            "datetime_start": str(datetime_values.min()),
            "datetime_end": str(datetime_values.max()),
            "nan_ratio": float(panel.isna().sum().sum() / max(1, panel.size)),
        }
        return summary

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return round(parsed, 6)

    @staticmethod
    def _classify_percentile(value: float | None, low: float, high: float) -> str:
        if value is None:
            return "unknown"
        if value <= low:
            return "low"
        if value >= high:
            return "high"
        return "normal"

    def summarize_market_regime(self) -> dict[str, Any]:
        """Summarize recent panel conditions for non-report hypothesis seeding."""
        panel = self._load_panel_data()
        if panel.empty:
            return {"status": "empty_panel", "text": "Market regime summary unavailable because panel data is empty."}

        datetime_values = pd.Index(panel.index.get_level_values("datetime")).sort_values().unique()
        recent_count = max(5, int(len(datetime_values) * 0.2))
        recent_datetimes = set(datetime_values[-recent_count:])
        recent_panel = panel[panel.index.get_level_values("datetime").isin(recent_datetimes)]
        if recent_panel.empty:
            recent_panel = panel

        return_column = "returns_24h" if "returns_24h" in panel.columns else ""
        volume_column = "volume_change_24h" if "volume_change_24h" in panel.columns else ""
        if not volume_column:
            for candidate in ("volume_quote", "volume", "turnover"):
                if candidate in panel.columns:
                    volume_column = candidate
                    break
        spread_column = "high_low_spread" if "high_low_spread" in panel.columns else ""

        metrics: dict[str, Any] = {
            "status": "ok",
            "lookback_datetime_count": int(len(set(recent_panel.index.get_level_values("datetime")))),
            "available_columns": {
                "return": return_column,
                "volume": volume_column,
                "spread": spread_column,
            },
        }

        if return_column:
            full_returns = pd.to_numeric(panel[return_column], errors="coerce")
            recent_returns = pd.to_numeric(recent_panel[return_column], errors="coerce")
            by_time_mean = recent_returns.groupby(level="datetime").mean()
            by_time_std = recent_returns.groupby(level="datetime").std()
            trend_strength = self._safe_float(by_time_mean.tail(min(24, len(by_time_mean))).mean())
            dispersion = self._safe_float(by_time_std.mean())
            realized_vol = self._safe_float(recent_returns.std())
            q_low, q_high = full_returns.quantile([0.33, 0.67]).tolist()
            metrics.update(
                {
                    "recent_mean_return": self._safe_float(recent_returns.mean()),
                    "recent_trend_strength": trend_strength,
                    "recent_cross_section_dispersion": dispersion,
                    "recent_realized_volatility": realized_vol,
                    "return_regime": self._classify_percentile(
                        self._safe_float(recent_returns.mean()),
                        float(q_low),
                        float(q_high),
                    ),
                }
            )

        if volume_column:
            full_volume = pd.to_numeric(panel[volume_column], errors="coerce")
            recent_volume = pd.to_numeric(recent_panel[volume_column], errors="coerce")
            q_low, q_high = full_volume.quantile([0.33, 0.67]).tolist()
            recent_volume_mean = self._safe_float(recent_volume.mean())
            metrics.update(
                {
                    "recent_volume_activity": recent_volume_mean,
                    "volume_regime": self._classify_percentile(recent_volume_mean, float(q_low), float(q_high)),
                }
            )

        if spread_column:
            full_spread = pd.to_numeric(panel[spread_column], errors="coerce")
            recent_spread = pd.to_numeric(recent_panel[spread_column], errors="coerce")
            q_low, q_high = full_spread.quantile([0.33, 0.67]).tolist()
            recent_spread_mean = self._safe_float(recent_spread.mean())
            metrics.update(
                {
                    "recent_spread_stress": recent_spread_mean,
                    "spread_regime": self._classify_percentile(recent_spread_mean, float(q_low), float(q_high)),
                }
            )

        text_parts = [
            f"recent return regime={metrics.get('return_regime', 'unknown')}",
            f"volume regime={metrics.get('volume_regime', 'unknown')}",
            f"spread regime={metrics.get('spread_regime', 'unknown')}",
            f"cross-section dispersion={metrics.get('recent_cross_section_dispersion', 'unknown')}",
            f"realized volatility={metrics.get('recent_realized_volatility', 'unknown')}",
        ]
        metrics["text"] = (
            "Dynamic market regime summary for hypothesis seeding: "
            + "; ".join(str(item) for item in text_parts)
            + ". Treat this as a current-condition hint, not as a fixed thesis."
        )
        return metrics

    @staticmethod
    def build_parametric_specifications(summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Seed hypothesis generation with concrete, testable parameter families."""
        features = {str(item) for item in list(summary.get("feature_columns", []))}
        specs: list[dict[str, Any]] = []

        def add(
            *,
            name: str,
            specification: str,
            feature_anchors: list[str],
            preferred_windows: list[int],
            rationale: str,
        ) -> None:
            available_anchors = [feature for feature in feature_anchors if feature in features]
            if available_anchors:
                specs.append(
                    {
                        "name": name,
                        "specification": specification,
                        "feature_anchors": available_anchors,
                        "preferred_windows": preferred_windows,
                        "rationale": rationale,
                    }
                )

        add(
            name="short_horizon_reversal",
            specification="1h/4h/24h trailing return extremes as short-horizon reversal candidates",
            feature_anchors=["returns_1h", "returns_4h", "returns_24h", "log_return_1h", "close"],
            preferred_windows=[1, 4, 24],
            rationale="Crypto short-horizon crowding can reverse after abrupt price displacement.",
        )
        add(
            name="medium_horizon_momentum",
            specification="10-20 period medium-term momentum or breakout continuation",
            feature_anchors=["close", "returns_12h", "returns_24h"],
            preferred_windows=[10, 20],
            rationale="Medium-horizon continuation can survive after very short-term noise is smoothed.",
        )
        add(
            name="high_low_range_stress",
            specification="10-period high/low range or spread stress gating for reversal/momentum reliability",
            feature_anchors=["high", "low", "high_low_spread", "close"],
            preferred_windows=[5, 10, 20],
            rationale="Wide high-low ranges can mark stress regimes where raw price signals need conditioning.",
        )
        add(
            name="volume_confirmation",
            specification="24h volume or quote-volume expansion as confirmation/exhaustion context",
            feature_anchors=["volume", "volume_quote", "volume_change_1h", "volume_change_24h"],
            preferred_windows=[4, 12, 24],
            rationale="Volume expansion distinguishes broad participation from fragile low-liquidity moves.",
        )
        add(
            name="volatility_regime_filter",
            specification="short/long volatility ratio for regime-conditioned signal weighting",
            feature_anchors=["close", "returns_24h", "high_low_spread"],
            preferred_windows=[5, 20, 60],
            rationale="Volatility regime filters can reduce false signals during stress or compression.",
        )
        return specs

    def build_initial_payload(self) -> dict[str, object]:
        summary = self.summarize_panel_data()
        market_regime_summary = self.summarize_market_regime()
        parametric_specifications = self.build_parametric_specifications(summary)
        rag_text = (
            f"crypto panel rows={summary['rows']}, symbols={summary['symbol_count']}, "
            f"time_range={summary['datetime_start']}~{summary['datetime_end']}, "
            f"features={', '.join(summary['feature_columns'])}, "
            f"targets={', '.join(summary['target_columns'])}, "
            f"validation_target={summary['validation_target_column']}."
        )
        return {
            "market": "crypto_cross_section",
            "scenario": "panel_data_cross_section_factor_mining",
            "direction": "discover cross-sectional alpha factors with robust turnover control",
            "rag_text": rag_text,
            "domain_dataset_summary": summary,
            "market_regime_summary": market_regime_summary,
            "parametric_specifications": parametric_specifications,
        }
