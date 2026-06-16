from __future__ import annotations

import pandas as pd
import pytest

from web.backend.services.runner import (
    _canonical_horizon,
    _horizon_to_minutes,
    _resolve_target_horizons,
    _standardize_panel_dataframe,
)


def test_standardize_panel_can_generate_all_common_fwd_targets() -> None:
    rows = []
    for symbol in ["BTC", "ETH"]:
        for i in range(30):
            rows.append(
                {
                    "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                    "symbol": symbol,
                    "close": 100.0 + i,
                    "volume": 1000.0 + i,
                }
            )
    raw = pd.DataFrame(rows)

    panel, info = _standardize_panel_dataframe(
        raw,
        prediction_horizon="24h",
        bar_frequency="60min",
        compute_targets=True,
        target_horizons=["1h", "4h", "12h", "24h"],
    )

    assert info["content_changed"] is True
    assert {"fwd_returns_1h", "fwd_returns_4h", "fwd_returns_12h", "fwd_returns_24h"}.issubset(panel.columns)
    assert panel["fwd_returns_1h"].notna().any()
    assert panel["fwd_returns_24h"].notna().any()


def test_horizon_helpers_canonicalize_minutes_and_hours() -> None:
    assert _horizon_to_minutes("5min") == 5
    assert _horizon_to_minutes("30m") == 30
    assert _horizon_to_minutes("4h") == 240
    assert _horizon_to_minutes("1d") == 1440
    assert _horizon_to_minutes("garbage") is None
    # < 60min keeps minute form; exact-hour multiples collapse to hours.
    assert _canonical_horizon("5min") == "5min"
    assert _canonical_horizon("60min") == "1h"
    assert _canonical_horizon("240min") == "4h"
    assert _canonical_horizon("1d") == "24h"
    assert _canonical_horizon("90min") == "90min"
    # Multi-select de-dupes 60min↔1h and preserves order; default adds 24h.
    assert _resolve_target_horizons(["5min", "30min", "60min", "4h", "24h"], "24h") == [
        "5min", "30min", "1h", "4h", "24h",
    ]
    assert _resolve_target_horizons(None, "4h") == ["4h", "24h"]


def test_standardize_panel_generates_minute_horizons_with_correct_offsets() -> None:
    # 5-minute bars; verify fwd offsets are bar-frequency aware for sub-hour horizons.
    rows = []
    for symbol in ["BTC", "ETH"]:
        for i in range(600):
            rows.append(
                {
                    "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * i),
                    "symbol": symbol,
                    "close": 100.0 + i,
                    "volume": 1000.0 + i,
                }
            )
    raw = pd.DataFrame(rows)

    panel, info = _standardize_panel_dataframe(
        raw,
        prediction_horizon="24h",
        bar_frequency="5min",
        compute_targets=True,
        target_horizons=["5min", "30min", "60min", "4h", "24h"],
    )

    # 60min canonicalizes to 1h — no separate fwd_returns_60min column.
    assert {"fwd_returns_5min", "fwd_returns_30min", "fwd_returns_1h", "fwd_returns_4h", "fwd_returns_24h"}.issubset(panel.columns)
    assert "fwd_returns_60min" not in panel.columns
    assert info["bar_minutes"] == 5
    # 5min fwd = 1 bar ahead: close[i+1]/close[i]-1. For BTC first row: 101/100-1 = 0.01.
    # 容差按 float32 量级（面板加载已降精度，见 downcast_float_columns）。
    btc = panel.xs("BTC", level="symbol")["fwd_returns_5min"].dropna()
    assert abs(float(btc.iloc[0]) - (101.0 / 100.0 - 1.0)) < 1e-5
    # 30min fwd = 6 bars ahead: close[i+6]/close[i]-1. BTC first: 106/100-1 = 0.06.
    btc30 = panel.xs("BTC", level="symbol")["fwd_returns_30min"].dropna()
    assert abs(float(btc30.iloc[0]) - (106.0 / 100.0 - 1.0)) < 1e-5


def test_run_standardization_requires_selected_target_column() -> None:
    raw = pd.DataFrame(
        [
            {"datetime": pd.Timestamp("2024-01-01 00:00:00"), "symbol": "BTC", "close": 100.0},
            {"datetime": pd.Timestamp("2024-01-01 00:05:00"), "symbol": "BTC", "close": 101.0},
        ]
    )

    with pytest.raises(ValueError, match="fwd_returns_4h"):
        _standardize_panel_dataframe(
            raw,
            prediction_horizon="4h",
            bar_frequency="5min",
            compute_targets=False,
        )
