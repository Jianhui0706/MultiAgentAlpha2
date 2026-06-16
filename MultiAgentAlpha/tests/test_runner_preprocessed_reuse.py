from __future__ import annotations

import pandas as pd

from web.backend.services import runner


def _panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3, freq="h"), ["BTC", "ETH"]],
        names=["datetime", "symbol"],
    )
    return pd.DataFrame(
        {
            "close": 100.0,
            "returns_1h": 0.0,
            "returns_4h": 0.0,
            "returns_12h": 0.0,
            "returns_24h": 0.0,
            "fwd_returns_24h": 0.01,
        },
        index=index,
    )


def test_run_reuses_single_preprocessed_panel_without_copy() -> None:
    path = str(runner.PROJECT_ROOT / "data" / "processed" / "panel_24h_ready.parquet")

    assert runner._should_reuse_preprocessed_panel(
        paths=[path],
        df=_panel(),
        target_col="fwd_returns_24h",
        start_date=None,
        end_date=None,
    )


def test_run_reuses_external_preprocessed_panel_without_copy() -> None:
    path = str(runner.PROJECT_ROOT.parent / "external_ready_panel.parquet")

    assert runner._should_reuse_preprocessed_panel(
        paths=[path],
        df=_panel(),
        target_col="fwd_returns_24h",
        start_date=None,
        end_date=None,
    )


def test_run_does_not_reuse_preprocessed_panel_when_window_is_requested() -> None:
    path = str(runner.PROJECT_ROOT / "data" / "processed" / "panel_24h_ready.parquet")

    assert not runner._should_reuse_preprocessed_panel(
        paths=[path],
        df=_panel(),
        target_col="fwd_returns_24h",
        start_date="2024-01-01",
        end_date=None,
    )
