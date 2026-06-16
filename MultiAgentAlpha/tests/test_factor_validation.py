from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_agents.infra.factor_validation import FactorValidationRequest, inspect_panel, run_validation


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=12, freq="h")
    symbols = [f"S{i}" for i in range(6)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    rows = []
    for ts_i, _date in enumerate(dates):
        for sym_i, _symbol in enumerate(symbols):
            close = 100 + ts_i + sym_i * 0.5
            rows.append({
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.5,
                "close": close,
                "volume": 1000 + ts_i * 10 + sym_i,
                "fwd_returns_1h": (sym_i - 2.5) * 0.001 + ts_i * 0.0001,
                "fwd_returns_24h": (sym_i - 2.5) * 0.0015 + ts_i * 0.0001,
                "fwd_returns_4h": (sym_i - 2.5) * 0.0012 + ts_i * 0.0001,
            })
    return pd.DataFrame(rows, index=index)


def test_factor_validation_generates_report(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    _panel().to_parquet(panel_path)

    preview = inspect_panel(str(panel_path))
    assert preview["fwd_columns"] == ["fwd_returns_1h", "fwd_returns_4h", "fwd_returns_24h"]
    assert preview["symbol_count"] == 6

    result = run_validation(
        FactorValidationRequest(
            panel_data_path=str(panel_path),
            expression="RANK($close)",
            factor_name="rank_close",
            target_column="fwd_returns_1h",
            rolling_window=4,
        ),
        reports_root=tmp_path / "reports",
    )

    assert result["metrics"]["coverage"] == 1.0
    assert "IC" in result["metrics"]
    assert Path(result["report_path"]).exists()
    assert Path(result["pdf_path"]).exists()
    assert (Path(result["report_dir"]).with_suffix(".zip")).exists()
    assert "cumulative_ic" in result["charts"]
    assert "ic_distribution" in result["charts"]
    assert "five_group_cumulative_return" in result["charts"]
    assert result["return_diagnostics"]["layer_return_table"]
    assert result["return_diagnostics"]["yearly_annualized_returns"]
