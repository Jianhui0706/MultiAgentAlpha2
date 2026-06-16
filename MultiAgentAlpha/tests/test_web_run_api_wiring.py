from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def test_start_run_forwards_frequency_horizon_and_window(monkeypatch, tmp_path) -> None:
    backend_path = Path(__file__).resolve().parents[1] / "web" / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from web.backend.api import runs

    captured: dict[str, object] = {}

    def fake_launch_run(*, run_id: str, params: dict, progress_callback) -> None:
        captured["run_id"] = run_id
        captured["params"] = dict(params)
        captured["progress_callback"] = progress_callback

    monkeypatch.setattr(runs.runner, "launch_run", fake_launch_run)
    monkeypatch.setattr(runs, "_db", lambda: tmp_path / "runs.db")

    result = asyncio.run(
        runs.start_run(
            runs.StartRunRequest(
                run_name="api-wiring",
                panel_data_paths=["data/processed/panel_ready.parquet"],
                start_date="2026-01-01 09:30:00",
                end_date="2026-02-01 15:45:00",
                prediction_horizon="30min",
                bar_frequency="5min",
                train_coverage_threshold=0.88,
                train_sharpe_threshold=1.7,
            )
        )
    )

    params = captured["params"]
    assert result["status"] == "started"
    assert params["panel_data_path"] == "data/processed/panel_ready.parquet"
    assert params["panel_data_paths"] == ["data/processed/panel_ready.parquet"]
    assert params["start_date"] == "2026-01-01 09:30:00"
    assert params["end_date"] == "2026-02-01 15:45:00"
    assert params["prediction_horizon"] == "30min"
    assert params["bar_frequency"] == "5min"
    assert params["train_coverage_threshold"] == 0.88
    assert params["train_sharpe_threshold"] == 1.7


def test_start_run_defaults_train_sharpe_threshold_to_one_point_five(monkeypatch, tmp_path) -> None:
    backend_path = Path(__file__).resolve().parents[1] / "web" / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from web.backend.api import runs

    captured: dict[str, object] = {}

    def fake_launch_run(*, run_id: str, params: dict, progress_callback) -> None:
        captured["params"] = dict(params)

    monkeypatch.setattr(runs.runner, "launch_run", fake_launch_run)
    monkeypatch.setattr(runs, "_db", lambda: tmp_path / "runs.db")

    asyncio.run(runs.start_run(runs.StartRunRequest(run_name="default-threshold")))

    assert captured["params"]["train_sharpe_threshold"] == 1.5
