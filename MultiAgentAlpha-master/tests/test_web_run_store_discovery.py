from __future__ import annotations

import sys
from pathlib import Path

from trading_agents.infra.run_store import RunStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "web" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api import runs as runs_api  # noqa: E402


def _write_factor_store(db_path: Path, *, run_id: str, factor_id: str, sharpe: float) -> None:
    store = RunStore(db_path)
    store.start_run(run_id=run_id, log_dir=str(db_path.parent / run_id), args={"run_name": run_id})
    store.upsert_factor(
        run_id=run_id,
        loop_index=1,
        trajectory_id=f"trajectory-{run_id}",
        factor_id=factor_id,
        factor={
            "factor_name": factor_id,
            "expression": "RANK($close)",
            "description": "",
            "implementation_status": "generated",
        },
        metrics={"sharpe": sharpe},
    )


def test_factor_top_discovers_timestamped_run_stores(monkeypatch, tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    older = artifacts_dir / "run_store_older.sqlite"
    newer = artifacts_dir / "run_store_newer.sqlite"
    _write_factor_store(older, run_id="run-low", factor_id="FactorLow", sharpe=1.0)
    _write_factor_store(newer, run_id="run-high", factor_id="FactorHigh", sharpe=5.0)

    monkeypatch.delenv("RUN_STORE_PATH", raising=False)
    monkeypatch.setattr(runs_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runs_api, "_DB_PATH", artifacts_dir / "run_store.sqlite")

    top = runs_api.get_top_factors(limit=2)

    assert [item["run_id"] for item in top] == ["run-high", "run-low"]
    assert [item["rank"] for item in top] == [1, 2]
    assert runs_api.get_factors("run-high")[0]["factor_name"] == "FactorHigh"


def test_factor_top_prefers_unified_run_store_when_present(monkeypatch, tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    unified = artifacts_dir / "run_store.sqlite"
    legacy = artifacts_dir / "run_store_legacy.sqlite"
    _write_factor_store(unified, run_id="run-unified", factor_id="FactorUnified", sharpe=1.0)
    _write_factor_store(legacy, run_id="run-legacy", factor_id="FactorLegacy", sharpe=5.0)

    monkeypatch.delenv("RUN_STORE_PATH", raising=False)
    monkeypatch.setattr(runs_api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runs_api, "_DB_PATH", unified)

    top = runs_api.get_top_factors(limit=5)

    assert [item["run_id"] for item in top] == ["run-unified"]
