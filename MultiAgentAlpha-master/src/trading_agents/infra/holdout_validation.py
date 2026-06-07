from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_agents.agents.backtest_runner_agent import QlibFactorBacktestEngine
from trading_agents.factor_runtime import calculate_factor_expression
from trading_agents.infra.factor_library_manager import FactorLibraryManager


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def build_holdout_validation_payload(
    *,
    candidates: list[dict[str, Any]],
    metric: str,
    threshold: float,
    train_window: dict[str, object],
    validation_metrics_by_factor: dict[str, dict[str, Any]] | None = None,
    test_metrics_by_factor: dict[str, dict[str, Any]] | None = None,
    validation_window: dict[str, object] | None = None,
    test_window: dict[str, object] | None = None,
    errors: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metric_name = str(metric).strip() or "sharpe"
    threshold_value = float(threshold)
    validation_metrics_lookup = dict(validation_metrics_by_factor or test_metrics_by_factor or {})
    validation_window_payload = dict(validation_window or test_window or {})
    rows: list[dict[str, Any]] = []
    for item in candidates:
        factor_name = str(item.get("factor_name", ""))
        train_metrics = dict(item.get("train_metrics", {}))
        validation_metrics = dict(validation_metrics_lookup.get(factor_name, {}))
        train_value = _finite_float(train_metrics.get(metric_name), 0.0)
        validation_value = _finite_float(validation_metrics.get(metric_name), 0.0)
        rows.append(
            {
                "factor_id": str(item.get("factor_id", "")),
                "factor_name": factor_name,
                "factor_expression": str(item.get("factor_expression", "")),
                "train_sharpe": train_value,
                "validation_sharpe": validation_value,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "passed_train_threshold": bool(train_value > threshold_value),
                "passed_validation_threshold": bool(validation_value > threshold_value),
            }
        )
    return {
        "status": "finished" if candidates else "no_train_candidates",
        "metric": metric_name,
        "threshold": threshold_value,
        "comparison": ">",
        "train_window": dict(train_window),
        "validation_window": validation_window_payload,
        "factor_count": len(rows),
        "factors": rows,
        "errors": dict(errors or {}),
    }


def run_holdout_validation(
    *,
    library_path: str | Path,
    panel_data_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
    experiment_ids: set[str] | None = None,
    run_ids: set[str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path) if config_path else (
        FactorLibraryManager.project_root() / "src" / "trading_agents" / "configs" / "qlib" / "backtest_engine.yaml"
    )
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid backtest engine config: {config_file}")

    engine = QlibFactorBacktestEngine(config, config_file)
    manager = FactorLibraryManager(library_path)
    candidates = manager.list_train_metric_candidates(
        metric=engine.train_filter_metric,
        threshold=engine.train_filter_min_value,
        experiment_ids=experiment_ids,
        run_ids=run_ids,
    )

    validation_metrics_by_factor: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, Any]] = {}
    if candidates:
        prepared = engine._prepare_panel(str(panel_data_path))
        if prepared is None:
            errors["_panel"] = {"status": "failed", "message": "invalid_panel_or_target_column"}
        else:
            panel, target_column = prepared
            engine._ensure_sample_windows_resolved(panel)
            label_series = pd.to_numeric(panel[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            validation_panel = panel.iloc[np.flatnonzero(engine._build_sample_window_mask(panel, engine.validation_window))].copy()
            engine._configure_sharpe_sampling(validation_panel)
            for item in candidates:
                factor_name = str(item.get("factor_name", ""))
                expression = str(item.get("factor_expression", ""))
                try:
                    signal_series = calculate_factor_expression(expression, panel, factor_name)
                    if signal_series is None:
                        raise ValueError("factor calculation returned None")
                    window_panel, validation_label, validation_signal = engine._slice_sample_window_inputs(
                        panel=panel,
                        label_series=label_series,
                        signal_series=signal_series,
                        window=engine.validation_window,
                    )
                    validation_metrics = engine._calculate_factor_metrics_bundle(
                        factor_name=f"{factor_name}_validation",
                        expression=expression,
                        signal_series=validation_signal,
                        label_series=validation_label,
                        panel=window_panel,
                    )
                    validation_metrics_by_factor[factor_name] = engine._round_metrics_dict(validation_metrics)
                except Exception as exc:
                    errors[factor_name] = {
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=6),
                    }

    payload = build_holdout_validation_payload(
        candidates=candidates,
        validation_metrics_by_factor=validation_metrics_by_factor,
        metric=engine.train_filter_metric,
        threshold=engine.train_filter_min_value,
        train_window=engine._format_sample_window(engine.train_window),
        validation_window=engine._format_sample_window(engine.validation_window),
        errors=errors,
    )
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload
