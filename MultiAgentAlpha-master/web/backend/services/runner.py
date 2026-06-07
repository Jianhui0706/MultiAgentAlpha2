from __future__ import annotations

import os
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)


_active_runs: dict[str, dict[str, Any]] = {}
_active_runs_lock = threading.Lock()

_DATE_FIELD_CANDIDATES = (
    "datetime",
    "date",
    "trade_date",
    "trading_date",
    "cal_date",
    "timestamp",
    "time",
    "open_time",   # Binance kline CSV
    "dt",
    "ds",
)
_SYMBOL_FIELD_CANDIDATES = (
    "symbol",
    "instrument",
    "ts_code",
    "code",
    "ticker",
    "sec_code",
    "security",
)


class RunCancelled(RuntimeError):
    """Raised when a user requests a running workflow to stop."""


def _parse_datetime_values(values: Any) -> Any:
    import pandas as pd

    raw = pd.Series(values).dropna()
    if raw.empty:
        raise ValueError("no non-empty date values")

    text = raw.astype(str).str.strip()
    compact_digits = text.str.fullmatch(r"\d{8}")
    if bool(compact_digits.mean() >= 0.8):
        dt = pd.to_datetime(text.where(compact_digits), format="%Y%m%d", errors="coerce")
    elif pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        max_value = numeric.dropna().abs().max()
        if pd.isna(max_value):
            dt = pd.to_datetime(raw, errors="coerce")
        elif max_value >= 10**17:
            dt = pd.to_datetime(numeric, unit="ns", errors="coerce")
        elif max_value >= 10**14:
            dt = pd.to_datetime(numeric, unit="us", errors="coerce")
        elif max_value >= 10**11:
            dt = pd.to_datetime(numeric, unit="ms", errors="coerce")
        elif max_value >= 10**9:
            dt = pd.to_datetime(numeric, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(raw, errors="coerce")
    else:
        dt = pd.to_datetime(raw, errors="coerce")

    dt = pd.Series(dt).dropna()
    if dt.empty:
        raise ValueError("no parseable date values")
    return dt


def _find_column(df: Any, candidates: tuple[str, ...]) -> Any | None:
    lower_to_column = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        column = lower_to_column.get(candidate)
        if column is not None:
            return column
    return None


def _load_single_file(path: str) -> "Any":
    """Load one parquet or CSV file into a DataFrame."""
    import pandas as pd
    from pathlib import Path as _Path
    fp = _Path(path)
    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
        # Binance-style kline CSV: if no symbol column, derive from filename
        if "symbol" not in df.columns and "ts_code" not in df.columns:
            stem = fp.stem  # e.g. "BTCUSDT_5m" or "BTCUSDT"
            symbol = stem.split("_")[0].upper()
            df["symbol"] = symbol
        return df
    return pd.read_parquet(fp)


def _file_summary(path: str, df: "Any") -> dict:
    """Return a per-file summary dict for the panel_data_loaded progress event."""
    import pandas as pd
    from pathlib import Path as _Path
    fp = _Path(path)
    # Try to detect date column for date range
    date_col = None
    for candidate in _DATE_FIELD_CANDIDATES:
        if candidate in df.columns:
            date_col = candidate
            break
    # If multi-indexed with datetime level, use that
    if date_col is None and hasattr(df.index, "names") and "datetime" in (df.index.names or []):
        dates = pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce")
        start = str(dates.min())[:19] if len(dates) else "?"
        end = str(dates.max())[:19] if len(dates) else "?"
    elif date_col:
        try:
            dates = _parse_datetime_values(df[date_col].dropna().values)
            start = str(dates.min())[:19]
            end = str(dates.max())[:19]
        except Exception:
            start = end = "?"
    else:
        start = end = "?"
    # Symbol(s) in this file
    symbol_col = None
    for candidate in _SYMBOL_FIELD_CANDIDATES:
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col:
        symbols = sorted(df[symbol_col].dropna().unique().tolist())
    elif hasattr(df.index, "names") and "symbol" in (df.index.names or []):
        symbols = sorted(df.index.get_level_values("symbol").unique().tolist())
    else:
        symbols = []
    return {
        "filename": fp.name,
        "rows": len(df),
        "symbols": symbols[:20],   # cap list length
        "symbol_count": len(symbols),
        "start": start,
        "end": end,
    }


def _load_panel_from_paths(paths: list[str]) -> tuple["Any", list[dict]]:
    """Load one or many parquet/CSV files and concatenate into a single DataFrame.

    Returns (combined_df, file_summaries) where file_summaries is a list of
    per-file metadata dicts suitable for emitting in a progress event.
    """
    import pandas as pd
    if not paths:
        raise ValueError("No panel data paths provided")
    summaries: list[dict] = []
    frames = []
    for p in paths:
        df = _load_single_file(p)
        summaries.append(_file_summary(p, df))
        frames.append(df)
    combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return combined, summaries


def _standardize_panel_dataframe(df: Any) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    info: dict[str, Any] = {"changed": False, "original_rows": int(len(df))}
    panel = df.copy()
    index_names = list(getattr(panel.index, "names", []) or [])
    has_standard_index = "datetime" in index_names and "symbol" in index_names

    if not has_standard_index:
        date_column = _find_column(panel, _DATE_FIELD_CANDIDATES)
        symbol_column = _find_column(panel, _SYMBOL_FIELD_CANDIDATES)
        if date_column is None:
            raise ValueError(f"could not detect date column; tried {', '.join(_DATE_FIELD_CANDIDATES)}")
        if symbol_column is None:
            raise ValueError(f"could not detect symbol column; tried {', '.join(_SYMBOL_FIELD_CANDIDATES)}")

        parsed_dates = _parse_datetime_values(panel[date_column])
        if len(parsed_dates) != len(panel):
            parsed_dates = pd.to_datetime(panel[date_column], errors="coerce")
        if pd.Series(parsed_dates).isna().any():
            raise ValueError(f"date column {date_column} contains unparseable values")

        panel = panel.assign(datetime=parsed_dates.values, symbol=panel[symbol_column].astype(str).values)
        drop_columns = [col for col in (date_column, symbol_column) if col in panel.columns and col not in ("datetime", "symbol")]
        panel = panel.drop(columns=drop_columns)
        panel = panel.set_index(["datetime", "symbol"]).sort_index()
        info.update({"changed": True, "date_source": str(date_column), "symbol_source": str(symbol_column)})
    else:
        panel = panel.sort_index()

    if "volume" not in panel.columns and "vol" in panel.columns:
        panel["volume"] = panel["vol"]
        info["changed"] = True
        info["volume_source"] = "vol"

    if "fwd_returns_4h" not in panel.columns:
        if "close" not in panel.columns:
            raise ValueError("panel is missing fwd_returns_4h and cannot derive it because close column is missing")
        close = pd.to_numeric(panel["close"], errors="coerce")
        panel["fwd_returns_4h"] = close.groupby(level="symbol").shift(-4) / close - 1.0
        info["changed"] = True
        info["target_source"] = "derived from close shift(-4)"

    # Derive common feature columns expected by LLM-generated expressions.
    # These names match the $variable tokens the hypothesis agent generates.
    if "close" in panel.columns:
        g = pd.to_numeric(panel["close"], errors="coerce").groupby(level="symbol")
        for periods, name in [(1, "returns_1h"), (4, "returns_4h"), (12, "returns_12h"), (24, "returns_24h")]:
            if name not in panel.columns:
                panel[name] = g.pct_change(periods)
    if "high" in panel.columns and "low" in panel.columns and "close" in panel.columns:
        if "high_low_spread" not in panel.columns:
            panel["high_low_spread"] = (
                pd.to_numeric(panel["high"], errors="coerce") - pd.to_numeric(panel["low"], errors="coerce")
            ) / pd.to_numeric(panel["close"], errors="coerce").replace(0, float("nan"))
    vol_col = "volume" if "volume" in panel.columns else ("vol" if "vol" in panel.columns else None)
    if vol_col:
        gv = pd.to_numeric(panel[vol_col], errors="coerce").groupby(level="symbol")
        for periods, name in [(1, "volume_change_1h"), (4, "volume_change_4h"), (24, "volume_change_24h")]:
            if name not in panel.columns:
                panel[name] = gv.pct_change(periods)
        if "volume_quote" not in panel.columns:
            if "amount" in panel.columns:
                panel["volume_quote"] = pd.to_numeric(panel["amount"], errors="coerce")
            else:
                panel["volume_quote"] = pd.to_numeric(panel[vol_col], errors="coerce")

    return panel, info


def get_active_run(run_id: str) -> dict[str, Any] | None:
    with _active_runs_lock:
        run = _active_runs.get(run_id)
        return dict(run) if run is not None else None


def request_stop_run(run_id: str) -> dict[str, Any]:
    with _active_runs_lock:
        run = _active_runs.get(run_id)
        if run is None:
            return {"run_id": run_id, "status": "not_active", "stop_requested": False}
        if run.get("status") != "running":
            return {"run_id": run_id, "status": run.get("status"), "stop_requested": bool(run.get("stop_requested"))}
        run["stop_requested"] = True
        run["stop_requested_at"] = datetime.now().isoformat()
        callback = run.get("progress_callback")

    if callable(callback):
        callback(
            {
                "stage": "run_stop_requested",
                "run_id": run_id,
                "message": "Stop requested by user. The workflow will stop at the next cancellation checkpoint.",
            },
            check_cancel=False,
        )
    return {"run_id": run_id, "status": "stopping", "stop_requested": True}


def launch_run(
    *,
    run_id: str,
    params: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None],
) -> None:
    """Launch the workflow in a background thread."""
    from trading_agents.domains import CryptoCrossSectionDomainAdapter
    from trading_agents.infra.run_store import RunStore
    from trading_agents.workflows import AlphaFactorMiningWorkflow, EvolutionConfig
    from trading_agents.infra.llm import GatewayModelClient, LLMGateway, RetryPolicy
    from trading_agents.infra.llm.providers import OpenAICompatibleProvider
    from main import (
        _append_jsonl,
        _compact_progress_payload,
        _enrich_progress_payload,
        _factor_id_for_storage,
        _first_dict,
        _save_loop_trace,
    )

    panel_data_paths_raw: list[str] = params.get("panel_data_paths") or []
    panel_data_path = params.get("panel_data_path", str(PROJECT_ROOT / "data" / "panel_data.parquet"))
    # Collect all paths: multi-path list wins over single path
    all_panel_paths: list[str] = panel_data_paths_raw if panel_data_paths_raw else [panel_data_path]
    loop_count = int(params.get("loop_count", 1))
    evolution_mode = bool(params.get("evolution_mode", False))
    evolution_profile = str(params.get("evolution_profile", "quantaalpha") or "quantaalpha").strip().lower() or "quantaalpha"
    research_report_pdf = str(params.get("research_report_pdf", "") or "")
    initial_hypothesis = str(params.get("initial_hypothesis", "") or "").strip()
    run_store_path = str(params.get("run_store_path", RunStore.default_path(PROJECT_ROOT)))
    start_date: str | None = params.get("start_date")  # e.g. "2022-03"
    end_date: str | None = params.get("end_date")      # e.g. "2023-09"
    log_dir = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id
    progress_file = log_dir / "progress.jsonl"
    run_store = RunStore(run_store_path)
    log_lock = threading.Lock()
    progress_sequence = 0
    run_store.start_run(run_id=run_id, log_dir=str(log_dir), args=dict(params))

    def _stop_requested() -> bool:
        with _active_runs_lock:
            return bool(_active_runs.get(run_id, {}).get("stop_requested"))

    def _check_cancelled() -> None:
        if _stop_requested():
            raise RunCancelled(f"run {run_id} cancelled by user")

    def _emit_progress(payload: dict[str, Any], *, check_cancel: bool = True) -> None:
        nonlocal progress_sequence
        if check_cancel:
            _check_cancelled()
        enriched = _compact_progress_payload(_enrich_progress_payload(dict(payload)))
        progress_sequence += 1
        enriched.setdefault("sequence", progress_sequence)
        enriched.setdefault("run_id", run_id)
        enriched.setdefault("timestamp", datetime.now().isoformat())
        with log_lock:
            _append_jsonl(progress_file, enriched)
        progress_callback(enriched)

    _active_runs[run_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "stop_requested": False,
        "progress_callback": _emit_progress,
    }

    _emit_progress(
        {
            "stage": "session_started",
            "loop_count": loop_count,
            "evolution_mode": evolution_mode,
            "evolution_profile": evolution_profile,
            "evolution_iterations": int(params.get("evolution_iterations", 5)),
            "evolution_num_initial_hypotheses": int(params.get("evolution_num_initial_hypotheses", 10)),
            "research_report_pdf": research_report_pdf or None,
        }
    )

    def _run() -> None:
        tmp_path: Path | None = None
        session = None
        try:
            _check_cancelled()
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("LLM_MODEL", "")
            base_url = os.getenv("OPENAI_BASE_URL", None) or None
            provider = OpenAICompatibleProvider(api_key=api_key, base_url=base_url)
            gateway = LLMGateway(
                provider=provider,
                retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
                default_model=model,
            )
            model_client = GatewayModelClient(gateway)

            workflow = AlphaFactorMiningWorkflow(
                model_client=model_client,
                progress_callback=_emit_progress,
            )

            # Normalize user-provided market tables into the workflow panel contract.
            _check_cancelled()
            import pandas as pd
            df, file_summaries = _load_panel_from_paths(all_panel_paths)
            _emit_progress({
                "stage": "panel_data_loaded",
                "file_count": len(file_summaries),
                "files": file_summaries,
                "total_rows": sum(s["rows"] for s in file_summaries),
            })
            df, normalization_info = _standardize_panel_dataframe(df)
            if start_date or end_date:
                dt_series = pd.to_datetime(df.index.get_level_values("datetime"))
                mask = pd.Series([True] * len(df), index=df.index)
                if start_date:
                    mask &= dt_series >= pd.Timestamp(start_date)
                if end_date:
                    # End of the selected month (inclusive)
                    end_ts = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    mask &= dt_series <= end_ts
                df = df[mask.values]
                _emit_progress({
                    "stage": "data_filter_applied",
                    "original_rows": int(normalization_info.get("original_rows", len(mask))),
                    "filtered_rows": len(df),
                    "start_date": start_date,
                    "end_date": end_date,
                })

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                tmp_path = Path(f.name)
            df.to_parquet(str(tmp_path))
            effective_path = str(tmp_path)
            _emit_progress({
                "stage": "panel_data_normalized",
                "panel_data_path": str(panel_data_path),
                "effective_panel_data_path": effective_path,
                "row_count": len(df),
                **normalization_info,
            })

            _check_cancelled()
            adapter = CryptoCrossSectionDomainAdapter(effective_path)
            payload = adapter.build_initial_payload()
            dataset_summary = payload.get("domain_dataset_summary", {})
            _emit_progress(
                {
                    "stage": "market_payload_finished",
                    "panel_data_path": str(panel_data_path),
                    "feature_count": len(list(dataset_summary.get("feature_columns", []))) if isinstance(dataset_summary, dict) else 0,
                    "target_count": len(list(dataset_summary.get("target_columns", []))) if isinstance(dataset_summary, dict) else 0,
                    "row_count": int(dataset_summary.get("rows", 0) or 0) if isinstance(dataset_summary, dict) else 0,
                    "symbol_count": int(dataset_summary.get("symbol_count", 0) or 0) if isinstance(dataset_summary, dict) else 0,
                    "datetime_start": str(dataset_summary.get("datetime_start", "")) if isinstance(dataset_summary, dict) else "",
                    "datetime_end": str(dataset_summary.get("datetime_end", "")) if isinstance(dataset_summary, dict) else "",
                }
            )

            _check_cancelled()
            if research_report_pdf:
                payload = workflow.build_report_driven_initial_payload(
                    initial_shared_payload=payload,
                    research_report_pdf=research_report_pdf,
                )
            if initial_hypothesis:
                payload["initial_hypothesis"] = initial_hypothesis

            _check_cancelled()
            if evolution_mode:
                evolution_iterations = max(1, int(params.get("evolution_iterations", 5)))
                profile_is_quantaalpha = evolution_profile == "quantaalpha"
                default_initial_directions = 10 if profile_is_quantaalpha else 5
                default_mutation_children = 10 if profile_is_quantaalpha else 3
                default_crossover_children = 10 if profile_is_quantaalpha else 2
                effective_initial_directions = max(
                    1,
                    int(params.get("evolution_num_initial_hypotheses", default_initial_directions)),
                )
                effective_mutation_tasks = max(
                    0,
                    int(params.get("evolution_mutation_tasks_per_round", default_mutation_children)),
                )
                effective_crossover_tasks = max(
                    0,
                    int(params.get("evolution_crossover_tasks_per_round", default_crossover_children)),
                )
                effective_max_rounds = max(
                    1,
                    int(params.get(
                        "evolution_max_rounds",
                        evolution_iterations * 2 + 1 if profile_is_quantaalpha else 6,
                    )),
                )
                session = workflow.run_evolution_session(
                    initial_shared_payload=payload,
                    initial_direction=initial_hypothesis or None,
                    evolution_config=EvolutionConfig(
                        profile=evolution_profile,
                        iterations=evolution_iterations,
                        phase_schedule="mutation_then_crossover" if profile_is_quantaalpha else "legacy",
                        max_rounds=effective_max_rounds,
                        num_initial_directions=effective_initial_directions,
                        mutation_tasks_per_round=effective_mutation_tasks,
                        crossover_tasks_per_round=effective_crossover_tasks,
                        children_per_mutation_phase=effective_mutation_tasks,
                        children_per_crossover_phase=effective_crossover_tasks,
                        factor_expressions_per_hypothesis=3,
                        evolved_directions_per_round=max(1, int(params.get("evolution_evolved_directions_per_round", 5))),
                        mutation_parent_top_n=max(1, int(params.get("evolution_mutation_parent_top_n", 10))),
                        crossover_parent_top_n=max(1, int(params.get("evolution_crossover_parent_top_n", 10))),
                        mutation_top_n=max(0, int(params.get("evolution_mutation_top_n", 3))),
                        crossover_group_size=max(1, int(params.get("evolution_crossover_group_size", 2))),
                        crossover_n_groups=max(0, int(params.get("evolution_crossover_n_groups", 2))),
                        parallel_enabled=bool(params.get("evolution_parallel_enabled", True)),
                        max_workers=max(1, int(params.get("evolution_max_workers", 5))),
                    ),
                )
            else:
                session = workflow.run_loop_session(
                    initial_shared_payload=payload,
                    loop_count=loop_count,
                )
            _check_cancelled()
            for idx, trace in enumerate(session.loop_traces, start=1):
                loop_summary, compact_final_context, agent_records, _debug_path = _save_loop_trace(log_dir, idx, trace)
                run_store.record_loop(
                    run_id=run_id,
                    loop_index=idx,
                    summary=loop_summary,
                    final_context=compact_final_context,
                )
                for agent_record in agent_records:
                    run_store.record_agent(
                        run_id=run_id,
                        loop_index=idx,
                        agent_index=int(agent_record.get("agent_index", 0) or 0),
                        payload=agent_record,
                    )
                trajectory_id = str(loop_summary.get("trajectory_id", ""))
                for factor in list(loop_summary.get("factors", []) or []):
                    if not isinstance(factor, dict):
                        continue
                    run_store.upsert_factor(
                        run_id=run_id,
                        loop_index=idx,
                        trajectory_id=trajectory_id,
                        factor_id=_factor_id_for_storage(trajectory_id, factor),
                        factor=factor,
                        metrics=_first_dict(factor.get("metrics", {})),
                    )
            for trajectory in session.trajectory_pool.get_all():
                run_store.upsert_trajectory(run_id=run_id, trajectory=trajectory)
            run_store.finish_run(run_id=run_id, status=session.status, summary={"run_id": run_id, "status": session.status})
            with _active_runs_lock:
                _active_runs[run_id]["status"] = session.status
        except RunCancelled as exc:
            with _active_runs_lock:
                _active_runs[run_id]["status"] = "cancelled"
                _active_runs[run_id]["error"] = str(exc)
            run_store.finish_run(run_id=run_id, status="cancelled", summary={"run_id": run_id, "status": "cancelled", "error": str(exc)})
            _emit_progress({"stage": "run_cancelled", "run_id": run_id, "error": str(exc)}, check_cancel=False)
        except Exception as exc:
            with _active_runs_lock:
                _active_runs[run_id]["status"] = "failed"
                _active_runs[run_id]["error"] = str(exc)
            run_store.finish_run(run_id=run_id, status="failed", summary={"run_id": run_id, "status": "failed", "error": str(exc)})
            _emit_progress({"stage": "session_failed", "run_id": run_id, "error": str(exc)}, check_cancel=False)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    thread = threading.Thread(target=_run, daemon=True, name=f"run-{run_id}")
    thread.start()
