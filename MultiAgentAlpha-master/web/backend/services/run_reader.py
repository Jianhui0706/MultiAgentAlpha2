from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _j(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row["name"]) == column_name for row in rows)


def list_runs(db_path: str | Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.run_id, r.status, r.started_at, r.ended_at, r.args_json, r.summary_json,
                   COUNT(DISTINCT f.factor_id) AS factor_count,
                   COUNT(DISTINCT l.loop_index) AS loop_count
            FROM runs r
            LEFT JOIN factors f ON f.run_id = r.run_id
            LEFT JOIN loops l ON l.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.started_at DESC
            """
        ).fetchall()
    result = []
    for row in rows:
        args = _j(row["args_json"]) or {}
        summary = _j(row["summary_json"]) or {}
        result.append(
            {
                "run_id": row["run_id"],
                "status": row["status"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "factor_count": row["factor_count"],
                "loop_count": row["loop_count"],
                "evolution_mode": args.get("evolution_mode", False),
                "loop_count_arg": args.get("loop_count", 1),
                "research_report_pdf": args.get("research_report_pdf", ""),
                "overview": summary.get("overview", {}),
            }
        )
    return result


def get_run(db_path: str | Path, run_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "log_dir": row["log_dir"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "args": _j(row["args_json"]) or {},
        "summary": _j(row["summary_json"]) or {},
    }


def get_loops(db_path: str | Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT loop_index, status, phase, round_idx, direction_id, trajectory_id,
                   parent_ids_json, error_count, agent_count, elapsed_seconds,
                   started_at, ended_at, summary_json
            FROM loops WHERE run_id = ?
            ORDER BY loop_index
            """,
            (run_id,),
        ).fetchall()
    result = []
    for row in rows:
        summary = _j(row["summary_json"]) or {}
        result.append(
            {
                "loop_index": row["loop_index"],
                "status": row["status"],
                "phase": row["phase"],
                "round_idx": row["round_idx"],
                "direction_id": row["direction_id"],
                "trajectory_id": row["trajectory_id"],
                "parent_ids": _j(row["parent_ids_json"]) or [],
                "error_count": row["error_count"],
                "agent_count": row["agent_count"],
                "elapsed_seconds": row["elapsed_seconds"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "hypothesis": summary.get("hypothesis", ""),
                "feedback_summary": summary.get("feedback_summary", ""),
                "factors": summary.get("factors", []),
                "metrics": summary.get("metrics", {}),
            }
        )
    return result


def get_factors(db_path: str | Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        metric_join = (
            "m.factor_id = f.factor_id AND m.run_id = f.run_id"
            if _has_column(conn, "factor_metrics", "run_id")
            else "m.factor_id = f.factor_id"
        )
        rows = conn.execute(
            f"""
            SELECT f.factor_id, f.loop_index, f.trajectory_id, f.factor_name, f.expression,
                   f.description, f.formulation, f.implementation_status,
                   MAX(CASE WHEN m.metric_name = 'sharpe' THEN m.metric_value END) AS sharpe,
                   MAX(CASE WHEN m.metric_name = 'IC' THEN m.metric_value END) AS IC,
                   MAX(CASE WHEN m.metric_name = 'Rank IC' THEN m.metric_value END) AS rank_ic,
                   MAX(CASE WHEN m.metric_name = 'ICIR' THEN m.metric_value END) AS icir,
                   MAX(CASE WHEN m.metric_name = 'Rank ICIR' THEN m.metric_value END) AS rank_icir,
                   MAX(CASE WHEN m.metric_name = 'coverage' THEN m.metric_value END) AS coverage,
                   MAX(CASE WHEN m.metric_name = 'turnover' THEN m.metric_value END) AS turnover,
                   MAX(CASE WHEN m.metric_name = 'long_short_mean_return'    THEN m.metric_value END) AS ls_return,
                   MAX(CASE WHEN m.metric_name = 'annualized_return'          THEN m.metric_value END) AS ann_return,
                   MAX(CASE WHEN m.metric_name = 'yearly_returns'             THEN m.metric_json  END) AS yearly_returns,
                   MAX(CASE WHEN m.metric_name = 'quality_gate_passed'        THEN m.metric_value END) AS quality_gate_passed,
                   MAX(CASE WHEN m.metric_name = 'quality_gate_threshold'     THEN m.metric_value END) AS quality_gate_threshold,
                   MAX(CASE WHEN m.metric_name = 'layer_1_mean_return'        THEN m.metric_value END) AS layer_1,
                   MAX(CASE WHEN m.metric_name = 'layer_2_mean_return'        THEN m.metric_value END) AS layer_2,
                   MAX(CASE WHEN m.metric_name = 'layer_3_mean_return'        THEN m.metric_value END) AS layer_3,
                   MAX(CASE WHEN m.metric_name = 'layer_4_mean_return'        THEN m.metric_value END) AS layer_4,
                   MAX(CASE WHEN m.metric_name = 'layer_5_mean_return'        THEN m.metric_value END) AS layer_5,
                   MAX(CASE WHEN m.metric_name = 'layer_monotonicity'         THEN m.metric_value END) AS layer_monotonicity,
                   MAX(CASE WHEN m.metric_name = 'ls_cumulative_series'       THEN m.metric_json  END) AS ls_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_1_cumulative_series'  THEN m.metric_json  END) AS layer_1_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_5_cumulative_series'  THEN m.metric_json  END) AS layer_5_cumulative_series,
                   l.phase, l.round_idx,
                   l.summary_json
            FROM factors f
            LEFT JOIN factor_metrics m ON {metric_join}
            LEFT JOIN loops l ON l.run_id = f.run_id AND l.loop_index = f.loop_index
            WHERE f.run_id = ?
            GROUP BY f.factor_id
            ORDER BY f.loop_index, sharpe DESC NULLS LAST
            """,
            (run_id,),
        ).fetchall()

    result = []
    for row in rows:
        loop_summary = _j(row["summary_json"]) or {}
        result.append(
            {
                "factor_id": row["factor_id"],
                "loop_index": row["loop_index"],
                "trajectory_id": row["trajectory_id"],
                "factor_name": row["factor_name"],
                "expression": row["expression"],
                "description": row["description"],
                "formulation": row["formulation"],
                "implementation_status": row["implementation_status"],
                "phase": row["phase"],
                "round_idx": row["round_idx"],
                # metrics
                "sharpe": row["sharpe"],
                "IC": row["IC"],
                "rank_ic": row["rank_ic"],
                "icir": row["icir"],
                "rank_icir": row["rank_icir"],
                "coverage": row["coverage"],
                "turnover": row["turnover"],
                "ls_return": row["ls_return"],
                "ann_return": row["ann_return"],
                "yearly_returns": _j(row["yearly_returns"]) if row["yearly_returns"] else None,
                "quality_gate_passed": row["quality_gate_passed"],
                "quality_gate_threshold": row["quality_gate_threshold"],
                "layer_1": row["layer_1"],
                "layer_2": row["layer_2"],
                "layer_3": row["layer_3"],
                "layer_4": row["layer_4"],
                "layer_5": row["layer_5"],
                "layer_monotonicity": row["layer_monotonicity"],
                "ls_cumulative_series": _j(row["ls_cumulative_series"]) if row["ls_cumulative_series"] else None,
                "layer_1_cumulative_series": _j(row["layer_1_cumulative_series"]) if row["layer_1_cumulative_series"] else None,
                "layer_5_cumulative_series": _j(row["layer_5_cumulative_series"]) if row["layer_5_cumulative_series"] else None,
                # source context from loop
                "hypothesis": loop_summary.get("hypothesis", ""),
                "feedback_summary": loop_summary.get("feedback_summary", ""),
            }
        )
    return result


def get_trajectories(db_path: str | Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT trajectory_id, phase, round_idx, direction_id, parent_ids_json,
                   hypothesis, reward_score, metrics_json
            FROM trajectories WHERE run_id = ?
            ORDER BY round_idx, direction_id
            """,
            (run_id,),
        ).fetchall()
    return [
        {
            "trajectory_id": row["trajectory_id"],
            "phase": row["phase"],
            "round_idx": row["round_idx"],
            "direction_id": row["direction_id"],
            "parent_ids": _j(row["parent_ids_json"]) or [],
            "hypothesis": row["hypothesis"],
            "reward_score": row["reward_score"],
            "metrics": _j(row["metrics_json"]) or {},
        }
        for row in rows
    ]


def get_top_factors(db_path: str | Path, limit: int = 30) -> list[dict[str, Any]]:
    """Return top factors across all runs, ranked by Sharpe descending."""
    with _connect(db_path) as conn:
        metric_join = (
            "m.factor_id = f.factor_id AND m.run_id = f.run_id"
            if _has_column(conn, "factor_metrics", "run_id")
            else "m.factor_id = f.factor_id"
        )
        rows = conn.execute(
            f"""
            SELECT f.factor_id, f.run_id, f.loop_index, f.trajectory_id,
                   f.factor_name, f.expression, f.description, f.implementation_status,
                   t.reward_score,
                   MAX(CASE WHEN m.metric_name = 'sharpe'                 THEN m.metric_value END) AS sharpe,
                   MAX(CASE WHEN m.metric_name = 'IC'                     THEN m.metric_value END) AS IC,
                   MAX(CASE WHEN m.metric_name = 'Rank IC'                THEN m.metric_value END) AS rank_ic,
                   MAX(CASE WHEN m.metric_name = 'ICIR'                   THEN m.metric_value END) AS icir,
                   MAX(CASE WHEN m.metric_name = 'Rank ICIR'              THEN m.metric_value END) AS rank_icir,
                   MAX(CASE WHEN m.metric_name = 'coverage'               THEN m.metric_value END) AS coverage,
                   MAX(CASE WHEN m.metric_name = 'turnover'               THEN m.metric_value END) AS turnover,
                   MAX(CASE WHEN m.metric_name = 'long_short_mean_return' THEN m.metric_value END) AS ls_return,
                   MAX(CASE WHEN m.metric_name = 'annualized_return'      THEN m.metric_value END) AS ann_return,
                   MAX(CASE WHEN m.metric_name = 'yearly_returns'         THEN m.metric_json  END) AS yearly_returns,
                   MAX(CASE WHEN m.metric_name = 'layer_1_mean_return'    THEN m.metric_value END) AS layer_1,
                   MAX(CASE WHEN m.metric_name = 'layer_2_mean_return'    THEN m.metric_value END) AS layer_2,
                   MAX(CASE WHEN m.metric_name = 'layer_3_mean_return'    THEN m.metric_value END) AS layer_3,
                   MAX(CASE WHEN m.metric_name = 'layer_4_mean_return'    THEN m.metric_value END) AS layer_4,
                   MAX(CASE WHEN m.metric_name = 'layer_5_mean_return'    THEN m.metric_value END) AS layer_5,
                   MAX(CASE WHEN m.metric_name = 'layer_monotonicity'     THEN m.metric_value END) AS layer_monotonicity,
                   MAX(CASE WHEN m.metric_name = 'ls_cumulative_series'    THEN m.metric_json  END) AS ls_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_1_cumulative_series' THEN m.metric_json END) AS layer_1_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_5_cumulative_series' THEN m.metric_json END) AS layer_5_cumulative_series,
                   r.args_json
            FROM factors f
            LEFT JOIN factor_metrics m ON {metric_join}
            LEFT JOIN trajectories t ON t.trajectory_id = f.trajectory_id AND t.run_id = f.run_id
            LEFT JOIN runs r ON r.run_id = f.run_id
            WHERE f.implementation_status NOT IN ('failed', 'error')
            GROUP BY f.run_id, f.factor_id
            HAVING reward_score IS NOT NULL OR sharpe IS NOT NULL
            ORDER BY COALESCE(t.reward_score, sharpe * 0.1, 0) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for rank, row in enumerate(rows, start=1):
        args = _j(row["args_json"]) or {}
        run_name = args.get("run_name", "") or row["run_id"]
        ls_series = _j(row["ls_cumulative_series"]) if row["ls_cumulative_series"] else None
        layer1_series = _j(row["layer_1_cumulative_series"]) if row["layer_1_cumulative_series"] else None
        layer5_series = _j(row["layer_5_cumulative_series"]) if row["layer_5_cumulative_series"] else None
        yearly = _j(row["yearly_returns"]) if row["yearly_returns"] else None
        result.append({
            "rank": rank,
            "factor_id": row["factor_id"],
            "run_id": row["run_id"],
            "run_name": run_name,
            "loop_index": row["loop_index"],
            "factor_name": row["factor_name"],
            "expression": row["expression"],
            "description": row["description"],
            "implementation_status": row["implementation_status"],
            "reward_score": row["reward_score"],
            "sharpe": row["sharpe"],
            "IC": row["IC"],
            "rank_ic": row["rank_ic"],
            "icir": row["icir"],
            "rank_icir": row["rank_icir"],
            "coverage": row["coverage"],
            "turnover": row["turnover"],
            "ls_return": row["ls_return"],
            "ann_return": row["ann_return"],
            "yearly_returns": yearly,
            "layer_1": row["layer_1"],
            "layer_2": row["layer_2"],
            "layer_3": row["layer_3"],
            "layer_4": row["layer_4"],
            "layer_5": row["layer_5"],
            "layer_monotonicity": row["layer_monotonicity"],
            "ls_cumulative_series": ls_series,
            "layer_1_cumulative_series": layer1_series,
            "layer_5_cumulative_series": layer5_series,
        })
    return result


def get_progress_events(log_dir: str, limit: int = 200) -> list[dict[str, Any]]:
    """Read last N events from progress.jsonl for a given run."""
    progress_file = Path(log_dir) / "progress.jsonl"
    if not progress_file.exists():
        return []
    lines = progress_file.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events
