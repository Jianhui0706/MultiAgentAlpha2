from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL 需要 -shm 文件的内存映射，在 exFAT/网络盘（移动硬盘常见）上会报
    # "disk I/O error"；跨机器拷贝项目时残留的 -wal/-shm 文件也会触发。
    # 失败时降级为默认 journal 模式，读取功能完全不受影响。
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.OperationalError:
            pass
    return conn


def _j(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _reward_breakdown(metrics: dict[str, Any], *, expression: str = "") -> dict[str, Any]:
    from trading_agents.infra.factor_library_manager import FactorLibraryManager

    factor_entry = {"factor_expression": expression} if expression else None
    return FactorLibraryManager.compute_reward_breakdown(metrics, factor_entry=factor_entry)


def _has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(row["name"]) == column_name for row in rows)


def _expression_identity(expression: object) -> str:
    return re.sub(r"\s+", "", str(expression or "").strip()).upper()


def _numeric_score(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _factor_dedupe_score(factor: dict[str, Any]) -> tuple[int, float]:
    for key, absolute in (
        ("factor_reward_score", False),
        ("reward_score", False),
        ("sharpe", False),
        ("rank_ic", True),
        ("IC", True),
        ("ann_return", False),
    ):
        value = _numeric_score(factor.get(key))
        if value is None:
            continue
        return (1, abs(value) if absolute else value)
    return (0, 0.0)


def _dedupe_factors_by_expression(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for factor in factors:
        identity = _expression_identity(factor.get("expression"))
        key = f"expr:{identity}" if identity else f"id:{factor.get('factor_id', '')}"
        if key not in metadata:
            metadata[key] = {"count": 0, "trajectory_ids": []}
        meta = metadata[key]
        meta["count"] = int(meta.get("count", 0)) + 1
        trajectory_id = str(factor.get("trajectory_id", "") or "").strip()
        trajectory_ids = list(meta.get("trajectory_ids", []))
        if trajectory_id and trajectory_id not in trajectory_ids:
            trajectory_ids.append(trajectory_id)
        meta["trajectory_ids"] = trajectory_ids

        if key not in kept:
            order.append(key)
            kept[key] = dict(factor)
            kept[key]["duplicate_expression_count"] = int(meta["count"])
            kept[key]["duplicate_trajectory_ids"] = list(trajectory_ids)
            continue

        current = kept[key]
        if _factor_dedupe_score(factor) > _factor_dedupe_score(current):
            replacement = dict(factor)
            replacement["duplicate_expression_count"] = int(meta["count"])
            replacement["duplicate_trajectory_ids"] = list(trajectory_ids)
            kept[key] = replacement
        else:
            current["duplicate_expression_count"] = int(meta["count"])
            current["duplicate_trajectory_ids"] = list(trajectory_ids)
    return [kept[key] for key in order]


def _leaderboard_score(factor: dict[str, Any]) -> float:
    for key in ("factor_reward_score", "reward_score"):
        value = _numeric_score(factor.get(key))
        if value is not None:
            return value
    sharpe = _numeric_score(factor.get("sharpe"))
    return sharpe * 0.1 if sharpe is not None else 0.0


def _leaderboard_sort_key(factor: dict[str, Any]) -> tuple[int, float]:
    return (1 if factor.get("admission_passed") is not False else 0, _leaderboard_score(factor))


def dedupe_factors_by_expression(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_factors_by_expression(factors)


def list_runs(db_path: str | Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.run_id, r.status, r.started_at, r.ended_at, r.args_json, r.summary_json,
                   COUNT(DISTINCT f.factor_id) AS factor_count,
                   COUNT(DISTINCT l.loop_index) AS loop_count,
                   COUNT(DISTINCT t.trajectory_id) AS trajectory_count
            FROM runs r
            LEFT JOIN factors f ON f.run_id = r.run_id
            LEFT JOIN loops l ON l.run_id = r.run_id
            LEFT JOIN trajectories t ON t.run_id = r.run_id
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
                "trajectory_count": row["trajectory_count"],
                "evolution_mode": args.get("evolution_mode", False),
                "loop_count_arg": args.get("loop_count", 1),
                "research_report_pdf": args.get("research_report_pdf", ""),
                "research_report_pdfs": args.get("research_report_pdfs", []),
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
        run_args_row = conn.execute(
            "SELECT args_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        # 容错配对：run_id 对得上就用；若任一边为空（旧数据迁移常把 metrics 的
        # run_id 填成空字符串），仍按 factor_id 读出指标，避免指标读不到显示成 "—"。
        metric_join = (
            "m.factor_id = f.factor_id AND (m.run_id = f.run_id OR m.run_id = '' OR f.run_id = '')"
            if _has_column(conn, "factor_metrics", "run_id")
            else "m.factor_id = f.factor_id"
        )
        rows = conn.execute(
            f"""
            SELECT f.factor_id, f.loop_index, f.trajectory_id, f.factor_name, f.expression,
                   f.description, f.formulation, f.implementation_status,
                   t.reward_score,
                   MAX(CASE WHEN m.metric_name = 'reward_score' THEN m.metric_value END) AS factor_reward_score,
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
                   MAX(CASE WHEN m.metric_name = 'valid_cross_section_mean'   THEN m.metric_value END) AS valid_cross_section_mean,
                   MAX(CASE WHEN m.metric_name = 'cross_section_universe_size' THEN m.metric_value END) AS cross_section_universe_size,
                   MAX(CASE WHEN m.metric_name = 'ic_win_rate'                THEN m.metric_value END) AS ic_win_rate,
                   MAX(CASE WHEN m.metric_name = 'rank_ic_win_rate'           THEN m.metric_value END) AS rank_ic_win_rate,
                   MAX(CASE WHEN m.metric_name = 'drawdown'                   THEN m.metric_value END) AS drawdown,
                   MAX(CASE WHEN m.metric_name = 'long_short_positive_ratio'  THEN m.metric_value END) AS long_short_positive_ratio,
                   MAX(CASE WHEN m.metric_name = 'layer_1_mean_return'        THEN m.metric_value END) AS layer_1,
                   MAX(CASE WHEN m.metric_name = 'layer_2_mean_return'        THEN m.metric_value END) AS layer_2,
                   MAX(CASE WHEN m.metric_name = 'layer_3_mean_return'        THEN m.metric_value END) AS layer_3,
                   MAX(CASE WHEN m.metric_name = 'layer_4_mean_return'        THEN m.metric_value END) AS layer_4,
                   MAX(CASE WHEN m.metric_name = 'layer_5_mean_return'        THEN m.metric_value END) AS layer_5,
                   MAX(CASE WHEN m.metric_name = 'layer_monotonicity'         THEN m.metric_value END) AS layer_monotonicity,
                   MAX(CASE WHEN m.metric_name = 'ls_cumulative_series'       THEN m.metric_json  END) AS ls_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_1_cumulative_series'  THEN m.metric_json  END) AS layer_1_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_2_cumulative_series'  THEN m.metric_json  END) AS layer_2_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_3_cumulative_series'  THEN m.metric_json  END) AS layer_3_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_4_cumulative_series'  THEN m.metric_json  END) AS layer_4_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_5_cumulative_series'  THEN m.metric_json  END) AS layer_5_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'benchmark_cumulative_series' THEN m.metric_json END) AS benchmark_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'ic_cumulative_series'       THEN m.metric_json  END) AS ic_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'rank_ic_cumulative_series'  THEN m.metric_json  END) AS rank_ic_cumulative_series,
                   l.phase, l.round_idx,
                   l.summary_json
            FROM factors f
            LEFT JOIN factor_metrics m ON {metric_join}
            LEFT JOIN loops l ON l.run_id = f.run_id AND l.loop_index = f.loop_index
            LEFT JOIN trajectories t ON t.run_id = f.run_id AND t.trajectory_id = f.trajectory_id
            WHERE f.run_id = ?
            GROUP BY f.factor_id
            ORDER BY f.loop_index, sharpe DESC NULLS LAST
            """,
            (run_id,),
        ).fetchall()

    # 入库判定：与因子库排行榜同口径的 9 项 admission gate（缺指标跳过该项）。
    # 前端用 admission_passed 区分「成功（达标且入库）」与「部分（达标未入库）」。
    from trading_agents.infra.factor_library_manager import FactorLibraryManager

    run_args = _j(run_args_row["args_json"]) if run_args_row else {}
    run_sharpe_threshold = _float_or_none((run_args or {}).get("train_sharpe_threshold"))

    result = []
    for row in rows:
        loop_summary = _j(row["summary_json"]) or {}
        reward_metrics = {
            "sharpe": row["sharpe"],
            "IC": row["IC"],
            "Rank IC": row["rank_ic"],
            "ICIR": row["icir"],
            "Rank ICIR": row["rank_icir"],
            "coverage": row["coverage"],
            "turnover": row["turnover"],
            "drawdown": row["drawdown"],
        }
        reward_breakdown = _reward_breakdown(reward_metrics, expression=str(row["expression"] or ""))
        factor_reward_score = _float_or_none(row["factor_reward_score"])
        if factor_reward_score is None:
            factor_reward_score = _float_or_none(reward_breakdown.get("score"))
        admission_gate = FactorLibraryManager.evaluate_admission_gate({
            "valid_cross_section_mean": row["valid_cross_section_mean"],
            "cross_section_universe_size": row["cross_section_universe_size"],
            "IC": row["IC"],
            "ic_win_rate": row["ic_win_rate"],
            "ICIR": row["icir"],
            "Rank ICIR": row["rank_icir"],
            "rank_ic_win_rate": row["rank_ic_win_rate"],
            "sharpe": row["sharpe"],
            "drawdown": row["drawdown"],
            "long_short_positive_ratio": row["long_short_positive_ratio"],
        }, sharpe_threshold=run_sharpe_threshold)
        admission_failed_checks = [
            str(check.get("name", ""))
            for check in admission_gate.get("checks", [])
            if not check.get("skipped", False) and not check.get("passed", True)
        ]
        result.append(
            {
                "admission_passed": bool(admission_gate.get("passed", False)),
                "admission_failed_checks": admission_failed_checks,
                "factor_id": row["factor_id"],
                "loop_index": row["loop_index"],
                "trajectory_id": row["trajectory_id"],
                "factor_name": row["factor_name"],
                "expression": row["expression"],
                "description": row["description"],
                "formulation": row["formulation"],
                "implementation_status": row["implementation_status"],
                "reward_score": row["reward_score"],
                "factor_reward_score": factor_reward_score,
                "reward_breakdown": reward_breakdown,
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
                "valid_cross_section_mean": row["valid_cross_section_mean"],
                "cross_section_universe_size": row["cross_section_universe_size"],
                "ic_win_rate": row["ic_win_rate"],
                "rank_ic_win_rate": row["rank_ic_win_rate"],
                "drawdown": row["drawdown"],
                "long_short_positive_ratio": row["long_short_positive_ratio"],
                "layer_1": row["layer_1"],
                "layer_2": row["layer_2"],
                "layer_3": row["layer_3"],
                "layer_4": row["layer_4"],
                "layer_5": row["layer_5"],
                "layer_monotonicity": row["layer_monotonicity"],
                "ls_cumulative_series": _j(row["ls_cumulative_series"]) if row["ls_cumulative_series"] else None,
                "layer_1_cumulative_series": _j(row["layer_1_cumulative_series"]) if row["layer_1_cumulative_series"] else None,
                "layer_2_cumulative_series": _j(row["layer_2_cumulative_series"]) if row["layer_2_cumulative_series"] else None,
                "layer_3_cumulative_series": _j(row["layer_3_cumulative_series"]) if row["layer_3_cumulative_series"] else None,
                "layer_4_cumulative_series": _j(row["layer_4_cumulative_series"]) if row["layer_4_cumulative_series"] else None,
                "layer_5_cumulative_series": _j(row["layer_5_cumulative_series"]) if row["layer_5_cumulative_series"] else None,
                "benchmark_cumulative_series": _j(row["benchmark_cumulative_series"]) if row["benchmark_cumulative_series"] else None,
                "ic_cumulative_series": _j(row["ic_cumulative_series"]) if row["ic_cumulative_series"] else None,
                "rank_ic_cumulative_series": _j(row["rank_ic_cumulative_series"]) if row["rank_ic_cumulative_series"] else None,
                # source context from loop
                "hypothesis": loop_summary.get("hypothesis", ""),
                "feedback_summary": loop_summary.get("feedback_summary", ""),
            }
        )
    return _dedupe_factors_by_expression(result)


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
        # 容错配对：run_id 对得上就用；若任一边为空（旧数据迁移常把 metrics 的
        # run_id 填成空字符串），仍按 factor_id 读出指标，避免指标读不到显示成 "—"。
        metric_join = (
            "m.factor_id = f.factor_id AND (m.run_id = f.run_id OR m.run_id = '' OR f.run_id = '')"
            if _has_column(conn, "factor_metrics", "run_id")
            else "m.factor_id = f.factor_id"
        )
        hidden_filter = (
            "AND COALESCE(f.leaderboard_hidden, 0) = 0"
            if _has_column(conn, "factors", "leaderboard_hidden")
            else ""
        )
        rows = conn.execute(
            f"""
            SELECT f.factor_id, f.run_id, f.loop_index, f.trajectory_id,
                   f.factor_name, f.expression, f.description, f.implementation_status,
                   t.reward_score,
                   MAX(CASE WHEN m.metric_name = 'reward_score'           THEN m.metric_value END) AS factor_reward_score,
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
                   MAX(CASE WHEN m.metric_name = 'layer_2_cumulative_series' THEN m.metric_json END) AS layer_2_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_3_cumulative_series' THEN m.metric_json END) AS layer_3_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_4_cumulative_series' THEN m.metric_json END) AS layer_4_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'layer_5_cumulative_series' THEN m.metric_json END) AS layer_5_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'benchmark_cumulative_series' THEN m.metric_json END) AS benchmark_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'ic_cumulative_series'    THEN m.metric_json  END) AS ic_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'rank_ic_cumulative_series' THEN m.metric_json END) AS rank_ic_cumulative_series,
                   MAX(CASE WHEN m.metric_name = 'valid_cross_section_mean'   THEN m.metric_value END) AS valid_cross_section_mean,
                   MAX(CASE WHEN m.metric_name = 'cross_section_universe_size' THEN m.metric_value END) AS cross_section_universe_size,
                   MAX(CASE WHEN m.metric_name = 'ic_win_rate'                THEN m.metric_value END) AS ic_win_rate,
                   MAX(CASE WHEN m.metric_name = 'rank_ic_win_rate'           THEN m.metric_value END) AS rank_ic_win_rate,
                   MAX(CASE WHEN m.metric_name = 'drawdown'                   THEN m.metric_value END) AS drawdown,
                   MAX(CASE WHEN m.metric_name = 'long_short_positive_ratio'  THEN m.metric_value END) AS long_short_positive_ratio,
                   r.args_json
            FROM factors f
            LEFT JOIN factor_metrics m ON {metric_join}
            LEFT JOIN trajectories t ON t.trajectory_id = f.trajectory_id AND t.run_id = f.run_id
            LEFT JOIN runs r ON r.run_id = f.run_id
            WHERE f.implementation_status NOT IN ('failed', 'error')
              {hidden_filter}
            GROUP BY f.run_id, f.factor_id
            HAVING reward_score IS NOT NULL OR sharpe IS NOT NULL
            ORDER BY COALESCE(factor_reward_score, t.reward_score, sharpe * 0.1, 0) DESC
            """,
        ).fetchall()

    # 排行榜硬门：只有通过 9 项入库标准的因子才上榜（缺指标则跳过该项判断）。
    # 每次运行的全部因子仍完整保留在 RunStore / 运行详情页，不受此过滤影响。
    from trading_agents.infra.factor_library_manager import FactorLibraryManager

    capped = max(1, int(limit))
    result = []
    for row in rows:
        args = _j(row["args_json"]) or {}
        run_sharpe_threshold = _float_or_none(args.get("train_sharpe_threshold"))
        reward_metrics = {
            "sharpe": row["sharpe"],
            "IC": row["IC"],
            "Rank IC": row["rank_ic"],
            "ICIR": row["icir"],
            "Rank ICIR": row["rank_icir"],
            "coverage": row["coverage"],
            "turnover": row["turnover"],
            "drawdown": row["drawdown"],
        }
        reward_breakdown = _reward_breakdown(reward_metrics, expression=str(row["expression"] or ""))
        factor_reward_score = _float_or_none(row["factor_reward_score"])
        if factor_reward_score is None:
            factor_reward_score = _float_or_none(reward_breakdown.get("score"))
        gate = FactorLibraryManager.evaluate_admission_gate({
            "valid_cross_section_mean": row["valid_cross_section_mean"],
            "cross_section_universe_size": row["cross_section_universe_size"],
            "IC": row["IC"],
            "ic_win_rate": row["ic_win_rate"],
            "ICIR": row["icir"],
            "Rank ICIR": row["rank_icir"],
            "rank_ic_win_rate": row["rank_ic_win_rate"],
            "sharpe": row["sharpe"],
            "drawdown": row["drawdown"],
            "long_short_positive_ratio": row["long_short_positive_ratio"],
        }, sharpe_threshold=run_sharpe_threshold)
        gate_passed = bool(gate.get("passed", False))
        # 资格线（与资格赛/运行记录同口径，OR）：Sharpe > 1.0 或 |IC| > 0.02。
        # 过资格线但未通过 9 项入库标准的因子也上榜（标记 admission_passed=False），
        # 让"达标未入库"的因子同样参与因子库展示与图表。
        sharpe_v = _float_or_none(row["sharpe"])
        ic_v = _float_or_none(row["IC"])
        qualifies = (sharpe_v is not None and sharpe_v > 1.0) or (ic_v is not None and abs(ic_v) > 0.02)
        if not gate_passed and not qualifies:
            continue
        run_name = args.get("run_name", "") or row["run_id"]
        ls_series = _j(row["ls_cumulative_series"]) if row["ls_cumulative_series"] else None
        layer1_series = _j(row["layer_1_cumulative_series"]) if row["layer_1_cumulative_series"] else None
        layer2_series = _j(row["layer_2_cumulative_series"]) if row["layer_2_cumulative_series"] else None
        layer3_series = _j(row["layer_3_cumulative_series"]) if row["layer_3_cumulative_series"] else None
        layer4_series = _j(row["layer_4_cumulative_series"]) if row["layer_4_cumulative_series"] else None
        layer5_series = _j(row["layer_5_cumulative_series"]) if row["layer_5_cumulative_series"] else None
        benchmark_series = _j(row["benchmark_cumulative_series"]) if row["benchmark_cumulative_series"] else None
        ic_series = _j(row["ic_cumulative_series"]) if row["ic_cumulative_series"] else None
        rank_ic_series = _j(row["rank_ic_cumulative_series"]) if row["rank_ic_cumulative_series"] else None
        yearly = _j(row["yearly_returns"]) if row["yearly_returns"] else None
        result.append({
            "rank": len(result) + 1,
            "admission_gate": gate,
            "admission_passed": gate_passed,
            "admission_failed_checks": [
                str(check.get("name", ""))
                for check in gate.get("checks", [])
                if not check.get("skipped", False) and not check.get("passed", True)
            ],
            "factor_id": row["factor_id"],
            "run_id": row["run_id"],
            "run_name": run_name,
            "loop_index": row["loop_index"],
            "factor_name": row["factor_name"],
            "expression": row["expression"],
            "description": row["description"],
            "implementation_status": row["implementation_status"],
            "reward_score": row["reward_score"],
            "factor_reward_score": factor_reward_score,
            "reward_breakdown": reward_breakdown,
            "sharpe": row["sharpe"],
            "IC": row["IC"],
            "rank_ic": row["rank_ic"],
            "icir": row["icir"],
            "rank_icir": row["rank_icir"],
            "coverage": row["coverage"],
            "turnover": row["turnover"],
            "valid_cross_section_mean": row["valid_cross_section_mean"],
            "cross_section_universe_size": row["cross_section_universe_size"],
            "ic_win_rate": row["ic_win_rate"],
            "rank_ic_win_rate": row["rank_ic_win_rate"],
            "drawdown": row["drawdown"],
            "long_short_positive_ratio": row["long_short_positive_ratio"],
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
            "layer_2_cumulative_series": layer2_series,
            "layer_3_cumulative_series": layer3_series,
            "layer_4_cumulative_series": layer4_series,
            "layer_5_cumulative_series": layer5_series,
            "benchmark_cumulative_series": benchmark_series,
            "ic_cumulative_series": ic_series,
            "rank_ic_cumulative_series": rank_ic_series,
        })
    deduped = sorted(_dedupe_factors_by_expression(result), key=_leaderboard_sort_key, reverse=True)
    for rank, factor in enumerate(deduped, start=1):
        factor["rank"] = rank
    return deduped[:capped]


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
