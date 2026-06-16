from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class RunStore:
    """SQLite-backed store for new workflow runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        # WAL 需要 -shm 文件的内存映射，在 exFAT/网络盘（移动硬盘常见）上会报
        # "disk I/O error"；跨机器拷贝项目时残留的 -wal/-shm 文件也会触发。
        # 失败时降级为默认 journal 模式（并发性能略降，功能不受影响）。
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError:
                pass
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT,
                    log_dir TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    args_json TEXT,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS loops (
                    run_id TEXT NOT NULL,
                    loop_index INTEGER NOT NULL,
                    status TEXT,
                    phase TEXT,
                    round_idx INTEGER,
                    direction_id INTEGER,
                    trajectory_id TEXT,
                    parent_ids_json TEXT,
                    error_count INTEGER,
                    agent_count INTEGER,
                    elapsed_seconds REAL,
                    started_at TEXT,
                    ended_at TEXT,
                    summary_json TEXT,
                    final_context_json TEXT,
                    PRIMARY KEY (run_id, loop_index)
                );

                CREATE TABLE IF NOT EXISTS agents (
                    run_id TEXT NOT NULL,
                    loop_index INTEGER NOT NULL,
                    agent_index INTEGER NOT NULL,
                    agent_name TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    elapsed_seconds REAL,
                    error TEXT,
                    input_keys_json TEXT,
                    output_keys_json TEXT,
                    PRIMARY KEY (run_id, loop_index, agent_index)
                );

                CREATE TABLE IF NOT EXISTS trajectories (
                    run_id TEXT,
                    trajectory_id TEXT,
                    phase TEXT,
                    round_idx INTEGER,
                    direction_id INTEGER,
                    parent_ids_json TEXT,
                    hypothesis TEXT,
                    reward_score REAL,
                    factors_json TEXT,
                    metrics_json TEXT,
                    extra_info_json TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (run_id, trajectory_id)
                );

                CREATE TABLE IF NOT EXISTS factors (
                    run_id TEXT,
                    factor_id TEXT,
                    loop_index INTEGER,
                    trajectory_id TEXT,
                    factor_name TEXT,
                    expression TEXT,
                    description TEXT,
                    formulation TEXT,
                    implementation_status TEXT,
                    signal_cache_key TEXT,
                    source_pickle_path TEXT,
                    payload_json TEXT,
                    leaderboard_hidden INTEGER NOT NULL DEFAULT 0,
                    leaderboard_hidden_at TEXT,
                    PRIMARY KEY (run_id, factor_id)
                );

                CREATE TABLE IF NOT EXISTS factor_metrics (
                    run_id TEXT NOT NULL DEFAULT '',
                    factor_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_json TEXT,
                    PRIMARY KEY (run_id, factor_id, metric_name)
                );

                CREATE TABLE IF NOT EXISTS llm_calls (
                    request_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    run_id TEXT,
                    loop_index INTEGER,
                    agent_name TEXT,
                    model TEXT,
                    prompt_chars INTEGER,
                    response_chars INTEGER,
                    message_count INTEGER,
                    error TEXT,
                    payload_json TEXT,
                    created_at TEXT,
                    PRIMARY KEY (request_id, phase, created_at)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    loop_index INTEGER,
                    kind TEXT,
                    path TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER,
                    compressed INTEGER,
                    created_at TEXT
                );
                """
            )
            self._migrate_schema(conn)

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return list(conn.execute(f"PRAGMA table_info({table_name})").fetchall())

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        trajectory_columns = self._table_columns(conn, "trajectories")
        trajectory_pk = {str(row["name"]) for row in trajectory_columns if int(row["pk"] or 0) > 0}
        if trajectory_pk != {"run_id", "trajectory_id"}:
            conn.executescript(
                """
                ALTER TABLE trajectories RENAME TO trajectories_legacy;
                CREATE TABLE trajectories (
                    run_id TEXT,
                    trajectory_id TEXT,
                    phase TEXT,
                    round_idx INTEGER,
                    direction_id INTEGER,
                    parent_ids_json TEXT,
                    hypothesis TEXT,
                    reward_score REAL,
                    factors_json TEXT,
                    metrics_json TEXT,
                    extra_info_json TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (run_id, trajectory_id)
                );
                INSERT OR REPLACE INTO trajectories (
                    run_id, trajectory_id, phase, round_idx, direction_id, parent_ids_json,
                    hypothesis, reward_score, factors_json, metrics_json, extra_info_json, updated_at
                )
                SELECT run_id, trajectory_id, phase, round_idx, direction_id, parent_ids_json,
                       hypothesis, reward_score, factors_json, metrics_json, extra_info_json, updated_at
                FROM trajectories_legacy;
                DROP TABLE trajectories_legacy;
                """
            )

        factor_columns = self._table_columns(conn, "factors")
        factor_pk = {str(row["name"]) for row in factor_columns if int(row["pk"] or 0) > 0}
        if factor_pk != {"run_id", "factor_id"}:
            conn.executescript(
                """
                ALTER TABLE factors RENAME TO factors_legacy;
                CREATE TABLE factors (
                    run_id TEXT,
                    factor_id TEXT,
                    loop_index INTEGER,
                    trajectory_id TEXT,
                    factor_name TEXT,
                    expression TEXT,
                    description TEXT,
                    formulation TEXT,
                    implementation_status TEXT,
                    signal_cache_key TEXT,
                    source_pickle_path TEXT,
                    payload_json TEXT,
                    leaderboard_hidden INTEGER NOT NULL DEFAULT 0,
                    leaderboard_hidden_at TEXT,
                    PRIMARY KEY (run_id, factor_id)
                );
                INSERT OR REPLACE INTO factors (
                    run_id, factor_id, loop_index, trajectory_id, factor_name, expression,
                    description, formulation, implementation_status, signal_cache_key,
                    source_pickle_path, payload_json, leaderboard_hidden, leaderboard_hidden_at
                )
                SELECT run_id, factor_id, loop_index, trajectory_id, factor_name, expression,
                       description, formulation, implementation_status, signal_cache_key,
                       source_pickle_path, payload_json, 0, NULL
                FROM factors_legacy;
                DROP TABLE factors_legacy;
                """
            )
        else:
            factor_names = {str(row["name"]) for row in factor_columns}
            if "leaderboard_hidden" not in factor_names:
                conn.execute("ALTER TABLE factors ADD COLUMN leaderboard_hidden INTEGER NOT NULL DEFAULT 0")
            if "leaderboard_hidden_at" not in factor_names:
                conn.execute("ALTER TABLE factors ADD COLUMN leaderboard_hidden_at TEXT")

        metric_columns = self._table_columns(conn, "factor_metrics")
        metric_names = {str(row["name"]) for row in metric_columns}
        metric_pk = {str(row["name"]) for row in metric_columns if int(row["pk"] or 0) > 0}
        if "run_id" not in metric_names or metric_pk != {"run_id", "factor_id", "metric_name"}:
            conn.executescript(
                """
                ALTER TABLE factor_metrics RENAME TO factor_metrics_legacy;
                CREATE TABLE factor_metrics (
                    run_id TEXT NOT NULL DEFAULT '',
                    factor_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_json TEXT,
                    PRIMARY KEY (run_id, factor_id, metric_name)
                );
                INSERT OR REPLACE INTO factor_metrics (run_id, factor_id, metric_name, metric_value, metric_json)
                SELECT COALESCE(f.run_id, ''), m.factor_id, m.metric_name, m.metric_value, m.metric_json
                FROM factor_metrics_legacy m
                LEFT JOIN factors f ON f.factor_id = m.factor_id;
                DROP TABLE factor_metrics_legacy;
                """
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def default_path(project_root: str | Path) -> Path:
        env_path = os.getenv("RUN_STORE_PATH", "").strip()
        if env_path:
            return Path(env_path)
        return Path(project_root) / "artifacts" / "run_store.sqlite"

    def start_run(self, *, run_id: str, log_dir: str, args: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, status, log_dir, started_at, args_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, "running", log_dir, datetime.now().isoformat(), self._json(args)),
            )

    def finish_run(self, *, run_id: str, status: str, summary: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, ended_at = ?, summary_json = ?
                WHERE run_id = ?
                """,
                (status, datetime.now().isoformat(), self._json(summary), run_id),
            )

    def record_loop(self, *, run_id: str, loop_index: int, summary: dict[str, Any], final_context: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO loops (
                    run_id, loop_index, status, phase, round_idx, direction_id, trajectory_id,
                    parent_ids_json, error_count, agent_count, elapsed_seconds, started_at,
                    ended_at, summary_json, final_context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    loop_index,
                    str(summary.get("status", "")),
                    str(summary.get("phase", "")),
                    int(summary.get("round_idx", 0) or 0),
                    int(summary.get("direction_id", 0) or 0),
                    str(summary.get("trajectory_id", "")),
                    self._json(summary.get("parent_ids", [])),
                    int(summary.get("error_count", 0) or 0),
                    int(summary.get("agent_count", 0) or 0),
                    float(summary.get("elapsed_seconds", 0.0) or 0.0),
                    str(summary.get("started_at", "")),
                    str(summary.get("ended_at", "")),
                    self._json(summary),
                    self._json(final_context),
                ),
            )

    def record_agent(self, *, run_id: str, loop_index: int, agent_index: int, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agents (
                    run_id, loop_index, agent_index, agent_name, started_at, ended_at,
                    elapsed_seconds, error, input_keys_json, output_keys_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    loop_index,
                    agent_index,
                    str(payload.get("agent_name", "")),
                    str(payload.get("started_at", "")),
                    str(payload.get("ended_at", "")),
                    float(payload.get("elapsed_seconds", 0.0) or 0.0),
                    str(payload.get("error", "") or ""),
                    self._json(payload.get("input_keys", [])),
                    self._json(payload.get("output_keys", [])),
                ),
            )

    def upsert_trajectory(self, *, run_id: str, trajectory: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories (
                    trajectory_id, run_id, phase, round_idx, direction_id, parent_ids_json,
                    hypothesis, reward_score, factors_json, metrics_json, extra_info_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(getattr(trajectory, "trajectory_id", "")),
                    run_id,
                    getattr(getattr(trajectory, "phase", ""), "value", str(getattr(trajectory, "phase", ""))),
                    int(getattr(trajectory, "round_idx", 0) or 0),
                    int(getattr(trajectory, "direction_id", 0) or 0),
                    self._json(list(getattr(trajectory, "parent_ids", []))),
                    str(getattr(trajectory, "hypothesis", "")),
                    float(trajectory.get_reward_score()) if hasattr(trajectory, "get_reward_score") else None,
                    self._json(getattr(trajectory, "factors", [])),
                    self._json(getattr(trajectory, "backtest_metrics", {})),
                    self._json(getattr(trajectory, "extra_info", {})),
                    datetime.now().isoformat(),
                ),
            )

    def clear_factors(self) -> int:
        """Delete every factor (and its metrics) from this store.

        Returns the number of factor rows removed. Used by the "清空因子库"
        action so the global leaderboard, which reads from the factors table,
        is actually emptied.
        """
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0])
            conn.execute("DELETE FROM factor_metrics")
            conn.execute("DELETE FROM factors")
        return count

    def hide_all_factors_from_leaderboard(self) -> int:
        """Hide all factors from the global leaderboard while preserving run history."""
        with self._connect() as conn:
            count = int(
                conn.execute("SELECT COUNT(*) FROM factors WHERE COALESCE(leaderboard_hidden, 0) = 0").fetchone()[0]
            )
            conn.execute(
                """
                UPDATE factors
                SET leaderboard_hidden = 1, leaderboard_hidden_at = ?
                WHERE COALESCE(leaderboard_hidden, 0) = 0
                """,
                (datetime.now().isoformat(),),
            )
        return count

    def hide_factor_from_leaderboard(self, *, run_id: str, factor_id: str) -> dict[str, Any]:
        """Hide a single factor from the leaderboard without deleting historical records."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT factor_name, expression FROM factors WHERE run_id = ? AND factor_id = ?",
                (run_id, factor_id),
            ).fetchone()
            if row is None:
                return {"hidden": 0, "factor_name": "", "expression": ""}
            cur = conn.execute(
                """
                UPDATE factors
                SET leaderboard_hidden = 1, leaderboard_hidden_at = ?
                WHERE run_id = ? AND factor_id = ?
                """,
                (datetime.now().isoformat(), run_id, factor_id),
            )
            return {
                "hidden": int(cur.rowcount or 0),
                "factor_name": str(row[0] or ""),
                "expression": str(row[1] or ""),
            }

    def delete_factor(self, *, run_id: str, factor_id: str) -> dict[str, Any]:
        """Delete a single factor and its metrics from this store."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT factor_name, expression FROM factors WHERE run_id = ? AND factor_id = ?",
                (run_id, factor_id),
            ).fetchone()
            if row is None:
                return {"deleted": 0, "factor_name": "", "expression": ""}
            conn.execute(
                "DELETE FROM factor_metrics WHERE factor_id = ? AND (run_id = ? OR run_id = '')",
                (factor_id, run_id),
            )
            cur = conn.execute("DELETE FROM factors WHERE run_id = ? AND factor_id = ?", (run_id, factor_id))
            return {
                "deleted": int(cur.rowcount or 0),
                "factor_name": str(row[0] or ""),
                "expression": str(row[1] or ""),
            }

    def upsert_factor(
        self,
        *,
        run_id: str,
        loop_index: int,
        trajectory_id: str,
        factor_id: str,
        factor: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO factors (
                    factor_id, run_id, loop_index, trajectory_id, factor_name, expression,
                    description, formulation, implementation_status, signal_cache_key,
                    source_pickle_path, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    factor_id,
                    run_id,
                    loop_index,
                    trajectory_id,
                    str(factor.get("factor_name", factor.get("name", ""))),
                    str(factor.get("expression", factor.get("factor_expression", ""))),
                    str(factor.get("description", "")),
                    str(factor.get("formulation", "")),
                    str(factor.get("implementation_status", factor.get("status", ""))),
                    str(factor.get("signal_cache_key", "")),
                    str(factor.get("source_pickle_path", "")),
                    self._json(factor),
                ),
            )
            for metric_name, metric_value in metrics.items():
                numeric_value = None
                if isinstance(metric_value, (int, float)):
                    numeric_value = float(metric_value)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO factor_metrics (run_id, factor_id, metric_name, metric_value, metric_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, factor_id, str(metric_name), numeric_value, self._json(metric_value)),
                )

    def record_llm_event(self, *, run_id: str, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            return
        request = payload.get("request", {}) if isinstance(payload.get("request", {}), dict) else {}
        if not request and isinstance(payload.get("request_summary", {}), dict):
            request = payload.get("request_summary", {})
        response = payload.get("response", {}) if isinstance(payload.get("response", {}), dict) else {}
        response_summary = payload.get("response_summary", {}) if isinstance(payload.get("response_summary", {}), dict) else {}
        length_summary = request.get("input_length_summary", {}) if isinstance(request, dict) else {}
        response_content = str(response.get("content", "")) if isinstance(response, dict) else ""
        response_chars = int(response_summary.get("content_chars", len(response_content)) or 0)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_calls (
                    request_id, phase, run_id, loop_index, agent_name, model, prompt_chars,
                    response_chars, message_count, error, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    str(payload.get("phase", "")),
                    run_id,
                    int(payload.get("loop_index", 0) or 0),
                    str(payload.get("agent_name", "")),
                    str(request.get("model", payload.get("model", ""))) if isinstance(request, dict) else "",
                    int(length_summary.get("total_chars", payload.get("prompt_chars", 0)) or 0),
                    response_chars,
                    int(length_summary.get("message_count", 0) or 0),
                    str(payload.get("error", "") or ""),
                    self._json(payload),
                    datetime.now().isoformat(),
                ),
            )

    def record_artifact(self, *, run_id: str, kind: str, path: str, loop_index: int = 0, size_bytes: int = 0, compressed: bool = False, sha256: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (run_id, loop_index, kind, path, sha256, size_bytes, compressed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, loop_index, kind, path, sha256, int(size_bytes), 1 if compressed else 0, datetime.now().isoformat()),
            )

    def load_loop_compatible(self, *, run_id: str, loop_index: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json, final_context_json FROM loops WHERE run_id = ? AND loop_index = ?",
                (run_id, loop_index),
            ).fetchone()
        if row is None:
            return None
        summary = json.loads(row[0] or "{}")
        final_context = json.loads(row[1] or "{}")
        return {
            "loop_index": loop_index,
            "record_count": int(summary.get("agent_count", 0) or 0),
            "records": [],
            "final_shared_context": final_context,
        }

    def load_trajectories(self) -> list[Any]:
        from trading_agents.schemas import RoundPhase, StrategyTrajectory

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT trajectory_id, phase, round_idx, direction_id, parent_ids_json,
                       hypothesis, factors_json, metrics_json, extra_info_json
                FROM trajectories
                ORDER BY round_idx, trajectory_id
                """
            ).fetchall()
        trajectories = []
        for row in rows:
            phase_text = str(row[1] or RoundPhase.ORIGINAL.value)
            if phase_text not in {item.value for item in RoundPhase}:
                phase_text = RoundPhase.ORIGINAL.value
            trajectories.append(
                StrategyTrajectory(
                    trajectory_id=str(row[0] or ""),
                    direction_id=int(row[3] or 0),
                    round_idx=int(row[2] or 0),
                    phase=RoundPhase(phase_text),
                    hypothesis=str(row[5] or ""),
                    factors=json.loads(row[6] or "[]"),
                    backtest_metrics=json.loads(row[7] or "{}"),
                    parent_ids=[str(item) for item in json.loads(row[4] or "[]")],
                    extra_info=json.loads(row[8] or "{}"),
                )
            )
        return trajectories
