"""运行单次 Loop 会话并输出每个 Agent 的精确耗时报告。

用法:
    uv run python bin/run_with_timing.py

可选环境变量:
    INITIAL_DIRECTION   -- 初始研究方向（必填）
    LOOP_COUNT          -- 循环次数，默认 2
    PANEL_DATA_PATH     -- 面板数据路径，默认 data/panel_data.parquet
    FACTOR_USE_ORIGINAL -- 设为 1 强制使用原始因子计算（对比测试）
    NO_REPORT_PDF       -- 设为 1 跳过 PDF 生成
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local")


def main() -> None:
    loop_count = int(os.getenv("LOOP_COUNT", "2"))
    initial_direction = os.getenv(
        "INITIAL_DIRECTION",
        "加密货币在过热时可能会发生反转，基于这个假设发掘盈利因子",
    ).strip()
    panel_data_path = os.getenv("PANEL_DATA_PATH", str(PROJECT_ROOT / "data" / "panel_data.parquet"))
    no_report_pdf = os.getenv("NO_REPORT_PDF", "0").strip().lower() in ("1", "true", "yes")

    print(f"{'=' * 70}")
    print(f"Run config: loop_count={loop_count}, panel={panel_data_path}")
    print(f"Initial direction: {initial_direction[:80]}...")
    print(f"{'=' * 70}")

    # ── Build workflow ──
    from trading_agents.domain.crypto_cross_section_adapter import CryptoCrossSectionDomainAdapter
    from trading_agents.workflows import AlphaFactorMiningWorkflow
    from trading_agents.core.model_client import ModelClient

    adapter = CryptoCrossSectionDomainAdapter(panel_data_path)
    payload = adapter.build_initial_payload()

    def _null_callback(_payload: dict) -> None:
        pass

    model_client = ModelClient()
    workflow = AlphaFactorMiningWorkflow(
        model_client=model_client,
        progress_callback=_null_callback,
    )

    # ── Run ──
    t_session_start = time.perf_counter()
    session = workflow.run_loop_session(
        loop_count=loop_count,
        initial_shared_payload=payload,
        initial_direction=initial_direction,
        stop_on_error=False,
        retry_per_round=0,
    )
    t_session_end = time.perf_counter()

    # ── Build timing report ──
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines: list[str] = []
    report_lines.append(f"{'=' * 70}")
    report_lines.append(f"Agent Timing Report")
    report_lines.append(f"Run ID: {run_id}")
    report_lines.append(f"Total session: {t_session_end - t_session_start:.1f}s")
    report_lines.append(f"{'=' * 70}")

    total_by_agent: dict[str, float] = {}

    for loop_idx, trace in enumerate(session.loop_traces, start=1):
        report_lines.append(f"\n--- Loop {loop_idx:03d} ---")
        loop_total = 0.0
        for record in trace.records:
            elapsed = getattr(record, "elapsed_seconds", 0.0) or 0.0
            status = "OK" if record.error is None else f"ERROR: {record.error[:60]}"
            report_lines.append(
                f"  {record.agent_name:30s}  {elapsed:8.2f}s  [{status}]"
            )
            loop_total += elapsed
            total_by_agent[record.agent_name] = total_by_agent.get(record.agent_name, 0.0) + elapsed
        report_lines.append(f"  {'Loop total':30s}  {loop_total:8.2f}s")

    report_lines.append(f"\n{'=' * 70}")
    report_lines.append(f"Summary across all loops:")
    for agent_name, total in total_by_agent.items():
        report_lines.append(f"  {agent_name:30s}  {total:8.2f}s")
    report_lines.append(f"  {'TOTAL':30s}  {t_session_end - t_session_start:8.2f}s")
    report_lines.append(f"{'=' * 70}")

    # ── Print to stdout ──
    report_text = "\n".join(report_lines)
    print(report_text)

    # ── Save report file ──
    report_dir = PROJECT_ROOT / "logs" / "timing_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"timing_{run_id}.txt"
    report_file.write_text(report_text, encoding="utf-8")

    # Also save as JSON for programmatic access
    json_data = {
        "run_id": run_id,
        "loop_count": loop_count,
        "session_total_seconds": round(t_session_end - t_session_start, 3),
        "loops": [],
        "total_by_agent": total_by_agent,
    }
    for loop_idx, trace in enumerate(session.loop_traces, start=1):
        loop_agents = []
        for record in trace.records:
            loop_agents.append({
                "agent_name": record.agent_name,
                "elapsed_seconds": getattr(record, "elapsed_seconds", 0.0) or 0.0,
                "error": record.error,
            })
        json_data["loops"].append({"loop_index": loop_idx, "agents": loop_agents})
    json_file = report_dir / f"timing_{run_id}.json"
    json_file.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print output locations ──
    print(f"\n--- Output locations ---")
    print(f"Timing report (txt):  {report_file}")
    print(f"Timing report (json): {json_file}")
    print(f"Factor library:       {PROJECT_ROOT / 'factor_library' / 'all_factors_library.json'}")
    print(f"Trajectory pool:      {PROJECT_ROOT / 'artifacts' / 'trajectory_pool.json'}")
    print(f"Session status:       {session.status}, loops={len(session.loop_traces)}, errors={session.total_errors}")


if __name__ == "__main__":
    main()
