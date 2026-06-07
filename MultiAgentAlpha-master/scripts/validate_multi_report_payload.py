#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.infra.research_report_loader import build_combined_research_report_payload
from trading_agents.infra.research_report_loader import build_research_report_payload
from trading_agents.workflows.alpha_factor_mining_workflow import AlphaFactorMiningWorkflow


def _dataset_summary() -> dict[str, Any]:
    return {
        "feature_columns": [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "volume_quote",
            "returns_4h",
            "returns_24h",
            "volume_change_24h",
            "high_low_spread",
        ],
        "validation_target_column": "fwd_returns_24h",
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    report_a = PROJECT_ROOT / "materials" / "20260329-dingbeili.pdf"
    report_b = PROJECT_ROOT / "materials" / "20170501-zeshi.pdf"
    _assert(report_a.exists(), f"missing fixture: {report_a}")
    _assert(report_b.exists(), f"missing fixture: {report_b}")

    single_payload = build_research_report_payload(report_a, dataset_summary=_dataset_summary())
    single_report = single_payload["research_report"]
    _assert(single_report["report_type"] == "pdf_research_report", "single-report type changed")
    _assert(bool(single_report.get("report_text")), "single-report text is empty")
    _assert(not single_report.get("source_reports"), "single-report payload should not require source_reports")

    combined_payload = build_combined_research_report_payload(
        [report_a, report_b],
        dataset_summary=_dataset_summary(),
        max_total_chars=16000,
        max_chars_per_report=5000,
    )
    combined_report = combined_payload["research_report"]
    report_text = str(combined_report.get("report_text", ""))
    source_reports = combined_report.get("source_reports", [])
    required_sections = [
        "Per-report thesis summaries:",
        "Common mechanisms or shared proxy families:",
        "Conflicting or caveated views:",
        "Transferable logic to crypto:",
        "Candidate proxy families:",
    ]
    _assert(combined_report.get("source", {}).get("type") == "combined_reports", "combined source type missing")
    _assert(combined_report.get("report_type") == "combined_pdf_research_reports", "combined report type missing")
    _assert(isinstance(source_reports, list) and len(source_reports) == 2, "combined source_reports must list every PDF")
    _assert(combined_report.get("multi_report_synthesis_status") == "step_3_compact_synthesis", "compact synthesis status missing")
    _assert(len(report_text) <= 16040, "combined report text exceeded compact bound")
    _assert(all(section in report_text for section in required_sections), "combined synthesis is missing required sections")
    _assert(len(str(combined_report.get("raw_report_text", ""))) > len(report_text), "combined report is not compacted")

    workflow_payload = AlphaFactorMiningWorkflow.build_multi_report_driven_initial_payload(
        initial_shared_payload={"domain_dataset_summary": _dataset_summary()},
        research_report_pdfs=[str(report_a), str(report_b)],
    )
    workflow_report = workflow_payload["research_report"]
    workflow_source_reports = workflow_report.get("source_reports", [])
    _assert(workflow_payload.get("research_input_mode") == "research_report_pdfs", "workflow did not enter multi-report mode")
    _assert(isinstance(workflow_source_reports, list) and len(workflow_source_reports) == 2, "workflow lost source report list")
    _assert("thesis_summary" in workflow_source_reports[0], "workflow lost enriched per-report summaries")
    _assert(workflow_report.get("multi_report_synthesis_status") == "step_3_compact_synthesis", "workflow lost synthesis status")

    print("multi-report payload validation passed")
    print(f"single_report_chars={len(str(single_report.get('report_text', '')))}")
    print(f"combined_report_chars={len(report_text)}")
    print("source_reports=" + ", ".join(str(item.get("file_name", "")) for item in source_reports if isinstance(item, dict)))


if __name__ == "__main__":
    main()
