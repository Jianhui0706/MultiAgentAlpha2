from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from trading_agents.infra.factor_validation import FactorValidationRequest, inspect_panel, run_validation

router = APIRouter(prefix="/api/factor-validation")


class FactorValidationPreviewRequest(BaseModel):
    panel_data_path: str


class FactorValidationRunRequest(BaseModel):
    panel_data_path: str
    expression: str
    factor_name: str = "manual_factor"
    target_column: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    rolling_window: int = 60
    rolling_icir_mode: str = "ewm"  # "ewm" (默认) | "simple"


def _asset_urls(report_id: str, charts: dict[str, str]) -> dict[str, str]:
    return {
        key: f"/report-assets/{report_id}/charts/{Path(path).name}"
        for key, path in charts.items()
    }


@router.post("/preview")
def preview_factor_validation(req: FactorValidationPreviewRequest) -> dict[str, Any]:
    try:
        return inspect_panel(req.panel_data_path)
    except Exception as exc:
        raise HTTPException(400, str(exc))


@router.post("/run")
def run_factor_validation(req: FactorValidationRunRequest) -> dict[str, Any]:
    try:
        result = run_validation(
            FactorValidationRequest(
                panel_data_path=req.panel_data_path,
                expression=req.expression,
                factor_name=req.factor_name or "manual_factor",
                target_column=req.target_column,
                start_datetime=req.start_datetime,
                end_datetime=req.end_datetime,
                rolling_window=max(2, int(req.rolling_window or 60)),
                rolling_icir_mode=(req.rolling_icir_mode or "ewm").strip().lower(),
            ),
            reports_root=PROJECT_ROOT / "reports",
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))

    report_id = str(result["report_id"])
    return {
        **result,
        "chart_urls": _asset_urls(report_id, dict(result.get("charts", {}))),
        "download_url": f"/api/factor-validation/{report_id}/download",
        "pdf_url": f"/api/factor-validation/{report_id}/pdf",
    }


@router.get("/{report_id}/download")
def download_factor_validation_report(report_id: str) -> FileResponse:
    if not report_id.startswith("factor_validation_") or "/" in report_id or "\\" in report_id:
        raise HTTPException(400, "invalid report id")
    zip_path = PROJECT_ROOT / "reports" / f"{report_id}.zip"
    if not zip_path.exists():
        raise HTTPException(404, "report archive not found")
    return FileResponse(
        str(zip_path),
        filename=f"{report_id}.zip",
        media_type="application/zip",
    )


@router.get("/{report_id}/pdf")
def download_factor_validation_pdf(report_id: str) -> FileResponse:
    if not report_id.startswith("factor_validation_") or "/" in report_id or "\\" in report_id:
        raise HTTPException(400, "invalid report id")
    pdf_path = PROJECT_ROOT / "reports" / report_id / "factor_validation_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "pdf report not found")
    return FileResponse(
        str(pdf_path),
        filename=f"{report_id}.pdf",
        media_type="application/pdf",
    )
