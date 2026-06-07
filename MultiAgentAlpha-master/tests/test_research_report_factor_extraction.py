from __future__ import annotations

from trading_agents.infra.research_report_loader import build_report_initial_seed
from trading_agents.infra.research_report_loader import extract_report_factor_candidates


def test_extract_report_factor_candidates_marks_runtime_expression_implementable() -> None:
    text = (
        "The report gives a direct factor formula: "
        "RANK(TS_ZSCORE($returns_24h, 24) / (TS_STD($returns_24h, 24) + 1e-8))."
    )

    result = extract_report_factor_candidates(
        text,
        feature_columns=["returns_24h", "close"],
        validation_target="fwd_returns_4h",
    )

    assert result["status"] == "extractable"
    assert result["fallback_thesis_required"] is False
    factor = result["extracted_factors"][0]
    assert factor["implementation_status"] == "implementable"
    assert factor["expression"] == "RANK(TS_ZSCORE($returns_24h,24)/(TS_STD($returns_24h,24)+1e-8))"
    assert factor["required_variables"] == ["$returns_24h"]


def test_extract_report_factor_candidates_rejects_unknown_features() -> None:
    text = "Report formula: RANK(TS_MEAN($unavailable_feature, 12))."

    result = extract_report_factor_candidates(
        text,
        feature_columns=["returns_24h"],
        validation_target="fwd_returns_4h",
    )

    assert result["status"] == "not_extractable"
    assert result["fallback_thesis_required"] is True
    assert result["extracted_factors"] == []
    assert result["rejected_candidates"][0]["implementation_status"] == "not_implementable"


def test_report_initial_seed_prefers_expression_then_falls_back_to_idea() -> None:
    extraction = {
        "extracted_factors": [
            {
                "expression": "RANK($close)",
                "implementation_status": "implementable",
            }
        ]
    }

    expression_seed = build_report_initial_seed(
        report_summary={"thesis_summary": "price pressure reverses"},
        factor_extraction=extraction,
        validation_target="fwd_returns_4h",
    )

    assert expression_seed["seed_type"] == "explicit_factor_expression"
    assert expression_seed["seed_text"] == "RANK($close)"

    idea_seed = build_report_initial_seed(
        report_summary={
            "thesis_summary": "volume-confirmed price exhaustion tends to reverse",
            "mechanism_summary": "crowded short-horizon demand exhausts liquidity",
            "proxy_candidates": ["price_momentum", "volume_confirmation"],
        },
        factor_extraction={"extracted_factors": []},
        validation_target="fwd_returns_4h",
    )

    assert idea_seed["seed_type"] == "report_idea"
    assert "volume-confirmed price exhaustion" in idea_seed["seed_text"]
