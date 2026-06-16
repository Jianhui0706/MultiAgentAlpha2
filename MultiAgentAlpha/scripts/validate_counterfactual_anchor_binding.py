from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.experiment_designer_agent import ExperimentDesignerAgent


def assert_counterfactuals_are_anchored(
    *,
    name: str,
    allowed_features: set[str],
    anchor_features: list[str],
    anchor_patterns: list[str],
    expected_count: int = 2,
) -> None:
    agent = ExperimentDesignerAgent("experiment_designer_agent")
    factors = agent._build_counterfactual_factor_candidates(
        count=expected_count,
        round_idx=7,
        allowed_input_features=allowed_features,
        anchor_input_features=anchor_features,
        anchor_expression_patterns=anchor_patterns,
        current_hypothesis="Mainline expects the anchor pattern to predict negative forward returns.",
        current_expected_direction="negative_forward_return",
    )
    if len(factors) != expected_count:
        raise AssertionError(f"{name}: expected {expected_count} factors, got {len(factors)}")
    for factor in factors:
        if factor.get("theory_alignment") != "counterfactual":
            raise AssertionError(f"{name}: factor is not counterfactual: {factor}")
        if not factor.get("retained_for_backtest"):
            raise AssertionError(f"{name}: factor is not retained for backtest: {factor}")
        if not factor.get("matched_anchor_features") and not factor.get("matched_anchor_pattern"):
            raise AssertionError(f"{name}: factor is not bound to current anchors: {factor}")
        if set(factor.get("expected_anchor_features", [])) != set(anchor_features):
            raise AssertionError(f"{name}: expected anchor features not preserved: {factor}")
        normalized_anchor_patterns = [item.replace("epsilon", "1e-8") for item in anchor_patterns]
        if factor.get("expected_anchor_patterns", []) != normalized_anchor_patterns:
            raise AssertionError(f"{name}: expected anchor patterns not preserved: {factor}")
        if "epsilon" in str(factor.get("expression", "")).lower():
            raise AssertionError(f"{name}: epsilon literal was not normalized: {factor}")
        warnings = factor.get("validation_warnings", [])
        if "counterfactual_economic_critique" not in warnings:
            raise AssertionError(f"{name}: missing economic critique warning: {factor}")
        if "positive_or_non_negative_forward_return" != factor.get("counterfactual_expected_direction"):
            raise AssertionError(f"{name}: counterfactual direction is not opposite: {factor}")
        if not factor.get("counterfactual_thesis") or not factor.get("counterfactual_mechanism"):
            raise AssertionError(f"{name}: missing counterfactual thesis/mechanism: {factor}")


def assert_counterfactual_fallbacks_are_complete(
    *,
    allowed_features: set[str],
) -> None:
    agent = ExperimentDesignerAgent("experiment_designer_agent")
    factors = agent._build_counterfactual_factor_candidates(
        count=2,
        round_idx=8,
        allowed_input_features=allowed_features,
        anchor_input_features=[],
        anchor_expression_patterns=[],
        current_hypothesis="Mainline expects positive forward returns.",
        current_expected_direction="positive_forward_return",
    )
    if factors:
        raise AssertionError(f"fallback: expected no hard-coded counterfactual templates, got {len(factors)}")


def assert_no_hardcoded_feature_pair_counterfactuals(
    *,
    allowed_features: set[str],
) -> None:
    agent = ExperimentDesignerAgent("experiment_designer_agent")
    factors = agent._build_counterfactual_factor_candidates(
        count=2,
        round_idx=9,
        allowed_input_features=allowed_features,
        anchor_input_features=["$returns_24h", "$volume_change_24h"],
        anchor_expression_patterns=[],
        current_hypothesis="Mainline expects negative forward returns.",
        current_expected_direction="negative_forward_return",
    )
    if factors:
        raise AssertionError(f"feature-pair fallback should be model-written, got local templates: {factors}")


def main() -> None:
    allowed_features = {
        "$close",
        "$vwap",
        "$returns_1h",
        "$returns_4h",
        "$returns_12h",
        "$returns_24h",
        "$volume_change_1h",
        "$volume_change_24h",
        "$volume_quote",
        "$high_low_spread",
        "$open_close_return",
    }
    assert_counterfactuals_are_anchored(
        name="pattern_and_regime_anchors",
        allowed_features=allowed_features,
        anchor_features=["$close", "$high_low_spread", "$volume_quote"],
        anchor_patterns=[
            "TS_RANK($high_low_spread,*)/TS_RANK(TS_MEAN($high_low_spread,*),*)",
            "TS_PCTCHANGE($close,6)/TS_STD($high_low_spread,6)",
        ],
    )
    assert_counterfactuals_are_anchored(
        name="vwap_divergence_anchors",
        allowed_features=allowed_features,
        anchor_features=["$close", "$vwap", "$returns_12h", "$volume_change_24h"],
        anchor_patterns=[
            "TS_PCTCHANGE($vwap,*)-TS_PCTCHANGE($close,*)",
            "TS_RANK($returns_12h,*)-TS_RANK($returns_24h,*)",
        ],
    )
    assert_counterfactuals_are_anchored(
        name="epsilon_pattern_normalization",
        allowed_features=allowed_features,
        anchor_features=["$close", "$volume_quote", "$returns_12h"],
        anchor_patterns=[
            "TS_PCTCHANGE($close,12)/(TS_PCTCHANGE($volume_quote,12)+epsilon)",
        ],
    )
    assert_counterfactual_fallbacks_are_complete(allowed_features=allowed_features)
    assert_no_hardcoded_feature_pair_counterfactuals(allowed_features=allowed_features)
    print("counterfactual anchor binding validation passed")


if __name__ == "__main__":
    main()
