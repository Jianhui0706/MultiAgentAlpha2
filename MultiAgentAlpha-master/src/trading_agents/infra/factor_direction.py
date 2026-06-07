from __future__ import annotations

from typing import Any


EXPECTED_FACTOR_RETURN_SIGN = {
    "top_divergence_strength": "negative",
    "bottom_divergence_strength": "positive",
    "mixed_divergence_score": "depends_on_encoding",
}


def canonical_expected_factor_return_sign(_value: Any = None) -> dict[str, str]:
    return dict(EXPECTED_FACTOR_RETURN_SIGN)


def normalize_expected_factor_return_sign(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    aliases = {
        "negative_forward_return": "negative",
        "positive_forward_return": "positive",
        "negative": "negative",
        "positive": "positive",
        "depends_on_encoding": "depends_on_encoding",
        "mixed_or_conditional": "depends_on_encoding",
        "conditional": "depends_on_encoding",
    }
    if text in aliases:
        return aliases[text]
    has_negative = "negative" in text
    has_positive = "positive" in text
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    if has_negative and has_positive:
        return "depends_on_encoding"
    return text


def expected_sign_from_hypothesis(hypothesis_structured: dict[str, Any]) -> str:
    raw = hypothesis_structured.get("expected_factor_return_sign")
    if isinstance(raw, dict):
        canonical = canonical_expected_factor_return_sign(raw)
        for key in ("default", "primary", "expected"):
            normalized = normalize_expected_factor_return_sign(raw.get(key))
            if normalized in {"negative", "positive", "depends_on_encoding"}:
                return normalized
        if any(key in raw for key in canonical):
            return "depends_on_encoding"
    normalized = normalize_expected_factor_return_sign(raw)
    if normalized in {"negative", "positive", "depends_on_encoding"}:
        return normalized
    return normalize_expected_factor_return_sign(hypothesis_structured.get("expected_direction"))


def expected_sign_for_factor(
    hypothesis_structured: dict[str, Any],
    *,
    factor_name: str = "",
    factor_expression: str = "",
    factor_description: str = "",
) -> str:
    raw = hypothesis_structured.get("expected_factor_return_sign")
    if not isinstance(raw, dict):
        return expected_sign_from_hypothesis(hypothesis_structured)

    canonical = canonical_expected_factor_return_sign(raw)
    context = " ".join([factor_name, factor_expression, factor_description]).lower()
    category_keywords = (
        (
            "top_divergence_strength",
            ("top", "顶部", "顶背离", "bearish", "overbought", "exhaustion", "deceleration"),
        ),
        (
            "bottom_divergence_strength",
            ("bottom", "底部", "底背离", "bullish", "oversold", "rebound"),
        ),
        (
            "mixed_divergence_score",
            ("mixed", "composite", "combo", "combined", "ratio", "spread"),
        ),
    )
    for category, keywords in category_keywords:
        if any(keyword in context for keyword in keywords):
            normalized = normalize_expected_factor_return_sign(canonical.get(category))
            if normalized in {"negative", "positive", "depends_on_encoding"}:
                return normalized
    return expected_sign_from_hypothesis(hypothesis_structured)


def rank_ic_sign(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "unknown"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def build_direction_diagnostics(
    *,
    expected_direction: Any,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    expected = normalize_expected_factor_return_sign(expected_direction)
    actual = rank_ic_sign(metrics.get("Rank IC"))
    if actual not in {"positive", "negative"}:
        aligned = None
    elif expected == "positive":
        aligned = actual == "positive"
    elif expected == "negative":
        aligned = actual == "negative"
    else:
        aligned = None
    recommended_usage = "as_is"
    if aligned is False and actual in {"positive", "negative"} and expected in {"positive", "negative"}:
        recommended_usage = "negate_factor"
    return {
        "expected_direction": expected,
        "actual_rank_ic_sign": actual,
        "direction_aligned": aligned,
        "recommended_usage": recommended_usage,
    }
