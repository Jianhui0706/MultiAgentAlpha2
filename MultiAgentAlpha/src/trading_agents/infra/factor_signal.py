from __future__ import annotations

from statistics import median
from typing import Any


NOISE_RANK_IC_THRESHOLD = 0.01
WEAK_RANK_IC_THRESHOLD = 0.025
MODERATE_RANK_IC_THRESHOLD = 0.04
COUNTERFACTUAL_STRONG_THRESHOLD = 0.025
COUNTERFACTUAL_MODERATE_THRESHOLD = 0.02
COUNTERFACTUAL_FLIP_THRESHOLD = 0.035
ICIR_OK_THRESHOLD = 0.05


def to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def classify_signal_strength(abs_rank_ic: Any) -> str:
    value = to_float_or_none(abs_rank_ic)
    if value is None:
        return "unknown"
    if value < NOISE_RANK_IC_THRESHOLD:
        return "noise"
    if value < WEAK_RANK_IC_THRESHOLD:
        return "weak"
    if value < MODERATE_RANK_IC_THRESHOLD:
        return "moderate"
    return "stronger"


def signal_action_for_strength(signal_strength: str) -> str:
    if signal_strength == "noise":
        return "replace_hypothesis_if_persistent"
    if signal_strength == "weak":
        return "revise_theory_or_proxy"
    if signal_strength == "moderate":
        return "tune_parameters"
    if signal_strength == "stronger":
        return "strengthen_hypothesis"
    return "insufficient_data"


def rank_ic_sign(value: Any) -> str:
    number = to_float_or_none(value)
    if number is None:
        return "unknown"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def build_factor_signal_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    rank_ic = to_float_or_none(metrics.get("Rank IC"))
    abs_rank_ic = abs(rank_ic) if rank_ic is not None else None
    signal_strength = classify_signal_strength(abs_rank_ic)
    return {
        "signal_strength": signal_strength,
        "signal_action": signal_action_for_strength(signal_strength),
        "abs_rank_ic": abs_rank_ic,
        "weak_signal_flag": abs_rank_ic < WEAK_RANK_IC_THRESHOLD if abs_rank_ic is not None else None,
    }


def build_loop_signal_diagnostics(
    *,
    factors: list[dict[str, Any]],
    per_factor_metrics: dict[str, dict[str, Any]],
    previous_loop_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    signs: set[str] = set()
    abs_icirs: list[float] = []
    group_values: dict[str, list[dict[str, Any]]] = {
        "mainline": [],
        "counterfactual": [],
        "adjacent": [],
    }
    counterfactual_states: dict[str, dict[str, Any]] = {}

    for factor in factors:
        if not isinstance(factor, dict):
            continue
        factor_name = str(factor.get("factor_name") or factor.get("name") or "").strip()
        metrics = per_factor_metrics.get(factor_name, {})
        if not isinstance(metrics, dict):
            metrics = {}
        rank_ic = to_float_or_none(metrics.get("Rank IC"))
        if rank_ic is None:
            continue
        abs_rank_ic = abs(rank_ic)
        signal = build_factor_signal_diagnostics(metrics)
        sign = rank_ic_sign(rank_ic)
        if sign in {"positive", "negative"}:
            signs.add(sign)
        rank_icir = to_float_or_none(metrics.get("Rank ICIR"))
        if rank_icir is None:
            rank_icir = to_float_or_none(metrics.get("ICIR"))
        if rank_icir is not None:
            abs_icirs.append(abs(rank_icir))
        alignment = str(factor.get("theory_alignment", "mainline") or "mainline")
        if alignment not in group_values:
            alignment = "mainline"
        if alignment == "counterfactual":
            counterfactual_states[factor_name] = {
                "counterfactual_thesis": str(factor.get("counterfactual_thesis", "")),
                "counterfactual_mechanism": str(factor.get("counterfactual_mechanism", "")),
                "counterfactual_critique": str(factor.get("counterfactual_critique", "")),
                "counterfactual_anchor": factor.get("counterfactual_anchor", {})
                if isinstance(factor.get("counterfactual_anchor", {}), dict)
                else {},
                "counterfactual_expected_direction": str(factor.get("counterfactual_expected_direction", "")),
            }
        group_values[alignment].append(
            {
                "factor_name": factor_name,
                "rank_ic": rank_ic,
                "abs_rank_ic": abs_rank_ic,
                "rank_icir": rank_icir,
                "abs_rank_icir": abs(rank_icir) if rank_icir is not None else None,
                "signal_strength": signal.get("signal_strength"),
            }
        )
        evaluated.append(
            {
                "factor_name": factor_name,
                "rank_ic": rank_ic,
                "rank_ic_sign": sign,
                "theory_alignment": alignment,
                **signal,
            }
        )

    evaluated_count = len(evaluated)
    weak_count = sum(1 for item in evaluated if item.get("weak_signal_flag") is True)
    noise_count = sum(1 for item in evaluated if item.get("signal_strength") == "noise")
    majority_weak = evaluated_count > 0 and weak_count > evaluated_count / 2
    all_weak = evaluated_count > 0 and weak_count == evaluated_count
    direction_stable = len(signs) <= 1 if evaluated_count > 0 else None
    median_abs_icir = median(abs_icirs) if abs_icirs else None
    icir_ok = median_abs_icir is not None and median_abs_icir >= ICIR_OK_THRESHOLD

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {
                "count": 0,
                "best_abs_rank_ic": None,
                "best_factor": "",
                "best_rank_ic": None,
                "best_rank_icir": None,
                "median_abs_rank_ic": None,
                "moderate_or_better_count": 0,
                "abs_rank_ic_values": [],
            }
        best = max(items, key=lambda item: float(item.get("abs_rank_ic") or 0.0))
        abs_values = [float(item.get("abs_rank_ic") or 0.0) for item in items]
        return {
            "count": len(items),
            "best_abs_rank_ic": best.get("abs_rank_ic"),
            "best_factor": best.get("factor_name", ""),
            "best_rank_ic": best.get("rank_ic"),
            "best_rank_icir": best.get("rank_icir"),
            "median_abs_rank_ic": median(abs_values),
            "moderate_or_better_count": sum(1 for value in abs_values if value >= WEAK_RANK_IC_THRESHOLD),
            "abs_rank_ic_values": abs_values,
        }

    group_performance = {name: summarize_group(items) for name, items in group_values.items()}
    mainline_best_abs = group_performance["mainline"]["best_abs_rank_ic"]
    counterfactual_best_abs = group_performance["counterfactual"]["best_abs_rank_ic"]
    adjacent_best_abs = group_performance["adjacent"]["best_abs_rank_ic"]
    counterfactual_best_factor = str(group_performance["counterfactual"]["best_factor"] or "")
    counterfactual_state = dict(counterfactual_states.get(counterfactual_best_factor, {}))
    if counterfactual_state:
        counterfactual_state["best_factor"] = counterfactual_best_factor
        counterfactual_state["challenge_level"] = None

    previous = previous_loop_diagnostics or []
    consecutive_weak_majority = 1 if majority_weak else 0
    if majority_weak:
        for item in reversed(previous):
            if item.get("majority_weak_signal") is True:
                consecutive_weak_majority += 1
            else:
                break

    counterfactual_edge = None
    if counterfactual_best_abs is not None and mainline_best_abs is not None:
        counterfactual_edge = float(counterfactual_best_abs) - float(mainline_best_abs)
    mainline_median_abs = group_performance["mainline"].get("median_abs_rank_ic")
    counterfactual_median_abs = group_performance["counterfactual"].get("median_abs_rank_ic")
    mainline_count = int(group_performance["mainline"].get("count") or 0)
    counterfactual_count = int(group_performance["counterfactual"].get("count") or 0)
    mainline_values = [
        float(value)
        for value in group_performance["mainline"].get("abs_rank_ic_values", [])
        if value is not None
    ]
    counterfactual_values = [
        float(value)
        for value in group_performance["counterfactual"].get("abs_rank_ic_values", [])
        if value is not None
    ]
    mainline_comparison_bar = (
        float(mainline_median_abs)
        if mainline_median_abs is not None
        else (max(mainline_values) if mainline_values else None)
    )
    counterfactual_members_above_mainline = (
        sum(1 for value in counterfactual_values if mainline_comparison_bar is not None and value > mainline_comparison_bar)
        if counterfactual_values
        else 0
    )
    counterfactual_majority_beats_mainline = (
        counterfactual_count >= 2
        and mainline_comparison_bar is not None
        and counterfactual_members_above_mainline > counterfactual_count / 2
    )
    counterfactual_median_beats_mainline = (
        counterfactual_count >= 2
        and mainline_count > 0
        and counterfactual_median_abs is not None
        and mainline_median_abs is not None
        and float(counterfactual_median_abs) > float(mainline_median_abs)
    )

    previous_counterfactual_levels = [
        str(item.get("counterfactual_challenge_level", "")).strip().lower()
        for item in previous
        if isinstance(item, dict)
    ]
    previous_counterfactual_median_edges = [
        item.get("counterfactual_median_edge_vs_mainline")
        for item in previous
        if isinstance(item, dict)
    ]
    prior_moderate_or_strong = 0
    for level in reversed(previous_counterfactual_levels):
        if level in {"moderate", "strong"}:
            prior_moderate_or_strong += 1
        elif level:
            break

    counterfactual_challenge_level = "none"
    prior_positive_median_edge = any(
        to_float_or_none(value) is not None and float(value) > 0.0
        for value in previous_counterfactual_median_edges[-1:]
    )
    counterfactual_median_edge = None
    if counterfactual_median_abs is not None and mainline_median_abs is not None:
        counterfactual_median_edge = float(counterfactual_median_abs) - float(mainline_median_abs)
    if mainline_count > 0 and counterfactual_count >= 2 and counterfactual_best_abs is not None:
        cf_best = float(counterfactual_best_abs)
        main_best = float(mainline_best_abs or 0.0)
        cf_icir = to_float_or_none(group_performance["counterfactual"].get("best_rank_icir"))
        main_icir = to_float_or_none(group_performance["mainline"].get("best_rank_icir"))
        cf_icir_better = (
            cf_icir is not None
            and (main_icir is None or abs(cf_icir) >= abs(main_icir))
        )
        if cf_best > main_best or counterfactual_median_beats_mainline:
            if (
                cf_best >= COUNTERFACTUAL_FLIP_THRESHOLD
                and cf_icir_better
                and (
                    counterfactual_majority_beats_mainline
                    or (counterfactual_median_beats_mainline and prior_positive_median_edge)
                )
            ):
                counterfactual_challenge_level = "strong"
            elif cf_best >= COUNTERFACTUAL_MODERATE_THRESHOLD:
                counterfactual_challenge_level = "moderate"
            elif cf_best >= NOISE_RANK_IC_THRESHOLD:
                counterfactual_challenge_level = "weak"
        if (
            counterfactual_challenge_level == "moderate"
            and prior_moderate_or_strong >= 1
            and counterfactual_median_beats_mainline
            and prior_positive_median_edge
        ):
            counterfactual_challenge_level = "strong"

    counterfactual_stronger = counterfactual_challenge_level in {"moderate", "strong"}
    if counterfactual_state:
        counterfactual_state["challenge_level"] = counterfactual_challenge_level
        counterfactual_state["best_abs_rank_ic"] = counterfactual_best_abs
    adjacent_best_group = (
        adjacent_best_abs is not None
        and (mainline_best_abs is None or float(adjacent_best_abs) > float(mainline_best_abs))
        and (counterfactual_best_abs is None or float(adjacent_best_abs) >= float(counterfactual_best_abs))
    )

    recommendation = "continue"
    revision_recommended = False
    revision_reason = ""
    majority_noise = evaluated_count > 0 and noise_count > evaluated_count / 2
    if counterfactual_challenge_level == "strong":
        revision_recommended = True
        recommendation = "hypothesis_revision"
        revision_reason = (
            f"Counterfactual group is strong: `{counterfactual_best_factor}` beats mainline with |Rank IC|={counterfactual_best_abs:.6f}, suggesting the thesis may need revision."
        )
    elif counterfactual_challenge_level == "moderate":
        revision_recommended = True
        recommendation = "proxy_reconstruction"
        revision_reason = (
            f"Counterfactual group is moderately stronger than mainline (best |Rank IC|={counterfactual_best_abs:.6f}); broaden proxy reconstruction before flipping the thesis."
        )
    elif adjacent_best_group:
        revision_recommended = True
        recommendation = "proxy_reconstruction"
        revision_reason = (
            f"Adjacent/proxy-expansion group beats mainline (best |Rank IC|={adjacent_best_abs:.6f}); keep the mechanism but move proxy family toward the adjacent winner."
        )
    elif majority_noise and consecutive_weak_majority >= 3:
        revision_recommended = True
        recommendation = "proxy_reconstruction"
        revision_reason = "连续三轮多数因子 |Rank IC| < 0.01，信号接近噪声；先重构代理变量，暂不直接替换假设。"
    elif majority_weak and consecutive_weak_majority >= 3:
        revision_recommended = True
        if direction_stable and icir_ok:
            recommendation = "parameter_tuning"
            revision_reason = "连续三轮多数因子 |Rank IC| < 0.025，但方向较稳定且 ICIR 尚可，优先继续调参。"
        elif direction_stable:
            recommendation = "proxy_reconstruction"
            revision_reason = "连续三轮多数因子 |Rank IC| < 0.025，方向虽稳定但稳定性不足，建议重构代理变量。"
        else:
            recommendation = "proxy_reconstruction"
            revision_reason = "连续三轮多数因子 |Rank IC| < 0.025 且方向不稳定；先重构代理变量，暂不直接替换假设。"
    elif majority_weak:
        recommendation = "parameter_tuning" if direction_stable else "proxy_reconstruction"
        revision_reason = "本轮多数因子 |Rank IC| < 0.025，先标记弱信号，暂不直接丢弃。"

    return {
        "evaluated_factor_count": evaluated_count,
        "weak_signal_count": weak_count,
        "noise_signal_count": noise_count,
        "majority_weak_signal": majority_weak,
        "majority_noise_signal": majority_noise,
        "all_factors_weak_signal": all_weak,
        "direction_stable": direction_stable,
        "median_abs_icir": median_abs_icir,
        "icir_ok": icir_ok,
        "consecutive_weak_majority_loops": consecutive_weak_majority,
        "counterfactual_stronger": counterfactual_stronger,
        "counterfactual_challenge_level": counterfactual_challenge_level,
        "counterfactual_edge_vs_mainline": counterfactual_edge,
        "counterfactual_median_edge_vs_mainline": counterfactual_median_edge,
        "counterfactual_members_above_mainline": counterfactual_members_above_mainline,
        "counterfactual_majority_beats_mainline": counterfactual_majority_beats_mainline,
        "counterfactual_best_abs_rank_ic": counterfactual_best_abs,
        "counterfactual_state": counterfactual_state,
        "mainline_best_abs_rank_ic": mainline_best_abs,
        "adjacent_best_abs_rank_ic": adjacent_best_abs,
        "group_performance": group_performance,
        "hypothesis_revision_recommended": revision_recommended,
        "recommended_action": recommendation,
        "revision_reason": revision_reason,
        "factor_signal_diagnostics": evaluated,
    }
