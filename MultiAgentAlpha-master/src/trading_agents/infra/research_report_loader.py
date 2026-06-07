from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from trading_agents.agents.tools.factor_expression_quality import is_parsable
from trading_agents.agents.tools.factor_expression_validator import get_supported_runtime_functions
from trading_agents.agents.tools.factor_expression_validator import normalize_window_function_aliases
from trading_agents.agents.tools.factor_expression_validator import validate_expression_symbols
from trading_agents.infra.pdf_reader import extract_pdf_text
from trading_agents.infra.report_cleaner import clean_research_report_text


_THESIS_KEYWORDS = (
    "thesis",
    "hypothesis",
    "conclusion",
    "alpha",
    "factor",
    "return",
    "momentum",
    "reversal",
    "divergence",
    "volume",
    "volatility",
    "收益",
    "因子",
    "结论",
    "假设",
    "背离",
    "动量",
    "反转",
    "成交",
    "波动",
)

_MECHANISM_KEYWORDS = (
    "mechanism",
    "because",
    "driven",
    "behavior",
    "liquidity",
    "risk",
    "premium",
    "confirmation",
    "exhaustion",
    "机制",
    "由于",
    "因为",
    "驱动",
    "流动性",
    "风险",
    "确认",
    "衰竭",
    "分歧",
)

_CONFLICT_KEYWORDS = (
    "however",
    "but",
    "risk",
    "caveat",
    "limitation",
    "not applicable",
    "可能",
    "但",
    "然而",
    "风险",
    "注意",
    "不适用",
    "长尾",
    "仅",
)

_PROXY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "price_momentum": ("price", "return", "momentum", "close", "收益", "价格", "动量", "涨跌"),
    "volume_confirmation": ("volume", "turnover", "成交", "放量", "缩量", "量能"),
    "volatility_regime": ("volatility", "spread", "range", "波动", "振幅", "价差"),
    "liquidity_pressure": ("liquidity", "交易", "流动性", "换手"),
    "reversal_or_exhaustion": ("reversal", "exhaustion", "divergence", "反转", "衰竭", "背离"),
}

_EXPRESSION_FUNCTION_MARKERS = (
    "RANK",
    "ZSCORE",
    "TS_",
    "DELAY",
    "DELTA",
    "ABS",
    "SIGN",
    "LOG",
    "SQRT",
    "MAX",
    "MIN",
    "MEAN",
    "STD",
    "PERCENTILE",
    "RSI",
    "MACD",
)


def _dataset_context(dataset_summary: dict[str, Any]) -> tuple[list[str], str]:
    feature_columns = [str(item) for item in list(dataset_summary.get("feature_columns", []))]
    validation_target = str(dataset_summary.get("validation_target_column", "fwd_returns_4h"))
    return feature_columns, validation_target


def _truncate_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _balanced_prefix(text: str) -> str:
    depth = 0
    started = False
    chars: list[str] = []
    for char in str(text or "").strip():
        if char == "(":
            depth += 1
            started = True
        elif char == ")":
            depth -= 1
        chars.append(char)
        if started and depth == 0:
            break
        if depth < 0:
            break
    return "".join(chars).strip()


def _normalize_report_expression(text: str, feature_columns: list[str]) -> str:
    expression = str(text or "").strip()
    expression = expression.strip("`'\"，。；;：:、 ")
    expression = re.sub(r"\s+", "", expression)
    feature_names = sorted({str(item).strip() for item in feature_columns if str(item).strip()}, key=len, reverse=True)
    for feature in feature_names:
        expression = re.sub(rf"(?<![$A-Za-z0-9_]){re.escape(feature)}(?![A-Za-z0-9_])", f"${feature}", expression)
    return normalize_window_function_aliases(expression)


def _candidate_expression_strings(text: str, *, max_candidates: int = 200) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for segment in re.split(r"[\n\r]", str(text or "")):
        compact = segment.strip()
        if not compact:
            continue
        upper = compact.upper()
        if not any(marker in upper for marker in _EXPRESSION_FUNCTION_MARKERS):
            continue
        for match in re.finditer(r"\b(?:RANK|ZSCORE|TS_[A-Z_]+|DELAY|DELTA|ABS|SIGN|LOG|SQRT|MAX|MIN|MEAN|STD|PERCENTILE|RSI|MACD)\s*\(", compact, flags=re.IGNORECASE):
            candidate = _balanced_prefix(compact[match.start() :])
            if candidate and candidate not in seen:
                if any(candidate != existing and candidate in existing for existing in seen):
                    continue
                seen.add(candidate)
                candidates.append(candidate)
    return candidates[: max(1, int(max_candidates))]


def extract_report_factor_candidates(
    text: str,
    *,
    feature_columns: list[str],
    validation_target: str,
    max_expression_length: int = 300,
    max_candidates: int = 200,
) -> dict[str, Any]:
    """Extract directly testable runtime-style factors from report text.

    This first version is intentionally conservative: only expressions that are
    already close to the local runtime DSL are marked implementable.
    """
    allowed_features = {f"${str(item).strip()}" for item in feature_columns if str(item).strip()}
    allowed_functions = get_supported_runtime_functions()
    extracted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()
    for index, raw_candidate in enumerate(_candidate_expression_strings(text, max_candidates=max_candidates), start=1):
        expression = _normalize_report_expression(raw_candidate, feature_columns)
        if not expression or expression in seen_normalized:
            continue
        seen_normalized.add(expression)
        reasons: list[str] = []
        if len(expression) > max_expression_length:
            reasons.append(f"expression too long: {len(expression)} > {max_expression_length}")
        parsable, parse_feedback = is_parsable(expression)
        if not parsable:
            reasons.append(parse_feedback)
        symbol_check = validate_expression_symbols(
            expression,
            allowed_features=allowed_features,
            allowed_functions=allowed_functions,
        )
        if not symbol_check.passed:
            reasons.append(symbol_check.to_feedback())

        variables = sorted(set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", expression)))
        missing_variables = sorted(item for item in variables if item not in allowed_features)
        implementable = not reasons
        row = {
            "name": f"report_extracted_factor_{index}",
            "factor_name": f"report_extracted_factor_{index}",
            "original_formula": raw_candidate,
            "translated_expression": expression,
            "expression": expression,
            "required_variables": variables,
            "supported_variables": sorted(item for item in variables if item in allowed_features),
            "missing_variables": missing_variables,
            "validation_target": validation_target,
            "implementation_status": "implementable" if implementable else "not_implementable",
            "validation_errors": reasons,
            "description": "Directly extracted from research report for first-pass reproduction.",
        }
        if reasons:
            rejected.append(row)
        else:
            extracted.append(row)
    if extracted and rejected:
        status = "partially_extractable"
    elif extracted:
        status = "extractable"
    else:
        status = "not_extractable"
    return {
        "status": status,
        "extraction_method": "runtime_dsl_pattern_v1",
        "extracted_factors": extracted,
        "rejected_candidates": rejected,
        "candidate_count": len(extracted) + len(rejected),
        "implementable_count": len(extracted),
        "fallback_thesis_required": not bool(extracted),
    }


def build_report_initial_seed(
    *,
    report_summary: dict[str, Any],
    factor_extraction: dict[str, Any],
    validation_target: str,
) -> dict[str, Any]:
    extracted = [
        dict(item)
        for item in list(factor_extraction.get("extracted_factors", []))
        if isinstance(item, dict) and str(item.get("implementation_status", "")) == "implementable"
    ]
    if extracted:
        factor = extracted[0]
        expression = str(factor.get("expression", "")).strip()
        return {
            "seed_type": "explicit_factor_expression",
            "seed_text": expression,
            "hypothesis_hint": (
                f"Use the report's explicit factor expression as the starting hypothesis: {expression}"
            ),
            "expression": expression,
            "source": "report_factor_extraction",
            "validation_target": validation_target,
        }

    thesis = _truncate_text(report_summary.get("thesis_summary", ""), 260)
    mechanism = _truncate_text(report_summary.get("mechanism_summary", ""), 220)
    proxies = [str(item).strip() for item in list(report_summary.get("proxy_candidates", [])) if str(item).strip()]
    seed_text = " ".join(
        item
        for item in [
            f"Thesis: {thesis}" if thesis else "",
            f"Mechanism: {mechanism}" if mechanism else "",
            f"Proxy families: {', '.join(proxies[:5])}" if proxies else "",
        ]
        if item
    ).strip()
    return {
        "seed_type": "report_idea",
        "seed_text": seed_text,
        "hypothesis_hint": seed_text or "Extract one transferable factor idea from the research report.",
        "proxy_candidates": proxies[:8],
        "source": "report_summary",
        "validation_target": validation_target,
    }


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+", normalized)
    return [item.strip() for item in parts if len(item.strip()) >= 12]


def _score_sentence(sentence: str, keywords: tuple[str, ...]) -> int:
    lowered = sentence.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def _pick_sentences(text: str, keywords: tuple[str, ...], *, limit: int, char_limit: int) -> list[str]:
    sentences = _split_sentences(text)
    scored = [
        (index, _score_sentence(sentence, keywords), sentence)
        for index, sentence in enumerate(sentences)
    ]
    selected = [
        sentence
        for _index, score, sentence in sorted(scored, key=lambda item: (-item[1], item[0]))
        if score > 0
    ][:limit]
    if not selected:
        selected = sentences[:limit]
    return [_truncate_text(sentence, char_limit) for sentence in selected if sentence]


def _infer_proxy_candidates(text: str, feature_columns: list[str]) -> list[str]:
    lowered = str(text or "").lower()
    candidates: list[str] = []
    for family, keywords in _PROXY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            candidates.append(family)
    feature_set = {str(item).strip() for item in feature_columns}
    if {"returns_24h", "returns_4h", "close"} & feature_set and "price_momentum" not in candidates:
        candidates.append("price_momentum")
    if {"volume", "volume_quote", "volume_change_24h"} & feature_set and "volume_confirmation" not in candidates:
        candidates.append("volume_confirmation")
    if {"high_low_spread"} & feature_set and "volatility_regime" not in candidates:
        candidates.append("volatility_regime")
    return candidates[:6]


def _summarize_clean_report(report: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    text = str(report.get("cleaned_report_text", ""))
    thesis = _pick_sentences(text, _THESIS_KEYWORDS, limit=2, char_limit=260)
    mechanisms = _pick_sentences(text, _MECHANISM_KEYWORDS, limit=2, char_limit=260)
    caveats = _pick_sentences(text, _CONFLICT_KEYWORDS, limit=2, char_limit=220)
    return {
        "thesis_summary": " ".join(thesis) if thesis else _truncate_text(text, 260),
        "mechanism_summary": " ".join(mechanisms) if mechanisms else "",
        "caveats_or_conflicts": caveats,
        "proxy_candidates": _infer_proxy_candidates(text, feature_columns),
        "compact_excerpt": _truncate_text(text, 700),
    }


def _common_items(list_values: list[list[str]], *, fallback: list[str], limit: int) -> list[str]:
    if not list_values:
        return fallback[:limit]
    counts: dict[str, int] = {}
    for values in list_values:
        for item in dict.fromkeys(values):
            counts[item] = counts.get(item, 0) + 1
    common = [item for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])) if count >= 2]
    return (common or fallback)[:limit]


def combine_report_payloads(
    reports: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    validation_target: str,
    max_total_chars: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Build a bounded deterministic synthesis for multiple cleaned reports."""
    enriched_reports: list[dict[str, Any]] = []
    proxy_lists: list[list[str]] = []
    for index, report in enumerate(reports, start=1):
        source = dict(report["source"])
        summary = _summarize_clean_report(report, feature_columns)
        proxy_lists.append(list(summary["proxy_candidates"]))
        enriched_reports.append(
            {
                "index": index,
                "file_path": str(source["file_path"]),
                "file_name": str(source["file_name"]),
                "page_count": int(source["page_count"]),
                "cleaned_char_count": len(str(report["cleaned_report_text"])),
                "raw_char_count": len(str(report["raw_report_text"])),
                "truncated": bool(report["truncated"]),
                **summary,
            }
        )
    fallback_proxies = _infer_proxy_candidates(" ".join(str(report.get("cleaned_report_text", "")) for report in reports), feature_columns)
    common_proxies = _common_items(proxy_lists, fallback=fallback_proxies, limit=6)
    conflict_lines = []
    for item in enriched_reports:
        caveats = item.get("caveats_or_conflicts", [])
        if caveats:
            conflict_lines.append(f"- Report {item['index']} ({item['file_name']}): {' '.join(caveats[:2])}")
    if not conflict_lines and len(enriched_reports) > 1:
        conflict_lines.append("- No explicit conflict was detected by the deterministic summarizer; the hypothesis agent should still compare report mechanisms before selecting one anchor.")

    sections: list[str] = [
        "Combined research report compact synthesis.",
        "Instruction: use the reports together, identify consensus and conflicts, then select one primary mechanism anchor for crypto cross-sectional factor mining.",
        "",
        "Source reports:",
    ]
    for item in enriched_reports:
        sections.append(f"- Report {item['index']}: {item['file_name']} ({item['page_count']} pages)")
    sections.extend(
        [
            "",
            "Per-report thesis summaries:",
        ]
    )
    for item in enriched_reports:
        sections.append(f"- Report {item['index']} thesis: {item['thesis_summary']}")
        if item.get("mechanism_summary"):
            sections.append(f"  Mechanism clues: {item['mechanism_summary']}")
    sections.extend(
        [
            "",
            "Common mechanisms or shared proxy families:",
            "- " + ", ".join(common_proxies) if common_proxies else "- No common proxy family detected; use report-specific thesis comparison.",
            "",
            "Conflicting or caveated views:",
            *conflict_lines,
            "",
            "Transferable logic to crypto:",
            "- Prefer mechanisms that can be expressed using continuous crypto panel features rather than original-market session, auction, price-limit, or industry-classification assumptions.",
            "- Treat report-specific claims as directional priors; validate them with cross-sectional Rank IC and grouped return behavior.",
            "",
            "Candidate proxy families:",
            "- " + ", ".join(common_proxies) if common_proxies else "- price_momentum, volume_confirmation, volatility_regime",
            "",
            f"Dataset features available for proxy reconstruction: {', '.join(feature_columns)}.",
            f"Validation target: {validation_target}.",
            "",
            "Compact source excerpts for audit:",
        ]
    )
    for item in enriched_reports:
        sections.append(f"- Report {item['index']} excerpt: {item['compact_excerpt']}")

    combined_report_text = "\n".join(sections).strip()
    truncated_combined = False
    if max_total_chars > 0 and len(combined_report_text) > max_total_chars:
        combined_report_text = combined_report_text[:max_total_chars].rstrip() + "\n\n[TRUNCATED_COMPACT_SYNTHESIS]"
        truncated_combined = True
    synthesis_metadata = {
        "format": "compact_deterministic_synthesis",
        "common_proxy_candidates": common_proxies,
        "combined_truncated": truncated_combined,
        "report_count": len(enriched_reports),
    }
    return combined_report_text, enriched_reports, synthesis_metadata


def _extract_clean_report(
    pdf_path: str | Path,
    *,
    max_chars: int,
) -> dict[str, Any]:
    extracted = extract_pdf_text(pdf_path)
    raw_report_text = str(extracted.get("text", "")).strip()
    cleaned = clean_research_report_text(raw_report_text)
    cleaned_report_text = str(cleaned.get("cleaned_text", "")).strip() or raw_report_text
    pages = list(extracted.get("pages", []))
    low_text_page_count = len(
        [page for page in pages if int(page.get("char_count", 0) or 0) < 120]
    )
    cleaning_stats = dict(cleaned.get("cleaning_stats", {}))
    cleaning_stats["page_count"] = int(extracted.get("page_count", 0) or 0)
    cleaning_stats["low_text_page_count"] = low_text_page_count
    cleaning_stats["image_heavy_report"] = bool(
        cleaning_stats["page_count"] and low_text_page_count >= max(1, cleaning_stats["page_count"] // 2)
    )
    truncated = False
    if max_chars > 0 and len(cleaned_report_text) > max_chars:
        cleaned_report_text = cleaned_report_text[:max_chars].rstrip() + "\n\n[TRUNCATED]"
        truncated = True
    return {
        "source": {
            "file_path": str(extracted["file_path"]),
            "file_name": str(extracted["file_name"]),
            "page_count": int(extracted["page_count"]),
        },
        "raw_report_text": raw_report_text,
        "cleaned_report_text": cleaned_report_text,
        "removed_snippets": list(cleaned.get("removed_snippets", [])),
        "cleaning_stats": cleaning_stats,
        "truncated": truncated,
    }


def build_research_report_payload(
    pdf_path: str | Path,
    *,
    dataset_summary: dict[str, Any],
    max_chars: int = 24000,
) -> dict[str, Any]:
    report = _extract_clean_report(pdf_path, max_chars=max_chars)
    source = dict(report["source"])
    cleaned_report_text = str(report["cleaned_report_text"])
    feature_columns, validation_target = _dataset_context(dataset_summary)
    prompt_text = (
        f"Research report source: {source['file_name']} ({source['page_count']} pages).\n"
        f"Dataset features available for proxy reconstruction: {', '.join(feature_columns)}.\n"
        f"Validation target: {validation_target}.\n\n"
        f"Cleaned report text:\n{cleaned_report_text}"
    )
    report_factor_extraction = extract_report_factor_candidates(
        cleaned_report_text,
        feature_columns=feature_columns,
        validation_target=validation_target,
    )
    report_summary = _summarize_clean_report(report, feature_columns)
    report_initial_seed = build_report_initial_seed(
        report_summary=report_summary,
        factor_extraction=report_factor_extraction,
        validation_target=validation_target,
    )
    return {
        "research_report": {
            "source": source,
            "report_type": "pdf_research_report",
            "report_text": cleaned_report_text,
            "raw_report_text": str(report["raw_report_text"]),
            "cleaned_report_text": cleaned_report_text,
            "removed_snippets": list(report["removed_snippets"]),
            "cleaning_stats": dict(report["cleaning_stats"]),
            "dataset_features": feature_columns,
            "validation_target": validation_target,
            "prompt_text": prompt_text,
            "report_factor_extraction": report_factor_extraction,
            "report_initial_seed": report_initial_seed,
        }
    }


def build_combined_research_report_payload(
    pdf_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    dataset_summary: dict[str, Any],
    max_total_chars: int = 24000,
    max_chars_per_report: int = 9000,
) -> dict[str, Any]:
    normalized_paths: list[str | Path] = []
    seen: set[str] = set()
    for item in pdf_paths:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized_paths.append(item)
    if not normalized_paths:
        raise ValueError("at least one research report PDF is required")
    if len(normalized_paths) == 1:
        return build_research_report_payload(
            normalized_paths[0],
            dataset_summary=dataset_summary,
            max_chars=max_total_chars,
        )

    feature_columns, validation_target = _dataset_context(dataset_summary)
    reports = [
        _extract_clean_report(path, max_chars=max_chars_per_report)
        for path in normalized_paths
    ]
    raw_text_parts: list[str] = []
    removed_snippets: list[Any] = []
    combined_cleaning_stats: dict[str, Any] = {
        "report_count": len(reports),
        "page_count": 0,
        "low_text_page_count": 0,
        "image_heavy_report_count": 0,
    }
    for report in reports:
        source = dict(report["source"])
        cleaning_stats = dict(report["cleaning_stats"])
        combined_cleaning_stats["page_count"] = int(combined_cleaning_stats["page_count"]) + int(source["page_count"])
        combined_cleaning_stats["low_text_page_count"] = int(combined_cleaning_stats["low_text_page_count"]) + int(
            cleaning_stats.get("low_text_page_count", 0) or 0
        )
        if cleaning_stats.get("image_heavy_report"):
            combined_cleaning_stats["image_heavy_report_count"] = int(
                combined_cleaning_stats["image_heavy_report_count"]
            ) + 1
        raw_text_parts.append(str(report["raw_report_text"]))
        removed_snippets.extend(list(report["removed_snippets"]))

    combined_report_text, source_reports, synthesis_metadata = combine_report_payloads(
        reports,
        feature_columns=feature_columns,
        validation_target=validation_target,
        max_total_chars=max_total_chars,
    )
    report_factor_extraction = extract_report_factor_candidates(
        "\n\n".join(str(report.get("cleaned_report_text", "")) for report in reports),
        feature_columns=feature_columns,
        validation_target=validation_target,
    )
    combined_summary = {
        "thesis_summary": " ".join(str(item.get("thesis_summary", "")) for item in source_reports if isinstance(item, dict)),
        "mechanism_summary": " ".join(str(item.get("mechanism_summary", "")) for item in source_reports if isinstance(item, dict)),
        "proxy_candidates": synthesis_metadata.get("common_proxy_candidates", []),
    }
    report_initial_seed = build_report_initial_seed(
        report_summary=combined_summary,
        factor_extraction=report_factor_extraction,
        validation_target=validation_target,
    )
    combined_cleaning_stats["combined_truncated"] = bool(synthesis_metadata.get("combined_truncated", False))
    source = {
        "type": "combined_reports",
        "file_path": "",
        "file_name": "combined_reports",
        "page_count": int(combined_cleaning_stats["page_count"]),
        "report_count": len(source_reports),
    }
    prompt_text = (
        f"Research report source: combined_reports ({len(source_reports)} reports, {source['page_count']} pages).\n"
        f"Dataset features available for proxy reconstruction: {', '.join(feature_columns)}.\n"
        f"Validation target: {validation_target}.\n\n"
        f"Combined cleaned report text:\n{combined_report_text}"
    )
    return {
        "research_report": {
            "source": source,
            "source_reports": source_reports,
            "report_type": "combined_pdf_research_reports",
            "report_text": combined_report_text,
            "raw_report_text": "\n\n".join(raw_text_parts),
            "cleaned_report_text": combined_report_text,
            "removed_snippets": removed_snippets,
            "cleaning_stats": combined_cleaning_stats,
            "dataset_features": feature_columns,
            "validation_target": validation_target,
            "prompt_text": prompt_text,
            "multi_report_input_count": len(source_reports),
            "multi_report_synthesis_status": "step_3_compact_synthesis",
            "multi_report_synthesis": synthesis_metadata,
            "report_factor_extraction": report_factor_extraction,
            "report_initial_seed": report_initial_seed,
        }
    }
