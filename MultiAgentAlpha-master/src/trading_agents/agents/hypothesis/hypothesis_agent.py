from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.agents.tools import get_supported_runtime_functions
from trading_agents.agents.tools import validate_expression_symbols
from trading_agents.agents.tools.factor_expression_quality import is_parsable
from trading_agents.infra.factor_direction import canonical_expected_factor_return_sign
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.prompts import PromptStore, render_prompt
from trading_agents.schemas import AgentContext, AgentResult, SharedContext


class HypothesisAgentV2(BaseAgent):
    _CONTINUITY_CONTROLLED_STAGES = {"parameter_tuning", "signal_enhancement", "resonance_filtering"}
    _SCHEMA_PLACEHOLDER_MARKERS = (
        "A SINGLE LINE OF TEXT",
        "List original-market",
        "raw supported input features only",
        "semantic anchor labels",
        "one of:",
    )

    _KNOWN_INPUT_FEATURES = {
        "$open",
        "$close",
        "$high",
        "$low",
        "$volume",
        "$volume_quote",
        "$returns_1h",
        "$returns_4h",
        "$returns_12h",
        "$returns_24h",
        "$log_return_1h",
        "$volume_change_1h",
        "$volume_change_24h",
        "$high_low_spread",
        "$vwap",
        "$open_close_return",
    }

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parent.parent.parent / "prompts")
        self._allowed_runtime_functions = get_supported_runtime_functions()

    def _is_input_length_error(self, error_msg: str) -> bool:
        error_indicators = [
            "input length",
            "context length",
            "maximum context",
            "token limit",
            "InvalidParameter",
            "Range of input length",
            "max_tokens",
            "too long",
        ]
        error_text = str(error_msg).lower()
        return any(indicator.lower() in error_text for indicator in error_indicators)

    def _is_transient_model_error(self, error_msg: str) -> bool:
        transient_indicators = [
            "apiconnectionerror",
            "connection error",
            "apitimeouterror",
            "request timed out",
            "timeout",
            "temporarily unavailable",
            "overloaded",
            "rate limit",
            "bad gateway",
            "service unavailable",
        ]
        error_text = str(error_msg).lower()
        return any(indicator in error_text for indicator in transient_indicators)

    def _render_hypothesis_history(self, history: list[dict[str, object]], prompt_meta: dict[str, object], limit: int) -> str:
        if not history:
            return ""
        history_entries = history[-limit:]
        template_value = prompt_meta.get("hypothesis_and_feedback", "")
        template = "\n".join(str(item) for item in template_value) if isinstance(template_value, list) else str(template_value)
        return render_prompt(template, {"history_entries": history_entries})

    def _build_hypothesis_and_feedback_text(
        self,
        history: list[dict[str, object]],
        direction: str,
        prompt_meta: dict[str, object],
        history_limit: int,
    ) -> str:
        if history:
            return self._render_hypothesis_history(history, prompt_meta, history_limit)
        if direction:
            return render_prompt(
                str(prompt_meta.get("potential_direction_transformation", "")),
                {"potential_direction": direction},
            )
        return str(prompt_meta.get("first_round_fallback", ""))

    def _normalize_string_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = re.split(r"[,/|]", value)
        else:
            raw_items = []
        items: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = str(item).strip()
            lowered = text.lower()
            if not text or lowered in seen:
                continue
            seen.add(lowered)
            items.append(text)
        return items

    def _default_allowed_mutation_types(self, refinement_stage: str) -> list[str]:
        defaults = {
            "initial_exploration": [
                "window_adjustment",
                "threshold_adjustment",
                "normalization_change",
                "signal_combination",
            ],
            "parameter_tuning": [
                "window_adjustment",
                "threshold_adjustment",
                "holding_period_adjustment",
                "normalization_change",
            ],
            "signal_enhancement": [
                "signal_combination",
                "regime_filter",
                "normalization_change",
                "factor_form_change",
            ],
            "hypothesis_replacement": [
                "variable_expansion",
                "factor_form_change",
                "hypothesis_reset",
            ],
        }
        return list(defaults.get(refinement_stage, defaults["parameter_tuning"]))

    def _normalize_input_features(self, value: object) -> list[str]:
        candidates = self._normalize_string_list(value)
        normalized: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            token = item.strip()
            lowered = token.lower()
            if not token.startswith("$") or lowered not in self._KNOWN_INPUT_FEATURES or lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(token)
        return normalized

    def _normalize_expression_patterns(self, value: object) -> list[str]:
        patterns = self._normalize_string_list(value)
        normalized: list[str] = []
        seen: set[str] = set()
        for item in patterns:
            token = re.sub(r"\s+", "", str(item).strip())
            lowered = token.lower()
            if not token or lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(token)
        return normalized

    def _materialize_expression_pattern(self, pattern: str) -> str:
        return re.sub(r"\*", "1", pattern)

    def _filter_supported_expression_patterns(self, patterns: list[str]) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        allowed_features = set(self._KNOWN_INPUT_FEATURES)
        for item in patterns:
            token = re.sub(r"\s+", "", str(item).strip())
            lowered = token.lower()
            if not token or lowered in seen:
                continue
            materialized = self._materialize_expression_pattern(token)
            parsable, _ = is_parsable(materialized)
            if not parsable:
                continue
            symbol_validation = validate_expression_symbols(
                materialized,
                allowed_features=allowed_features,
                allowed_functions=self._allowed_runtime_functions,
            )
            if not symbol_validation.passed:
                continue
            seen.add(lowered)
            filtered.append(token)
        return filtered

    def _normalize_market_structure_dependencies(self, value: object) -> list[str]:
        return self._normalize_string_list(value)

    def _normalize_selected_parametric_specifications(self, value: object) -> list[str]:
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = re.split(r"[,/|]", value)
        else:
            raw_items = []
        selected: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            if isinstance(item, dict):
                text = str(item.get("name", item.get("specification", ""))).strip()
            else:
                text = str(item).strip()
            lowered = text.lower()
            if not text or lowered in seen:
                continue
            seen.add(lowered)
            selected.append(text)
        return selected

    def _normalize_crypto_adaptation_summary(self, value: object) -> dict[str, object]:
        payload = value if isinstance(value, dict) else {}
        return {
            "transferable_core": str(payload.get("transferable_core", "")).strip(),
            "non_transferable_parts": self._normalize_string_list(payload.get("non_transferable_parts", [])),
            "crypto_rewrite": str(payload.get("crypto_rewrite", "")).strip(),
        }

    def _normalize_mechanism_anchor(self, payload: dict[str, object]) -> dict[str, object]:
        raw_anchor = payload.get("mechanism_anchor", {})
        anchor = raw_anchor if isinstance(raw_anchor, dict) else {}
        name = str(anchor.get("name", "") or payload.get("mechanism_anchor_name", "")).strip()
        summary = str(anchor.get("summary", "") or payload.get("mechanism", "")).strip()
        expected_direction = str(anchor.get("expected_direction", "") or payload.get("expected_direction", "")).strip()
        core_concepts = self._normalize_string_list(anchor.get("core_concepts", []))
        if not name:
            mechanism_text = f"{payload.get('mechanism', '')} {payload.get('thesis', '')}".lower()
            if "volume" in mechanism_text and ("divergence" in mechanism_text or "confirmation" in mechanism_text):
                name = "price_volume_divergence"
            elif "momentum" in mechanism_text:
                name = "momentum_structure"
            else:
                name = "report_mechanism"
        if not core_concepts:
            text = f"{name} {summary}".lower()
            if "divergence" in text:
                core_concepts.append("price_volume_divergence")
            if "volume" in text:
                core_concepts.append("volume_confirmation")
            if "exhaust" in text or "reversion" in text or "reversal" in text:
                core_concepts.append("momentum_exhaustion")
                core_concepts.append("mean_reversion")
        return {
            "name": name,
            "summary": summary,
            "expected_direction": expected_direction,
            "core_concepts": list(dict.fromkeys(core_concepts)),
        }

    def _text_tokens(self, value: object) -> set[str]:
        return {item for item in re.split(r"[^a-zA-Z0-9_]+", str(value).lower()) if len(item) >= 3}

    def _overlap_score(self, current: object, previous: object) -> float:
        current_tokens = self._text_tokens(current)
        previous_tokens = self._text_tokens(previous)
        if not current_tokens or not previous_tokens:
            return 0.0
        return len(current_tokens & previous_tokens) / max(1, len(current_tokens | previous_tokens))

    def _contains_schema_placeholder(self, value: object) -> bool:
        if isinstance(value, dict):
            return any(self._contains_schema_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_schema_placeholder(item) for item in value)
        text = str(value or "")
        return any(marker.lower() in text.lower() for marker in self._SCHEMA_PLACEHOLDER_MARKERS)

    def _clean_schema_placeholder_text(self, value: object) -> str:
        text = str(value or "").strip()
        if self._contains_schema_placeholder(text):
            return ""
        return text

    def _validate_no_schema_placeholders(self, payload: dict[str, object]) -> None:
        critical_fields = {
            "hypothesis": payload.get("hypothesis", ""),
            "thesis": payload.get("thesis", ""),
            "mechanism": payload.get("mechanism", ""),
            "expected_direction": payload.get("expected_direction", ""),
        }
        bad_fields = [key for key, value in critical_fields.items() if self._contains_schema_placeholder(value)]
        if bad_fields:
            raise ValueError(f"schema placeholder leaked into hypothesis fields: {bad_fields}")

    def _build_continuity_diagnostics(
        self,
        *,
        current_payload: dict[str, object],
        previous_structured: dict[str, object],
        research_stage: str,
    ) -> dict[str, object]:
        stage = str(research_stage or "").strip() or "unknown"
        previous_anchor = previous_structured.get("mechanism_anchor", {})
        if not isinstance(previous_anchor, dict):
            previous_anchor = {}
        current_anchor = current_payload.get("mechanism_anchor", {})
        if not isinstance(current_anchor, dict):
            current_anchor = {}
        previous_proxy = previous_structured.get("proxy_reconstruction_plan", {})
        current_proxy = current_payload.get("proxy_reconstruction_plan", {})
        if not isinstance(previous_proxy, dict):
            previous_proxy = {}
        if not isinstance(current_proxy, dict):
            current_proxy = {}

        controlled = stage in self._CONTINUITY_CONTROLLED_STAGES
        revision_review = stage == "thesis_revision"
        if stage == "resonance_filtering":
            mechanism_threshold = 0.35
            proxy_threshold = 0.15
            combined_threshold = 0.4
        else:
            mechanism_threshold = 0.45
            proxy_threshold = 0.3
            combined_threshold = 0.5

        current_name = str(current_anchor.get("name", "")).strip()
        previous_name = str(previous_anchor.get("name", "")).strip()
        current_direction = str(current_anchor.get("expected_direction", "")).strip().lower()
        previous_direction = str(previous_anchor.get("expected_direction", "")).strip().lower()
        direction_switch_detected = bool(previous_direction and current_direction and previous_direction != current_direction)
        direction_switch_allowed = not direction_switch_detected
        mechanism_name_changed = bool(previous_name and current_name and previous_name != current_name)
        mechanism_summary_overlap = self._overlap_score(current_anchor.get("summary", ""), previous_anchor.get("summary", ""))
        mechanism_concept_overlap = self._overlap_score(
            " ".join(self._normalize_string_list(current_anchor.get("core_concepts", []))),
            " ".join(self._normalize_string_list(previous_anchor.get("core_concepts", []))),
        )
        direction_match = 1.0 if not previous_direction or not current_direction or previous_direction == current_direction else 0.0
        mechanism_continuity_score = round((mechanism_summary_overlap + mechanism_concept_overlap + direction_match) / 3.0, 4)

        current_features = set(self._normalize_input_features(current_proxy.get("input_features", [])))
        previous_features = set(self._normalize_input_features(previous_proxy.get("input_features", [])))
        proxy_feature_overlap = len(current_features & previous_features) / max(1, len(current_features | previous_features))
        current_families = " ".join(self._normalize_string_list(current_proxy.get("core_proxy_family", [])))
        previous_families = " ".join(self._normalize_string_list(previous_proxy.get("core_proxy_family", [])))
        family_overlap = self._overlap_score(current_families, previous_families)
        proxy_continuity_score = round((proxy_feature_overlap + family_overlap) / 2.0, 4)
        score = round((mechanism_continuity_score + proxy_continuity_score) / 2.0, 4)
        drift_detected = bool(
            controlled
            and (
                direction_switch_detected
                or mechanism_name_changed
                or mechanism_continuity_score < mechanism_threshold
                or (proxy_continuity_score < proxy_threshold and score < combined_threshold)
            )
        )
        return {
            "stage": stage,
            "checked": controlled,
            "revision_review": revision_review,
            "allow_proxy_family_shift": False if controlled else True,
            "revision_allowed": revision_review,
            "revision_allowed_reason": (
                "thesis_revision stage was requested by feedback; drift is inspected and logged but not hard-enforced"
                if revision_review
                else None
            ),
            "mechanism_threshold": mechanism_threshold if controlled else None,
            "proxy_threshold": proxy_threshold if controlled else None,
            "combined_threshold": combined_threshold if controlled else None,
            "direction_switch_detected": direction_switch_detected,
            "direction_switch_allowed": direction_switch_allowed,
            "mechanism_name_changed": mechanism_name_changed,
            "drift_detected": drift_detected,
            "score": score,
            "mechanism_continuity_score": mechanism_continuity_score,
            "proxy_continuity_score": proxy_continuity_score,
            "mechanism_summary_overlap": round(mechanism_summary_overlap, 4),
            "mechanism_concept_overlap": round(mechanism_concept_overlap, 4),
            "direction_match": direction_match,
            "proxy_feature_overlap": round(proxy_feature_overlap, 4),
            "family_overlap": round(family_overlap, 4),
            "continuity_enforced": drift_detected,
            "continuity_enforcement_reason": "drift_detected" if drift_detected else None,
        }

    def _apply_continuity_enforcement(
        self,
        *,
        converted: dict[str, object],
        previous_structured: dict[str, object],
        continuity_diagnostics: dict[str, object],
    ) -> None:
        """Keep natural-language thesis fields aligned when continuity clamps the mechanism."""
        previous_anchor = previous_structured.get("mechanism_anchor", {})
        if isinstance(previous_anchor, dict) and previous_anchor:
            converted["mechanism_anchor"] = dict(previous_anchor)
            previous_direction = str(previous_anchor.get("expected_direction", "")).strip()
            if previous_direction:
                converted["expected_direction"] = previous_direction

        natural_language_fields: list[str] = []
        for field in ("hypothesis", "thesis", "mechanism"):
            previous_value = self._clean_schema_placeholder_text(previous_structured.get(field, ""))
            if previous_value:
                converted[field] = previous_value
                natural_language_fields.append(field)

        enforced_fields = ["mechanism_anchor", "expected_direction"]
        enforced_fields.extend(natural_language_fields)
        continuity_diagnostics["enforced_fields"] = list(dict.fromkeys(enforced_fields))
        continuity_diagnostics["natural_language_enforced"] = bool(natural_language_fields)
        continuity_diagnostics["natural_language_enforced_fields"] = natural_language_fields

    def _normalize_proxy_reconstruction_plan(
        self,
        value: object,
        *,
        fallback_input_features: list[str],
        validation_target: str,
    ) -> dict[str, object]:
        payload = value if isinstance(value, dict) else {}
        input_features = self._normalize_input_features(payload.get("input_features", []))
        if not input_features:
            input_features = list(fallback_input_features)
        return {
            "input_features": input_features,
            "core_proxy_family": self._normalize_string_list(payload.get("core_proxy_family", [])),
            "supporting_proxy_family": self._normalize_string_list(payload.get("supporting_proxy_family", [])),
            "validation_target": validation_target,
        }

    def _append_diagnostic_event(
        self,
        private_context: AgentContext,
        *,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> None:
        diagnostics = private_context.payload.setdefault("hypothesis_diagnostics", [])
        if not isinstance(diagnostics, list):
            diagnostics = []
            private_context.payload["hypothesis_diagnostics"] = diagnostics
        payload: dict[str, object] = {"stage": stage}
        if details:
            payload.update(details)
        diagnostics.append(payload)
        if len(diagnostics) > 20:
            del diagnostics[:-20]

    @staticmethod
    def _truncate_text(value: str, limit: int = 500) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "...[TRUNCATED]"

    def _truncate_research_report_text(self, report_text: str, char_limit: int) -> str:
        text = str(report_text or "").strip()
        if char_limit <= 0 or len(text) <= char_limit:
            return text
        truncated = text[:char_limit]
        last_break = max(truncated.rfind("\n\n"), truncated.rfind("\n"), truncated.rfind("。"), truncated.rfind(". "))
        if last_break >= max(200, int(char_limit * 0.6)):
            truncated = truncated[: last_break + 1]
        return truncated.rstrip() + "\n...[TRUNCATED]"

    def _next_report_char_limit(self, current_limit: int) -> int:
        if current_limit > 12000:
            return 12000
        if current_limit > 8000:
            return 8000
        if current_limit > 6000:
            return 6000
        if current_limit > 4000:
            return 4000
        if current_limit > 2500:
            return 2500
        if current_limit > 1500:
            return 1500
        return max(0, current_limit - 500)

    def _infer_verifiable_anchors(self, core_variables: list[str]) -> dict[str, list[str]]:
        input_features: list[str] = []
        expression_patterns: list[str] = []

        def add_feature(name: str) -> None:
            if name not in input_features:
                input_features.append(name)

        def add_pattern(pattern: str) -> None:
            compact = re.sub(r"\s+", "", pattern)
            if compact not in expression_patterns:
                expression_patterns.append(compact)

        for raw_item in core_variables:
            item = str(raw_item).strip()
            lowered = item.lower()
            if not item:
                continue
            if item.startswith("$"):
                add_feature(item)
                continue
            if "quote_volume" in lowered or "volume_quote" in lowered:
                add_feature("$volume_quote")
            if "volume" in lowered and "$volume_quote" not in input_features:
                add_feature("$volume")
            if "momentum" in lowered or "return" in lowered:
                add_feature("$close")
                add_pattern("TS_PCTCHANGE($close,*)")
            if "volatility" in lowered or lowered.startswith("vol_") or "_vol" in lowered:
                add_feature("$close")
                add_pattern("TS_STD($close,*)")
            if "zscore" in lowered:
                add_pattern("TS_ZSCORE($close,*)")
            if "rsi" in lowered:
                add_feature("$close")
                add_pattern("RSI($close,*)")
            if "vwap" in lowered:
                add_feature("$vwap")
            if "open_close" in lowered:
                add_feature("$open_close_return")
            if "high_low" in lowered or "spread" in lowered:
                add_feature("$high_low_spread")
            if "volume_change" in lowered:
                add_feature("$volume_change_1h")
                add_feature("$volume_change_24h")
            if "log_return" in lowered:
                add_feature("$log_return_1h")

        return {
            "input_features": input_features,
            "expression_patterns": expression_patterns,
        }

    def _normalize_evolution_state(
        self,
        payload: dict[str, object],
        previous_state: dict[str, object],
        round_index: int,
    ) -> dict[str, object]:
        raw_state = payload.get("hypothesis_evolution_state", {})
        if not isinstance(raw_state, dict):
            raw_state = {}
        valid_stages = {
            "initial_exploration",
            "parameter_tuning",
            "signal_enhancement",
            "hypothesis_replacement",
        }
        valid_mutation_types = {
            "window_adjustment",
            "threshold_adjustment",
            "holding_period_adjustment",
            "normalization_change",
            "signal_combination",
            "regime_filter",
            "factor_form_change",
            "variable_expansion",
            "hypothesis_reset",
        }
        core_variables = self._normalize_string_list(raw_state.get("core_variables", []))
        if not core_variables:
            core_variables = self._normalize_string_list(previous_state.get("core_variables", []))
        refinement_stage = str(raw_state.get("refinement_stage", "")).strip().lower()
        if refinement_stage not in valid_stages:
            previous_stage = str(previous_state.get("refinement_stage", "")).strip().lower()
            if previous_stage in valid_stages:
                refinement_stage = previous_stage
            else:
                refinement_stage = "initial_exploration" if round_index <= 0 else "parameter_tuning"
        allowed_mutation_types = [
            item
            for item in self._normalize_string_list(raw_state.get("allowed_mutation_types", []))
            if item in valid_mutation_types
        ]
        if not allowed_mutation_types:
            previous_allowed = [
                item
                for item in self._normalize_string_list(previous_state.get("allowed_mutation_types", []))
                if item in valid_mutation_types
            ]
            allowed_mutation_types = previous_allowed or self._default_allowed_mutation_types(refinement_stage)
        raw_anchors = raw_state.get("verifiable_anchors", {})
        if not isinstance(raw_anchors, dict):
            raw_anchors = {}
        previous_anchors = previous_state.get("verifiable_anchors", {})
        if not isinstance(previous_anchors, dict):
            previous_anchors = {}
        input_features = self._normalize_input_features(raw_anchors.get("input_features", []))
        if not input_features:
            input_features = self._normalize_input_features(previous_anchors.get("input_features", []))
        expression_patterns = self._normalize_expression_patterns(raw_anchors.get("expression_patterns", []))
        if not expression_patterns:
            expression_patterns = self._normalize_expression_patterns(previous_anchors.get("expression_patterns", []))
        inferred_anchors = self._infer_verifiable_anchors(core_variables)
        expression_patterns = self._filter_supported_expression_patterns(expression_patterns)
        if not input_features:
            input_features = inferred_anchors["input_features"]
        if not expression_patterns:
            expression_patterns = self._filter_supported_expression_patterns(inferred_anchors["expression_patterns"])
        return {
            "core_variables": core_variables,
            "verifiable_anchors": {
                "input_features": input_features,
                "expression_patterns": expression_patterns,
            },
            "refinement_stage": refinement_stage,
            "allowed_mutation_types": allowed_mutation_types,
        }

    def _convert_response(
        self,
        raw_text: str,
        *,
        previous_state: dict[str, object],
        previous_structured: dict[str, object],
        history: list[dict[str, object]],
        round_index: int,
        validation_target: str,
        research_stage: str,
    ) -> dict[str, object]:
        payload = parse_json_object(raw_text)
        self._validate_no_schema_placeholders(payload)
        evolution_state = self._normalize_evolution_state(payload, previous_state, round_index)
        fallback_input_features = list(dict.fromkeys(evolution_state["verifiable_anchors"]["input_features"]))
        converted = {
            "hypothesis": self._clean_schema_placeholder_text(payload.get("hypothesis", "")),
            "thesis": self._clean_schema_placeholder_text(payload.get("thesis", payload.get("hypothesis", ""))),
            "mechanism": self._clean_schema_placeholder_text(payload.get("mechanism", "")),
            "expected_direction": self._clean_schema_placeholder_text(payload.get("expected_direction", "")),
            "expected_factor_return_sign": canonical_expected_factor_return_sign(
                payload.get("expected_factor_return_sign")
            ),
            "market_structure_dependencies": self._normalize_market_structure_dependencies(
                payload.get("market_structure_dependencies", [])
            ),
            "crypto_adaptation_summary": self._normalize_crypto_adaptation_summary(
                payload.get("crypto_adaptation_summary", {})
            ),
            "selected_parametric_specifications": self._normalize_selected_parametric_specifications(
                payload.get("selected_parametric_specifications", [])
            ),
            "proxy_reconstruction_plan": self._normalize_proxy_reconstruction_plan(
                payload.get("proxy_reconstruction_plan", {}),
                fallback_input_features=fallback_input_features,
                validation_target=validation_target,
            ),
            "concise_knowledge": self._clean_schema_placeholder_text(payload.get("concise_knowledge", "")),
            "concise_observation": self._clean_schema_placeholder_text(payload.get("concise_observation", "")),
            "concise_justification": self._clean_schema_placeholder_text(payload.get("concise_justification", "")),
            "concise_specification": self._clean_schema_placeholder_text(payload.get("concise_specification", "")),
            "hypothesis_evolution_state": evolution_state,
        }
        converted["mechanism_anchor"] = self._normalize_mechanism_anchor({**payload, **converted})
        continuity_previous_structured = dict(previous_structured)
        if not continuity_previous_structured.get("mechanism_anchor"):
            for item in reversed(history):
                item_structured = item.get("hypothesis_structured", {})
                if not isinstance(item_structured, dict):
                    continue
                item_anchor = item_structured.get("mechanism_anchor", {})
                if isinstance(item_anchor, dict) and item_anchor:
                    continuity_previous_structured = dict(item_structured)
                    break
        continuity_diagnostics = self._build_continuity_diagnostics(
            current_payload=converted,
            previous_structured=continuity_previous_structured,
            research_stage=research_stage,
        )
        if continuity_diagnostics.get("continuity_enforced"):
            previous_anchor = continuity_previous_structured.get("mechanism_anchor", {})
            if not isinstance(previous_anchor, dict) or not previous_anchor:
                for item in reversed(history):
                    item_structured = item.get("hypothesis_structured", {})
                    if not isinstance(item_structured, dict):
                        continue
                    item_anchor = item_structured.get("mechanism_anchor", {})
                    if isinstance(item_anchor, dict) and item_anchor:
                        previous_anchor = item_anchor
                        continuity_diagnostics["anchor_fallback_source"] = "hypothesis_feedback_history"
                        break
            if isinstance(previous_anchor, dict) and previous_anchor:
                self._apply_continuity_enforcement(
                    converted=converted,
                    previous_structured=continuity_previous_structured,
                    continuity_diagnostics=continuity_diagnostics,
                )
            else:
                continuity_diagnostics["enforced_fields"] = []
                continuity_diagnostics["natural_language_enforced"] = False
                continuity_diagnostics["natural_language_enforced_fields"] = []
        converted["continuity_diagnostics"] = continuity_diagnostics
        return converted

    def _run_once(
        self,
        *,
        private_context: AgentContext,
        prompt_meta: dict[str, Any],
        system_prompt_template: str,
        user_prompt_template: str,
        scenario: str,
        direction: str,
        rag_text: str,
        shared_market_regime_summary: object,
        history: list[dict[str, object]],
        history_limit: int,
        report_char_limit: int,
        previous_state: dict[str, object],
        previous_structured: dict[str, object],
        research_report: dict[str, object],
        research_stage: str,
        research_stage_reason: str,
        phase: str,
        direction_id: int,
        evolution_guidance: str,
        parent_trajectory_summaries: list[dict[str, object]],
        mutation_type: str,
        parametric_specifications: list[object],
        model_client: ModelClient,
    ) -> tuple[dict[str, object], str, dict[str, object]]:
        hypothesis_and_feedback = self._build_hypothesis_and_feedback_text(
            history,
            direction,
            prompt_meta,
            history_limit,
        )
        report_source = research_report.get("source", {}) if isinstance(research_report, dict) else {}
        source_reports = research_report.get("source_reports", []) if isinstance(research_report, dict) else []
        if not isinstance(source_reports, list):
            source_reports = []
        source_report_names = [
            str(item.get("file_name", "") or item.get("file_path", "")).strip()
            for item in source_reports
            if isinstance(item, dict) and str(item.get("file_name", "") or item.get("file_path", "")).strip()
        ]
        dataset_features = research_report.get("dataset_features", []) if isinstance(research_report, dict) else []
        validation_target = str(research_report.get("validation_target", "fwd_returns_4h")) if isinstance(research_report, dict) else "fwd_returns_4h"
        report_initial_seed = research_report.get("report_initial_seed", {}) if isinstance(research_report, dict) else {}
        if not isinstance(report_initial_seed, dict):
            report_initial_seed = {}
        report_source_payload = dict(report_source) if isinstance(report_source, dict) else {}
        if source_reports:
            report_source_payload["source_reports"] = source_reports
        synthesis_status = (
            str(research_report.get("multi_report_synthesis_status", "")).strip()
            if isinstance(research_report, dict)
            else ""
        )
        synthesis_metadata = (
            research_report.get("multi_report_synthesis", {})
            if isinstance(research_report, dict)
            else {}
        )
        if not isinstance(synthesis_metadata, dict):
            synthesis_metadata = {}
        rendered_output_format = render_prompt(
            str(prompt_meta.get("hypothesis_output_format", "")),
            {"validation_target": validation_target},
        )
        rendered_specification = render_prompt(
            str(prompt_meta.get("hypothesis_specification", "")),
            {"validation_target": validation_target},
        )
        context = {
            "targets": str(prompt_meta.get("targets", "factors")),
            "scenario": scenario,
            "hypothesis_output_format": rendered_output_format,
            "hypothesis_specification": rendered_specification,
            "current_evolution_state_json": str(previous_state if previous_state else {}),
            "round": len(history),
            "round_index": len(history),
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": rag_text if rag_text else "",
            "market_regime_summary_json": json.dumps(shared_market_regime_summary, ensure_ascii=False, indent=2)
            if isinstance(shared_market_regime_summary, dict)
            else "{}",
            "market_regime_summary_text": str(shared_market_regime_summary.get("text", "")).strip()
            if isinstance(shared_market_regime_summary, dict)
            else "",
            "use_market_regime_for_seed": bool(
                isinstance(shared_market_regime_summary, dict)
                and not str(research_report.get("report_text", "") if isinstance(research_report, dict) else "").strip()
                and len(history) == 0
                and str(phase or "original").strip().lower() in {"", "original"}
            ),
            "research_report_text": self._truncate_research_report_text(
                str(research_report.get("report_text", "")) if isinstance(research_report, dict) else "",
                report_char_limit,
            ),
            "research_report_type": str(research_report.get("report_type", "")) if isinstance(research_report, dict) else "",
            "research_report_source_json": json.dumps(report_source_payload if report_source_payload else {}, ensure_ascii=False),
            "research_report_source_reports_json": json.dumps(source_reports, ensure_ascii=False),
            "research_report_source_names_text": ", ".join(source_report_names),
            "report_initial_seed_json": json.dumps(report_initial_seed, ensure_ascii=False),
            "report_initial_seed_text": str(report_initial_seed.get("hypothesis_hint", "") or report_initial_seed.get("seed_text", "")).strip(),
            "report_initial_seed_type": str(report_initial_seed.get("seed_type", "")).strip(),
            "dataset_features_json": json.dumps(dataset_features if dataset_features else [], ensure_ascii=False),
            "parametric_specifications_json": json.dumps(parametric_specifications, ensure_ascii=False, indent=2),
            "validation_target": validation_target,
            "research_stage": str(research_stage or "").strip(),
            "research_stage_reason": str(research_stage_reason or "").strip(),
            "phase": str(phase or "").strip(),
            "direction_id": int(direction_id),
            "evolution_guidance": str(evolution_guidance or "").strip(),
            "parent_trajectory_summaries_json": json.dumps(parent_trajectory_summaries, ensure_ascii=False, indent=2),
            "mutation_type": str(mutation_type or "").strip(),
        }
        self._append_diagnostic_event(
            private_context,
            stage="prompt_context_built",
            details={
                "history_length": len(history),
                "has_research_report": bool(context["research_report_text"]),
                "has_evolution_guidance": bool(context["evolution_guidance"]),
                "report_char_limit": report_char_limit,
                "report_chars_used": len(context["research_report_text"]),
                "research_report_type": context["research_report_type"],
                "source_report_count": len(source_reports),
                "source_report_names": source_report_names,
                "combined_report": str(report_source_payload.get("type", "")) == "combined_reports",
                "multi_report_synthesis_status": synthesis_status,
                "multi_report_synthesis_format": str(synthesis_metadata.get("format", "")),
                "common_proxy_candidates": list(synthesis_metadata.get("common_proxy_candidates", []))
                if isinstance(synthesis_metadata.get("common_proxy_candidates", []), list)
                else [],
                "validation_target": validation_target,
                "dataset_feature_count": len(dataset_features),
                "parametric_specification_count": len(parametric_specifications),
                "has_market_regime_summary": bool(context["market_regime_summary_text"]),
                "use_market_regime_for_seed": bool(context["use_market_regime_for_seed"]),
            },
        )
        system_prompt = render_prompt(system_prompt_template, context)
        user_prompt = render_prompt(user_prompt_template, context)
        self._append_diagnostic_event(
            private_context,
            stage="model_call_started",
            details={
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "source_report_names": source_report_names,
                "multi_report_synthesis_status": synthesis_status,
            },
        )
        started_at = time.perf_counter()
        try:
            raw_response = model_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=True,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            self._append_diagnostic_event(
                private_context,
                stage="model_call_failed",
                details={
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                },
            )
            raise RuntimeError(f"Hypothesis agent model call failed: {exc}") from exc
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        self._append_diagnostic_event(
            private_context,
            stage="model_response_received",
            details={
                "elapsed_ms": elapsed_ms,
                "response_chars": len(raw_response),
                "response_preview": self._truncate_text(raw_response),
            },
        )
        try:
            hypothesis_payload = self._convert_response(
                raw_response,
                previous_state=previous_state,
                previous_structured=previous_structured,
                history=history,
                round_index=len(history),
                validation_target=validation_target,
                research_stage=research_stage,
            )
        except Exception as exc:
            self._append_diagnostic_event(
                private_context,
                stage="response_parse_failed",
                details={
                    "error": str(exc),
                    "response_preview": self._truncate_text(raw_response),
                },
            )
            raise RuntimeError(f"Hypothesis agent response parse failed: {exc}") from exc
        self._append_diagnostic_event(
            private_context,
            stage="response_parsed",
            details={
                "output_keys": sorted(hypothesis_payload.keys()),
            },
        )
        return (
            hypothesis_payload,
            raw_response,
            context,
        )

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        prompt_bundle = self.prompt_store.load("hypothesis_agent")
        prompt_meta = prompt_bundle.metadata
        scenario = str(shared_context.payload.get("scenario", ""))
        direction = str(shared_context.payload.get("direction", ""))
        rag_text = str(shared_context.payload.get("rag_text", ""))
        shared_market_regime_summary = shared_context.payload.get("market_regime_summary", {})
        history = list(shared_context.payload.get("hypothesis_feedback_history", []))
        research_report = dict(shared_context.payload.get("research_report", {}))
        research_stage = str(shared_context.payload.get("research_stage", "")).strip()
        research_stage_reason = str(shared_context.payload.get("research_stage_reason", "")).strip()
        parametric_specifications = [
            item
            for item in list(shared_context.payload.get("parametric_specifications", []))
            if isinstance(item, (dict, str))
        ]
        phase = str(shared_context.payload.get("phase", "original"))
        direction_id = int(shared_context.payload.get("direction_id", 0) or 0)
        evolution_guidance = str(shared_context.payload.get("evolution_guidance", "")).strip()
        parent_trajectory_summaries = [
            dict(item)
            for item in list(shared_context.payload.get("parent_trajectory_summaries", []))
            if isinstance(item, dict)
        ]
        mutation_type = str(shared_context.payload.get("mutation_type", "")).strip()
        previous_state = dict(shared_context.payload.get("hypothesis_structured", {})).get("hypothesis_evolution_state", {})
        previous_structured = dict(shared_context.payload.get("hypothesis_structured", {}))
        if not isinstance(previous_state, dict):
            previous_state = {}
        history_limit = int(prompt_meta.get("default_history_limit", 6))
        min_history_limit = int(prompt_meta.get("min_history_limit", 1))
        max_parse_retries = int(prompt_meta.get("max_parse_retries", 2))
        report_text = str(research_report.get("report_text", "")) if isinstance(research_report, dict) else ""
        report_char_limit = len(report_text)
        min_report_char_limit = 1200 if report_text else 0
        parse_retries_used = 0
        last_error: Exception | None = None
        while history_limit >= min_history_limit:
            try:
                hypothesis_payload, raw_response, context = self._run_once(
                    private_context=private_context,
                    prompt_meta=prompt_meta,
                    system_prompt_template=prompt_bundle.system_prompt,
                    user_prompt_template=prompt_bundle.user_prompt,
                    scenario=scenario,
                    direction=direction,
                    rag_text=rag_text,
                    shared_market_regime_summary=shared_market_regime_summary,
                    history=history,
                    history_limit=history_limit,
                    report_char_limit=report_char_limit,
                    previous_state=previous_state,
                    previous_structured=previous_structured,
                    research_report=research_report,
                    research_stage=research_stage,
                    research_stage_reason=research_stage_reason,
                    phase=phase,
                    direction_id=direction_id,
                    evolution_guidance=evolution_guidance,
                    parent_trajectory_summaries=parent_trajectory_summaries,
                    mutation_type=mutation_type,
                    parametric_specifications=parametric_specifications,
                    model_client=model_client,
                )
                private_context.payload["hypothesis_round_count"] = private_context.payload.get("hypothesis_round_count", 0) + 1
                private_context.payload["last_hypothesis_prompt_context"] = context
                private_context.payload["last_hypothesis_payload"] = hypothesis_payload
                private_context.payload["last_hypothesis_raw_response"] = raw_response
                return AgentResult(
                    shared_updates={
                        "hypothesis": hypothesis_payload["hypothesis"],
                        "hypothesis_reasoning": hypothesis_payload["concise_justification"],
                        "hypothesis_structured": hypothesis_payload,
                    },
                    artifacts={"agent": self.name},
                )
            except Exception as exc:
                last_error = exc
                retryable_response_error = (
                    "response parse failed" in str(exc).lower()
                    or "schema placeholder leaked" in str(exc).lower()
                )
                if retryable_response_error and parse_retries_used < max_parse_retries:
                    parse_retries_used += 1
                    self._append_diagnostic_event(
                        private_context,
                        stage="response_retry_scheduled",
                        details={
                            "retry_index": parse_retries_used,
                            "max_parse_retries": max_parse_retries,
                            "error": str(exc),
                        },
                    )
                    continue
                if self._is_input_length_error(str(exc)) and history_limit > min_history_limit:
                    self._append_diagnostic_event(
                        private_context,
                        stage="load_shedding_history",
                        details={
                            "reason": "input_length",
                            "previous_history_limit": history_limit,
                            "next_history_limit": history_limit - 1,
                            "report_char_limit": report_char_limit,
                            "error": str(exc),
                        },
                    )
                    history_limit -= 1
                    continue
                if self._is_transient_model_error(str(exc)):
                    if report_char_limit > min_report_char_limit:
                        next_report_char_limit = max(min_report_char_limit, self._next_report_char_limit(report_char_limit))
                        if next_report_char_limit < report_char_limit:
                            self._append_diagnostic_event(
                                private_context,
                                stage="load_shedding_report",
                                details={
                                    "reason": "transient_model_error",
                                    "previous_report_char_limit": report_char_limit,
                                    "next_report_char_limit": next_report_char_limit,
                                    "history_limit": history_limit,
                                    "error": str(exc),
                                },
                            )
                            report_char_limit = next_report_char_limit
                            continue
                    if history_limit > min_history_limit:
                        self._append_diagnostic_event(
                            private_context,
                            stage="load_shedding_history",
                            details={
                                "reason": "transient_model_error",
                                "previous_history_limit": history_limit,
                                "next_history_limit": history_limit - 1,
                                "report_char_limit": report_char_limit,
                                "error": str(exc),
                            },
                        )
                        history_limit -= 1
                        continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Hypothesis generation failed")
