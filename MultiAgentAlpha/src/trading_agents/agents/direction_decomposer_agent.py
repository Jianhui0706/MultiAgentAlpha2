from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.prompts import PromptStore, render_prompt
from trading_agents.schemas import AgentContext, AgentResult, SharedContext


class DirectionDecomposerAgent(BaseAgent):
    """初始化假设生成智能体（LLM 驱动）。

    核心职责：从初始种子直接生成多个互补、可执行的初始假设。
    在遗传进化流程中被两处使用：
    1. 第 0 轮：生成 N 个 diversified initial hypotheses
    2. 兼容旧 joint-seed 模式：将进化种子拆解为 N 个下一代假设
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parent.parent / "prompts")

    def _load_prompt_bundle(self) -> dict[str, Any]:
        """从共享 PromptStore 加载 prompt 模板。"""
        return self.prompt_store.load_mapping("direction_decomposer_agent")

    @staticmethod
    def _normalize_direction(item: object, fallback_id: int) -> dict[str, object] | None:
        """校验并规范化旧版方向项。

        要求 direction 字段非空，提取 direction_id、title、rationale、
        expected_signal_family、risk_notes 等字段。
        无效项返回 None。
        """
        if not isinstance(item, dict):
            return None
        direction = str(item.get("direction", "")).strip()
        if not direction:
            return None
        try:
            direction_id = int(item.get("direction_id", fallback_id))
        except Exception:
            direction_id = fallback_id
        return {
            "direction_id": direction_id,
            "title": str(item.get("title", f"direction_{direction_id}")).strip() or f"direction_{direction_id}",
            "direction": direction,
            "rationale": str(item.get("rationale", "")).strip(),
            "expected_signal_family": str(item.get("expected_signal_family", "")).strip(),
            "risk_notes": str(item.get("risk_notes", "")).strip(),
            "diversity_axis": str(item.get("diversity_axis", "")).strip(),
        }

    @classmethod
    def normalize_directions(cls, payload: dict[str, Any], *, target_count: int) -> list[dict[str, object]]:
        """将 LLM 返回的旧版 directions payload 规范化为兼容列表。

        处理逻辑：
        1. 逐个校验方向项，过滤无效项
        2. 按方向文本去重（大小写不敏感）
        3. 重新编号 direction_id（连续递增）
        4. 截断到 target_count 个
        """
        raw_directions = payload.get("directions", [])
        if not isinstance(raw_directions, list):
            raw_directions = []
        directions: list[dict[str, object]] = []
        seen: set[str] = set()
        for idx, item in enumerate(raw_directions):
            normalized = cls._normalize_direction(item, idx)
            if normalized is None:
                continue
            direction_text = str(normalized["direction"]).strip().lower()
            if direction_text in seen:
                continue
            seen.add(direction_text)
            normalized["direction_id"] = len(directions)
            directions.append(normalized)
            if len(directions) >= target_count:
                break
        return directions

    @staticmethod
    def _normalize_string_list(value: object) -> list[str]:
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = [item.strip() for item in value.split(",")]
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

    @staticmethod
    def _canonical_text(value: object) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"[^\w$]+", "", text, flags=re.UNICODE)

    @classmethod
    def _flatten_text_parts(cls, value: object) -> list[str]:
        if isinstance(value, dict):
            parts: list[str] = []
            for key in sorted(value):
                parts.extend(cls._flatten_text_parts(value.get(key)))
            return parts
        if isinstance(value, list):
            parts = []
            for item in value:
                parts.extend(cls._flatten_text_parts(item))
            return parts
        text = str(value or "").strip()
        return [text] if text else []

    @classmethod
    def _hypothesis_family_key(cls, item: dict[str, object]) -> str:
        proxy_plan = item.get("proxy_reconstruction_plan", {})
        if not isinstance(proxy_plan, dict):
            proxy_plan = {}
        parts: list[str] = []
        for field in (
            "input_features",
            "core_proxy_family",
            "supporting_proxy_family",
            "validation_target",
        ):
            parts.extend(cls._flatten_text_parts(proxy_plan.get(field)))
        parts.extend(cls._flatten_text_parts(item.get("selected_parametric_specifications", [])))
        parts.extend(
            [
                str(item.get("expected_direction", "")).strip(),
                str(item.get("expected_signal_family", "")).strip(),
                str(item.get("diversity_axis", "")).strip(),
            ]
        )
        canonical_parts = [cls._canonical_text(part) for part in parts if cls._canonical_text(part)]
        # Avoid over-filtering sparse LLM outputs that do not provide enough structured anchors.
        if len(canonical_parts) < 2:
            return ""
        return "|".join(sorted(set(canonical_parts)))

    @classmethod
    def _similarity_text(cls, item: dict[str, object]) -> str:
        parts = [
            str(item.get("hypothesis", "")).strip(),
            str(item.get("thesis", "")).strip(),
            str(item.get("mechanism", "")).strip(),
            str(item.get("expected_signal_family", "")).strip(),
            str(item.get("diversity_axis", "")).strip(),
        ]
        parts.extend(cls._flatten_text_parts(item.get("selected_parametric_specifications", [])))
        parts.extend(cls._flatten_text_parts(item.get("proxy_reconstruction_plan", {})))
        return cls._canonical_text(" ".join(part for part in parts if part))

    @staticmethod
    def _char_ngrams(text: str, size: int = 3) -> set[str]:
        if not text:
            return set()
        if len(text) <= size:
            return {text}
        return {text[idx : idx + size] for idx in range(0, len(text) - size + 1)}

    @classmethod
    def _too_similar(cls, left: str, right: str, *, threshold: float = 0.92) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True
        left_grams = cls._char_ngrams(left)
        right_grams = cls._char_ngrams(right)
        if not left_grams or not right_grams:
            return False
        return len(left_grams & right_grams) / len(left_grams | right_grams) >= threshold

    @classmethod
    def _novelty_hypothesis_items(cls, novelty_context: dict[str, Any] | None) -> list[dict[str, object]]:
        if not isinstance(novelty_context, dict):
            return []
        raw_items: list[object] = []
        for key in (
            "recent_hypotheses",
            "top_reward_hypotheses",
            "avoid_hypotheses",
            "historical_hypotheses",
        ):
            value = novelty_context.get(key, [])
            if isinstance(value, list):
                raw_items.extend(value)
        items: list[dict[str, object]] = []
        for raw in raw_items:
            if isinstance(raw, dict):
                item = dict(raw)
            else:
                item = {"hypothesis": str(raw)}
            if not str(item.get("hypothesis", item.get("direction", ""))).strip():
                continue
            items.append(item)
        return items

    @classmethod
    def _seed_seen_from_novelty_context(
        cls,
        novelty_context: dict[str, Any] | None,
        *,
        seen_text: set[str],
        seen_family: set[str],
        seen_similarity_texts: list[str],
    ) -> int:
        seeded = 0
        for idx, item in enumerate(cls._novelty_hypothesis_items(novelty_context)):
            normalized = cls._normalize_initial_hypothesis(item, idx)
            if normalized is None:
                continue
            hypothesis_text = cls._canonical_text(normalized["hypothesis"])
            if hypothesis_text:
                seen_text.add(hypothesis_text)
            family_key = cls._hypothesis_family_key(normalized)
            if family_key:
                seen_family.add(family_key)
            similarity_text = cls._similarity_text(normalized)
            if similarity_text:
                seen_similarity_texts.append(similarity_text)
            seeded += 1
        return seeded

    @classmethod
    def _normalize_initial_hypothesis(cls, item: object, fallback_id: int) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        hypothesis = str(item.get("hypothesis", item.get("direction", ""))).strip()
        if not hypothesis:
            return None
        try:
            direction_id = int(item.get("direction_id", fallback_id))
        except Exception:
            direction_id = fallback_id
        proxy_plan = item.get("proxy_reconstruction_plan", {})
        if not isinstance(proxy_plan, dict):
            proxy_plan = {}
        evolution_state = item.get("hypothesis_evolution_state", {})
        if not isinstance(evolution_state, dict):
            evolution_state = {}
        return {
            "direction_id": direction_id,
            "title": str(item.get("title", f"hypothesis_{direction_id}")).strip() or f"hypothesis_{direction_id}",
            "hypothesis": hypothesis,
            "thesis": str(item.get("thesis", hypothesis)).strip() or hypothesis,
            "mechanism": str(item.get("mechanism", "")).strip(),
            "expected_direction": str(item.get("expected_direction", "")).strip(),
            "proxy_reconstruction_plan": dict(proxy_plan),
            "selected_parametric_specifications": cls._normalize_string_list(
                item.get("selected_parametric_specifications", [])
            ),
            "hypothesis_evolution_state": dict(evolution_state),
            "rationale": str(item.get("rationale", item.get("concise_justification", ""))).strip(),
            "concise_observation": str(item.get("concise_observation", "")).strip(),
            "concise_justification": str(item.get("concise_justification", item.get("rationale", ""))).strip(),
            "concise_specification": str(item.get("concise_specification", "")).strip(),
            "diversity_axis": str(item.get("diversity_axis", "")).strip(),
            "expected_signal_family": str(item.get("expected_signal_family", "")).strip(),
            "risk_notes": str(item.get("risk_notes", "")).strip(),
        }

    @classmethod
    def normalize_initial_hypotheses(
        cls,
        payload: dict[str, Any],
        *,
        target_count: int,
        novelty_context: dict[str, Any] | None = None,
    ) -> list[dict[str, object]]:
        raw_hypotheses = payload.get("initial_hypotheses", payload.get("hypotheses", []))
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []
        hypotheses: list[dict[str, object]] = []
        seen_text: set[str] = set()
        seen_family: set[str] = set()
        seen_similarity_texts: list[str] = []
        cls._seed_seen_from_novelty_context(
            novelty_context,
            seen_text=seen_text,
            seen_family=seen_family,
            seen_similarity_texts=seen_similarity_texts,
        )
        for idx, item in enumerate(raw_hypotheses):
            normalized = cls._normalize_initial_hypothesis(item, idx)
            if normalized is None:
                continue
            hypothesis_text = cls._canonical_text(normalized["hypothesis"])
            if hypothesis_text in seen_text:
                continue
            family_key = cls._hypothesis_family_key(normalized)
            if family_key and family_key in seen_family:
                continue
            similarity_text = cls._similarity_text(normalized)
            if any(cls._too_similar(similarity_text, previous) for previous in seen_similarity_texts):
                continue
            seen_text.add(hypothesis_text)
            if family_key:
                seen_family.add(family_key)
            if similarity_text:
                seen_similarity_texts.append(similarity_text)
            normalized["direction_id"] = len(hypotheses)
            hypotheses.append(normalized)
            if len(hypotheses) >= target_count:
                break
        return hypotheses

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        """执行初始化假设生成：调用 LLM 生成 N 个聚焦、可执行假设。

        流程：
        1. 从 SharedContext 提取 initial_direction、target_direction_count、
           数据集摘要、研究报告等上下文
        2. 渲染 prompt 模板
        3. 调用 LLM（json_mode=True）获取初始假设列表
        4. 规范化并校验假设数量，不足时交由 runner fallback 补齐

        SharedContext 预期字段：
        - initial_direction: 作为 A0 输入的宽泛种子文本
        - target_direction_count: 目标初始假设数量（沿用旧字段名）
        - domain_dataset_summary_json: 数据集摘要（可选）
        - research_report_json: 研究报告（可选）
        """
        prompt_bundle = self._load_prompt_bundle()
        target_count = int(shared_context.payload.get("target_direction_count", 5) or 5)
        novelty_context = shared_context.payload.get("novelty_context", {})
        if not isinstance(novelty_context, dict):
            novelty_context = {}
        context = {
            "initial_direction": str(shared_context.payload.get("initial_direction", shared_context.payload.get("direction", ""))).strip(),
            "target_direction_count": target_count,
            "scenario": str(shared_context.payload.get("scenario", "")),
            "market": str(shared_context.payload.get("market", "")),
            "market_regime_summary_json": json.dumps(
                shared_context.payload.get("market_regime_summary", {}),
                ensure_ascii=False,
                indent=2,
            ),
            "domain_dataset_summary_json": json.dumps(
                shared_context.payload.get("domain_dataset_summary", {}),
                ensure_ascii=False,
                indent=2,
            ),
            "research_report_json": json.dumps(
                shared_context.payload.get("research_report", {}),
                ensure_ascii=False,
                indent=2,
            ),
            "novelty_context_json": json.dumps(
                novelty_context,
                ensure_ascii=False,
                indent=2,
            ),
            "parametric_specifications_json": json.dumps(
                shared_context.payload.get("parametric_specifications", []),
                ensure_ascii=False,
                indent=2,
            ),
        }
        system_prompt = render_prompt(str(prompt_bundle.get("system_prompt", "")), context)
        user_prompt = render_prompt(str(prompt_bundle.get("user_prompt", "")), context)
        raw_response = model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
        payload = parse_json_object(raw_response)
        raw_initial_hypotheses = payload.get("initial_hypotheses", payload.get("hypotheses", []))
        raw_initial_hypothesis_count = len(raw_initial_hypotheses) if isinstance(raw_initial_hypotheses, list) else 0
        raw_directions = payload.get("directions", [])
        raw_direction_count = len(raw_directions) if isinstance(raw_directions, list) else 0
        initial_hypotheses = self.normalize_initial_hypotheses(
            payload,
            target_count=target_count,
            novelty_context=novelty_context,
        )
        directions: list[dict[str, object]]
        if initial_hypotheses:
            directions = [
                {
                    "direction_id": int(item.get("direction_id", idx)),
                    "title": str(item.get("title", f"hypothesis_{idx}")),
                    "direction": str(item.get("hypothesis", "")),
                    "rationale": str(item.get("rationale", item.get("concise_justification", ""))),
                    "expected_signal_family": str(item.get("expected_signal_family", "")),
                    "risk_notes": str(item.get("risk_notes", "")),
                    "diversity_axis": str(item.get("diversity_axis", "")),
                }
                for idx, item in enumerate(initial_hypotheses)
            ]
        else:
            directions = self.normalize_directions(payload, target_count=target_count)
        # 论文对初始假设数量是容忍的：只要拿到 >=1 个有效项就接受，不足部分交由
        # EvolutionRunner 的确定性 fallback（_diversified_directions）补齐。
        # 仅当一个有效项都没有时才视为失败抛错。
        if not directions:
            raise RuntimeError("Initialization agent returned 0 valid hypotheses.")
        shortfall = max(0, target_count - len(directions))
        private_context.payload["last_direction_decomposer_prompt_context"] = context
        private_context.payload["last_direction_decomposer_raw_response"] = raw_response
        private_context.payload["last_direction_decomposer_payload"] = payload
        novelty_factor_expressions = novelty_context.get("recent_factor_expressions", [])
        novelty_expression_count = len(novelty_factor_expressions) if isinstance(novelty_factor_expressions, list) else 0
        novelty_hypothesis_count = len(self._novelty_hypothesis_items(novelty_context))
        return AgentResult(
            shared_updates={
                "decomposed_directions": directions,
                "initial_hypotheses": initial_hypotheses,
                "direction_decomposer_mode": "llm",
            },
            artifacts={
                "agent": self.name,
                "direction_count": len(directions),
                "initial_hypothesis_count": len(initial_hypotheses),
                "raw_direction_count": raw_direction_count,
                "raw_initial_hypothesis_count": raw_initial_hypothesis_count,
                "novelty_context_hypothesis_count": novelty_hypothesis_count,
                "novelty_context_expression_count": novelty_expression_count,
                "deduplicated_initial_hypothesis_count": max(
                    0,
                    raw_initial_hypothesis_count - len(initial_hypotheses),
                ),
                "shortfall": shortfall,
            },
        )
