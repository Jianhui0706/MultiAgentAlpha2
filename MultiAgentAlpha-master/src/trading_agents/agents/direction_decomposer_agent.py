from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.prompts import render_prompt
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
        self.prompt_file = Path(__file__).resolve().parent.parent / "prompts" / "direction_decomposer_agent.json"

    def _load_prompt_bundle(self) -> dict[str, Any]:
        """从 JSON 文件加载 prompt 模板。"""
        return json.loads(self.prompt_file.read_text(encoding="utf-8"))

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
    def normalize_initial_hypotheses(cls, payload: dict[str, Any], *, target_count: int) -> list[dict[str, object]]:
        raw_hypotheses = payload.get("initial_hypotheses", payload.get("hypotheses", []))
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []
        hypotheses: list[dict[str, object]] = []
        seen: set[str] = set()
        for idx, item in enumerate(raw_hypotheses):
            normalized = cls._normalize_initial_hypothesis(item, idx)
            if normalized is None:
                continue
            hypothesis_text = str(normalized["hypothesis"]).strip().lower()
            if hypothesis_text in seen:
                continue
            seen.add(hypothesis_text)
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
        initial_hypotheses = self.normalize_initial_hypotheses(payload, target_count=target_count)
        directions = self.normalize_directions(payload, target_count=target_count)
        if not directions and initial_hypotheses:
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
        # 论文对初始假设数量是容忍的：只要拿到 >=1 个有效项就接受，不足部分交由
        # EvolutionRunner 的确定性 fallback（_diversified_directions）补齐。
        # 仅当一个有效项都没有时才视为失败抛错。
        if not directions:
            raise RuntimeError("Initialization agent returned 0 valid hypotheses.")
        shortfall = max(0, target_count - len(directions))
        private_context.payload["last_direction_decomposer_prompt_context"] = context
        private_context.payload["last_direction_decomposer_raw_response"] = raw_response
        private_context.payload["last_direction_decomposer_payload"] = payload
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
                "shortfall": shortfall,
            },
        )
