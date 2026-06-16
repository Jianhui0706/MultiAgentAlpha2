from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.prompts import PromptStore, render_prompt
from trading_agents.schemas import AgentContext, AgentResult, SharedContext


class EvolutionAgent(BaseAgent):
    """进化规划智能体（LLM 驱动）。

    核心职责：分析轨迹池中的 best/recent/failed 轨迹，规划下一轮的
    变异任务（mutation_tasks）和交叉任务（crossover_tasks）。

    输入：轨迹摘要（best/recent/failed 三类）、轮次信息、任务配额
    输出：JSON 格式的进化计划，包含 mutation_tasks 和 crossover_tasks 列表
    """

    # 允许的变异类型枚举（用于校验 LLM 输出）
    _ALLOWED_MUTATION_TYPES = {
        "parameter_mutation",    # 参数变异
        "expression_mutation",   # 表达式变异
        "risk_mutation",         # 风控变异
        "hypothesis_mutation",   # 假说变异
    }
    _ALLOWED_SEGMENTS = {
        "hypothesis_generation",
        "factor_expression",
        "implementation",
        "evaluation_feedback",
    }
    _ALLOWED_TARGET_GAPS = {
        "improve_sharpe",
        "improve_ic",
        "improve_rank_ic",
        "reduce_turnover",
        "reduce_complexity",
    }

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.prompt_store = PromptStore(Path(__file__).resolve().parent.parent / "prompts")

    def _load_prompt_bundle(self) -> dict[str, Any]:
        """从共享 PromptStore 加载 prompt 模板。"""
        return self.prompt_store.load_mapping("evolution_agent")

    @staticmethod
    def _compact_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _normalize_parent_ids(value: object) -> list[str]:
        """将 LLM 返回的 parent_ids 规范化为去重的字符串列表。"""
        if not isinstance(value, list):
            return []
        parent_ids: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in parent_ids:
                parent_ids.append(text)
        return parent_ids

    @classmethod
    def _normalize_segment(cls, value: object, *, fallback: str) -> str:
        text = str(value or "").strip()
        return text if text in cls._ALLOWED_SEGMENTS else fallback

    @classmethod
    def _normalize_target_gap(cls, value: object, *, fallback: str = "improve_sharpe") -> str:
        text = str(value or "").strip()
        return text if text in cls._ALLOWED_TARGET_GAPS else fallback

    @classmethod
    def _normalize_preserve_segments(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        segments: list[str] = []
        for item in value:
            segment = cls._normalize_segment(item, fallback="")
            if segment and segment not in segments:
                segments.append(segment)
        return segments

    @staticmethod
    def _normalize_source_segments(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        segments: list[dict[str, object]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            parent_id = str(item.get("parent_id", item.get("trajectory_id", ""))).strip()
            segment = str(item.get("segment", "")).strip()
            if not parent_id or not segment:
                continue
            segments.append(
                {
                    "parent_id": parent_id,
                    "segment": segment,
                    "inherit": str(item.get("inherit", item.get("rationale", ""))).strip(),
                }
            )
        return segments

    @classmethod
    def _normalize_mutation_task(cls, item: object, fallback_id: int) -> dict[str, object] | None:
        """校验并规范化单个变异任务。

        校验规则：
        - 必须是 dict 且包含有效的 new_direction
        - 必须有至少一个有效的 parent_id
        - mutation_type 必须在 _ALLOWED_MUTATION_TYPES 中，否则默认 hypothesis_mutation

        无效任务返回 None。
        """
        if not isinstance(item, dict):
            return None
        new_direction = str(item.get("new_direction", item.get("direction", ""))).strip()
        if not new_direction:
            return None
        parent_ids = cls._normalize_parent_ids(item.get("parent_ids", []))
        if not parent_ids:
            return None
        try:
            direction_id = int(item.get("direction_id", fallback_id))
        except Exception:
            direction_id = fallback_id
        mutation_type = str(item.get("mutation_type", "hypothesis_mutation")).strip()
        if mutation_type not in cls._ALLOWED_MUTATION_TYPES:
            mutation_type = "hypothesis_mutation"
        return {
            "direction_id": direction_id,
            "parent_ids": parent_ids,
            "new_direction": new_direction,
            "mutation_type": mutation_type,
            "rationale": str(item.get("rationale", "")).strip(),
            "target_gap": cls._normalize_target_gap(item.get("target_gap", item.get("objective", ""))),
            "metric_objective": str(item.get("metric_objective", item.get("objective", ""))).strip(),
            "preserve": str(item.get("preserve", "")).strip(),
            "change": str(item.get("change", "")).strip(),
            "target_segment": cls._normalize_segment(
                item.get("target_segment", ""),
                fallback=cls._default_target_segment(mutation_type),
            ),
            "fault_diagnosis": str(item.get("fault_diagnosis", item.get("diagnosis", ""))).strip(),
            "preserve_segments": cls._normalize_preserve_segments(item.get("preserve_segments", [])),
            "rewrite_instruction": str(item.get("rewrite_instruction", item.get("change", ""))).strip(),
        }

    @classmethod
    def _normalize_crossover_task(cls, item: object, fallback_id: int) -> dict[str, object] | None:
        """校验并规范化单个交叉任务。

        与变异任务类似，但要求至少 2 个有效的 parent_id。
        额外提取 fusion_logic 字段（融合逻辑说明）。
        """
        if not isinstance(item, dict):
            return None
        new_direction = str(item.get("new_direction", item.get("direction", ""))).strip()
        if not new_direction:
            return None
        parent_ids = cls._normalize_parent_ids(item.get("parent_ids", []))
        if len(parent_ids) < 2:
            return None
        try:
            direction_id = int(item.get("direction_id", fallback_id))
        except Exception:
            direction_id = fallback_id
        return {
            "direction_id": direction_id,
            "parent_ids": parent_ids,
            "new_direction": new_direction,
            "fusion_logic": str(item.get("fusion_logic", "")).strip(),
            "rationale": str(item.get("rationale", "")).strip(),
            "target_gap": cls._normalize_target_gap(item.get("target_gap", item.get("objective", ""))),
            "metric_objective": str(item.get("metric_objective", item.get("objective", ""))).strip(),
            "source_segments": cls._normalize_source_segments(item.get("source_segments", [])),
            "fusion_plan": str(item.get("fusion_plan", item.get("fusion_logic", ""))).strip(),
        }

    @staticmethod
    def _default_target_segment(mutation_type: str) -> str:
        if mutation_type == "hypothesis_mutation":
            return "hypothesis_generation"
        if mutation_type in {"expression_mutation", "parameter_mutation", "risk_mutation"}:
            return "factor_expression"
        return "factor_expression"

    @classmethod
    def normalize_plan(cls, payload: dict[str, Any]) -> dict[str, list[dict[str, object]]]:
        """将 LLM 返回的原始 payload 规范化为标准的进化计划格式。

        从 payload 中提取 mutation_tasks 和 crossover_tasks 列表，
        逐个校验和规范化，过滤掉无效项。

        Returns:
            {"mutation_tasks": [...], "crossover_tasks": [...]}
        """
        raw_mutation_tasks = payload.get("mutation_tasks", [])
        raw_crossover_tasks = payload.get("crossover_tasks", [])
        if not isinstance(raw_mutation_tasks, list):
            raw_mutation_tasks = []
        if not isinstance(raw_crossover_tasks, list):
            raw_crossover_tasks = []
        mutation_tasks = [
            task
            for idx, item in enumerate(raw_mutation_tasks)
            if (task := cls._normalize_mutation_task(item, 100 + idx)) is not None
        ]
        crossover_tasks = [
            task
            for idx, item in enumerate(raw_crossover_tasks)
            if (task := cls._normalize_crossover_task(item, 200 + idx)) is not None
        ]
        return {
            "mutation_tasks": mutation_tasks,
            "crossover_tasks": crossover_tasks,
        }

    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        """执行进化规划：调用 LLM 分析轨迹池，生成变异/交叉计划。

        流程：
        1. 从 SharedContext 中提取 round_idx、任务配额、轨迹摘要等上下文
        2. 渲染 prompt 模板（注入轨迹摘要 JSON）
        3. 调用 LLM（json_mode=True）获取结构化进化计划
        4. 解析并规范化计划，过滤无效任务
        5. 将计划写入 SharedContext 的 evolution_agent_plan 字段

        SharedContext 预期字段：
        - round_idx: 当前轮次
        - mutation_target_count / crossover_target_count: 任务配额
        - primary_metric: 主要评估指标
        - best_trajectories / recent_trajectories / failed_trajectories: 轨迹摘要列表
        """
        prompt_bundle = self._load_prompt_bundle()
        execution_failures = list(shared_context.payload.get("execution_failures", []) or [])
        context = {
            "round_idx": int(shared_context.payload.get("round_idx", 0) or 0),
            "mutation_target_count": int(shared_context.payload.get("mutation_target_count", 0) or 0),
            "crossover_target_count": int(shared_context.payload.get("crossover_target_count", 0) or 0),
            "primary_metric": str(shared_context.payload.get("primary_metric", "Rank IC")),
            "evolution_profile": str(shared_context.payload.get("evolution_profile", "")),
            "phase_schedule": str(shared_context.payload.get("phase_schedule", "")),
            "best_trajectories_json": self._compact_json(shared_context.payload.get("best_trajectories", [])),
            "recent_trajectories_json": self._compact_json(shared_context.payload.get("recent_trajectories", [])),
            "failed_trajectories_json": self._compact_json(shared_context.payload.get("failed_trajectories", [])),
            # 执行错误（agent 崩溃、LLM 失败等），区别于回测指标差的轨迹
            "execution_failures_json": self._compact_json(execution_failures),
        }
        system_prompt = render_prompt(str(prompt_bundle.get("system_prompt", "")), context)
        user_prompt = render_prompt(str(prompt_bundle.get("user_prompt", "")), context)
        raw_response = model_client.generate(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
        payload = parse_json_object(raw_response)
        plan = self.normalize_plan(payload)
        private_context.payload["last_evolution_agent_prompt_context"] = context
        private_context.payload["last_evolution_agent_prompt_chars"] = {
            "system": len(system_prompt),
            "user": len(user_prompt),
            "total": len(system_prompt) + len(user_prompt),
        }
        private_context.payload["last_evolution_agent_raw_response"] = raw_response
        private_context.payload["last_evolution_agent_payload"] = payload
        return AgentResult(
            shared_updates={
                "evolution_agent_plan": plan,
                "evolution_planner_mode": "llm",
            },
            artifacts={
                "agent": self.name,
                "mutation_task_count": len(plan["mutation_tasks"]),
                "crossover_task_count": len(plan["crossover_tasks"]),
            },
        )
