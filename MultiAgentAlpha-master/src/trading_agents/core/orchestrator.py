from __future__ import annotations

import time
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Callable, Iterable

from trading_agents.core.base_agent import BaseAgent
from trading_agents.core.context_assembler import ContextAssembler
from trading_agents.core.context_policy import ContextPolicyRegistry
from trading_agents.core.context_store import ContextStore
from trading_agents.core.model_client import (
    ModelClient,
    reset_current_agent_name,
    set_current_agent_name,
)
from trading_agents.core.trace_store import TraceStore
from trading_agents.schemas import LoopTrace, SharedContext, TraceRecord


class Orchestrator(ABC):
    """编排器抽象基类：定义单轮执行入口的统一接口。"""
    @abstractmethod
    def run_one_loop(self, initial_shared_context: SharedContext | None = None) -> LoopTrace:
        """执行一轮 agent 链路，返回包含完整记录的 LoopTrace。"""
        raise NotImplementedError


class SequentialLoopOrchestrator(Orchestrator):
    """顺序编排器：按 agents 列表顺序逐个执行，并维护上下文与轨迹。"""
    def __init__(
        self,
        *,
        agents: Iterable[BaseAgent],
        context_store: ContextStore,
        model_client: ModelClient,
        policy_registry: ContextPolicyRegistry | None = None,
        trace_store: TraceStore | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.agents = list(agents)
        self.context_store = context_store
        self.model_client = model_client
        self.policy_registry = policy_registry or ContextPolicyRegistry()
        self.policy_registry.validate_agents(agent.name for agent in self.agents)
        self.trace_store = trace_store
        self.progress_callback = progress_callback
        self.context_assembler = ContextAssembler(
            policy_registry=self.policy_registry,
            trace_store=self.trace_store,
        )

    def _notify_progress(self, payload: dict[str, object]) -> None:
        """安全触发进度回调：回调异常被吞掉，不影响主流程。"""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            return

    def _strip_recent_trace(self, payload: object) -> object:
        if isinstance(payload, dict):
            return {
                key: self._strip_recent_trace(value)
                for key, value in payload.items()
                if key != "recent_trace"
            }
        if isinstance(payload, list):
            return [self._strip_recent_trace(item) for item in payload]
        if isinstance(payload, tuple):
            return [self._strip_recent_trace(item) for item in payload]
        return payload

    @staticmethod
    def _snapshot(payload: object) -> object:
        try:
            return deepcopy(payload)
        except Exception:
            return payload

    def run_one_loop(self, initial_shared_context: SharedContext | None = None) -> LoopTrace:
        """执行单轮顺序链路，失败即短路，成功则返回完整轨迹与最终共享上下文。"""
        return self._run_agents(initial_shared_context=initial_shared_context, start_agent_name=None)

    def run_from_agent(
        self,
        start_agent_name: str,
        initial_shared_context: SharedContext | None = None,
    ) -> LoopTrace:
        """从指定 agent 开始执行后缀链路，用于轨迹级局部 mutation。"""
        return self._run_agents(initial_shared_context=initial_shared_context, start_agent_name=start_agent_name)

    def _run_agents(
        self,
        *,
        initial_shared_context: SharedContext | None,
        start_agent_name: str | None,
    ) -> LoopTrace:
        """执行完整链路或从指定 agent 开始的后缀链路。"""
        if initial_shared_context is not None:
            self.context_store.set_shared_context(initial_shared_context)

        shared_context = self.context_store.get_shared_context()
        trace = LoopTrace()
        agents = list(self.agents)
        if start_agent_name:
            names = [agent.name for agent in agents]
            if start_agent_name not in names:
                raise ValueError(f"unknown start_agent_name: {start_agent_name}")
            agents = agents[names.index(start_agent_name) :]

        for agent in agents:
            private_context = self.context_store.get_private_context(agent.name)
            shared_context_for_agent = self.context_assembler.build_shared_view(agent.name, shared_context)
            recent_trace = self.context_assembler.get_recent_records(agent.name)
            private_context.payload["recent_trace"] = [
                {
                    "agent_name": record.agent_name,
                    "input_snapshot": self._strip_recent_trace(record.input_snapshot),
                    "output_snapshot": self._strip_recent_trace(record.output_snapshot),
                    "error": record.error,
                }
                for record in recent_trace
            ]
            input_snapshot = {
                "private_context": self._snapshot(private_context.payload),
                "shared_context": self._snapshot(shared_context_for_agent.payload),
            }
            started_at = datetime.now()
            t_start = time.perf_counter()
            error: str | None = None
            output_snapshot: dict[str, object] = {}
            current_agent_token = set_current_agent_name(agent.name)
            self._notify_progress(
                {
                    "stage": "agent_started",
                    "agent_name": agent.name,
                    "started_at": started_at.isoformat(),
                    "input_snapshot": input_snapshot,
                }
            )

            try:
                result = agent.run(
                    private_context=private_context,
                    shared_context=shared_context_for_agent,
                    model_client=self.model_client,
                )
                filtered_updates = self.policy_registry.apply_write_filter(agent.name, result.shared_updates)
                shared_context.payload.update(filtered_updates)
                output_snapshot = {
                    "shared_updates": self._snapshot(filtered_updates),
                    "artifacts": self._snapshot(result.artifacts),
                    "private_context": self._snapshot(private_context.payload),
                    "shared_context": self._snapshot(shared_context.payload),
                }
                self._notify_progress(
                    {
                        "stage": "agent_finished",
                        "agent_name": agent.name,
                        "started_at": started_at.isoformat(),
                        "finished_at": datetime.now().isoformat(),
                        "elapsed_seconds": round(time.perf_counter() - t_start, 3),
                        "output_snapshot": output_snapshot,
                    }
                )
            except Exception as exc:
                error = str(exc)
                output_snapshot = {
                    "private_context": self._snapshot(private_context.payload),
                    "shared_context": self._snapshot(shared_context.payload),
                }
                self._notify_progress(
                    {
                        "stage": "agent_failed",
                        "agent_name": agent.name,
                        "started_at": started_at.isoformat(),
                        "failed_at": datetime.now().isoformat(),
                        "elapsed_seconds": round(time.perf_counter() - t_start, 3),
                        "error": error,
                        "output_snapshot": output_snapshot,
                    }
                )
            finally:
                reset_current_agent_name(current_agent_token)
                ended_at = datetime.now()
                elapsed = round(time.perf_counter() - t_start, 3)
                self.context_store.set_private_context(agent.name, private_context)
                record = TraceRecord(
                    agent_name=agent.name,
                    started_at=started_at,
                    ended_at=ended_at,
                    elapsed_seconds=elapsed,
                    input_snapshot=input_snapshot,
                    output_snapshot=output_snapshot,
                    error=error,
                )
                trace.records.append(record)
                if self.trace_store is not None:
                    self.trace_store.add_record(record)

            if error is not None:
                break

        self.context_store.set_shared_context(shared_context)
        trace.final_shared_context = self._snapshot(shared_context.payload)
        return trace
