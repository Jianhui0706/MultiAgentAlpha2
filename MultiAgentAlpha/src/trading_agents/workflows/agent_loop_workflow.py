from __future__ import annotations

from typing import Callable, Iterable

from trading_agents.core import BaseAgent, ContextPolicyRegistry, ModelClient, SequentialLoopOrchestrator
from trading_agents.infra import InMemoryContextStore, InMemoryTraceStore
from trading_agents.schemas import LoopTrace, SharedContext
from trading_agents.workflows.loop_runtime import LoopPolicy, LoopRunner, LoopSession


class AgentLoopWorkflow:
    def __init__(
        self,
        *,
        agents: Iterable[BaseAgent],
        policy_registry: ContextPolicyRegistry | None = None,
        model_client: ModelClient | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if model_client is None:
            raise RuntimeError("model_client is required. Please configure a real LLM gateway client.")
        self.trace_store = InMemoryTraceStore()
        self.context_store = InMemoryContextStore()
        self.orchestrator = SequentialLoopOrchestrator(
            agents=list(agents),
            context_store=self.context_store,
            model_client=model_client,
            policy_registry=policy_registry,
            trace_store=self.trace_store,
            progress_callback=progress_callback,
        )
        self.loop_runner = LoopRunner(
            orchestrator=self.orchestrator,
            context_store=self.context_store,
            progress_callback=progress_callback,
        )

    def run(self, initial_shared_payload: dict[str, object] | None = None) -> LoopTrace:
        return self.orchestrator.run_one_loop(SharedContext(payload=dict(initial_shared_payload or {})))

    def run_loops(
        self,
        *,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        stop_on_error: bool = True,
        retry_per_round: int = 0,
    ) -> LoopSession:
        return self.loop_runner.run(
            initial_shared_payload=initial_shared_payload,
            policy=LoopPolicy(
                max_loops=loop_count,
                stop_on_error=stop_on_error,
                retry_per_round=retry_per_round,
            ),
        )
