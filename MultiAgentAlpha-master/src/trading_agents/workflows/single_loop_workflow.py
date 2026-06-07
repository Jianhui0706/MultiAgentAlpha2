from __future__ import annotations

from trading_agents.schemas import LoopTrace
from trading_agents.workflows.alpha_factor_mining_workflow import AlphaFactorMiningWorkflow


class SingleLoopWorkflow:
    def __init__(self) -> None:
        self.workflow = AlphaFactorMiningWorkflow()

    def run(self, initial_shared_payload: dict[str, object] | None = None) -> LoopTrace:
        return self.workflow.run(initial_shared_payload)
