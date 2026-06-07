from __future__ import annotations

from trading_agents.core import BaseAgent, ModelClient
from trading_agents.schemas import AgentContext, AgentResult, SharedContext


class HypothesisAgent(BaseAgent):
    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        message = model_client.generate(
            system_prompt="You are hypothesis agent.",
            user_prompt="Generate one hypothesis.",
            json_mode=False,
        )
        private_context.payload["last_hypothesis"] = message
        return AgentResult(shared_updates={"hypothesis": message}, artifacts={"agent": self.name})


class StrategyBuilderAgent(BaseAgent):
    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        hypothesis = str(shared_context.payload.get("hypothesis", ""))
        plan = f"plan_for::{hypothesis}"
        private_context.payload["last_plan"] = plan
        return AgentResult(shared_updates={"plan": plan}, artifacts={"agent": self.name})


class EvaluatorAgent(BaseAgent):
    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        plan = str(shared_context.payload.get("plan", ""))
        metrics = {"sharpe": 0.5, "turnover": 0.2, "drawdown": 0.1}
        private_context.payload["last_metrics"] = metrics
        return AgentResult(
            shared_updates={"execution": {"plan": plan, "metrics": metrics}},
            artifacts={"agent": self.name},
        )


class ReviewerAgent(BaseAgent):
    def run(
        self,
        *,
        private_context: AgentContext,
        shared_context: SharedContext,
        model_client: ModelClient,
    ) -> AgentResult:
        execution = dict(shared_context.payload.get("execution", {}))
        feedback = {
            "summary": "stub_review_completed",
            "score": 0.7,
            "actions": ["refine signal", "reduce turnover"],
            "execution": execution,
        }
        private_context.payload["last_feedback"] = feedback
        return AgentResult(shared_updates={"feedback": feedback}, artifacts={"agent": self.name})
