from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from trading_agents.agents import (
    BacktestRunnerAgent,
    ExperimentDesignerAgent,
    FactorCoderAgent,
    FeedbackSummarizerAgent,
    HypothesisAgentV2,
)
from trading_agents.core import ModelClient
from trading_agents.domains import CryptoCrossSectionDomainAdapter
from trading_agents.infra.factor_library_manager import FactorLibraryManager
from trading_agents.infra.research_report_loader import build_combined_research_report_payload
from trading_agents.infra.research_report_loader import build_research_report_payload
from trading_agents.schemas import LoopTrace, SharedContext
from trading_agents.workflows.agent_loop_workflow import AgentLoopWorkflow
from trading_agents.workflows.context_policies import build_alpha_factor_mining_policy_registry
from trading_agents.workflows.evolution_runner import EvolutionConfig, EvolutionRunner
from trading_agents.workflows.loop_runtime import LoopSession


class AlphaFactorMiningWorkflow(AgentLoopWorkflow):
    def __init__(
        self,
        *,
        model_client: ModelClient | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        super().__init__(
            agents=[
                HypothesisAgentV2("hypothesis_agent"),
                ExperimentDesignerAgent("experiment_designer_agent"),
                FactorCoderAgent("factor_coder_agent"),
                BacktestRunnerAgent("backtest_runner_agent"),
                FeedbackSummarizerAgent("feedback_summarizer_agent"),
            ],
            policy_registry=build_alpha_factor_mining_policy_registry(),
            model_client=model_client,
            progress_callback=progress_callback,
        )
        self.last_session: LoopSession | None = None
        self.evolution_runner = EvolutionRunner(
            orchestrator=self.orchestrator,
            context_store=self.context_store,
            model_client=model_client,
            progress_callback=progress_callback,
        )

    @staticmethod
    def _apply_initial_direction(
        payload: dict[str, object] | None,
        initial_direction: str | None,
    ) -> dict[str, object]:
        merged_payload = dict(payload or {})
        if initial_direction is not None:
            stripped = initial_direction.strip()
            if stripped:
                merged_payload["direction"] = stripped
        return merged_payload

    @staticmethod
    def _apply_report_driven_mode(
        payload: dict[str, object] | None,
        *,
        research_report_pdf: str | Path | None = None,
        research_report_pdfs: Sequence[str | Path] | None = None,
    ) -> dict[str, object]:
        merged_payload = dict(payload or {})
        if not merged_payload.get("research_report"):
            return merged_payload
        report_paths = AlphaFactorMiningWorkflow._normalize_report_paths(
            [item for item in list(research_report_pdfs or [])] + ([research_report_pdf] if research_report_pdf else [])
        )
        source = report_paths[0] if report_paths else str(
            ((merged_payload.get("research_report", {}) or {}).get("source", {}) or {}).get("file_path", "")
        )
        merged_payload.setdefault("workflow_mode", "report_driven_factor_mining")
        merged_payload.setdefault(
            "research_input_mode",
            "research_report_pdfs" if len(report_paths) > 1 else "research_report_pdf",
        )
        merged_payload["report_driven"] = True
        if source:
            merged_payload["report_driven_source"] = source
        if report_paths:
            merged_payload["report_driven_sources"] = report_paths
            research_report = merged_payload.get("research_report", {})
            if isinstance(research_report, dict):
                existing_source_reports = research_report.get("source_reports", [])
                if isinstance(existing_source_reports, list) and existing_source_reports:
                    enriched_source_reports = []
                    for index, source_report in enumerate(existing_source_reports):
                        if isinstance(source_report, dict):
                            enriched_item = dict(source_report)
                        else:
                            enriched_item = {}
                        if index < len(report_paths):
                            enriched_item.setdefault("file_path", report_paths[index])
                            enriched_item.setdefault("file_name", Path(report_paths[index]).name)
                        enriched_item["primary_payload_source"] = index == 0
                        enriched_source_reports.append(enriched_item)
                    research_report["source_reports"] = enriched_source_reports
                else:
                    research_report["source_reports"] = [
                        {
                            "file_path": item,
                            "file_name": Path(item).name,
                            "primary_payload_source": index == 0,
                        }
                        for index, item in enumerate(report_paths)
                    ]
                research_report["multi_report_input_count"] = len(report_paths)
                research_report.setdefault(
                    "multi_report_synthesis_status",
                    "step_2_combined_payload" if len(report_paths) > 1 else "single_report",
                )
                merged_payload["research_report"] = research_report
        return merged_payload

    @staticmethod
    def _normalize_report_paths(report_paths: Sequence[str | Path | None]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in report_paths:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @classmethod
    def build_report_driven_initial_payload(
        cls,
        *,
        initial_shared_payload: dict[str, object] | None = None,
        research_report_pdf: str | Path,
    ) -> dict[str, object]:
        payload = dict(initial_shared_payload or {})
        dataset_summary = dict(payload.get("domain_dataset_summary", {}))
        payload.update(
            build_research_report_payload(
                research_report_pdf,
                dataset_summary=dataset_summary,
            )
        )
        return cls._apply_report_driven_mode(payload, research_report_pdf=research_report_pdf)

    @classmethod
    def build_multi_report_driven_initial_payload(
        cls,
        *,
        initial_shared_payload: dict[str, object] | None = None,
        research_report_pdfs: Sequence[str | Path],
    ) -> dict[str, object]:
        report_paths = cls._normalize_report_paths(research_report_pdfs)
        if not report_paths:
            return dict(initial_shared_payload or {})
        payload = dict(initial_shared_payload or {})
        dataset_summary = dict(payload.get("domain_dataset_summary", {}))
        payload.update(
            build_combined_research_report_payload(
                report_paths,
                dataset_summary=dataset_summary,
            )
        )
        return cls._apply_report_driven_mode(
            payload,
            research_report_pdf=report_paths[0],
            research_report_pdfs=report_paths,
        )

    def run(
        self,
        initial_shared_payload: dict[str, object] | None = None,
        *,
        initial_direction: str | None = None,
    ) -> LoopTrace:
        payload = self._apply_initial_direction(initial_shared_payload, initial_direction)
        initial_context = SharedContext(payload=payload)
        trace = self.orchestrator.run_one_loop(initial_context)
        factor_library = FactorLibraryManager(FactorLibraryManager.default_library_path())
        factor_library.add_from_shared_payload(trace.final_shared_context)
        return trace

    def run_loops(
        self,
        *,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
    ) -> list[LoopTrace]:
        shared_payload = self._apply_initial_direction(initial_shared_payload, initial_direction)
        session = super().run_loops(
            loop_count=loop_count,
            initial_shared_payload=shared_payload,
        )
        self.last_session = session
        return list(session.loop_traces)

    def run_loop_session(
        self,
        *,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
        stop_on_error: bool = True,
        retry_per_round: int = 0,
    ) -> LoopSession:
        shared_payload = self._apply_initial_direction(initial_shared_payload, initial_direction)
        session = super().run_loops(
            loop_count=loop_count,
            initial_shared_payload=shared_payload,
            stop_on_error=stop_on_error,
            retry_per_round=retry_per_round,
        )
        self.last_session = session
        return session

    def run_evolution_session(
        self,
        *,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
        evolution_config: EvolutionConfig | None = None,
    ) -> LoopSession:
        """运行遗传迭代工作流，同时保持普通循环入口不变。"""
        session = self.evolution_runner.run(
            initial_shared_payload=initial_shared_payload,
            initial_direction=initial_direction,
            config=evolution_config or EvolutionConfig(),
        )
        self.last_session = session
        return session

    def run_report_driven(
        self,
        *,
        research_report_pdf: str | Path,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
    ) -> LoopTrace:
        payload = self.build_report_driven_initial_payload(
            initial_shared_payload=initial_shared_payload,
            research_report_pdf=research_report_pdf,
        )
        return self.run(
            initial_shared_payload=payload,
            initial_direction=initial_direction,
        )

    def run_report_driven_loops(
        self,
        *,
        research_report_pdf: str | Path,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
    ) -> list[LoopTrace]:
        payload = self.build_report_driven_initial_payload(
            initial_shared_payload=initial_shared_payload,
            research_report_pdf=research_report_pdf,
        )
        return self.run_loops(
            loop_count=loop_count,
            initial_shared_payload=payload,
            initial_direction=initial_direction,
        )

    def run_report_driven_loop_session(
        self,
        *,
        research_report_pdf: str | Path,
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
        stop_on_error: bool = True,
        retry_per_round: int = 0,
    ) -> LoopSession:
        payload = self.build_report_driven_initial_payload(
            initial_shared_payload=initial_shared_payload,
            research_report_pdf=research_report_pdf,
        )
        return self.run_loop_session(
            loop_count=loop_count,
            initial_shared_payload=payload,
            initial_direction=initial_direction,
            stop_on_error=stop_on_error,
            retry_per_round=retry_per_round,
        )

    def run_multi_report_driven_loop_session(
        self,
        *,
        research_report_pdfs: Sequence[str | Path],
        loop_count: int,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
        stop_on_error: bool = True,
        retry_per_round: int = 0,
    ) -> LoopSession:
        payload = self.build_multi_report_driven_initial_payload(
            initial_shared_payload=initial_shared_payload,
            research_report_pdfs=research_report_pdfs,
        )
        return self.run_loop_session(
            loop_count=loop_count,
            initial_shared_payload=payload,
            initial_direction=initial_direction,
            stop_on_error=stop_on_error,
            retry_per_round=retry_per_round,
        )

    def run_with_crypto_panel(
        self,
        panel_data_path: str | Path,
        *,
        loop_count: int = 5,
        initial_shared_payload: dict[str, object] | None = None,
        initial_direction: str | None = None,
    ) -> list[LoopTrace]:
        adapter = CryptoCrossSectionDomainAdapter(panel_data_path)
        payload = adapter.build_initial_payload()
        if initial_shared_payload:
            payload.update(initial_shared_payload)
        return self.run_loops(
            loop_count=loop_count,
            initial_shared_payload=payload,
            initial_direction=initial_direction,
        )
