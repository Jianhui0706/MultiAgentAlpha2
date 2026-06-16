from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_agents.agents.hypothesis import HypothesisAgentV2
from trading_agents.core import ModelClient
from trading_agents.domains.crypto_cross_section_adapter import CryptoCrossSectionDomainAdapter
from trading_agents.schemas import AgentContext, SharedContext


class CapturingClient(ModelClient):
    def __init__(self) -> None:
        self.user_prompt = ""

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        self.user_prompt = user_prompt
        return json.dumps(
            {
                "hypothesis": "Recent high dispersion should make volatility-adjusted reversal signals stronger.",
                "thesis": "Recent market dispersion can reveal short-horizon cross-sectional reversal.",
                "mechanism": "Crowded short-term moves reverse when dispersion and volatility are elevated.",
                "expected_direction": "negative_forward_return",
                "expected_factor_return_sign": {
                    "top_divergence_strength": "negative",
                    "bottom_divergence_strength": "positive",
                    "mixed_divergence_score": "depends_on_encoding",
                },
                "market_structure_dependencies": ["continuous crypto trading"],
                "crypto_adaptation_summary": {
                    "transferable_core": "Cross-sectional extremes can revert.",
                    "non_transferable_parts": [],
                    "crypto_rewrite": "Use recent crypto panel dispersion as the conditioning context.",
                },
                "selected_parametric_specifications": ["short_horizon_reversal", "high_low_range_stress"],
                "proxy_reconstruction_plan": {
                    "input_features": ["$returns_24h", "$high_low_spread"],
                    "core_proxy_family": ["short_term_reversal"],
                    "supporting_proxy_family": ["spread_stress"],
                    "validation_target": "fwd_returns_4h",
                },
                "concise_knowledge": "Extreme short-term moves can reverse when volatility is elevated.",
                "concise_observation": "Dynamic regime summary indicates current dispersion.",
                "concise_justification": "The hypothesis links current regime to a testable proxy family.",
                "concise_specification": "Rank recent returns adjusted by spread or volatility.",
                "hypothesis_evolution_state": {
                    "core_variables": ["returns_24h", "high_low_spread"],
                    "verifiable_anchors": {
                        "input_features": ["$returns_24h", "$high_low_spread"],
                        "expression_patterns": ["RANK($returns_24h)"],
                    },
                    "refinement_stage": "initial_exploration",
                    "allowed_mutation_types": ["window_adjustment", "normalization_change"],
                },
            },
            ensure_ascii=False,
        )


def _panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=8, freq="D"), ["BTC", "ETH", "SOL"]],
        names=["datetime", "symbol"],
    )
    rows = len(index)
    return pd.DataFrame(
        {
            "close": range(rows),
            "returns_24h": [(-1) ** i * i / 100 for i in range(rows)],
            "volume_change_24h": [i / 50 for i in range(rows)],
            "high_low_spread": [0.01 + i / 1000 for i in range(rows)],
            "fwd_returns_4h": [0.001 * i for i in range(rows)],
        },
        index=index,
    )


def test_adapter_adds_dynamic_market_regime_summary(monkeypatch) -> None:
    adapter = CryptoCrossSectionDomainAdapter(Path("dummy.parquet"))
    monkeypatch.setattr(adapter, "_load_panel_data", _panel)

    payload = adapter.build_initial_payload()

    regime = payload["market_regime_summary"]
    assert isinstance(regime, dict)
    assert regime["status"] == "ok"
    assert "Dynamic market regime summary" in regime["text"]
    assert regime["available_columns"]["return"] == "returns_24h"
    specs = payload["parametric_specifications"]
    assert isinstance(specs, list)
    assert any(item["name"] == "high_low_range_stress" for item in specs)
    assert any("10-period high/low" in item["specification"] for item in specs)


def test_hypothesis_agent_uses_market_regime_only_for_first_non_report_seed() -> None:
    agent = HypothesisAgentV2("hypothesis_agent")
    client = CapturingClient()
    private_context = AgentContext()
    shared_context = SharedContext(
        payload={
            "scenario": "panel_data_cross_section_factor_mining",
            "direction": "discover cross-sectional alpha factors with robust turnover control",
            "rag_text": "static dataset summary",
            "market_regime_summary": {
                "text": "Dynamic market regime summary for hypothesis seeding: high dispersion.",
                "recent_cross_section_dispersion": 0.12,
            },
            "parametric_specifications": [
                {
                    "name": "high_low_range_stress",
                    "specification": "10-period high/low range",
                    "feature_anchors": ["high_low_spread"],
                }
            ],
        }
    )

    agent.run(private_context=private_context, shared_context=shared_context, model_client=client)

    assert "Dynamic Market Regime Summary" in client.user_prompt
    assert "Parametric Specifications" in client.user_prompt
    assert "10-period high/low range" in client.user_prompt
    assert private_context.payload["last_hypothesis_prompt_context"]["use_market_regime_for_seed"] is True
    assert private_context.payload["last_hypothesis_payload"]["selected_parametric_specifications"] == [
        "short_horizon_reversal",
        "high_low_range_stress",
    ]

    client_with_history = CapturingClient()
    agent.run(
        private_context=AgentContext(),
        shared_context=SharedContext(
            payload={
                **shared_context.payload,
                "hypothesis_feedback_history": [{"hypothesis": "old", "feedback_summary": "done"}],
            }
        ),
        model_client=client_with_history,
    )

    assert "Dynamic Market Regime Summary" not in client_with_history.user_prompt
