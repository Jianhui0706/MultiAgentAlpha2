#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from trading_agents.agents.factor_coder_agent import FactorCoderAgent
from trading_agents.core import ModelClient
from trading_agents.schemas import AgentContext, SharedContext


class SequentialClient(ModelClient):
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[tuple[str, str, bool]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        self.prompts.append((system_prompt, user_prompt, json_mode))
        if not self.payloads:
            raise AssertionError("No payload left for model call")
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)


class temporary_env:
    def __init__(self, updates: dict[str, str]) -> None:
        self.updates = dict(updates)
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.updates.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, previous in self.previous.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _iter_user_prompts(prompts: list[tuple[str, str, bool]]) -> Iterator[str]:
    for _system_prompt, user_prompt, _json_mode in prompts:
        yield user_prompt


def main() -> None:
    client = SequentialClient(
        [
            "No comment found",
            {
                "final_decision": True,
                "final_feedback": "Expression accepted after runtime checks.",
            },
        ]
    )
    private_context = AgentContext()
    shared_context = SharedContext(
        payload={
            "scenario": "panel_data_cross_section_factor_mining",
            "domain_dataset_summary": {
                "feature_columns": ["close", "volume"],
                "validation_target_column": "fwd_returns_4h",
            },
            "experiment_spec": {
                "factors": [
                    {
                        "factor_name": "rolling_max_close",
                        "description": "rolling max of close",
                        "formulation": "MAX($close, 5)",
                        "expression": "MAX($close, 5)",
                        "variables": {"$close": "close price"},
                    }
                ]
            },
            "factor_coder_max_loop": 1,
            "factor_coder_fail_task_trial_limit": 1,
            "factor_coder_fast_path_enabled": False,
        }
    )

    with temporary_env({"FAST_MODE": "false"}):
        result = FactorCoderAgent("factor_coder_agent").run(
            private_context=private_context,
            shared_context=shared_context,
            model_client=client,
        )

    generation_summary = private_context.payload.get("factor_coder_generation_skill_context_summary", {})
    retry_summary = private_context.payload.get("last_factor_coder_skill_context_summary", {})
    rule_files_used = list(private_context.payload.get("factor_coder_skill_rule_files_used", []))
    user_prompts = list(_iter_user_prompts(client.prompts))

    _assert(result.shared_updates["calculation_report"]["generated_factors"] == 1, "expected one generated factor")
    _assert(generation_summary.get("mode") == "generation", "generation skill context missing")
    _assert("DELAY" in list(generation_summary.get("function_rule_names", [])), "generation rule pack should include DELAY")
    _assert(retry_summary.get("mode") == "retry", "retry skill context missing")
    _assert(list(retry_summary.get("function_rule_names", [])) == ["TS_MAX"], "retry should target TS_MAX only")
    _assert(any("Project-local FactorCoder rule context" in prompt for prompt in user_prompts), "FactorCoder rule context not injected")
    _assert(any("Function Rule: TS_MAX" in prompt for prompt in user_prompts), "TS_MAX rule not injected into retry prompt")
    _assert(all("Function Rule: DELAY" not in prompt for prompt in user_prompts), "retry prompt should not carry unrelated DELAY rule")

    print("factor_coder skill injection validation passed")
    print("generation_summary=" + json.dumps(generation_summary, ensure_ascii=False))
    print("retry_summary=" + json.dumps(retry_summary, ensure_ascii=False))
    print("rule_files_used=" + json.dumps(rule_files_used, ensure_ascii=False))
    print("llm_call_count=" + str(len(client.prompts)))


if __name__ == "__main__":
    main()
