#!/usr/bin/env python3
"""Inspect prompt files loaded by the shared PromptStore.

The repository now keeps long prompt templates in Markdown front matter format
where practical. This helper intentionally avoids rewriting prompt text; edit
the source prompt files directly, then use this script to confirm that each
JSON-mode prompt still explicitly asks for JSON output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = PROJECT_ROOT / "src" / "trading_agents" / "prompts"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_agents.prompts import PromptStore  # noqa: E402


DEFAULT_JSON_MODE_PROMPTS = (
    "hypothesis_agent",
    "feedback_summarizer_agent",
    "direction_decomposer_agent",
    "evolution_agent",
)


def prompt_mentions_json(prompt_name: str, *, prompt_root: Path = PROMPT_ROOT) -> bool:
    bundle = PromptStore(prompt_root).load(prompt_name)
    combined = f"{bundle.system_prompt}\n{bundle.user_prompt}".lower()
    return "json" in combined


def check_prompts(prompt_names: list[str], *, prompt_root: Path = PROMPT_ROOT) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for prompt_name in prompt_names:
        try:
            results[prompt_name] = prompt_mentions_json(prompt_name, prompt_root=prompt_root)
        except Exception as exc:
            print(f"ERROR {prompt_name}: {exc}")
            results[prompt_name] = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that JSON-mode prompt templates mention JSON output explicitly."
    )
    parser.add_argument(
        "--file",
        default="hypothesis_agent",
        help="Prompt key to check when --all is not set.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=PROMPT_ROOT,
        help="Prompt directory. Defaults to src/trading_agents/prompts.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check the known JSON-mode prompt templates.",
    )
    args = parser.parse_args()

    prompt_names = list(DEFAULT_JSON_MODE_PROMPTS) if args.all else [args.file]
    results = check_prompts(prompt_names, prompt_root=args.base_path)
    for prompt_name, passed in results.items():
        status = "ok" if passed else "missing-json-keyword"
        print(f"{status}: {prompt_name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
