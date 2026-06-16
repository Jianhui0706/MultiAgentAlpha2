from pathlib import Path

from trading_agents.prompts import PromptStore


def test_prompt_store_loads_markdown_front_matter_and_sections(tmp_path: Path) -> None:
    prompt_file = tmp_path / "sample_agent.md"
    prompt_file.write_text(
        """---
max_json_parse_retries: 2
targets: factors
---

# System Prompt

Return JSON for {{ target }}.

# User Prompt

Use the supplied context.

# Output Format

{"answer": "中文文本"}
""",
        encoding="utf-8",
    )

    bundle = PromptStore(tmp_path).load("sample_agent")

    assert bundle.system_prompt == "Return JSON for {{ target }}."
    assert bundle.user_prompt == "Use the supplied context."
    assert bundle.metadata["max_json_parse_retries"] == 2
    assert bundle.metadata["targets"] == "factors"
    assert bundle.metadata["output_format"] == '{"answer": "中文文本"}'


def test_prompt_store_loads_repository_markdown_prompts() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "src" / "trading_agents" / "prompts"
    store = PromptStore(prompt_root)

    hypothesis = store.load("hypothesis_agent")
    feedback = store.load("feedback_summarizer_agent")

    assert "{{ hypothesis_output_format }}" in hypothesis.system_prompt
    assert "hypothesis_output_format" in hypothesis.metadata
    assert feedback.metadata["max_json_parse_retries"] == 3
    assert "feedback_output_format" in feedback.metadata
