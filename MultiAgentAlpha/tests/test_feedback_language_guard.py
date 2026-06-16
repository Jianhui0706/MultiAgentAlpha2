import pytest

from trading_agents.agents.feedback.feedback_summarizer_agent import FeedbackSummarizerAgent


def test_feedback_language_guard_rejects_korean_and_japanese_scripts() -> None:
    agent = FeedbackSummarizerAgent("feedback_summarizer_agent")

    with pytest.raises(ValueError, match="disallowed non-Chinese language script"):
        agent._validate_no_disallowed_language_scripts({"New Hypothesis": "거래량 괴리를 계속 조정한다."})

    with pytest.raises(ValueError, match="disallowed non-Chinese language script"):
        agent._validate_no_disallowed_language_scripts({"Reasoning": "短期反転を確認する。"})


def test_feedback_language_guard_allows_chinese_and_machine_identifiers() -> None:
    agent = FeedbackSummarizerAgent("feedback_summarizer_agent")

    agent._validate_no_disallowed_language_scripts(
        {
            "Observations": "当前因子方向较弱，但量价背离仍可继续调参。",
            "Semantic Diagnosis": "parameter_not_tuned",
            "Replace Best Result": "no",
        }
    )
