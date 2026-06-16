import pytest

from trading_agents.agents.hypothesis.hypothesis_agent import HypothesisAgentV2


def test_hypothesis_language_guard_rejects_korean_and_japanese_scripts() -> None:
    agent = HypothesisAgentV2("hypothesis_agent")

    with pytest.raises(ValueError, match="disallowed non-Chinese language script"):
        agent._validate_no_disallowed_language_scripts({"hypothesis": "거래량 괴리가 단기 반전을 예측한다."})

    with pytest.raises(ValueError, match="disallowed non-Chinese language script"):
        agent._validate_no_disallowed_language_scripts({"thesis": "モメンタム乖離は短期反転を示す。"})


def test_hypothesis_language_guard_allows_chinese_and_machine_identifiers() -> None:
    agent = HypothesisAgentV2("hypothesis_agent")

    agent._validate_no_disallowed_language_scripts(
        {
            "hypothesis": "量价背离越强，未来短期收益越低。",
            "expected_direction": "negative_forward_return",
            "proxy_reconstruction_plan": {
                "input_features": ["$close", "$volume_quote"],
                "expression_patterns": ["TS_PCTCHANGE($close,*)"],
            },
        }
    )
