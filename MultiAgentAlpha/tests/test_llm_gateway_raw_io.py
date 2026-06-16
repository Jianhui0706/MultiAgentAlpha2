from __future__ import annotations

from trading_agents.infra.llm.gateway import LLMGateway
from trading_agents.infra.llm.models import ChatRequest, ChatResponse
from trading_agents.infra.llm.provider import ChatProvider


class DummyProvider(ChatProvider):
    def create_chat_completion(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="{}", finish_reason="stop", raw={"ok": True})


def test_llm_gateway_response_event_uses_request_id_without_full_request() -> None:
    events = []
    gateway = LLMGateway(provider=DummyProvider(), raw_io_logger=events.append)

    gateway.create_chat_completion(system_prompt="system prompt", user_prompt="user prompt")

    request_event = next(item for item in events if item["phase"] == "request")
    response_event = next(item for item in events if item["phase"] == "response")

    assert request_event["request_id"] == response_event["request_id"]
    assert "request" in request_event
    assert "request" not in response_event
    assert response_event["request_summary"]["input_length_summary"]["total_chars"] > 0
