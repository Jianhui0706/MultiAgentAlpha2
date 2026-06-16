from __future__ import annotations

from typing import Any

from trading_agents.infra.llm.models import ChatRequest, ChatResponse
from trading_agents.infra.llm.provider import ChatProvider


def _format_openai_exception(exc: Exception) -> str:
    parts = [f"{exc.__class__.__name__}: {exc}"]
    for attr in ("status_code", "request_id", "code", "type"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            parts.append(f"{attr}={value}")
    body = getattr(exc, "body", None)
    if body not in (None, ""):
        parts.append(f"body={str(body)[:1000]}")
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            parts.append(f"response_status_code={status_code}")
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            parts.append(f"response_text={text[:1000]}")
    return "; ".join(parts)


class OpenAICompatibleProvider(ChatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAICompatibleProvider") from exc
        # 显式设置单次请求超时，避免上游无响应时按 SDK 默认 600s 干等；
        # 关闭 SDK 内置重试（max_retries=0），重试统一交给外层 LLMGateway/run_with_retry，
        # 否则 SDK 内部重试会与外层重试相乘，导致一次调用可干等数十分钟。
        self._timeout = float(timeout)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self._timeout,
            max_retries=max_retries,
        )

    def create_chat_completion(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.model is not None:
            kwargs["model"] = request.model
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(_format_openai_exception(exc)) from exc
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        return ChatResponse(
            content=content,
            finish_reason=finish_reason,
            raw=response.model_dump(),
        )
