from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from trading_agents.core.model_client import get_current_agent_name, get_current_loop_index
from trading_agents.infra.llm.cache import InMemoryResponseCache
from trading_agents.infra.llm.models import ChatMessage, ChatRequest, ChatResponse
from trading_agents.infra.llm.parser import parse_json_object
from trading_agents.infra.llm.provider import ChatProvider
from trading_agents.infra.llm.retry import RetryPolicy, run_with_retry


@dataclass
class LLMGateway:
    """统一封装 LLM 调用链：缓存、重试、续写与 JSON 输出支持。"""
    provider: ChatProvider
    retry_policy: RetryPolicy = RetryPolicy()
    cache: InMemoryResponseCache | None = None
    default_model: str | None = None
    max_continue_rounds: int = 3
    raw_io_logger: Callable[[dict[str, Any]], None] | None = None

    def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> ChatResponse:
        """发起一次对话补全请求并返回标准化 ChatResponse。"""
        if json_mode:
            system_prompt, user_prompt = self._ensure_json_instruction(
                system_prompt, user_prompt)
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        request = ChatRequest(
            messages=messages,
            model=model or self.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return self._create_with_cache_retry_and_continue(request, retry_policy=retry_policy)

    def _ensure_json_instruction(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        """为要求 JSON 模式的兼容接口补充显式 JSON 指令。"""
        combined = f"{system_prompt}\n{user_prompt}".lower()
        if "json" in combined:
            return system_prompt, user_prompt
        json_instruction = (
            "Return a valid JSON object only. "
            "The response must be strict JSON with double-quoted keys and no extra markdown."
        )
        updated_system_prompt = f"{system_prompt}\n\n{json_instruction}".strip(
        )
        return updated_system_prompt, user_prompt

    def create_chat_completion_as_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> dict[str, Any]:
        """以 JSON 模式请求并解析为对象，解析失败会抛出异常。"""
        response = self.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_policy=retry_policy,
        )
        return parse_json_object(response.content)

    def _create_with_cache_retry_and_continue(
        self,
        request: ChatRequest,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> ChatResponse:
        """内部执行链路：先查缓存，再重试调用，必要时自动续写。"""
        request_id = self._request_id(request)
        if self.cache is not None:
            cached_response = self.cache.get(request)
            if cached_response is not None:
                self._emit_raw_io(
                    {
                        "request_id": request_id,
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "cache_hit",
                        "request_summary": self._request_summary(request),
                        "response": self._response_to_raw(cached_response),
                    }
                )
                return cached_response

        attempt = 0

        def _call_provider() -> ChatResponse:
            nonlocal attempt
            attempt += 1
            self._emit_raw_io(
                {
                    "request_id": request_id,
                    "loop_index": get_current_loop_index(),
                    "agent_name": get_current_agent_name(),
                    "phase": "request",
                    "attempt": attempt,
                    "request": self._request_to_raw(request),
                }
            )
            try:
                response = self.provider.create_chat_completion(request)
            except Exception as exc:
                self._emit_raw_io(
                    {
                        "request_id": request_id,
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "error",
                        "attempt": attempt,
                        "request_summary": self._request_summary(request),
                        "error": str(exc),
                        "error_details": self._exception_to_raw(exc),
                    }
                )
                raise
            self._emit_raw_io(
                {
                    "request_id": request_id,
                    "loop_index": get_current_loop_index(),
                    "agent_name": get_current_agent_name(),
                    "phase": "response",
                    "attempt": attempt,
                    "request_summary": self._request_summary(request),
                    "response": self._response_to_raw(response),
                }
            )
            return response

        response = run_with_retry(
            action=_call_provider,
            policy=retry_policy or self.retry_policy,
            on_retry=lambda retry_no, exc, delay, transient: self._emit_raw_io(
                {
                    "request_id": request_id,
                    "loop_index": get_current_loop_index(),
                    "agent_name": get_current_agent_name(),
                    "phase": "retry_scheduled",
                    "retry_no": retry_no,
                    "delay_seconds": delay,
                    "transient": transient,
                    "error": str(exc),
                    "error_details": self._exception_to_raw(exc),
                    "request_summary": self._request_summary(request),
                }
            ),
        )
        merged_response = self._auto_continue_if_needed(request, response)

        if self.cache is not None:
            self.cache.set(request, merged_response)
        return merged_response

    def _auto_continue_if_needed(self, request: ChatRequest, response: ChatResponse) -> ChatResponse:
        """当 finish_reason=length 时自动发起 continue，最多 max_continue_rounds 次。"""
        if response.finish_reason != "length":
            return response

        content_parts = [response.content]
        current_request = request
        current_response = response
        rounds = 0
        while current_response.finish_reason == "length" and rounds < self.max_continue_rounds:
            next_messages = list(current_request.messages) + [
                ChatMessage(role="assistant",
                            content=current_response.content),
                ChatMessage(role="user", content="continue"),
            ]
            next_request = ChatRequest(
                messages=next_messages,
                model=current_request.model,
                max_tokens=current_request.max_tokens,
                temperature=current_request.temperature,
                json_mode=current_request.json_mode,
                metadata=current_request.metadata,
            )
            next_request_id = self._request_id(next_request)
            attempt = 0

            def _call_provider_continue() -> ChatResponse:
                nonlocal attempt
                attempt += 1
                self._emit_raw_io(
                    {
                        "request_id": next_request_id,
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "request_continue",
                        "continue_round": rounds + 1,
                        "attempt": attempt,
                        "request": self._request_to_raw(next_request),
                    }
                )
                try:
                    response = self.provider.create_chat_completion(
                        next_request)
                except Exception as exc:
                    self._emit_raw_io(
                        {
                            "request_id": next_request_id,
                            "loop_index": get_current_loop_index(),
                            "agent_name": get_current_agent_name(),
                            "phase": "error_continue",
                            "continue_round": rounds + 1,
                            "attempt": attempt,
                            "request_summary": self._request_summary(next_request),
                            "error": str(exc),
                            "error_details": self._exception_to_raw(exc),
                        }
                    )
                    raise
                self._emit_raw_io(
                    {
                        "request_id": next_request_id,
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "response_continue",
                        "continue_round": rounds + 1,
                        "attempt": attempt,
                        "request_summary": self._request_summary(next_request),
                        "response": self._response_to_raw(response),
                    }
                )
                return response

            current_response = run_with_retry(
                action=_call_provider_continue,
                policy=self.retry_policy,
                on_retry=lambda retry_no, exc, delay, transient: self._emit_raw_io(
                    {
                        "request_id": next_request_id,
                        "loop_index": get_current_loop_index(),
                        "agent_name": get_current_agent_name(),
                        "phase": "retry_scheduled_continue",
                        "continue_round": rounds + 1,
                        "retry_no": retry_no,
                        "delay_seconds": delay,
                        "transient": transient,
                        "error": str(exc),
                        "error_details": self._exception_to_raw(exc),
                        "request_summary": self._request_summary(next_request),
                    }
                ),
            )
            content_parts.append(current_response.content)
            current_request = next_request
            rounds += 1

        return ChatResponse(
            content="".join(content_parts),
            finish_reason=current_response.finish_reason,
            raw=current_response.raw,
        )

    def _emit_raw_io(self, payload: dict[str, Any]) -> None:
        if self.raw_io_logger is None:
            return
        try:
            self.raw_io_logger(payload)
        except Exception:
            return

    def _request_to_raw(self, request: ChatRequest) -> dict[str, Any]:
        return {
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            **self._request_summary(request),
        }

    def _request_summary(self, request: ChatRequest) -> dict[str, Any]:
        message_lengths = [
            {
                "role": message.role,
                "chars": len(message.content or ""),
            }
            for message in request.messages
        ]
        system_chars = sum(item["chars"] for item in message_lengths if item["role"] == "system")
        user_chars = sum(item["chars"] for item in message_lengths if item["role"] == "user")
        total_chars = sum(item["chars"] for item in message_lengths)
        return {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "json_mode": request.json_mode,
            "metadata": dict(request.metadata),
            "message_lengths": message_lengths,
            "input_length_summary": {
                "system_chars": system_chars,
                "user_chars": user_chars,
                "total_chars": total_chars,
                "message_count": len(request.messages),
            },
        }

    def _request_id(self, request: ChatRequest) -> str:
        payload = {
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "json_mode": request.json_mode,
            "metadata": dict(request.metadata),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _response_to_raw(self, response: ChatResponse) -> dict[str, Any]:
        return {
            "content": response.content,
            "finish_reason": response.finish_reason,
            "raw": response.raw,
        }

    def _exception_to_raw(self, exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        for attr in ("status_code", "request_id", "code", "type"):
            value = getattr(exc, attr, None)
            if value not in (None, ""):
                details[attr] = value
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                details["response_status_code"] = status_code
            text = getattr(response, "text", None)
            if isinstance(text, str) and text:
                details["response_text_preview"] = text[:1000]
        body = getattr(exc, "body", None)
        if body not in (None, ""):
            details["body_preview"] = str(body)[:1000]
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            details["cause_type"] = cause.__class__.__name__
            details["cause_message"] = str(cause)
        return details
