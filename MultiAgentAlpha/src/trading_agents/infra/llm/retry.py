from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略配置：次数与固定等待间隔。"""
    max_retries: int = 3
    wait_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_wait_seconds: float = 8.0


def is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "connection error",
        "timeout",
        "temporarily unavailable",
        "overloaded_error",
        "rate limit",
        "http_code': '529'",
        '"http_code": "529"',
        "http code: 529",
        "bad gateway",
        "service unavailable",
    )
    return any(marker in text for marker in transient_markers)


def run_with_retry(
    action: Callable[[], T],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, Exception, float, bool], None] | None = None,
) -> T:
    """按策略执行重试；超过次数后抛出最后一次异常。"""
    last_error: Exception | None = None
    for retry_i in range(policy.max_retries):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            transient = is_transient_error(exc)
            should_retry = retry_i < policy.max_retries - 1 and transient
            if should_retry:
                delay = min(
                    policy.wait_seconds * (policy.backoff_multiplier ** retry_i),
                    policy.max_wait_seconds,
                )
                if on_retry is not None:
                    on_retry(retry_i + 1, exc, delay, transient)
                time.sleep(delay + random.uniform(0.0, min(0.25, delay * 0.1)))
                continue
            if retry_i < policy.max_retries - 1 and not is_transient_error(exc):
                break
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry failed without explicit exception")
