"""Lightweight retry/backoff for flaky network calls (Yahoo Finance rate limits, etc.)."""

import logging
import random
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Substrings that mark a transient, worth-retrying failure (rate limit / auth crumb / network).
_RETRYABLE_MARKERS = (
    "401", "403", "429",
    "unauthorized", "too many requests", "rate limit", "rate-limit",
    "timed out", "timeout", "temporarily", "connection", "remote end",
    "curl", "ssl",
)


def is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def with_retry(retries: int = 3, base_delay: float = 0.8, max_delay: float = 8.0):
    """Decorator: retry the wrapped call on transient errors with exponential backoff + jitter.

    Non-retryable errors (and the final attempt) propagate unchanged.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - we re-raise unless retryable
                    if attempt >= retries or not is_retryable(exc):
                        raise
                    sleep = min(max_delay, delay) + random.uniform(0, 0.4)
                    logger.warning(
                        "Retryable error in %s (%s); retry %d/%d in %.1fs",
                        getattr(fn, "__name__", "call"), exc, attempt + 1, retries, sleep,
                    )
                    time.sleep(sleep)
                    delay *= 2
        return wrapper
    return decorator
