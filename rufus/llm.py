"""
Ollama LLM client with production reliability patterns.

Circuit breaker  — after N consecutive failures the breaker opens and all
                   calls immediately return a fallback instead of waiting.
                   After a recovery timeout it enters half-open state and
                   probes with one request.  (Closed → Open → Half-Open)

Retry + backoff  — transient Ollama errors are retried up to MAX_RETRIES
                   times with exponential backoff + jitter to avoid the
                   thundering herd problem on recovery.

Timeout          — each chat call is bounded by TIMEOUT_SECS so a slow
                   Ollama instance never hangs the whole query pipeline.
                   "Slow services are worse than dead services."

Graceful fallback — when the circuit is open the caller receives None for
                    non-streaming calls so the app can show products without
                    an LLM answer rather than an error page.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Iterator

import ollama

DEFAULT_MODEL   = "qwen3.5:latest"  # 6.6 GB; better instruction-following than 1.7b
TIMEOUT_SECS    = 60       # max time to wait for a single chat call
MAX_RETRIES     = 2        # attempts after first failure (total = 3)
BASE_BACKOFF    = 0.5      # seconds, doubled each retry
FAILURE_THRESH  = 3        # consecutive failures before breaker opens
RECOVERY_SECS   = 60       # seconds before half-open probe


class _CircuitBreaker:
    """Minimal circuit breaker: Closed → Open → Half-Open."""

    def __init__(self, failure_threshold: int, recovery_timeout: float) -> None:
        self._threshold = failure_threshold
        self._recovery  = recovery_timeout
        self._failures  = 0
        self._state     = "closed"   # closed | open | half_open
        self._opened_at: float = 0.0
        self._lock      = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._recovery:
                    self._state = "half_open"
                    return True
                return False
            # half_open: let one request through
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state    = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= self._threshold:
                self._state    = "open"
                self._opened_at = time.monotonic()
                print(f"[circuit_breaker] Ollama breaker OPEN "
                      f"(failures={self._failures}, recovery={self._recovery}s)")

    @property
    def state(self) -> str:
        return self._state


_breaker = _CircuitBreaker(FAILURE_THRESH, RECOVERY_SECS)


def _chat_with_retry(model: str, messages: list[dict], stream: bool) -> any:
    """Single attempt wrapped in retry loop with exponential backoff + jitter."""
    # Do NOT set think:False for qwen3 — it routes output to `thinking` field
    # and leaves `content` empty. Without it, thinking goes to `thinking` field
    # and the final answer goes to `content` (correct behaviour).
    opts: dict = {"timeout": TIMEOUT_SECS, "num_ctx": 2048, "num_predict": 256, "temperature": 0}
    # think=False as a direct kwarg (NOT inside options) suppresses CoT for qwen3.
    # Passing it inside options routes output to the thinking field instead.
    kwargs = {"options": opts, "keep_alive": "60m", "think": False}

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            sleep = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            print(f"[llm] retry {attempt}/{MAX_RETRIES} in {sleep:.2f}s")
            time.sleep(sleep)
        try:
            if stream:
                return ollama.chat(model=model, messages=messages, stream=True, **kwargs)
            resp = ollama.chat(model=model, messages=messages, **kwargs)
            return resp.message.content or ""
        except Exception as exc:
            last_exc = exc
            print(f"[llm] attempt {attempt+1} failed: {exc}")
    raise last_exc


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> str | Iterator[str] | None:
        """
        Returns:
          str           — full response text (stream=False, circuit closed)
          Iterator[str] — token stream        (stream=True,  circuit closed)
          None          — circuit is open, caller should degrade gracefully
        """
        if not _breaker.allow():
            print(f"[llm] circuit OPEN — returning None for graceful degradation")
            return None

        try:
            result = _chat_with_retry(self.model, messages, stream)
            _breaker.record_success()
            if stream:
                return self._wrap_stream(result)
            return result
        except Exception as exc:
            _breaker.record_failure()
            print(f"[llm] all retries exhausted: {exc}")
            return None

    def _wrap_stream(self, raw_stream) -> Iterator[str]:
        """Yield content tokens only — thinking tokens go to chunk.message.thinking
        and are intentionally skipped so CoT never leaks into the UI."""
        try:
            for chunk in raw_stream:
                token = chunk.message.content
                if token:
                    yield token
            _breaker.record_success()
        except Exception as exc:
            _breaker.record_failure()
            print(f"[llm] stream error: {exc}")

    @property
    def circuit_state(self) -> str:
        return _breaker.state
