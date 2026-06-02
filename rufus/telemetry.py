"""
Structured observability for Rufus — RED method.

RED = Rate · Errors · Duration  (per component, per intent type)

Every query emits one JSON log line with all timing breakdowns and a
correlation ID so you can trace a single request across log lines.

Usage:
  from rufus.telemetry import span, log_query

  with span("classify") as s:
      result = classify(...)
  # s.elapsed_ms is set automatically

  log_query(session_id, intent, ms_dict, error=None)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

_log = logging.getLogger("rufus")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)


@dataclass
class Span:
    name: str
    start: float = field(default_factory=time.perf_counter)
    elapsed_ms: int = 0
    error: str | None = None

    def finish(self, error: str | None = None) -> "Span":
        self.elapsed_ms = int((time.perf_counter() - self.start) * 1000)
        self.error = error
        return self


@contextmanager
def span(name: str):
    s = Span(name=name)
    try:
        yield s
    except Exception as exc:
        s.finish(error=str(exc))
        raise
    finally:
        if not s.elapsed_ms:
            s.finish()


def log_query(
    session_id: str,
    intent: str,
    timings: dict[str, int],
    *,
    n_products: int = 0,
    cache_hit: bool = False,
    error: str | None = None,
    model: str = "",
) -> str:
    """Emit one structured JSON log line per query.  Returns a correlation ID."""
    correlation_id = uuid.uuid4().hex[:12]
    record = {
        "event":          "query",
        "correlation_id": correlation_id,
        "session_id":     session_id[:12],
        "intent":         intent,
        "model":          model,
        "n_products":     n_products,
        "cache_hit":      cache_hit,
        "error":          error,
        "timings_ms":     timings,
        "total_ms":       sum(timings.values()),
        "ts":             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _log.info(json.dumps(record))
    return correlation_id


# In-process RED counter (resets on restart; for production use Prometheus/StatsD)
class _Metrics:
    def __init__(self) -> None:
        self.total_queries   = 0
        self.error_queries   = 0
        self.total_ms        = 0
        self._durations: list[int] = []

    def record(self, total_ms: int, error: bool) -> None:
        self.total_queries += 1
        self.total_ms += total_ms
        if error:
            self.error_queries += 1
        self._durations.append(total_ms)
        if len(self._durations) > 1000:
            self._durations.pop(0)

    @property
    def error_rate(self) -> float:
        return self.error_queries / self.total_queries if self.total_queries else 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total_queries if self.total_queries else 0.0

    @property
    def p99_ms(self) -> int:
        if not self._durations:
            return 0
        s = sorted(self._durations)
        return s[int(len(s) * 0.99)]

    def summary(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "error_rate":    f"{self.error_rate:.1%}",
            "avg_ms":        int(self.avg_ms),
            "p99_ms":        self.p99_ms,
        }


metrics = _Metrics()
