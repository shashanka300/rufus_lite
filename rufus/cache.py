"""
Query result cache — Cache-Aside pattern.

Caches BGE-M3 embedding vectors and post-fusion search results so repeated
queries skip the full retrieval pipeline.  80/20 rule: shopping queries are
highly repetitive ("wireless headphones", "running shoes") so cache hit rates
are high in practice.

Two cache layers:
  1. Embedding cache: query text → 1024-dim float vector  (no expiry — vectors
     are deterministic for a given model)
  2. Result cache:    (query, top_k) → list[Product]       (TTL = 5 min, so
     re-ingestion or ranking changes propagate quickly)

Both caches use an LRU eviction policy with configurable max size.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Thread-safe LRU cache with optional TTL."""

    def __init__(self, maxsize: int = 512, ttl: float | None = None) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, *args) -> str:
        raw = str(args).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def get(self, *args) -> Any | None:
        key = self._key(*args)
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            value, ts = self._cache[key]
            if self._ttl is not None and (time.monotonic() - ts) > self._ttl:
                del self._cache[key]
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, *args, value: Any) -> None:
        key = self._key(*args[:-1]) if len(args) > 1 else self._key(args[0])
        # last arg is the value when called as set(k1, k2, ..., value=v)
        key = self._key(*args)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.monotonic())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def put(self, key_args: tuple, value: Any) -> None:
        """Alternative API: cache.put((k1, k2), value)."""
        k = self._key(*key_args)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
            self._cache[k] = (value, time.monotonic())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def fetch(self, key_args: tuple) -> Any | None:
        """Alternative API: cache.fetch((k1, k2))."""
        k = self._key(*key_args)
        with self._lock:
            if k not in self._cache:
                self.misses += 1
                return None
            value, ts = self._cache[k]
            if self._ttl is not None and (time.monotonic() - ts) > self._ttl:
                del self._cache[k]
                self.misses += 1
                return None
            self._cache.move_to_end(k)
            self.hits += 1
            return value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# Module-level singletons
embedding_cache = LRUCache(maxsize=2048, ttl=None)      # vectors never expire
result_cache    = LRUCache(maxsize=512,  ttl=5 * 60)    # results expire in 5 min
rerank_cache    = LRUCache(maxsize=512,  ttl=5 * 60)
