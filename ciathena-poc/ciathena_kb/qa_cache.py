"""
ciathena_kb.qa_cache
--------------------
Session-scoped Q&A result cache. Keyed by normalized query string.
Uses generation-based invalidation: bumping the generation counter makes
all existing entries stale without walking the cache. Stale entries are
lazily evicted on get().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    query: str
    route: dict[str, Any]
    graded_chunks: list[dict[str, Any]]
    answer: str
    citations: list[str]
    generation: int


class QACache:

    def __init__(self, max_entries: int = 100):
        self._store: dict[str, CacheEntry] = {}
        self._generation: int = 0
        self._max_entries: int = max_entries
        self._hit_count: int = 0
        self._miss_count: int = 0

    @staticmethod
    def _normalize_key(query: str) -> str:
        return " ".join(query.lower().split())

    def get(self, query: str) -> CacheEntry | None:
        key = self._normalize_key(query)
        entry = self._store.get(key)
        if entry is not None and entry.generation == self._generation:
            self._hit_count += 1
            return entry
        if entry is not None:
            del self._store[key]
        self._miss_count += 1
        return None

    def put(self, query: str, route: dict[str, Any],
            graded_chunks: list[dict[str, Any]],
            answer: str, citations: list[str]) -> None:
        if len(self._store) >= self._max_entries:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        key = self._normalize_key(query)
        self._store[key] = CacheEntry(
            query=query, route=route, graded_chunks=graded_chunks,
            answer=answer, citations=citations, generation=self._generation,
        )

    def invalidate(self) -> None:
        self._generation += 1

    @property
    def stats(self) -> dict[str, int]:
        valid = sum(1 for e in self._store.values()
                    if e.generation == self._generation)
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "entries": valid,
            "generation": self._generation,
        }

    def clear(self) -> None:
        self._store.clear()
        self._hit_count = 0
        self._miss_count = 0
