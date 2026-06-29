"""
ciathena_kb.feedback_store
--------------------------
Stores user feedback (thumbs up/down) for each assistant answer.
Same storage pattern as chat_history.py — blob-first, local fallback.

Storage layout:
  - Local: .chroma/feedback/{session_id}.json
  - Blob:  feedback/{session_id}.json
"""

from __future__ import annotations

import json
import pathlib
import os
from datetime import datetime, timezone
from typing import Any


FEEDBACK_DIR = ".chroma/feedback"
BLOB_PREFIX = "feedback/"
MAX_ENTRIES = 500
NEGATIVE_THRESHOLD = 3


class FeedbackStore:

    def __init__(self, session_id: str, blob_client: Any = None):
        self._session_id = session_id
        self._blob = blob_client
        self._local_dir = pathlib.Path(
            os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
        ) / "feedback"
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _blob_path(self) -> str:
        return f"{BLOB_PREFIX}{self._session_id}.json"

    def _local_path(self) -> pathlib.Path:
        return self._local_dir / f"{self._session_id}.json"

    def _load(self) -> None:
        data = None
        if self._blob:
            try:
                raw = self._blob._container.get_blob_client(
                    self._blob_path()
                ).download_blob().readall()
                data = json.loads(raw)
            except Exception:
                pass

        if data is None:
            path = self._local_path()
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

        if data and isinstance(data.get("entries"), list):
            self._entries = data["entries"]

    def _save(self) -> None:
        payload = json.dumps({
            "session_id": self._session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self._entries),
            "entries": self._entries,
        }, ensure_ascii=False, indent=2).encode("utf-8")

        if self._blob:
            try:
                self._blob._container.upload_blob(
                    self._blob_path(), payload, overwrite=True,
                )
                return
            except Exception:
                pass

        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._local_path().write_bytes(payload)

    def append(
        self,
        message_id: str,
        query: str,
        answer: str,
        rating: int,
        reason: str = "",
        artifacts_cited: list[str] | None = None,
    ) -> None:
        """Record a feedback entry. rating=1 for thumbs up, -1 for thumbs down."""
        self._entries.append({
            "message_id": message_id,
            "query": query,
            "answer": answer[:500],
            "rating": rating,
            "reason": reason,
            "artifacts_cited": artifacts_cited or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._entries) > MAX_ENTRIES:
            self._entries = self._entries[-MAX_ENTRIES:]
        self._save()

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def count_negative_for_query(self, query: str) -> int:
        """Count how many negative ratings a query has received."""
        normalized = query.strip().lower()
        return sum(
            1 for e in self._entries
            if e.get("rating", 0) < 0 and e.get("query", "").strip().lower() == normalized
        )

    def should_invalidate_cache(self, query: str) -> bool:
        """Return True when a query has hit the negative-feedback threshold."""
        return self.count_negative_for_query(query) >= NEGATIVE_THRESHOLD
