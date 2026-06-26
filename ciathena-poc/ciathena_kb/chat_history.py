"""
ciathena_kb.chat_history
------------------------
Persistent chat history storage. Saves conversation messages to a JSON file
(local or blob-backed) keyed by session ID. Survives Streamlit page reloads.

Storage layout:
  - Local: .chroma/chat_history/{session_id}.json
  - Blob:  chat_history/{session_id}.json
"""

from __future__ import annotations

import json
import pathlib
import os
from datetime import datetime, timezone
from typing import Any


HISTORY_DIR = ".chroma/chat_history"
BLOB_PREFIX = "chat_history/"
MAX_MESSAGES = 200


class ChatHistoryStore:

    def __init__(self, session_id: str, blob_client: Any = None):
        self._session_id = session_id
        self._blob = blob_client
        self._local_dir = pathlib.Path(
            os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
        ) / "chat_history"
        self._messages: list[dict[str, Any]] = []
        self._load()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

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

        if data and isinstance(data.get("messages"), list):
            self._messages = data["messages"]

    def _save(self) -> None:
        payload = json.dumps({
            "session_id": self._session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "message_count": len(self._messages),
            "messages": self._messages,
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

    def append(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        if len(self._messages) > MAX_MESSAGES:
            self._messages = self._messages[-MAX_MESSAGES:]
        self._save()

    def clear(self) -> None:
        self._messages = []
        self._save()

    def list_sessions(self) -> list[dict[str, str]]:
        """List all saved sessions (id + last updated)."""
        sessions: list[dict[str, str]] = []

        if self._blob:
            try:
                for blob in self._blob._container.list_blobs(
                    name_starts_with=BLOB_PREFIX
                ):
                    name = blob.name[len(BLOB_PREFIX):]
                    if name.endswith(".json"):
                        sid = name[:-5]
                        sessions.append({"id": sid, "source": "blob"})
            except Exception:
                pass

        if self._local_dir.exists():
            for f in sorted(self._local_dir.glob("*.json")):
                sid = f.stem
                if not any(s["id"] == sid for s in sessions):
                    sessions.append({"id": sid, "source": "local"})

        return sessions

    def delete(self) -> None:
        """Delete this session's history file."""
        if self._blob:
            try:
                self._blob._container.get_blob_client(
                    self._blob_path()
                ).delete_blob()
            except Exception:
                pass

        path = self._local_path()
        if path.exists():
            path.unlink(missing_ok=True)
