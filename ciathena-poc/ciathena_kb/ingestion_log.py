"""
ciathena_kb.ingestion_log
-------------------------
Tracks which artifacts have been ingested, at which version, and when.
Stored as a JSON manifest alongside the Chroma DB so the two stay in sync.

Used to:
  - Skip re-embedding unchanged artifacts (saves cost)
  - Detect version bumps
  - Show the team what's loaded in the Streamlit sidebar
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

from .loader import Artifact

DEFAULT_LOG_PATH = ".chroma/ingestion_log.json"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _is_blob_path(path: str) -> bool:
    return path.startswith("blob://")


def _artifact_entry(
    artifact: Artifact, chunk_count: int, embedding_model: str,
    file_hash: str | None = None,
) -> dict[str, Any]:
    if file_hash is None:
        if artifact.source_path and not _is_blob_path(artifact.source_path):
            file_hash = _file_hash(artifact.source_path)
        else:
            file_hash = ""
    return {
        "artifact_id": artifact.artifact_id,
        "title": artifact.envelope.get("title", ""),
        "content_version": str(artifact.envelope.get("content_version", "")),
        "schema_version": str(artifact.envelope.get("schema_version", "")),
        "usecase": artifact.envelope.get("usecase", ""),
        "component_type": artifact.component_type,
        "review_status": artifact.envelope.get("review_status", ""),
        "layer": artifact.envelope.get("layer", ""),
        "file_hash": file_hash,
        "source_path": artifact.source_path,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


class IngestionLog:
    def __init__(self, log_path: str | None = None):
        self._path = pathlib.Path(log_path or os.environ.get("CHROMA_PERSIST_DIR", ".chroma"))
        if self._path.is_dir() or not str(self._path).endswith(".json"):
            self._path = self._path / "ingestion_log.json"
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = {e["artifact_id"]: e for e in data.get("artifacts", [])}
            except (json.JSONDecodeError, KeyError):
                self._entries = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "artifact_count": len(self._entries),
            "artifacts": list(self._entries.values()),
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def needs_reingest(
        self, artifact: Artifact, current_hash: str | None = None,
    ) -> tuple[bool, str]:
        """Check if an artifact needs re-ingestion.
        Pass current_hash for blob artifacts (where we already have the bytes hash).
        Returns (needs_reingest, reason)."""
        aid = artifact.artifact_id
        if aid not in self._entries:
            return True, "new artifact"

        prev = self._entries[aid]
        curr_version = str(artifact.envelope.get("content_version", ""))
        prev_version = prev.get("content_version", "")
        if curr_version != prev_version:
            return True, f"version changed: {prev_version} -> {curr_version}"

        if current_hash is not None:
            curr_hash = current_hash
        elif artifact.source_path and not _is_blob_path(artifact.source_path):
            curr_hash = _file_hash(artifact.source_path)
        else:
            curr_hash = ""

        prev_hash = prev.get("file_hash", "")
        if curr_hash and prev_hash and curr_hash != prev_hash:
            return True, "file content changed (hash mismatch)"

        return False, "unchanged"

    def record(
        self, artifact: Artifact, chunk_count: int, embedding_model: str,
        file_hash: str | None = None,
    ) -> None:
        """Record a successful ingestion. Pass file_hash for blob artifacts."""
        self._entries[artifact.artifact_id] = _artifact_entry(
            artifact, chunk_count, embedding_model, file_hash=file_hash,
        )
        self._save()

    def remove(self, artifact_id: str) -> None:
        """Remove an artifact from the log (e.g. when deleted)."""
        self._entries.pop(artifact_id, None)
        self._save()

    def clear(self) -> None:
        """Clear the entire log."""
        self._entries = {}
        self._save()

    def get_all(self) -> list[dict[str, Any]]:
        """Return all log entries, sorted by artifact_id."""
        return sorted(self._entries.values(), key=lambda e: e.get("artifact_id", ""))

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        return self._entries.get(artifact_id)

    @property
    def count(self) -> int:
        return len(self._entries)
