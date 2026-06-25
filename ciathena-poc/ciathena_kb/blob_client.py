"""
ciathena_kb.blob_client
-----------------------
Azure Blob Storage client for artifact YAML files and prompt templates.

Opt-in via env vars: when AZURE_BLOB_CONNECTION_STRING and
AZURE_BLOB_CONTAINER_NAME are set, artifacts are stored in / loaded from
Azure Blob Storage. When absent, callers fall back to local disk.

Follows the same pattern as embedder.py / llm.py: get_blob_client() returns
None when env vars are missing, and callers branch on that.

Artifact versioning: every upload stores the file under artifacts/<name> (the
"current" copy used for ingestion) and also saves a timestamped snapshot under
versions/<name>/<YYYYMMDD_HHMMSS>_<name> for audit trail.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _blob_env(key: str) -> str | None:
    return os.environ.get(f"AZURE_BLOB_{key}")


PROMPTS_PREFIX = "prompts/"
ARTIFACTS_PREFIX = "artifacts/"
VERSIONS_PREFIX = "versions/"


class ArtifactBlobClient:
    """Thin wrapper around azure.storage.blob for artifact YAML files."""

    def __init__(self, connection_string: str, container_name: str):
        from azure.storage.blob import BlobServiceClient

        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container_name = container_name
        self._container = self._service.get_container_client(container_name)

    @property
    def container_name(self) -> str:
        return self._container_name

    def ensure_container(self) -> None:
        """Create the container if it doesn't exist."""
        try:
            self._container.get_container_properties()
        except Exception:
            self._container.create_container()

    # ---- Artifact CRUD (versioned) ----

    def upload(self, blob_name: str, data: bytes) -> str:
        """Upload artifact bytes under artifacts/ and save a timestamped version."""
        current_path = f"{ARTIFACTS_PREFIX}{blob_name}"
        self._container.upload_blob(current_path, data, overwrite=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_path = f"{VERSIONS_PREFIX}{blob_name}/{ts}_{blob_name}"
        self._container.upload_blob(version_path, data, overwrite=True)

        return f"blob://{self._container_name}/{current_path}"

    def download(self, blob_name: str) -> bytes:
        """Download an artifact's current content."""
        path = blob_name if blob_name.startswith(ARTIFACTS_PREFIX) else f"{ARTIFACTS_PREFIX}{blob_name}"
        blob = self._container.get_blob_client(path)
        return blob.download_blob().readall()

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifact blob names (without the artifacts/ prefix)."""
        search = f"{ARTIFACTS_PREFIX}{prefix}"
        names: list[str] = []
        for blob in self._container.list_blobs(name_starts_with=search):
            name = blob.name
            if name.endswith(".yml") or name.endswith(".yaml"):
                short = name[len(ARTIFACTS_PREFIX):]
                names.append(short)
        return sorted(names)

    def list_versions(self, blob_name: str) -> list[dict[str, str]]:
        """List timestamped versions of an artifact, newest first."""
        search = f"{VERSIONS_PREFIX}{blob_name}/"
        versions: list[dict[str, str]] = []
        for blob in self._container.list_blobs(name_starts_with=search):
            parts = blob.name.rsplit("/", 1)[-1]
            ts_part = parts.split("_", 2)
            timestamp = f"{ts_part[0]}_{ts_part[1]}" if len(ts_part) >= 2 else ""
            versions.append({"path": blob.name, "timestamp": timestamp})
        return sorted(versions, key=lambda v: v["timestamp"], reverse=True)

    def exists(self, blob_name: str) -> bool:
        path = f"{ARTIFACTS_PREFIX}{blob_name}"
        blob = self._container.get_blob_client(path)
        return blob.exists()

    def delete(self, blob_name: str) -> None:
        path = f"{ARTIFACTS_PREFIX}{blob_name}"
        blob = self._container.get_blob_client(path)
        blob.delete_blob()

    # ---- Prompt CRUD ----

    def upload_prompt(self, prompt_name: str, text: str) -> str:
        """Save a prompt template to blob. Returns the blob path."""
        path = f"{PROMPTS_PREFIX}{prompt_name}.txt"
        self._container.upload_blob(path, text.encode("utf-8"), overwrite=True)
        return path

    def download_prompt(self, prompt_name: str) -> str | None:
        """Download a prompt template. Returns None if not found."""
        path = f"{PROMPTS_PREFIX}{prompt_name}.txt"
        try:
            blob = self._container.get_blob_client(path)
            return blob.download_blob().readall().decode("utf-8")
        except Exception:
            return None

    def list_prompts(self) -> list[str]:
        """List prompt names (without prefix/extension)."""
        names: list[str] = []
        for blob in self._container.list_blobs(name_starts_with=PROMPTS_PREFIX):
            name = blob.name[len(PROMPTS_PREFIX):]
            if name.endswith(".txt"):
                names.append(name[:-4])
        return sorted(names)


def get_blob_client() -> ArtifactBlobClient | None:
    """Return a blob client if env vars are configured, else None."""
    conn_str = _blob_env("CONNECTION_STRING")
    container = _blob_env("CONTAINER_NAME") or "ciathena-artifacts"

    if not conn_str:
        print("  Blob storage: not configured (no AZURE_BLOB_CONNECTION_STRING)")
        return None

    try:
        client = ArtifactBlobClient(conn_str, container)
        client.ensure_container()
        print(f"  Blob storage: connected → container '{container}'")
        return client
    except Exception as e:
        print(f"  Blob storage: connection failed ({e}), falling back to local")
        return None
