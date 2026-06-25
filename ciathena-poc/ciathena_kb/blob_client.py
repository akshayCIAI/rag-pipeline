"""
ciathena_kb.blob_client
-----------------------
Azure Blob Storage client for artifact YAML files.

Opt-in via env vars: when AZURE_BLOB_CONNECTION_STRING and
AZURE_BLOB_CONTAINER_NAME are set, artifacts are stored in / loaded from
Azure Blob Storage. When absent, callers fall back to local disk.

Follows the same pattern as embedder.py / llm.py: get_blob_client() returns
None when env vars are missing, and callers branch on that.
"""

from __future__ import annotations

import os
from typing import Any


def _blob_env(key: str) -> str | None:
    return os.environ.get(f"AZURE_BLOB_{key}")


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

    def upload(self, blob_name: str, data: bytes) -> str:
        """Upload bytes and return the blob URI."""
        self._container.upload_blob(blob_name, data, overwrite=True)
        return f"blob://{self._container_name}/{blob_name}"

    def download(self, blob_name: str) -> bytes:
        """Download a blob's content as bytes."""
        blob = self._container.get_blob_client(blob_name)
        return blob.download_blob().readall()

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List blob names matching *.yml / *.yaml."""
        names: list[str] = []
        for blob in self._container.list_blobs(name_starts_with=prefix or None):
            name = blob.name
            if name.endswith(".yml") or name.endswith(".yaml"):
                names.append(name)
        return sorted(names)

    def exists(self, blob_name: str) -> bool:
        blob = self._container.get_blob_client(blob_name)
        return blob.exists()

    def delete(self, blob_name: str) -> None:
        blob = self._container.get_blob_client(blob_name)
        blob.delete_blob()


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
