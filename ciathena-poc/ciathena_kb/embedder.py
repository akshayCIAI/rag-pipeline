"""
ciathena_kb.embedder
--------------------
Embedding provider for the PoC.

Primary: Azure OpenAI embeddings (text-embedding-3-large by default).
Fallback: a deterministic local hash embedding, used automatically when Azure
env vars are absent, so the PoC runs end-to-end with zero credentials. The
fallback is NOT semantic — it exists only so the pipeline is demonstrable
offline. Set the Azure env vars for real retrieval quality.

CRITICAL (Scenario B rule): the model used here to embed documents at build
time MUST be identical to the model used to embed live queries at runtime.
The model name is recorded so a mismatch is detectable.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _emb_env(key: str) -> str | None:
    """Read an embedding env var, trying the embedding-specific name first,
    then falling back to the shared AZURE_OPENAI_* name."""
    return os.environ.get(f"AZURE_OPENAI_EMBEDDING_{key}") or os.environ.get(f"AZURE_OPENAI_{key}")


class AzureOpenAIEmbedder:
    """Real embedder. Reads env vars with AZURE_OPENAI_EMBEDDING_ prefix first,
    falls back to shared AZURE_OPENAI_ prefix. Requires:
        AZURE_OPENAI_EMBEDDING_ENDPOINT  (or AZURE_OPENAI_ENDPOINT)
        AZURE_OPENAI_EMBEDDING_API_KEY   (or AZURE_OPENAI_API_KEY)
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    Optional:
        AZURE_OPENAI_EMBEDDING_API_VERSION (default 2024-02-01)
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        from openai import AzureOpenAI

        self.model_name = model_name
        self._deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        self._client = AzureOpenAI(
            azure_endpoint=_emb_env("ENDPOINT"),
            api_key=_emb_env("API_KEY"),
            api_version=_emb_env("API_VERSION") or "2024-02-01",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._deployment, input=texts)
        return [d.embedding for d in resp.data]


class FakeHashEmbedder:
    """Deterministic, dependency-free fallback. Maps text -> a fixed-dim vector
    via hashed token buckets, L2-normalized. Good enough to demonstrate the
    plumbing and metadata filtering; not semantically meaningful."""

    def __init__(self, dim: int = 256, model_name: str = "fake-hash-256"):
        self.dim = dim
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder(model_name: str = "text-embedding-3-large") -> Embedder:
    """Return Azure embedder if env is configured, else the fake fallback."""
    has_deployment = bool(os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
    has_endpoint = bool(_emb_env("ENDPOINT"))
    has_key = bool(_emb_env("API_KEY"))
    if has_deployment and has_endpoint and has_key:
        try:
            return AzureOpenAIEmbedder(model_name=model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[embedder] Azure init failed ({exc}); using fake embedder.")
    else:
        print("[embedder] Azure env not set; using deterministic fake embedder. "
              "Set AZURE_OPENAI_* for real embeddings.")
    return FakeHashEmbedder()
