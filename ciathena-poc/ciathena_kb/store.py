"""
ciathena_kb.store
-----------------
Chroma-backed vector store for the PoC. Wraps ingest and retrieval.

Retrieval applies metadata filters (usecase, layer, component_type, personas,
review_status) BEFORE the vector similarity search -- this is the Scenario-B
behaviour where our own node, not a managed service, controls filtering.

Supports:
  - General-layer OR-merge (retrieve usecase chunks AND General chunks together)
  - $in multi-value filters for component_type
  - Persistent Chroma by default (.chroma dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import chromadb

from .chunker import Chunk
from .embedder import Embedder

DEFAULT_PERSIST_DIR = "./.chroma"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class KnowledgeStore:
    def __init__(self, embedder: Embedder, collection_name: str = "ciathena_kb",
                 persist_dir: str | None = None):
        self.embedder = embedder
        name = "".join(ch for ch in collection_name if ch.isalnum() or ch in "._-")
        if len(name) < 3:
            name = f"col_{name}"
        resolved_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)
        try:
            self._client = chromadb.PersistentClient(path=resolved_dir)
        except Exception:
            self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})

    def ingest(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            vectors = self.embedder.embed(texts)
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors,
                documents=texts,
                metadatas=[c.metadata for c in batch],
            )
            total += len(batch)
        return total

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        ids = self._col.get()["ids"]
        if ids:
            self._col.delete(ids=ids)

    def _build_where(self, filters: dict[str, Any] | None,
                     include_general: bool = False) -> dict | None:
        """Build a Chroma where clause from filters.

        Supports:
          - Simple equality: {"usecase": "MMM"}
          - Multi-value ($in): {"component_type": ["playbook", "methodology"]}
          - General OR-merge: when include_general=True AND usecase is set,
            retrieves both the specified usecase AND "General" layer chunks.
        """
        clauses: list[dict] = []
        merged: dict[str, Any] = {"review_status": "approved"}
        if filters:
            merged.update(filters)

        for key, val in merged.items():
            if val is None:
                continue
            if key == "usecase" and include_general and val != "General":
                clauses.append({"$or": [
                    {"usecase": {"$eq": val}},
                    {"usecase": {"$eq": "General"}},
                ]})
            elif isinstance(val, list):
                if len(val) == 1:
                    clauses.append({key: {"$eq": val[0]}})
                elif len(val) > 1:
                    clauses.append({key: {"$in": val}})
            else:
                clauses.append({key: {"$eq": val}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def retrieve(self, query: str, top_k: int = 5,
                 filters: dict[str, Any] | None = None,
                 include_general: bool = True) -> list[RetrievedChunk]:
        where = self._build_where(filters, include_general=include_general)
        q_vec = self.embedder.embed([query])[0]
        res = self._col.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(RetrievedChunk(
                chunk_id=cid, text=doc, metadata=meta or {},
                score=round(1.0 - float(dist), 4)))
        return out
