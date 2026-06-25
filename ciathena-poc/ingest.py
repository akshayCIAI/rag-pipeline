"""
ingest.py -- Load, validate, chunk, embed, and ingest artifacts into persistent Chroma.

Smart re-ingestion: skips unchanged artifacts (compares content_version + file hash).

Usage:
    python ingest.py                              # auto: blob if configured, else local
    python ingest.py --clear                      # wipe everything and re-ingest from scratch
    python ingest.py --source local               # force local artifacts/ directory
    python ingest.py --source blob                # force blob storage
    python ingest.py --artifacts-dir ./my_dir     # custom local artifacts directory
"""

from __future__ import annotations

import argparse
import pathlib

from dotenv import load_dotenv
load_dotenv()

from ciathena_kb import (
    load_artifacts, load_artifact_from_bytes, chunk_artifact,
    get_embedder, KnowledgeStore, IngestionLog, get_blob_client,
)
from ciathena_kb.ingestion_log import _bytes_hash


def _load_from_blob(blob):
    """Download all YAML artifacts from blob storage. Returns (artifacts, hashes)."""
    blob_names = blob.list_artifacts()
    artifacts = []
    hashes = {}
    for name in blob_names:
        data = blob.download(name)
        uri = f"blob://{blob.container_name}/{name}"
        artifact = load_artifact_from_bytes(data, source_name=uri)
        artifacts.append(artifact)
        hashes[artifact.artifact_id] = _bytes_hash(data)
    return artifacts, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ciATHENA artifacts into Chroma")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Path to local artifacts directory")
    parser.add_argument("--clear", action="store_true", help="Clear everything and re-ingest from scratch")
    parser.add_argument("--source", choices=["auto", "blob", "local"], default="auto",
                        help="Artifact source: auto (blob if configured, else local), blob, or local")
    args = parser.parse_args()

    print("=" * 70)
    print("ciATHENA Knowledge Spine — Ingestion")
    print("=" * 70)

    print("\n[1/5] Checking blob storage...")
    blob = get_blob_client()

    use_blob = (args.source == "blob") or (args.source == "auto" and blob is not None)
    if args.source == "blob" and blob is None:
        print("  ERROR: --source blob but no AZURE_BLOB_CONNECTION_STRING configured")
        return

    print(f"\n[2/5] Loading artifacts (source: {'blob' if use_blob else 'local'})...")
    blob_hashes: dict[str, str] = {}
    if use_blob:
        artifacts, blob_hashes = _load_from_blob(blob)
    else:
        artifacts_dir = pathlib.Path(args.artifacts_dir)
        artifacts = load_artifacts(artifacts_dir)

    for a in artifacts:
        src = "blob" if use_blob else "local"
        print(f"  {a.artifact_id:40s} type={a.component_type:12s} usecase={a.envelope['usecase']} [{src}]")
    print(f"  Total: {len(artifacts)} artifacts")

    print("\n[3/5] Initializing embedder + store + ingestion log...")
    embedder = get_embedder()
    store = KnowledgeStore(embedder=embedder)
    log = IngestionLog()
    print(f"  Embedding model: {embedder.model_name}")
    print(f"  Collection count before: {store.count()}")
    print(f"  Ingestion log entries: {log.count}")

    if args.clear:
        print("  Clearing collection + ingestion log...")
        store.clear()
        log.clear()
        print(f"  Collection count after clear: {store.count()}")

    print("\n[4/5] Checking versions + ingesting...")
    total_ingested = 0
    total_skipped = 0

    for a in artifacts:
        current_hash = blob_hashes.get(a.artifact_id)
        needs, reason = log.needs_reingest(a, current_hash=current_hash)
        if not needs and not args.clear:
            print(f"  SKIP  {a.artifact_id:40s} ({reason})")
            total_skipped += 1
            continue

        chunks = chunk_artifact(a)
        if chunks:
            store.ingest(chunks)
        log.record(a, chunk_count=len(chunks), embedding_model=embedder.model_name,
                   file_hash=current_hash)
        print(f"  INGEST {a.artifact_id:40s} v{a.envelope.get('content_version', '?')} "
              f"({len(chunks)} chunks) [{reason}]")
        total_ingested += 1

    print(f"\n[5/5] Summary")
    print(f"  Source:   {'Azure Blob Storage' if use_blob else 'local directory'}")
    print(f"  Ingested: {total_ingested} artifacts")
    print(f"  Skipped:  {total_skipped} artifacts (unchanged)")
    print(f"  Collection count after: {store.count()}")
    print(f"  Ingestion log entries: {log.count}")

    print("\nDone. Run chat.py to query.")


if __name__ == "__main__":
    main()
