"""
demo.py -- end-to-end PoC runner.

    load artifacts -> validate -> chunk -> embed -> ingest into Chroma
    -> run sample retrievals (with and without usecase filtering)
    -> run the LangGraph node if langgraph is installed.

Run:
    cd ciathena-poc
    python demo.py
"""

from __future__ import annotations

import pathlib

from dotenv import load_dotenv
load_dotenv()

from ciathena_kb import (
    chunk_all, get_embedder, load_artifacts, make_retrieval_node,
    KnowledgeStore,
)

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    banner("1. LOAD + VALIDATE ARTIFACTS")
    artifacts = load_artifacts(ARTIFACTS_DIR)
    for a in artifacts:
        print(f"  loaded {a.artifact_id:32s} type={a.component_type:12s} "
              f"usecase={a.envelope['usecase']}")

    banner("2. CHUNK (one item per chunk; whole-doc types skipped)")
    chunks = chunk_all(artifacts)
    for c in chunks:
        preview = c.text.replace("\n", " ")[:60]
        print(f"  chunk {c.chunk_id:46s} | {preview}...")
    print(f"\n  total chunks: {len(chunks)}")

    banner("3. EMBED + INGEST INTO CHROMA")
    embedder = get_embedder(model_name="text-embedding-3-large")
    store = KnowledgeStore(embedder=embedder, collection_name="poc")
    n = store.ingest(chunks)
    print(f"  ingested {n} chunks; collection count = {store.count()}")
    print(f"  embedding model recorded: {embedder.model_name}")

    node = make_retrieval_node(store)

    banner("4a. RETRIEVE -- no usecase filter")
    out = node({"user_query": "best metric for a launch brand adoption",
                "top_k": 3})
    for r in out["retrieved_chunks"]:
        print(f"  {r['score']:.3f}  {r['component_type']:11s}  {r['chunk_id']}")

    banner("4b. RETRIEVE -- usecase=MMM filter (KOL/other usecases excluded)")
    out = node({"user_query": "where should I move my budget across channels",
                "usecase": "MMM", "top_k": 3})
    for r in out["retrieved_chunks"]:
        print(f"  {r['score']:.3f}  {r['component_type']:11s}  {r['chunk_id']}")

    banner("4c. RETRIEVE -- narrow to playbook component_type only")
    out = node({"user_query": "is this channel saturated",
                "usecase": "MMM", "component_type": "playbook", "top_k": 3})
    for r in out["retrieved_chunks"]:
        print(f"  {r['score']:.3f}  {r['component_type']:11s}  {r['chunk_id']}")

    banner("5. LANGGRAPH NODE (if langgraph installed)")
    try:
        from ciathena_kb import build_demo_graph
        graph = build_demo_graph(store)
        result = graph.invoke({"user_query": "what is gross to net", "top_k": 2})
        print("  graph ran; retrieved:")
        for r in result["retrieved_chunks"]:
            print(f"    {r['score']:.3f}  {r['chunk_id']}")
    except ImportError:
        print("  langgraph not installed; node is still usable directly "
              "(see make_retrieval_node). Install: pip install langgraph")


if __name__ == "__main__":
    main()
