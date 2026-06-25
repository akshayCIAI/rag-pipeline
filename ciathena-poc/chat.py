"""
chat.py -- Interactive CLI for the ciATHENA domain intelligence agent.

Usage:
    python chat.py                              # interactive REPL
    python chat.py --query "what is gross to net"  # single query
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from ciathena_kb import (
    load_artifacts, get_embedder, get_chat_llm,
    KnowledgeStore, build_agent_graph,
)

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"

SEPARATOR = "─" * 70


def print_result(state: dict) -> None:
    route = state.get("route", {})
    graded = state.get("graded_chunks", [])
    answer = state.get("answer", "")
    citations = state.get("citations", [])

    # Route info
    print(f"\n{SEPARATOR}")
    print("  Route")
    print(SEPARATOR)
    if not route.get("in_domain", True):
        print(f"  usecase:        {route.get('usecase', 'unknown')}")
        print(f"  in_domain:      false")
    else:
        print(f"  usecase:        {route.get('usecase', 'General')}")
        ct = route.get("component_types", [])
        print(f"  component_type: {', '.join(ct) if ct else '(all)'} (soft)")
        print(f"  intent:         {route.get('intent', 'definition')}")
        rq = route.get("rewritten_query", "")
        if rq:
            print(f"  rewritten:      {rq}")

    # Retrieved chunks
    if graded:
        print(f"\n{SEPARATOR}")
        print(f"  Retrieved chunks ({len(graded)})")
        print(SEPARATOR)
        for i, c in enumerate(graded, 1):
            score = c.get("score", 0)
            cid = c.get("chunk_id", "")
            ct = c.get("component_type", "")
            uc = c.get("usecase", "")
            print(f"  {i}. [{score:.3f}]  {cid:46s} {ct} / {uc}")

    # Answer
    print(f"\n{SEPARATOR}")
    print("  Answer")
    print(SEPARATOR)
    for line in answer.split("\n"):
        print(f"  {line}")

    if citations:
        print(f"\n  Citations: {', '.join(citations)}")
    print()


def run_single(graph, query: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  ciATHENA Knowledge Spine")
    print(f"{'═' * 70}")
    print(f"\n  Query: {query}")
    result = graph.invoke({"user_query": query})
    print_result(result)


def run_repl(graph) -> None:
    print(f"\n{'═' * 70}")
    print(f"  ciATHENA Knowledge Spine — Interactive Mode")
    print(f"  Type your question, or 'quit' / 'exit' to stop.")
    print(f"{'═' * 70}")
    history: list[dict[str, str]] = []
    while True:
        try:
            query = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        result = graph.invoke({
            "user_query": query,
            "conversation_history": history[-10:],
        })
        print_result(result)
        answer = result.get("answer", "")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="ciATHENA domain intelligence agent")
    parser.add_argument("--query", "-q", help="Single query (omit for interactive REPL)")
    args = parser.parse_args()

    print("Loading artifacts + building graph...")
    artifacts = load_artifacts(ARTIFACTS_DIR)
    embedder = get_embedder()
    store = KnowledgeStore(embedder=embedder)

    if store.count() == 0:
        print("WARNING: Chroma collection is empty. Run ingest.py first!")
        sys.exit(1)

    llm = get_chat_llm()
    graph = build_agent_graph(store=store, artifacts=artifacts, llm=llm)
    print(f"Ready. Store has {store.count()} chunks. LLM: {llm.model_name}")

    if args.query:
        run_single(graph, args.query)
    else:
        run_repl(graph)


if __name__ == "__main__":
    main()
