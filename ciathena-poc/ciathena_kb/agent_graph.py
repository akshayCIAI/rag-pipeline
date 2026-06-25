"""
ciathena_kb.agent_graph
-----------------------
Assembles the full agentic RAG LangGraph:

    router → retrieve → rerank → generate

Conditional edges are structured so a Level-2 self-correction loop (relax
filters / rewrite query on weak context) drops in cleanly later.
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.graph import StateGraph, START, END

from .catalog import build_routing_catalog
from .generate_node import make_generate_node
from .llm import ChatLLM, get_chat_llm
from .loader import Artifact
from .rerank_node import make_rerank_node
from .retrieval_node import AgentState, make_retrieval_node
from .router_node import make_router_node
from .store import KnowledgeStore


def _should_generate(state: dict[str, Any]) -> str:
    """Conditional edge: skip generation if out-of-domain."""
    route = state.get("route", {})
    if not route.get("in_domain", True):
        return "decline"
    return "continue"


def build_agent_graph(
    store: KnowledgeStore,
    artifacts: list[Artifact],
    llm: ChatLLM | None = None,
    candidate_pool: int | None = None,
    top_k: int | None = None,
):
    """Build and compile the full agentic RAG graph."""
    if llm is None:
        llm = get_chat_llm()

    pool = candidate_pool or int(os.environ.get("RETRIEVAL_CANDIDATE_POOL", "12"))
    k = top_k or int(os.environ.get("RETRIEVAL_TOP_K", "4"))

    catalog = build_routing_catalog(artifacts)

    router = make_router_node(llm, catalog)
    retriever = make_retrieval_node(store, candidate_pool=pool)
    reranker = make_rerank_node(llm, top_k=k)
    generator = make_generate_node(llm)

    decline_node = lambda state: {
        "answer": (
            "I don't have approved knowledge artifacts covering this topic. "
            "This may be outside the currently loaded corpus, or the relevant "
            "artifacts haven't been ingested yet."
        ),
        "citations": [],
        "graded_chunks": [],
    }

    g = StateGraph(AgentState)
    g.add_node("route", router)
    g.add_node("retrieve", retriever)
    g.add_node("rerank", reranker)
    g.add_node("generate", generator)
    g.add_node("decline", decline_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", _should_generate, {
        "continue": "retrieve",
        "decline": "decline",
    })
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    g.add_edge("decline", END)

    return g.compile()
