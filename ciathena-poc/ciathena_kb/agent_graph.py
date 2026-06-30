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
from .query_expander_node import make_query_expander_node
from .rerank_node import make_rerank_node, HIGH_CONFIDENCE_THRESHOLD, INTENT_PREFERRED_TYPES, INTENT_BOOST, _deduplicate_chunks
from .retrieval_node import AgentState, make_retrieval_node
from .router_node import make_router_node
from .store import KnowledgeStore
from .validation_node import make_validation_node


def _should_generate(state: dict[str, Any]) -> str:
    """Conditional edge: skip generation if out-of-domain."""
    route = state.get("route", {})
    if not route.get("in_domain", True):
        return "decline"
    return "continue"


def _should_rerank(state: dict[str, Any]) -> str:
    """Skip rerank when all top retrieved chunks are high-confidence."""
    chunks = state.get("retrieved_chunks", [])
    top_k = int(os.environ.get("RETRIEVAL_TOP_K", "4"))
    top_chunks = chunks[:top_k]
    if top_chunks and all(c.get("score", 0) >= HIGH_CONFIDENCE_THRESHOLD for c in top_chunks):
        return "skip"
    return "rerank"


def build_agent_graph(
    store: KnowledgeStore,
    artifacts: list[Artifact],
    llm: ChatLLM | None = None,
    candidate_pool: int | None = None,
    top_k: int | None = None,
    prompts: dict[str, str] | None = None,
):
    """Build and compile the full agentic RAG graph."""
    if llm is None:
        llm = get_chat_llm()

    pool = candidate_pool or int(os.environ.get("RETRIEVAL_CANDIDATE_POOL", "12"))
    k = top_k or int(os.environ.get("RETRIEVAL_TOP_K", "4"))
    p = prompts or {}

    catalog = build_routing_catalog(artifacts)

    router = make_router_node(llm, catalog, system_prompt=p.get("router_system"))
    expander = make_query_expander_node(llm, system_prompt=p.get("query_expander"))
    retriever = make_retrieval_node(store, candidate_pool=pool)
    reranker = make_rerank_node(llm, top_k=k, system_prompt=p.get("rerank_grading"))
    generator = make_generate_node(llm, system_prompt=p.get("generate_system"))
    validator = make_validation_node(llm, system_prompt=p.get("validation_grounding"))

    decline_node = lambda state: {
        "answer": (
            "I don't have approved knowledge artifacts covering this topic. "
            "This may be outside the currently loaded corpus, or the relevant "
            "artifacts haven't been ingested yet."
        ),
        "citations": [],
        "graded_chunks": [],
        "fallback_chunks": [],
    }

    def skip_rerank_node(state: dict[str, Any]) -> dict[str, Any]:
        chunks = list(state.get("retrieved_chunks", []))
        intent = state.get("route", {}).get("intent", "definition")
        preferred = INTENT_PREFERRED_TYPES.get(intent, set())
        chunks.sort(
            key=lambda c: c.get("score", 0) + (INTENT_BOOST if c.get("component_type", "") in preferred else 0),
            reverse=True,
        )
        return {"graded_chunks": _deduplicate_chunks(chunks)[:k], "fallback_chunks": []}

    g = StateGraph(AgentState)
    g.add_node("route", router)
    g.add_node("expand_queries", expander)
    g.add_node("retrieve", retriever)
    g.add_node("rerank", reranker)
    g.add_node("skip_rerank", skip_rerank_node)
    g.add_node("generate", generator)
    g.add_node("validation", validator)
    g.add_node("decline", decline_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", _should_generate, {
        "continue": "expand_queries",
        "decline": "decline",
    })
    g.add_edge("expand_queries", "retrieve")
    g.add_conditional_edges("retrieve", _should_rerank, {
        "rerank": "rerank",
        "skip": "skip_rerank",
    })
    g.add_edge("rerank", "generate")
    g.add_edge("skip_rerank", "generate")
    g.add_edge("generate", "validation")
    g.add_edge("validation", END)
    g.add_edge("decline", END)

    return g.compile()


def build_pre_generate_graph(
    store: KnowledgeStore,
    artifacts: list[Artifact],
    llm: ChatLLM | None = None,
    candidate_pool: int | None = None,
    top_k: int | None = None,
    prompts: dict[str, str] | None = None,
):
    """Build graph that runs route → retrieve → rerank only (no generate).

    Used with streaming: run this graph to get routed + reranked chunks,
    then stream the generate step separately via make_stream_generate().
    """
    if llm is None:
        llm = get_chat_llm()

    pool = candidate_pool or int(os.environ.get("RETRIEVAL_CANDIDATE_POOL", "12"))
    k = top_k or int(os.environ.get("RETRIEVAL_TOP_K", "4"))
    p = prompts or {}

    catalog = build_routing_catalog(artifacts)

    router = make_router_node(llm, catalog, system_prompt=p.get("router_system"))
    expander = make_query_expander_node(llm, system_prompt=p.get("query_expander"))
    retriever = make_retrieval_node(store, candidate_pool=pool)
    reranker = make_rerank_node(llm, top_k=k, system_prompt=p.get("rerank_grading"))

    decline_node = lambda state: {
        "answer": (
            "I don't have approved knowledge artifacts covering this topic. "
            "This may be outside the currently loaded corpus, or the relevant "
            "artifacts haven't been ingested yet."
        ),
        "citations": [],
        "graded_chunks": [],
        "fallback_chunks": [],
    }

    def skip_rerank_node(state: dict[str, Any]) -> dict[str, Any]:
        chunks = list(state.get("retrieved_chunks", []))
        intent = state.get("route", {}).get("intent", "definition")
        preferred = INTENT_PREFERRED_TYPES.get(intent, set())
        chunks.sort(
            key=lambda c: c.get("score", 0) + (INTENT_BOOST if c.get("component_type", "") in preferred else 0),
            reverse=True,
        )
        return {"graded_chunks": _deduplicate_chunks(chunks)[:k], "fallback_chunks": []}

    g = StateGraph(AgentState)
    g.add_node("route", router)
    g.add_node("expand_queries", expander)
    g.add_node("retrieve", retriever)
    g.add_node("rerank", reranker)
    g.add_node("skip_rerank", skip_rerank_node)
    g.add_node("decline", decline_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", _should_generate, {
        "continue": "expand_queries",
        "decline": "decline",
    })
    g.add_edge("expand_queries", "retrieve")
    g.add_conditional_edges("retrieve", _should_rerank, {
        "rerank": "rerank",
        "skip": "skip_rerank",
    })
    g.add_edge("rerank", END)
    g.add_edge("skip_rerank", END)
    g.add_edge("decline", END)

    return g.compile()
