"""
ciathena_kb.retrieval_node
--------------------------
The LangGraph-compatible retrieval node. Reads the router's filter decisions
from state and retrieves from the knowledge store with metadata pre-filtering
and General-layer OR-merge.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict

from .store import KnowledgeStore, RetrievedChunk


class AgentState(TypedDict, total=False):
    # inputs
    user_query: str
    # set by router
    route: dict[str, Any]
    # set by retrieval
    retrieved_chunks: list[dict[str, Any]]
    knowledge_context: str
    # set by reranker
    graded_chunks: list[dict[str, Any]]
    # set by generator
    answer: str
    citations: list[str]


def make_retrieval_node(store: KnowledgeStore, candidate_pool: int = 12) -> Callable:
    """Return a LangGraph node that retrieves using the router's decisions."""

    def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
        route = state.get("route", {})

        # Support both agentic flow (route dict from router) and direct calls
        # (usecase/component_type/top_k at top level, for demo.py compat)
        query = route.get("rewritten_query") or state["user_query"]
        usecase = route.get("usecase") or state.get("usecase")
        component_types = route.get("component_types", [])
        if not component_types and state.get("component_type"):
            component_types = [state["component_type"]]
        top_k = state.get("top_k", candidate_pool)

        filters: dict[str, Any] = {}
        if usecase and usecase != "General":
            filters["usecase"] = usecase
        if component_types:
            filters["component_type"] = component_types if len(component_types) > 1 else component_types[0]

        hits: list[RetrievedChunk] = store.retrieve(
            query, top_k=top_k, filters=filters, include_general=True,
        )

        retrieved = [{
            "chunk_id": h.chunk_id,
            "score": h.score,
            "usecase": h.metadata.get("usecase"),
            "component_type": h.metadata.get("component_type"),
            "artifact_id": h.metadata.get("artifact_id"),
            "text": h.text,
        } for h in hits]

        context = "\n\n---\n\n".join(
            f"[{h.metadata.get('component_type')} | {h.metadata.get('artifact_id')} "
            f"| score={h.score}]\n{h.text}" for h in hits)

        return {"retrieved_chunks": retrieved, "knowledge_context": context}

    return retrieval_node


def build_demo_graph(store: KnowledgeStore):
    """Build a one-node LangGraph that runs the retrieval node, if langgraph is
    available. Returns a compiled graph. Raises ImportError if langgraph absent.
    """
    from langgraph.graph import StateGraph, START, END

    node = make_retrieval_node(store)
    g = StateGraph(AgentState)
    g.add_node("retrieve", node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", END)
    return g.compile()
