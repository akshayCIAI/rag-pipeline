"""
ciathena_kb.query_expander_node
-------------------------------
Generates 3-4 semantically diverse query variations to boost recall.
Sets state["expanded_queries"] so retrieval_node loops over all of them
and merges results by chunk_id before reranking.
"""

from __future__ import annotations

from typing import Any, Callable

from .llm import ChatLLM, FakeChatLLM


EXPANDER_SYSTEM_PROMPT = """\
You are a query expansion assistant for a pharma commercial analytics knowledge base.
Given a user question, generate 3 semantically diverse variations that preserve
the original meaning while using different phrasing, synonyms, or perspectives.
Each variation will be embedded and used to search a vector store independently —
so each MUST be a standalone, complete, self-contained question.

Rules:
- Keep domain terminology accurate (GTN, MMM, HCP, DTC, etc.)
- Vary the angle: try a definitional, a procedural, and a contextual variation
- Do NOT add information not implied by the original question
- Keep each variation concise (under 25 words)

Respond ONLY with valid JSON (no markdown):
{"queries": ["variation 1", "variation 2", "variation 3"]}"""


def make_query_expander_node(llm: ChatLLM) -> Callable:
    """Return a LangGraph node that generates query variations for multi-query retrieval."""
    use_llm = not isinstance(llm, FakeChatLLM)

    def query_expander_node(state: dict[str, Any]) -> dict[str, Any]:
        if not use_llm:
            return {"expanded_queries": []}

        route = state.get("route", {})
        if not route.get("in_domain", True):
            return {"expanded_queries": []}

        base_query = route.get("rewritten_query") or state.get("user_query", "")
        if not base_query:
            return {"expanded_queries": []}

        messages = [
            {"role": "system", "content": EXPANDER_SYSTEM_PROMPT},
            {"role": "user", "content": base_query},
        ]

        try:
            result = llm.chat_json(messages, temperature=0.3)
            queries = result.get("queries", [])
            if not isinstance(queries, list):
                queries = []
            queries = [q for q in queries if isinstance(q, str) and q.strip()][:4]
        except Exception:
            queries = []

        return {"expanded_queries": queries}

    return query_expander_node
