"""
ciathena_kb.rerank_node
-----------------------
Post-retrieval filtering: drops chunks below a cosine threshold and optionally
uses the LLM to grade relevance (batched). Keeps the top_k best.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm import ChatLLM, FakeChatLLM

GRADE_SYSTEM_PROMPT = """\
You are a relevance grader for a pharma commercial analytics knowledge base.
Given a user question and a retrieved chunk, decide if the chunk is relevant
to answering the question.

Be GENEROUS with relevance:
- If the question is broad (e.g. "what is MMM"), any chunk that describes a
  component, step, or aspect of that topic IS relevant.
- A chunk does not need to directly define the term — if it explains part of
  the methodology, a use case, or a related concept, mark it relevant.
- Only mark irrelevant if the chunk is truly about a different, unrelated topic.

Respond ONLY with valid JSON (no markdown):
{{"relevant": true/false, "reason": "one-line explanation"}}
"""

COSINE_THRESHOLD = 0.15


def make_rerank_node(llm: ChatLLM, top_k: int = 4) -> Callable:
    """Return a LangGraph node that reranks/filters retrieved chunks."""
    use_llm_grading = not isinstance(llm, FakeChatLLM)

    def rerank_node(state: dict[str, Any]) -> dict[str, Any]:
        route = state.get("route", {})
        user_query = state.get("user_query", "")
        chunks = state.get("retrieved_chunks", [])

        if not route.get("in_domain", True):
            return {"graded_chunks": []}

        grading_query = route.get("rewritten_query") or user_query

        above_threshold = [c for c in chunks if c.get("score", 0) >= COSINE_THRESHOLD]

        if use_llm_grading and above_threshold:
            graded: list[dict[str, Any]] = []
            for chunk in above_threshold[:top_k * 2]:
                messages = [
                    {"role": "system", "content": GRADE_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Question: {grading_query}\n\n"
                        f"Chunk ({chunk.get('component_type', '')} / "
                        f"{chunk.get('artifact_id', '')}):\n{chunk.get('text', '')}"
                    )},
                ]
                try:
                    result = llm.chat_json(messages, temperature=0)
                    if result.get("relevant", False):
                        graded.append(chunk)
                except Exception:
                    graded.append(chunk)
            final = graded[:top_k]
        else:
            final = above_threshold[:top_k]

        return {"graded_chunks": final}

    return rerank_node
