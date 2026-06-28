"""
ciathena_kb.rerank_node
-----------------------
Post-retrieval filtering: drops chunks below a cosine threshold, auto-passes
high-confidence chunks, and batch-grades borderline chunks via a single LLM call.
Keeps the top_k best.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm import ChatLLM, FakeChatLLM

from .prompt_manager import DEFAULT_PROMPTS

GRADE_SYSTEM_PROMPT = DEFAULT_PROMPTS["rerank_grading"]

COSINE_THRESHOLD = 0.15
HIGH_CONFIDENCE_THRESHOLD = 0.7
SCORE_GAP_THRESHOLD = 0.1
INTENT_BOOST = 0.03

INTENT_PREFERRED_TYPES: dict[str, set[str]] = {
    "definition": {"concept", "methodology"},
    "how-to": {"methodology", "process_flow"},
    "advisory": {"playbook", "process_flow", "anomaly"},
    "comparison": {"concept", "methodology", "sttm_mapping"},
}


def _deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the highest-scoring chunk per artifact_id to ensure diversity."""
    seen_artifacts: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        aid = chunk.get("artifact_id", "")
        if aid not in seen_artifacts:
            seen_artifacts[aid] = 0
        seen_artifacts[aid] += 1
        if seen_artifacts[aid] <= 2:
            result.append(chunk)
    return result


def make_rerank_node(llm: ChatLLM, top_k: int = 4, system_prompt: str | None = None) -> Callable:
    """Return a LangGraph node that reranks/filters retrieved chunks."""
    use_llm_grading = not isinstance(llm, FakeChatLLM)
    grade_prompt = system_prompt or GRADE_SYSTEM_PROMPT

    def _finalize(chunks: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
        """Apply intent-aware sorting, dedup, and truncate to top_k."""
        preferred = INTENT_PREFERRED_TYPES.get(intent, set())

        def _sort_key(c: dict[str, Any]) -> float:
            score = c.get("score", 0)
            if c.get("component_type", "") in preferred:
                score += INTENT_BOOST
            return score

        chunks.sort(key=_sort_key, reverse=True)
        return _deduplicate_chunks(chunks)[:top_k]

    def rerank_node(state: dict[str, Any]) -> dict[str, Any]:
        route = state.get("route", {})
        user_query = state.get("user_query", "")
        chunks = state.get("retrieved_chunks", [])
        intent = route.get("intent", "definition")

        if not route.get("in_domain", True):
            return {"graded_chunks": []}

        grading_query = route.get("rewritten_query") or user_query

        above_threshold = [c for c in chunks if c.get("score", 0) >= COSINE_THRESHOLD]

        if not use_llm_grading or not above_threshold:
            return {"graded_chunks": _finalize(above_threshold, intent)}

        candidates = above_threshold[:top_k * 2]

        high_confidence = [c for c in candidates if c.get("score", 0) >= HIGH_CONFIDENCE_THRESHOLD]
        needs_grading = [c for c in candidates if c.get("score", 0) < HIGH_CONFIDENCE_THRESHOLD]

        if not needs_grading:
            return {"graded_chunks": _finalize(high_confidence, intent)}

        scores = [c.get("score", 0) for c in candidates]
        if len(scores) > top_k:
            gap = scores[top_k - 1] - scores[top_k]
            if gap >= SCORE_GAP_THRESHOLD:
                return {"graded_chunks": _finalize(candidates, intent)}

        graded = list(high_confidence)

        if needs_grading:
            chunk_descriptions = []
            for i, chunk in enumerate(needs_grading):
                chunk_descriptions.append(
                    f"[{i}] ({chunk.get('component_type', '')} / "
                    f"{chunk.get('artifact_id', '')})\n{chunk.get('text', '')}"
                )
            chunks_block = "\n---\n".join(chunk_descriptions)

            messages = [
                {"role": "system", "content": grade_prompt},
                {"role": "user", "content": (
                    f"Question: {grading_query}\n\n"
                    f"Chunks to grade:\n{chunks_block}"
                )},
            ]
            try:
                result = llm.chat_json(messages, temperature=0)
                verdicts = result.get("results", [])
                for v in verdicts:
                    idx = v.get("index")
                    if v.get("relevant", False) and isinstance(idx, int) and 0 <= idx < len(needs_grading):
                        graded.append(needs_grading[idx])
            except Exception:
                graded.extend(needs_grading)

        return {"graded_chunks": _finalize(graded, intent)}

    return rerank_node
