"""
ciathena_kb.generate_node
-------------------------
Grounded answer generation with citations. Refuses when no approved context
survives filtering. Obeys ai_routing_note from the artifacts.
"""

from __future__ import annotations

from typing import Any, Callable

from .llm import ChatLLM

GENERATE_SYSTEM_PROMPT = """\
You are ciATHENA, a domain intelligence agent for pharma life-sciences
commercial analytics. Generate a clear, accurate answer GROUNDED ONLY in the
provided knowledge chunks.

RULES:
1. Use ONLY information from the chunks below. Do NOT hallucinate or add
   information not present in the chunks.
2. Cite every claim with [artifact_id::chunk_item_id] at the end of the
   sentence or paragraph. The chunk_id contains both parts joined by "::".
3. If the chunks do not contain enough information to answer the question,
   say so explicitly — do NOT guess.
4. Match the response depth to the intent:
   - definition: concise explanation of what it is (and what it is NOT)
   - how-to: step-by-step guidance with context from methodology chunks
   - advisory: actionable recommendation grounded in playbook logic
   - comparison: structured comparison with differences highlighted
5. Keep answers focused and structured. Use bullet points for lists.

USER INTENT: {intent}

KNOWLEDGE CHUNKS:
{chunks}
"""

NO_CONTEXT_RESPONSE = (
    "I don't have approved knowledge artifacts covering this topic. "
    "This may be outside the currently loaded corpus, or the relevant "
    "artifacts haven't been ingested yet."
)


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"--- Chunk {i} [{c.get('chunk_id', '')}] "
            f"({c.get('component_type', '')} / {c.get('usecase', '')}) "
            f"score={c.get('score', 0):.3f} ---\n"
            f"{c.get('text', '')}"
        )
    return "\n\n".join(parts)


def make_generate_node(llm: ChatLLM) -> Callable:
    """Return a LangGraph node that generates a grounded answer."""

    def generate_node(state: dict[str, Any]) -> dict[str, Any]:
        route = state.get("route", {})
        user_query = state.get("user_query", "")
        graded = state.get("graded_chunks", [])

        if not route.get("in_domain", True) or not graded:
            return {
                "answer": NO_CONTEXT_RESPONSE,
                "citations": [],
            }

        intent = route.get("intent", "definition")
        chunks_text = _format_chunks(graded)

        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT.format(
                intent=intent, chunks=chunks_text)},
            {"role": "user", "content": user_query},
        ]

        answer = llm.chat(messages, temperature=0.1)

        citations = sorted({c.get("chunk_id", "") for c in graded if c.get("chunk_id")})

        return {
            "answer": answer,
            "citations": citations,
        }

    return generate_node
