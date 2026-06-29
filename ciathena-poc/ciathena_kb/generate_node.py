"""
ciathena_kb.generate_node
-------------------------
Grounded answer generation with citations. Refuses when no approved context
survives filtering. Obeys ai_routing_note from the artifacts.
"""

from __future__ import annotations

from typing import Any, Callable, Generator

from .llm import ChatLLM

from .prompt_manager import DEFAULT_PROMPTS

GENERATE_SYSTEM_PROMPT = DEFAULT_PROMPTS["generate_system"]

NO_CONTEXT_RESPONSE = (
    "I don't have approved knowledge artifacts covering this topic. "
    "This may be outside the currently loaded corpus, or the relevant "
    "artifacts haven't been ingested yet."
)

FALLBACK_DISCLAIMER = (
    "⚠️ *No high-confidence approved content was found for this query. "
    "The following answer is based on the closest available context in the knowledge base "
    "— treat it as indicative only and verify with your data team before acting.*\n\n"
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


def make_generate_node(llm: ChatLLM, system_prompt: str | None = None) -> Callable:
    """Return a LangGraph node that generates a grounded answer."""
    gen_prompt = system_prompt or GENERATE_SYSTEM_PROMPT

    def generate_node(state: dict[str, Any]) -> dict[str, Any]:
        route = state.get("route", {})
        user_query = state.get("user_query", "")
        graded = state.get("graded_chunks", [])

        if not route.get("in_domain", True):
            return {"answer": NO_CONTEXT_RESPONSE, "citations": [], "is_fallback": False}

        if not graded:
            fallback = state.get("fallback_chunks", [])
            if fallback:
                intent = route.get("intent", "definition")
                chunks_text = _format_chunks(fallback)
                messages = [
                    {"role": "system", "content": gen_prompt.format(
                        intent=intent, chunks=chunks_text)},
                ]
                history = state.get("conversation_history", [])
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": user_query})
                raw_answer = llm.chat(messages, temperature=0.1)
                answer = FALLBACK_DISCLAIMER + raw_answer
                citations = sorted({c.get("chunk_id", "") for c in fallback if c.get("chunk_id")})
                return {"answer": answer, "citations": citations, "is_fallback": True}
            return {"answer": NO_CONTEXT_RESPONSE, "citations": [], "is_fallback": False}

        intent = route.get("intent", "definition")
        chunks_text = _format_chunks(graded)

        messages = [
            {"role": "system", "content": gen_prompt.format(
                intent=intent, chunks=chunks_text)},
        ]
        history = state.get("conversation_history", [])
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        answer = llm.chat(messages, temperature=0.1)
        citations = sorted({c.get("chunk_id", "") for c in graded if c.get("chunk_id")})

        return {"answer": answer, "citations": citations, "is_fallback": False}

    return generate_node


def make_stream_generate(llm: ChatLLM, system_prompt: str | None = None) -> Callable[..., Generator[str, None, None]]:
    """Return a callable that streams the answer token-by-token."""
    gen_prompt = system_prompt or GENERATE_SYSTEM_PROMPT

    def stream_generate(
        route: dict[str, Any],
        graded_chunks: list[dict[str, Any]],
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        fallback_chunks: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        if not route.get("in_domain", True):
            yield NO_CONTEXT_RESPONSE
            return

        effective = graded_chunks or []
        is_fallback = not effective and bool(fallback_chunks)
        if is_fallback:
            effective = fallback_chunks or []

        if not effective:
            yield NO_CONTEXT_RESPONSE
            return

        if is_fallback:
            yield FALLBACK_DISCLAIMER

        intent = route.get("intent", "definition")
        chunks_text = _format_chunks(effective)

        messages = [
            {"role": "system", "content": gen_prompt.format(
                intent=intent, chunks=chunks_text)},
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_query})

        yield from llm.chat_stream(messages, temperature=0.1)

    return stream_generate
