"""
ciathena_kb.generate_node
-------------------------
Grounded answer generation with citations. Refuses when no approved context
survives filtering. Obeys ai_routing_note from the artifacts.

Base-model fallback: when the question is in-domain but no relevant chunk
survives filtering, and the base-model fallback is enabled, the answer is
generated from the LLM's own general pharma knowledge (closed-book), clearly
labelled and uncited. This REPLACES the older weak-chunk soft fallback.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Generator

from .llm import ChatLLM

from .prompt_manager import DEFAULT_PROMPTS

GENERATE_SYSTEM_PROMPT = DEFAULT_PROMPTS["generate_system"]
BASE_MODEL_SYSTEM_PROMPT = DEFAULT_PROMPTS["base_model_system"]

NO_CONTEXT_RESPONSE = (
    "I don't have approved knowledge artifacts covering this topic. "
    "This may be outside the currently loaded corpus, or the relevant "
    "artifacts haven't been ingested yet."
)

BASE_MODEL_DISCLAIMER = (
    "🧠 *Not found in the governed knowledge base. The answer below is from the "
    "base model's general knowledge — it is **not** from approved artifacts and "
    "carries no citations. Verify with your data team before acting.*\n\n"
)

# Appended to the grounded system prompt when the base-model fallback is ON, so
# the model can FILL GAPS the chunks don't cover (e.g. one side of a comparison)
# with clearly-labelled general knowledge instead of only saying "not available".
AUGMENTATION_CLAUSE = """

── GAP-FILLING (base-model fallback is ON) ──
The knowledge chunks above may not cover every part of the question. This AMENDS
rule 1: for aspects the chunks do NOT cover, you MAY use your own general
pharma / life-sciences knowledge to give a genuinely useful answer instead of
only writing "not available in provided chunks".
STRICT REQUIREMENTS when you do this:
- Cite chunks with [artifact_id::chunk_item_id] for every claim taken FROM the chunks.
- Clearly mark every general-knowledge addition. In prose, prefix it with
  "🧠 (general knowledge):". In a table cell, write "🧠 general knowledge" —
  never a fabricated citation.
- NEVER contradict the chunks, and NEVER invent citations, client names, or
  specific numeric figures (thresholds, decay rates, dollar amounts).
- Prefer grounded content; use general knowledge only to fill genuine gaps.
"""


def base_model_fallback_default() -> bool:
    """Default toggle for callers without a UI (e.g. chat.py). Overridable per
    request via state['base_model_enabled']; env ENABLE_BASE_MODEL_FALLBACK."""
    return os.environ.get("ENABLE_BASE_MODEL_FALLBACK", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def _base_model_messages(base_prompt: str, intent: str, user_query: str,
                         history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": base_prompt.format(intent=intent)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    return messages


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


def make_generate_node(llm: ChatLLM, system_prompt: str | None = None,
                       base_model_prompt: str | None = None) -> Callable:
    """Return a LangGraph node that generates a grounded answer."""
    gen_prompt = system_prompt or GENERATE_SYSTEM_PROMPT
    base_prompt = base_model_prompt or BASE_MODEL_SYSTEM_PROMPT

    def generate_node(state: dict[str, Any]) -> dict[str, Any]:
        route = state.get("route", {})
        user_query = state.get("user_query", "")
        graded = state.get("graded_chunks", [])
        base_enabled = state.get("base_model_enabled", base_model_fallback_default())

        if not route.get("in_domain", True):
            return {"answer": NO_CONTEXT_RESPONSE, "citations": [], "is_fallback": False,
                    "is_base_model": False, "fallback_chunks": []}

        if not graded:
            # In-domain but nothing relevant retrieved. If the base-model fallback
            # is enabled, answer from the LLM's own general pharma knowledge
            # (closed-book, uncited, disclaimed); otherwise decline.
            if base_enabled:
                intent = route.get("intent", "definition")
                messages = _base_model_messages(
                    base_prompt, intent, user_query, state.get("conversation_history", []))
                raw_answer = llm.chat(messages, temperature=0.2)
                answer = BASE_MODEL_DISCLAIMER + raw_answer
                return {"answer": answer, "citations": [], "is_fallback": False,
                        "is_base_model": True, "fallback_chunks": []}
            return {"answer": NO_CONTEXT_RESPONSE, "citations": [], "is_fallback": False,
                    "is_base_model": False, "fallback_chunks": []}

        intent = route.get("intent", "definition")
        chunks_text = _format_chunks(graded)

        system_content = gen_prompt.format(intent=intent, chunks=chunks_text)
        if base_enabled:  # allow labelled general-knowledge gap-filling
            system_content += AUGMENTATION_CLAUSE
        messages = [{"role": "system", "content": system_content}]
        history = state.get("conversation_history", [])
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        answer = llm.chat(messages, temperature=0.1)
        citations = sorted({c.get("chunk_id", "") for c in graded if c.get("chunk_id")})

        return {"answer": answer, "citations": citations, "is_fallback": False,
                "is_base_model": False}

    return generate_node


def make_stream_generate(llm: ChatLLM, system_prompt: str | None = None,
                         base_model_prompt: str | None = None) -> Callable[..., Generator[str, None, None]]:
    """Return a callable that streams the answer token-by-token."""
    gen_prompt = system_prompt or GENERATE_SYSTEM_PROMPT
    base_prompt = base_model_prompt or BASE_MODEL_SYSTEM_PROMPT

    def stream_generate(
        route: dict[str, Any],
        graded_chunks: list[dict[str, Any]],
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        fallback_chunks: list[dict[str, Any]] | None = None,  # deprecated; ignored
        base_model_enabled: bool | None = None,
    ) -> Generator[str, None, None]:
        if not route.get("in_domain", True):
            yield NO_CONTEXT_RESPONSE
            return

        enabled = base_model_fallback_default() if base_model_enabled is None else base_model_enabled
        effective = graded_chunks or []

        if not effective:
            # In-domain but nothing relevant retrieved → base-model fallback or decline.
            if not enabled:
                yield NO_CONTEXT_RESPONSE
                return
            yield BASE_MODEL_DISCLAIMER
            intent = route.get("intent", "definition")
            messages = _base_model_messages(base_prompt, intent, user_query, conversation_history)
            yield from llm.chat_stream(messages, temperature=0.2)
            return

        intent = route.get("intent", "definition")
        chunks_text = _format_chunks(effective)

        system_content = gen_prompt.format(intent=intent, chunks=chunks_text)
        if enabled:  # allow labelled general-knowledge gap-filling for uncovered parts
            system_content += AUGMENTATION_CLAUSE
        messages = [{"role": "system", "content": system_content}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_query})

        yield from llm.chat_stream(messages, temperature=0.1)

    return stream_generate
