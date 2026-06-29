"""
ciathena_kb.validation_node
----------------------------
Post-generation answer quality check.
  1. Citation existence check (no LLM) — flags [artifact::chunk] refs that don't
     match any chunk in the context.
  2. LLM grounding check — flags claims not supported by the provided chunks.
     Skipped for fallback answers (already disclaimed) and when using FakeChatLLM.

Sets state["validation_result"] = {"passed": bool, "issues": list[str]}.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .llm import ChatLLM, FakeChatLLM


GROUNDING_SYSTEM_PROMPT = """\
You are an answer quality checker for a pharma commercial analytics knowledge base.
Given an answer and the knowledge chunks it was grounded in, determine whether the
answer makes claims NOT supported by those chunks.

Guidelines:
- Minor paraphrasing and inference from stated facts are acceptable.
- Only flag clear hallucinations or claims that directly contradict the chunks.
- If the answer is short or only restates chunk content, it almost certainly passes.

Respond ONLY with valid JSON (no markdown):
{"passed": true, "issues": []}
or
{"passed": false, "issues": ["brief description of unsupported claim"]}"""


def _extract_cited_ids(answer: str) -> list[str]:
    return re.findall(r"\[([^\]]+::[^\]]+)\]", answer)


def make_validation_node(llm: ChatLLM) -> Callable:
    """Return a LangGraph node that validates the generated answer."""
    use_llm = not isinstance(llm, FakeChatLLM)

    def validation_node(state: dict[str, Any]) -> dict[str, Any]:
        answer = state.get("answer", "")
        graded = state.get("graded_chunks", [])
        fallback = state.get("fallback_chunks", [])
        effective = graded or fallback

        if not answer or not effective:
            return {"validation_result": {"passed": True, "issues": []}}

        # 1. Citation existence check (instant, no LLM)
        valid_ids = {c.get("chunk_id", "") for c in effective}
        bad_citations = [c for c in _extract_cited_ids(answer) if c not in valid_ids]
        issues: list[str] = []
        if bad_citations:
            issues.append(f"Unresolved citations: {', '.join(bad_citations)}")

        # 2. LLM grounding check — skip for fallback answers (already warned)
        if use_llm and not state.get("is_fallback", False):
            from .generate_node import _format_chunks
            chunks_text = _format_chunks(effective[:4])
            messages = [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Answer to validate:\n{answer}\n\n"
                    f"Knowledge chunks it was grounded in:\n{chunks_text}"
                )},
            ]
            try:
                result = llm.chat_json(messages, temperature=0)
                if not result.get("passed", True):
                    issues.extend(result.get("issues", []))
            except Exception:
                pass

        return {"validation_result": {"passed": len(issues) == 0, "issues": issues}}

    return validation_node
