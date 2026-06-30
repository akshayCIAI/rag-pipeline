"""
ciathena_kb.validation_node
----------------------------
Post-generation answer quality check.
  1. Citation existence check (no LLM) — flags [artifact::chunk] refs that don't
     match any chunk in the context.
  2. Intent-aware LLM grounding check — evaluates the answer against per-intent
     quality rubrics and flags unsupported claims or gaps.
     Skipped for fallback answers (already disclaimed) and when using FakeChatLLM.

Sets state["validation_result"] = {
    "verdict": "pass" | "warn" | "fail",
    "passed": bool,
    "issues": list[str],
    "reason": str,
    "suggestion": str,
}.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .llm import ChatLLM, FakeChatLLM
from .prompt_manager import DEFAULT_PROMPTS


INTENT_RUBRICS: dict[str, str] = {
    "definition": (
        "Must clearly define the term and its key components. "
        "Warn if the answer is vague or circular. Fail if the definition contradicts the chunks."
    ),
    "how-to": (
        "Must include actionable steps or a clear process. "
        "Warn if steps are vague or incomplete. Fail if steps contradict chunk content."
    ),
    "advisory": (
        "Must give a concrete, grounded recommendation. "
        "Warn if advice is generic without referencing specific metrics or thresholds from the chunks. "
        "Fail if the recommendation contradicts playbook content."
    ),
    "comparison": (
        "Must compare at least two concepts on defined dimensions. "
        "Warn if one side is missing or underdeveloped. Fail if a comparison claim contradicts chunks."
    ),
}

_DEFAULT_RUBRIC = (
    "Answer must be grounded in the chunks. Warn if vague. Fail if it contradicts chunks."
)

_VALIDATION_PROMPT = DEFAULT_PROMPTS.get("validation_grounding") or """\
You are an answer quality checker for a pharma commercial analytics knowledge base.

DETECTED INTENT: {intent}
QUALITY RUBRIC: {rubric}

TASK: Evaluate whether the generated answer satisfies the rubric and is grounded
in the knowledge chunks provided.

Guidelines:
- Minor paraphrasing and inference from stated facts are acceptable.
- Only flag clear hallucinations or claims that directly contradict the chunks.
- A "warn" verdict means the answer is mostly correct but has gaps for the stated intent.
- A "fail" verdict means the answer makes claims that contradict the chunks.
- If the answer is short or only restates chunk content, it almost certainly passes.

QUESTION: {question}

KNOWLEDGE CHUNKS:
{chunks}

Respond ONLY with valid JSON (no markdown):
{{"verdict": "pass", "passed": true, "issues": [], "reason": "brief summary", "suggestion": ""}}
or with issues:
{{"verdict": "warn", "passed": false, "issues": ["specific gap"], "reason": "brief summary", "suggestion": "actionable tip"}}
or for contradictions:
{{"verdict": "fail", "passed": false, "issues": ["specific contradiction"], "reason": "brief summary", "suggestion": "actionable tip"}}"""


def _extract_cited_ids(answer: str) -> list[str]:
    return re.findall(r"\[([^\]]+::[^\]]+)\]", answer)


def validate_answer(
    llm: ChatLLM,
    route: dict[str, Any],
    effective_chunks: list[dict[str, Any]],
    answer: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Intent-aware validation. Returns verdict dict with verdict/passed/issues/reason/suggestion."""
    val_prompt = system_prompt or _VALIDATION_PROMPT

    # 1. Citation existence check (instant, no LLM)
    valid_ids = {c.get("chunk_id", "") for c in effective_chunks}
    bad_citations = [c for c in _extract_cited_ids(answer) if c not in valid_ids]
    issues: list[str] = []
    if bad_citations:
        issues.append(f"Unresolved citations: {', '.join(bad_citations)}")

    # 2. LLM grounding check with intent-specific rubric
    if not isinstance(llm, FakeChatLLM):
        from .generate_node import _format_chunks
        intent = route.get("intent", "definition")
        rubric = INTENT_RUBRICS.get(intent, _DEFAULT_RUBRIC)
        question = route.get("rewritten_query", "") or ""
        chunks_text = _format_chunks(effective_chunks[:4])

        prompt_text = val_prompt.format(
            intent=intent,
            rubric=rubric,
            question=question,
            chunks=chunks_text,
        )

        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Answer to validate:\n{answer}"},
        ]
        try:
            result = llm.chat_json(messages, temperature=0)
            verdict = result.get("verdict", "pass")
            llm_passed = result.get("passed", True)
            llm_issues = result.get("issues", [])
            reason = result.get("reason", "")
            suggestion = result.get("suggestion", "")

            if isinstance(llm_issues, list):
                issues.extend([i for i in llm_issues if isinstance(i, str)])

            if verdict in ("warn", "fail") or not llm_passed:
                effective_verdict = verdict if verdict in ("pass", "warn", "fail") else "warn"
                return {
                    "verdict": effective_verdict,
                    "passed": effective_verdict == "pass",
                    "issues": issues,
                    "reason": reason,
                    "suggestion": suggestion,
                }
        except Exception:
            pass

    # Citation issues only (no LLM ran or LLM returned pass)
    if issues:
        return {
            "verdict": "warn",
            "passed": False,
            "issues": issues,
            "reason": f"Citation issues found: {'; '.join(issues)}",
            "suggestion": "Check that cited artifact/chunk IDs exist in the ingested corpus.",
        }

    return {"verdict": "pass", "passed": True, "issues": [], "reason": "", "suggestion": ""}


def make_validation_node(llm: ChatLLM, system_prompt: str | None = None) -> Callable:
    """Return a LangGraph node that validates the generated answer."""
    use_llm = not isinstance(llm, FakeChatLLM)

    def validation_node(state: dict[str, Any]) -> dict[str, Any]:
        answer = state.get("answer", "")
        graded = state.get("graded_chunks", [])
        fallback = state.get("fallback_chunks", [])
        effective = graded or fallback

        if not answer or not effective:
            return {"validation_result": {
                "verdict": "pass", "passed": True, "issues": [], "reason": "", "suggestion": ""
            }}

        # Skip LLM grounding for fallback answers (already disclaimed)
        if state.get("is_fallback", False) or not use_llm:
            # Still run citation check
            valid_ids = {c.get("chunk_id", "") for c in effective}
            bad = [c for c in _extract_cited_ids(answer) if c not in valid_ids]
            if bad:
                return {"validation_result": {
                    "verdict": "warn",
                    "passed": False,
                    "issues": [f"Unresolved citations: {', '.join(bad)}"],
                    "reason": "Unresolved citation references in answer.",
                    "suggestion": "Check that cited artifact/chunk IDs exist in the ingested corpus.",
                }}
            return {"validation_result": {
                "verdict": "pass", "passed": True, "issues": [], "reason": "", "suggestion": ""
            }}

        route = state.get("route", {})
        result = validate_answer(llm, route, effective, answer, system_prompt=system_prompt)
        return {"validation_result": result}

    return validation_node
