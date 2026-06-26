"""
ciathena_kb.router_node
-----------------------
LLM-powered query router. Analyzes the raw user question and decides:
  - in_domain (bool): is this answerable from the corpus?
  - usecase (str): which usecase to filter on (or "General")
  - component_types (list[str]): soft filter suggestions
  - intent (str): definition / how-to / advisory / comparison
  - rewritten_query (str): cleaned/expanded query for embedding search
"""

from __future__ import annotations

from typing import Any

from .llm import ChatLLM

from .prompt_manager import DEFAULT_PROMPTS

ROUTER_SYSTEM_PROMPT = DEFAULT_PROMPTS["router_system"]


def make_router_node(llm: ChatLLM, catalog: str, system_prompt: str | None = None):
    """Return a LangGraph node function that routes the user query."""
    prompt_template = system_prompt or ROUTER_SYSTEM_PROMPT

    def router_node(state: dict[str, Any]) -> dict[str, Any]:
        user_query = state["user_query"]

        messages = [
            {"role": "system", "content": prompt_template.format(catalog=catalog)},
        ]
        history = state.get("conversation_history", [])
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        route = llm.chat_json(messages, temperature=0)

        return {
            "route": {
                "in_domain": route.get("in_domain", True),
                "usecase": route.get("usecase", "General"),
                "component_types": route.get("component_types", []),
                "intent": route.get("intent", "definition"),
                "rewritten_query": route.get("rewritten_query", user_query),
            },
        }

    return router_node
