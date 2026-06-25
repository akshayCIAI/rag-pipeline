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

ROUTER_SYSTEM_PROMPT = """\
You are the query router for the ciATHENA Knowledge Spine, a pharma life-sciences
commercial analytics platform. Your job is to analyze a user question and decide
how to retrieve relevant knowledge artifacts.

You have access to this routing catalog describing the available knowledge:

{catalog}

INSTRUCTIONS:
1. Determine if the question is answerable from the corpus above (in_domain).
   IMPORTANT: Be GENEROUS with in_domain — if the question relates to ANY usecase
   listed in the catalog, or to pharma / life-sciences / commercial analytics
   topics that COULD be covered by the loaded artifacts, set in_domain=true.
   Only set in_domain=false for questions clearly outside pharma commercial
   analytics (e.g. cooking recipes, sports scores, general trivia).
   When in doubt, set in_domain=true and let retrieval decide relevance.
2. Identify the most relevant usecase. If the question mentions or relates to a
   specific usecase (e.g. "MMM", "media mix modeling"), return that usecase.
   If it's about a general pharma concept, return "General".
3. Suggest component_types to soft-filter. Only include types you're confident
   about — an empty list means "search all types."
4. Classify the intent: "definition" (what is X), "how-to" (how do I do X),
   "advisory" (what should I do / where should I move), "comparison" (X vs Y).
5. Rewrite the query to be more specific and embedding-friendly. Expand
   abbreviations, add domain synonyms where helpful.

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "in_domain": true/false,
  "usecase": "General" or specific usecase,
  "component_types": [],
  "intent": "definition|how-to|advisory|comparison",
  "rewritten_query": "expanded query text"
}}
"""


def make_router_node(llm: ChatLLM, catalog: str):
    """Return a LangGraph node function that routes the user query."""

    def router_node(state: dict[str, Any]) -> dict[str, Any]:
        user_query = state["user_query"]

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(catalog=catalog)},
            {"role": "user", "content": user_query},
        ]

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
