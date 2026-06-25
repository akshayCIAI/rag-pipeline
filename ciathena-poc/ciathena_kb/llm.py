"""
ciathena_kb.llm
---------------
Azure OpenAI chat client for the agentic layer (router + answer generation).

Reads AZURE_OPENAI_CHAT_* env vars first, falls back to shared AZURE_OPENAI_*.
When no creds are set, returns a FakeChatLLM that echoes a stub response so the
pipeline wiring is demonstrable offline.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]
RETRYABLE_STATUS_CODES = {500, 502, 503, 504, 429}


def _chat_env(key: str) -> str | None:
    return os.environ.get(f"AZURE_OPENAI_CHAT_{key}") or os.environ.get(f"AZURE_OPENAI_{key}")


class ChatLLM(Protocol):
    model_name: str
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...
    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict: ...


class AzureChatLLM:
    """Azure OpenAI chat completions. Reads:
        AZURE_OPENAI_CHAT_ENDPOINT    (or AZURE_OPENAI_ENDPOINT)
        AZURE_OPENAI_CHAT_API_KEY     (or AZURE_OPENAI_API_KEY)
        AZURE_OPENAI_CHAT_DEPLOYMENT
    Optional:
        AZURE_OPENAI_CHAT_API_VERSION (default 2024-02-01)
    """

    def __init__(self, model_name: str = "gpt-4o"):
        from openai import AzureOpenAI

        self.model_name = model_name
        self._deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        self._client = AzureOpenAI(
            azure_endpoint=_chat_env("ENDPOINT"),
            api_key=_chat_env("API_KEY"),
            api_version=_chat_env("API_VERSION") or "2024-02-01",
        )

    def _call_with_retry(self, **create_kwargs: Any) -> Any:
        from openai import APIStatusError, BadRequestError
        for attempt in range(MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**create_kwargs)
            except BadRequestError as e:
                if "temperature" in str(e) and "temperature" in create_kwargs:
                    print(f"  [llm] Model does not support temperature param, retrying without it...")
                    create_kwargs.pop("temperature", None)
                    return self._client.chat.completions.create(**create_kwargs)
                raise
            except APIStatusError as e:
                if e.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    print(f"  [llm] Azure returned {e.status_code}, retrying in {wait}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        resp = self._call_with_retry(
            model=self._deployment, messages=messages, **kwargs,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        resp = self._call_with_retry(
            model=self._deployment, messages=messages,
            response_format={"type": "json_object"}, **kwargs,
        )
        return json.loads(resp.choices[0].message.content or "{}")


class FakeChatLLM:
    """Offline stub. Returns a canned response so the pipeline wiring is testable
    without credentials. NOT useful for real answers."""

    def __init__(self, model_name: str = "fake-chat-stub"):
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "[offline stub] No chat LLM configured. Set AZURE_OPENAI_CHAT_* env vars."

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        return {
            "in_domain": True,
            "usecase": "General",
            "component_types": [],
            "intent": "definition",
            "rewritten_query": messages[-1].get("content", "") if messages else "",
        }


def get_chat_llm(model_name: str = "gpt-4o") -> ChatLLM:
    """Return Azure chat LLM if env is configured, else the fake stub."""
    has_deployment = bool(os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"))
    has_endpoint = bool(_chat_env("ENDPOINT"))
    has_key = bool(_chat_env("API_KEY"))
    if has_deployment and has_endpoint and has_key:
        try:
            return AzureChatLLM(model_name=model_name)
        except Exception as exc:
            print(f"[llm] Azure chat init failed ({exc}); using fake stub.")
    else:
        print("[llm] Azure chat env not set; using offline stub. "
              "Set AZURE_OPENAI_CHAT_* for real routing and answers.")
    return FakeChatLLM()
