# CLAUDE.md

> **Version 0.7** — Working copy for Release 7 (2026-06-30)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The **ciATHENA Knowledge Spine** — an agentic RAG pipeline for a pharma life-sciences commercial-analytics platform. It answers domain questions (e.g. "what is gross-to-net", "what is MMM", "where should I move budget across channels") by retrieving from a corpus of governed YAML knowledge artifacts with LLM-powered routing, relevance grading, and grounded answer generation with citations.

This is **"Scenario B"**: there is no managed/Foundry agent. *Our own code* controls embedding, metadata filtering, retrieval, routing, and generation — exposed as a LangGraph pipeline.

## Layout note

The git/working root is `ciathena-poc/`, but the runnable project is the **nested** `ciathena-poc/ciathena-poc/` directory. All commands below run from there. The Word doc `ciATHENA_Developer_Scope_PoC.docx` at the outer root is the authoritative scope spec.

## Commands

```bash
cd ciathena-poc                          # the INNER project dir
pip install -r requirements.txt          # includes pyyaml, chromadb, openai, langgraph, streamlit, azure-storage-blob

# Ingest artifacts into Chroma (smart re-ingest skips unchanged)
python ingest.py                         # auto: blob if configured, else local artifacts/
python ingest.py --clear                 # wipe + re-ingest from scratch
python ingest.py --source blob           # force blob storage source
python ingest.py --source local          # force local artifacts/ directory

# Chat
python chat.py                           # interactive REPL
python chat.py --query "what is mmm"     # single query

# Streamlit demo
streamlit run app.py                     # full UI with upload, chat, ingestion log

# Legacy retrieval-only demo
python demo.py                           # end-to-end: load → chunk → embed → ingest → retrieve
```

- **Offline by default.** With no Azure env vars set, a deterministic fake-hash embedder and stub chat LLM are used so the whole pipeline runs with zero credentials. Retrieval *ordering* and generated answers are only meaningful with real Azure deployments.
- **Real embeddings:** set `AZURE_OPENAI_EMBEDDING_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_API_KEY`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` (see `.env.example`).
- **Real chat:** set `AZURE_OPENAI_CHAT_ENDPOINT`, `AZURE_OPENAI_CHAT_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT`. Embeddings and chat can live on **different Azure resources**.
- **Blob storage (optional):** set `AZURE_BLOB_CONNECTION_STRING` and `AZURE_BLOB_CONTAINER_NAME` to store/load artifacts from Azure Blob Storage instead of local `artifacts/`.
- There is **no test suite, linter, or build step** configured. Acceptance is defined behaviorally in the scope doc §6.

## Architecture

The agentic RAG pipeline has six stages assembled as a LangGraph in `agent_graph.py`:

```
route → expand_queries → retrieve → rerank (or skip) → generate → validation
```

Performance optimizations: conditional rerank bypass (skips LLM call when all top-k chunks have cosine >= 0.7 or score gap >= 0.1), compact routing catalog (merged triggers, truncated covers), embedding LRU cache (512 entries, avoids redundant Azure API calls), and progressive `st.status()` updates during pipeline execution.

Retrieval quality: intent-aware reranking (router's intent boosts matching component_types — definition→concept/methodology, advisory→playbook/process_flow), chunk deduplication (max 2 chunks per artifact_id for diverse context), multi-query expansion (3 query variations increase recall), self-query metadata filtering (router outputs `chroma_filter` dict validated against known keys).

Resilience: **soft fallback** — when no chunks clear `COSINE_THRESHOLD` (0.15), `rerank_node` sets `fallback_chunks` instead of returning nothing; `generate_node` uses these with a ⚠️ disclaimer rather than hard-declining. **Output validation** — `validation_node` checks citation existence and LLM grounding after generation. **Feedback loop** — thumbs up/down in Streamlit UI stored via `FeedbackStore`; repeated negatives on the same query invalidate the Q&A cache.

Supporting modules under `ciathena_kb/`:

- **`loader.py`** — parses each artifact YAML into an `Artifact(envelope, body)`. Files may use `---` separators; all YAML docs in a file are *merged* into one mapping, then split into envelope vs body by the known `ENVELOPE_KEYS` set. Validates against closed controlled vocabularies. Also provides `load_artifact_from_bytes()` for parsing from blob downloads.
- **`chunker.py`** — the **universal chunking rule**: one item under an artifact's body list = exactly one `Chunk`. Whole-doc types are **not** chunked. Embedded text = human-readable fields **plus recall boosters** (`synonyms`, `disambiguation_triggers`, `trigger_patterns`). Envelope fields become filterable metadata.
- **`embedder.py`** — `AzureOpenAIEmbedder` (default `text-embedding-3-large`) with a `FakeHashEmbedder` fallback. Uses per-service env vars (`AZURE_OPENAI_EMBEDDING_*`) with fallback to shared `AZURE_OPENAI_*`. **Critical invariant:** the model used to *ingest* must equal the model used to embed *queries*. Single-query embeddings are LRU-cached (512 entries) to avoid redundant Azure API calls on repeated queries.
- **`llm.py`** — `AzureChatLLM` with a `FakeChatLLM` fallback. Display label reads from `AZURE_OPENAI_CHAT_DEPLOYMENT` env var. Uses per-service env vars (`AZURE_OPENAI_CHAT_*`). Includes **retry logic** (3 attempts, exponential backoff) for transient Azure errors (500, 502, 503, 504, 429). Auto-strips `temperature` parameter for models that don't support it (e.g. reasoning/o-series). Provides `chat_stream()` for token-by-token streaming via Azure OpenAI's `stream=True`.
- **`store.py`** — Chroma vector store wrapper. **Metadata filters are applied BEFORE vector search** (pre-filtering). Supports General OR-merge (always retrieves General-layer alongside usecase-specific chunks) and governance gate (`review_status="approved"`). Detects Streamlit Cloud (`/mount/src` or `STREAMLIT_SHARING_MODE`) and uses in-memory `chromadb.Client()` directly; locally falls back to in-memory if `PersistentClient` fails.
- **`catalog.py`** — builds a routing catalog from artifact metadata (trigger_patterns, disambiguation_triggers, synonyms, item names) for injection into the router prompt. Compact mode (default) merges synonyms/disambiguation into triggers and truncates covers lists to reduce token count.
- **`router_node.py`** — LLM-powered query router. Analyzes user question against the routing catalog and outputs `{in_domain, usecase, component_types, intent, rewritten_query, chroma_filter}`. Generous with `in_domain` — only marks false for clearly unrelated topics. Accepts optional `system_prompt` override from `PromptManager`. Injects `conversation_history` from state so follow-up queries are resolved into self-contained rewritten queries. `chroma_filter` keys are validated against `VALID_FILTER_KEYS` before reaching the store.
- **`retrieval_node.py`** — LangGraph retrieval node. **Multi-query**: loops over `state["expanded_queries"]` when set, merges results by `chunk_id`. **Self-query**: uses `route["chroma_filter"]` when present instead of building filters from `usecase`/`component_types`. Also supports direct calls (top-level state keys, for `demo.py` compat).
- **`rerank_node.py`** — post-retrieval filtering with adaptive strategy: chunks with cosine >= `HIGH_CONFIDENCE_THRESHOLD` (0.7) auto-pass; when score gap between top-k and (k+1)-th chunk exceeds `SCORE_GAP_THRESHOLD` (0.1), returns top-k directly without LLM; otherwise batch-grades borderline chunks (0.15–0.7) in a single LLM call. Uses the **rewritten query** (not raw) for grading. Generous grading for broad questions. Accepts optional `system_prompt` override. **Intent-aware ranking**: applies `INTENT_BOOST` (0.03) to chunks whose `component_type` matches the router's intent. **Chunk deduplication**: limits to 2 chunks per `artifact_id`. **Soft fallback**: when `above_threshold` is empty, returns `{"graded_chunks": [], "fallback_chunks": <top-k sorted by score>}` instead of an empty result.
- **`generate_node.py`** — grounded answer generation with `[artifact_id::chunk_id]` citations. When `graded_chunks` is empty but `fallback_chunks` is set, generates with a ⚠️ disclaimer prefix and sets `is_fallback=True`. Refuses when neither is available. Accepts optional `system_prompt` override. Also provides `make_stream_generate()` for token-by-token streaming in the Streamlit UI (accepts `fallback_chunks` parameter). Both paths inject `conversation_history` for coherent multi-turn answers.
- **`query_expander_node.py`** — LLM generates 3 semantically diverse query variations from the rewritten query. Sets `state["expanded_queries"]` for multi-query retrieval. Skips when `FakeChatLLM` or out-of-domain.
- **`validation_node.py`** — post-generation quality check. (1) Citation existence check (no LLM): flags `[artifact::chunk]` refs not in the effective context. (2) LLM grounding check: flags claims not supported by the chunks. Sets `state["validation_result"] = {"passed": bool, "issues": list}`. Skipped for fallback answers.
- **`agent_graph.py`** — assembles the full LangGraph with conditional edges: in_domain check (skip generation if out-of-domain) and rerank bypass (skip LLM grading when all top-k chunks are high-confidence). Accepts optional `prompts` dict to pass prompt overrides to all nodes. Also provides `build_pre_generate_graph()` that stops after rerank (used with streaming). Streamlit UI uses `graph.stream()` for progressive status updates.
- **`blob_client.py`** — Azure Blob Storage client for artifact YAML files and prompt templates. `get_blob_client()` returns None when env vars missing (opt-in, same pattern as embedder). Artifacts stored under `artifacts/` prefix with timestamped version snapshots under `versions/`. Prompts stored under `prompts/` prefix.
- **`prompt_manager.py`** — manages pipeline prompt templates (router, rerank, generate). Loads from blob when available, falls back to built-in defaults. Prompts editable via Streamlit UI without redeploying. Router prompt includes `chroma_filter` field in JSON schema (instruction 7).
- **`ingestion_log.py`** — tracks ingested artifacts in `.chroma/ingestion_log.json`. Smart re-ingestion: compares `content_version` + file hash (supports both local files and blob URIs via `_bytes_hash`).
- **`qa_cache.py`** — session-scoped Q&A result cache. Keyed by normalized query string (lowercase, whitespace-collapsed). Uses generation-based invalidation: bumping a counter makes all entries stale without walking the cache. FIFO eviction at `max_entries` (default 100). Caches all standalone queries regardless of conversation position; vague follow-ups ("tell me more", "explain that") are detected by `is_followup_query()` heuristic and skipped.
- **`chat_history.py`** — persistent chat history storage. `ChatHistoryStore(session_id, blob_client=None)` saves conversation messages to a JSON file keyed by session ID. Storage: blob `chat_history/{session_id}.json` or local `.chroma/chat_history/{session_id}.json`. Survives page reloads via `st.query_params["session"]` URL parameter. MAX_MESSAGES=200 with FIFO truncation. Methods: `append(message)`, `clear()`, `delete()`, `list_sessions()`.
- **`feedback_store.py`** — stores user thumbs up/down feedback per assistant message. `FeedbackStore(session_id, blob_client=None)` persists to blob `feedback/{session_id}.json` or local `.chroma/feedback/{session_id}.json`. MAX_ENTRIES=500. `should_invalidate_cache(query)` returns True when a query accumulates `NEGATIVE_THRESHOLD` (3) negative ratings — triggers `QACache.invalidate()` so the next request re-runs the full pipeline.

### The artifact contract (most important domain concept)

Every artifact is one YAML doc = **envelope** (identity + retrieval-filtering + governance) + **body** (type-specific content). Two axes drive retrieval filtering:
- **`layer`**: `general` → `usecase` → `udm` → `client_facing` (general concepts are client-agnostic; higher layers `inherits_from` lower ones).
- **`component_type`**: `concept`, `methodology`, `process_flow`, `sttm_mapping`, `dq_rule`, `playbook`, `anomaly`, `dataset_catalog` (chunked) vs the whole-doc types above (not chunked).

The mapping from `component_type` → which body list to chunk, which fields are embeddable text, which are recall boosters, and the per-item id field all live as dicts at the top of `loader.py` and `chunker.py`. **Adding a new component_type means editing those dicts**, not just adding a file.

## Conventions

- Pure-Python, dependency-light. The loader/chunker need no LLM. Modules degrade gracefully when optional deps (`openai`, `langgraph`, `azure-storage-blob`) or Azure creds are absent.
- Per-service env var pattern: `AZURE_OPENAI_EMBEDDING_*` / `AZURE_OPENAI_CHAT_*` / `AZURE_BLOB_*`, each with fallback to shared prefix where applicable.
- Sample artifacts in `artifacts/` are illustrative for format validation, **not approved knowledge** — the field *structure* is fixed by the standard; the data team replaces the values.

## Documentation rule (MANDATORY)

**Always update `CLAUDE.md` and `ciathena-poc/README.md` before every commit.**

- `CLAUDE.md` — update the version header date, architecture section, and relevant module descriptions to reflect any new/modified modules or pipeline changes.
- `README.md` — update the version header date, "Agentic RAG flow" diagram, "Streamlit demo" features list, "What this PoC covers" paragraph, and add a changelog entry for the new version.

This rule applies to every code change, no matter how small. The two files must always reflect the current state of the codebase.
