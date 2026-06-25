# CLAUDE.md

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

The agentic RAG pipeline has four stages assembled as a LangGraph in `agent_graph.py`:

```
route → retrieve → rerank → generate
```

Supporting modules under `ciathena_kb/`:

- **`loader.py`** — parses each artifact YAML into an `Artifact(envelope, body)`. Files may use `---` separators; all YAML docs in a file are *merged* into one mapping, then split into envelope vs body by the known `ENVELOPE_KEYS` set. Validates against closed controlled vocabularies. Also provides `load_artifact_from_bytes()` for parsing from blob downloads.
- **`chunker.py`** — the **universal chunking rule**: one item under an artifact's body list = exactly one `Chunk`. Whole-doc types are **not** chunked. Embedded text = human-readable fields **plus recall boosters** (`synonyms`, `disambiguation_triggers`, `trigger_patterns`). Envelope fields become filterable metadata.
- **`embedder.py`** — `AzureOpenAIEmbedder` (default `text-embedding-3-large`) with a `FakeHashEmbedder` fallback. Uses per-service env vars (`AZURE_OPENAI_EMBEDDING_*`) with fallback to shared `AZURE_OPENAI_*`. **Critical invariant:** the model used to *ingest* must equal the model used to embed *queries*.
- **`llm.py`** — `AzureChatLLM` (default `gpt-4o`) with a `FakeChatLLM` fallback. Uses per-service env vars (`AZURE_OPENAI_CHAT_*`). Includes **retry logic** (3 attempts, exponential backoff) for transient Azure errors (500, 502, 503, 504, 429).
- **`store.py`** — Chroma vector store wrapper. **Metadata filters are applied BEFORE vector search** (pre-filtering). Supports General OR-merge (always retrieves General-layer alongside usecase-specific chunks) and governance gate (`review_status="approved"`). Detects Streamlit Cloud (`/mount/src` or `STREAMLIT_SHARING_MODE`) and uses in-memory `chromadb.Client()` directly; locally falls back to in-memory if `PersistentClient` fails.
- **`catalog.py`** — builds a routing catalog from artifact metadata (trigger_patterns, disambiguation_triggers, synonyms, item names) for injection into the router prompt.
- **`router_node.py`** — LLM-powered query router. Analyzes user question against the routing catalog and outputs `{in_domain, usecase, component_types, intent, rewritten_query}`. Generous with `in_domain` — only marks false for clearly unrelated topics. Accepts optional `system_prompt` override from `PromptManager`.
- **`retrieval_node.py`** — LangGraph retrieval node. Reads from `route` dict (agentic flow) or top-level state keys (backward compat for `demo.py`).
- **`rerank_node.py`** — post-retrieval filtering: cosine threshold + LLM relevance grading. Uses the **rewritten query** (not raw) for grading. Generous grading for broad questions. Accepts optional `system_prompt` override.
- **`generate_node.py`** — grounded answer generation with `[artifact_id::chunk_id]` citations. Refuses when no approved context survives filtering. Accepts optional `system_prompt` override.
- **`agent_graph.py`** — assembles the full LangGraph with conditional edge (in_domain check). Accepts optional `prompts` dict to pass prompt overrides to all nodes.
- **`blob_client.py`** — Azure Blob Storage client for artifact YAML files and prompt templates. `get_blob_client()` returns None when env vars missing (opt-in, same pattern as embedder). Artifacts stored under `artifacts/` prefix with timestamped version snapshots under `versions/`. Prompts stored under `prompts/` prefix.
- **`prompt_manager.py`** — manages pipeline prompt templates (router, rerank, generate). Loads from blob when available, falls back to built-in defaults. Prompts editable via Streamlit UI without redeploying.
- **`ingestion_log.py`** — tracks ingested artifacts in `.chroma/ingestion_log.json`. Smart re-ingestion: compares `content_version` + file hash (supports both local files and blob URIs via `_bytes_hash`).

### The artifact contract (most important domain concept)

Every artifact is one YAML doc = **envelope** (identity + retrieval-filtering + governance) + **body** (type-specific content). Two axes drive retrieval filtering:
- **`layer`**: `general` → `usecase` → `udm` → `client_facing` (general concepts are client-agnostic; higher layers `inherits_from` lower ones).
- **`component_type`**: `concept`, `methodology`, `process_flow`, `sttm_mapping`, `dq_rule`, `playbook`, `anomaly` (chunked) vs the whole-doc types above (not chunked).

The mapping from `component_type` → which body list to chunk, which fields are embeddable text, which are recall boosters, and the per-item id field all live as dicts at the top of `loader.py` and `chunker.py`. **Adding a new component_type means editing those dicts**, not just adding a file.

## Conventions

- Pure-Python, dependency-light. The loader/chunker need no LLM. Modules degrade gracefully when optional deps (`openai`, `langgraph`, `azure-storage-blob`) or Azure creds are absent.
- Per-service env var pattern: `AZURE_OPENAI_EMBEDDING_*` / `AZURE_OPENAI_CHAT_*` / `AZURE_BLOB_*`, each with fallback to shared prefix where applicable.
- Sample artifacts in `artifacts/` are illustrative for format validation, **not approved knowledge** — the field *structure* is fixed by the standard; the data team replaces the values.
