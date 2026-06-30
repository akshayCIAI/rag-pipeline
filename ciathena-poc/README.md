# ciATHENA Knowledge Spine — Domain Intelligence Agent (Scenario B)

> **v0.7** — Working copy for Release 7 (2026-06-30)

An agentic RAG pipeline for the ciATHENA Knowledge Spine: load governed YAML
artifacts, embed, ingest into a local Chroma vector DB, and answer pharma
life-sciences commercial analytics questions with **LLM-powered query routing,
metadata pre-filtering, relevance grading, and grounded answer generation with
citations**.

This is **Scenario B**: no managed/Foundry agent. Our own code controls
embedding, metadata filtering, retrieval, routing, and generation — exposed as
a LangGraph pipeline.

## Layout

```
ciathena-poc/
  ciathena_kb/              # core package
    loader.py               # parse + validate artifact YAML (envelope + body)
    chunker.py              # one list-item -> one chunk (the universal rule)
    embedder.py             # Azure OpenAI embedder + offline fake fallback
    store.py                # Chroma vector store: ingest + metadata-filtered retrieval (persistent or in-memory fallback)
    llm.py                  # Azure OpenAI chat client + offline stub + retry logic
    catalog.py              # builds routing catalog from artifact metadata
    router_node.py          # LLM query router (infers usecase, component_type, intent)
    retrieval_node.py       # LangGraph retrieval node (metadata pre-filter + General OR-merge)
    rerank_node.py          # post-retrieval relevance grading + threshold; soft fallback when no chunks clear threshold
    generate_node.py        # grounded answer generation with citations; fallback generation with ⚠️ disclaimer
    query_expander_node.py  # LLM generates 3 query variations for multi-query retrieval
    validation_node.py      # citation existence check + LLM grounding check after generation
    agent_graph.py          # LangGraph: route → expand_queries → retrieve → rerank → generate → validation
    ingestion_log.py        # version-aware ingestion tracking (skip unchanged artifacts)
    blob_client.py          # Azure Blob Storage client for artifacts + prompts (optional, versioned)
    prompt_manager.py       # prompt template manager (blob-backed with built-in defaults)
    qa_cache.py             # session-scoped Q&A result cache (generation-based invalidation)
    chat_history.py         # persistent chat history storage (blob or local, keyed by session ID)
    feedback_store.py       # thumbs up/down feedback storage; triggers cache invalidation on repeated negatives
  artifacts/                # 3 sample artifacts (concept, methodology, playbook)
  ingest.py                 # CLI: smart re-ingest + blob/local source + --clear flag
  chat.py                   # CLI: interactive or single-query agentic Q&A (with history)
  app.py                    # Streamlit demo: chat + upload + prompt editor + ingestion log + Q&A cache
  demo.py                   # original retrieval-only end-to-end runner
  .env.example              # env var template — copy to .env
  requirements.txt
```

## Quick start

```bash
cd ciathena-poc
pip install -r requirements.txt

# 1. Copy and fill in your Azure credentials
cp .env.example .env        # then edit .env with your values

# 2. Ingest artifacts into persistent Chroma
python ingest.py            # smart re-ingest (skips unchanged artifacts)
python ingest.py --clear    # wipe + re-ingest from scratch

# 3. Chat (CLI or Streamlit)
python chat.py                                    # interactive REPL
python chat.py --query "what is gross to net"     # single query
streamlit run app.py                              # Streamlit demo UI
```

## Environment variables

Copy `.env.example` to `.env`. Embeddings and chat can live on **different Azure
resources** — each has its own endpoint/key/deployment vars.

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | Azure OpenAI endpoint for embeddings |
| `AZURE_OPENAI_EMBEDDING_API_KEY` | API key for embeddings resource |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment name (e.g. `ciathena-text-embedding-3-large`) |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | Optional, defaults to `2024-02-01` |
| `AZURE_OPENAI_CHAT_ENDPOINT` | Azure OpenAI endpoint for chat model |
| `AZURE_OPENAI_CHAT_API_KEY` | API key for chat resource |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_CHAT_API_VERSION` | Optional, defaults to `2024-02-01` |
| `AZURE_BLOB_CONNECTION_STRING` | Azure Blob Storage connection string (optional) |
| `AZURE_BLOB_CONTAINER_NAME` | Blob container name (default `ciathena-artifacts`) |
| `CHROMA_PERSIST_DIR` | Chroma storage path (default `./.chroma`) |
| `RETRIEVAL_CANDIDATE_POOL` | Chunks retrieved before rerank (default `12`) |
| `RETRIEVAL_TOP_K` | Chunks kept after rerank (default `4`) |
| `HISTORY_MAX_TURNS` | Q&A turns kept for follow-up context (default `5`) |

With **no env vars set**, the pipeline runs offline using a deterministic fake
embedder and a stubbed chat LLM — useful for wiring/smoke tests. Retrieval
ordering and generated answers are only meaningful with real Azure deployments.

> **Scenario B rule:** the embedding model used to ingest documents MUST equal
> the model used to embed live queries at runtime. The model name is recorded
> on every chunk and artifact envelope so a mismatch is detectable.

## Agentic RAG flow

```
user_query
    │
    ▼
1. ROUTER (LLM + routing catalog)
   → in_domain? · usecase · component_type(s) · intent · rewritten_query · chroma_filter
    │
    ▼ (in_domain=false → decline immediately)
2. EXPAND QUERIES (LLM)
   → 3 semantically diverse query variations for wider recall
   → skipped when FakeChatLLM (offline mode)
    │
    ▼
3. RETRIEVE (persistent Chroma)
   self-query: uses chroma_filter from router if present
   else builds filter from: usecase ∈ {chosen, General}  (OR-merge)
                             review_status = "approved"   (governance)
                             component_type ∈ chosen      (soft)
   multi-query: loops over all expanded queries, merges by chunk_id
    │
    ▼
4. RERANK / DOCUMENT-FILTER
   high-confidence chunks (cosine ≥ 0.7) auto-pass
   score-gap skip: top-k returned directly when gap ≥ 0.1
   borderline chunks batch-graded in 1 LLM call → top-k
   soft fallback: if no chunks clear threshold (0.15), returns
                  fallback_chunks (best available low-score chunks)
    │
    ▼
5. GENERATE (LLM, streamed)
   grounded answer + citations [artifact_id::chunk_id]
   tokens stream to UI in real time
   if fallback_chunks used: prepends ⚠️ disclaimer, sets is_fallback=True
   refuses when neither graded nor fallback chunks available
    │
    ▼
6. VALIDATION (chat.py only, not in streaming Streamlit path)
   citation existence check: verifies [artifact::chunk] refs match context
   LLM grounding check: flags claims not supported by provided chunks
   sets validation_result = {"passed": bool, "issues": list}
```

The router uses a **routing catalog** built at startup from artifact metadata
(trigger patterns, disambiguation triggers, AI routing notes, synonyms, item
names) — so the LLM routes with precision, not guesswork.

## Storage architecture

| Component | Stores | Purpose |
|---|---|---|
| **Azure Blob Storage** (optional) | YAML artifacts (versioned) + prompt templates | Shared cloud repository with timestamp-based version history |
| **ChromaDB** (local) | Vector embeddings + metadata | Similarity search to find relevant chunks |
| **Azure OpenAI** | — | LLM for routing, grading, and answer generation |

Without blob vars set, artifacts are loaded from the local `artifacts/` directory.

## CLI commands

### `ingest.py` — smart re-ingest with version tracking

```bash
python ingest.py                          # auto: blob if configured, else local
python ingest.py --source blob            # force blob storage source
python ingest.py --source local           # force local artifacts/ directory
python ingest.py --artifacts-dir ./my_dir # custom local artifacts directory
python ingest.py --clear                  # wipe everything and re-ingest from scratch
```

Smart re-ingestion compares each artifact's `content_version` field and file hash
against the ingestion log (`.chroma/ingestion_log.json`). Unchanged artifacts are
skipped, saving embedding cost.

### `chat.py` — query the knowledge base

```bash
python chat.py                                          # interactive REPL
python chat.py --query "what is gross to net"           # single query
python chat.py -q "where should I move budget"          # short flag
```

### `demo.py` — retrieval-only demo (original)

```bash
python demo.py    # end-to-end: load → chunk → embed → ingest → retrieve
```

## Sample artifacts

Three artifacts exercise different body shapes and cross-usecase filtering:

| File | component_type | usecase | Chunks |
|---|---|---|---|
| `gen-concept-pharma-001.yml` | concept | General | TRx, NBRx, Gross-to-Net |
| `mmm-methodology-core-001.yml` | methodology | MMM | adstock, saturation, model selection |
| `mmm-playbook-roi-001.yml` | playbook | MMM | ROI by channel, saturation check |

These are illustrative content for format validation, not approved knowledge.
The field structure is fixed by the artifact standard; the data team replaces
the values.

## Integrate the retrieval node into your existing graph

```python
from ciathena_kb import (
    load_artifacts, chunk_all, get_embedder,
    KnowledgeStore, make_retrieval_node,
)

store = KnowledgeStore(get_embedder())
store.ingest(chunk_all(load_artifacts("artifacts")))

retrieval_node = make_retrieval_node(store)
graph.add_node("retrieve", retrieval_node)
```

Or use the full agentic graph:

```python
from ciathena_kb import (
    load_artifacts, get_embedder, get_chat_llm,
    KnowledgeStore, build_agent_graph,
)

artifacts = load_artifacts("artifacts")
store = KnowledgeStore(get_embedder())
llm = get_chat_llm()
graph = build_agent_graph(store=store, artifacts=artifacts, llm=llm)

result = graph.invoke({"user_query": "what is gross to net"})
print(result["answer"])
```

## Streamlit demo (`app.py`)

```bash
streamlit run app.py
```

Features:
- **Chat interface** — ask questions, see answers streamed in real time with citations, routing details, and retrieved chunks; supports follow-up questions with conversation history (last 5 turns); persistent chat history survives page reloads via URL session ID; "New conversation" button starts a fresh session
- **Feedback buttons** — 👍/👎 after every answer (current and historical); ratings persisted to `.chroma/feedback/` or blob; 3 dislikes on the same query automatically invalidates the Q&A cache
- **Soft fallback display** — when no high-confidence chunks found, answer shows ⚠️ disclaimer; chunks expander shows "Fallback context" header with the low-score chunks used
- **Routing debug expander** — shows usecase, intent, rewritten query, metadata filter (`chroma_filter`), and expanded query variations per response
- **Upload artifacts** — drag-and-drop `.yml`/`.yaml` files in the sidebar; auto-validates and smart-ingests (skips unchanged, only embeds new/modified)
- **Auto-ingest on startup** — pulls all artifacts from blob on boot, compares hashes, only embeds what changed (saves embedding cost)
- **Re-ingest all** — one-click wipe + re-ingest from the sidebar
- **Prompt management** — edit router, rerank, and generate prompts directly in the sidebar; saves to blob so changes persist without redeploying
- **Ingestion log** — expandable entries showing version, type, layer, chunk count, hash, source (blob/local), timestamp
- **Status panel** — shows embedder/LLM/blob connection status, model names, storage mode, chunk count
- **Graceful error handling** — shows friendly message when Azure is temporarily unavailable instead of crashing

## Artifact upload + versioning

Teams can add new YAML artifacts at any time:

1. **CLI**: drop `.yml` files into `artifacts/`, run `python ingest.py` — only new/changed files are embedded
2. **Streamlit**: drag-drop files in the sidebar upload panel — validates, saves to blob (or local), smart-ingests (skips unchanged)
3. **Blob storage**: when configured, all uploaded artifacts are stored in Azure Blob Storage for team-wide access

### Blob storage layout

```
ciathena-artifacts/           # container
  artifacts/                  # current versions (used for ingestion)
    gen-concept-pharma-001.yml
    mmm-methodology-core-001.yml
  versions/                   # timestamped snapshots (audit trail)
    gen-concept-pharma-001.yml/
      20260625_043000_gen-concept-pharma-001.yml
      20260625_091500_gen-concept-pharma-001.yml
    mmm-methodology-core-001.yml/
      20260625_043000_mmm-methodology-core-001.yml
  prompts/                    # editable prompt templates
    router_system.txt
    rerank_grading.txt
    generate_system.txt
```

Every upload saves the file under `artifacts/` (the "current" copy used for ingestion) and
also creates a timestamped snapshot under `versions/<filename>/` for audit trail.

### Smart re-ingestion

The ingestion log (`.chroma/ingestion_log.json`) tracks every artifact's `content_version`,
file content hash, chunk count, and embedding model. On startup and on upload, the pipeline
compares hashes — only new or changed artifacts are embedded, saving Azure OpenAI cost.

## Prompt management

Pipeline prompts (router, rerank, generate) can be edited in the Streamlit sidebar without
code changes or redeployment:

1. Open the **Prompt Management** section in the sidebar
2. Edit the prompt text in the text area
3. Click **Save** — the prompt is stored in blob and used immediately
4. Click **Reset to default** to restore the built-in prompt

When blob is not configured, edits apply to the current session only. Prompts fall back to
built-in defaults when no blob copy exists.

## Resilience

- **LLM retry logic**: transient Azure errors (500, 502, 503, 504, 429) are retried up to 3 times with exponential backoff (2s, 5s, 10s) before failing; auto-strips `temperature` parameter for models that don't support it (e.g. reasoning/o-series)
- **Graceful degradation**: no blob vars = local mode, no Azure creds = offline mode with fake embedder/LLM
- **ChromaDB fallback**: `PersistentClient` is used locally; on Streamlit Cloud (detected via `/mount/src` or `STREAMLIT_SHARING_MODE`), uses in-memory `chromadb.Client()` directly to avoid Rust-binding initialization failures
- **Artifact validation**: invalid YAML files (wrong `component_type`, missing fields, malformed YAML syntax) are skipped with a warning instead of crashing the app; skipped files shown in the sidebar
- **Streamlit error handling**: Azure outages show a friendly warning instead of a traceback crash
- **Adaptive rerank**: rerank step uses a single batched LLM call instead of one per chunk; chunks with cosine >= 0.7 auto-pass; when score gap between top-k and next chunk exceeds 0.1, returns top-k directly without LLM call; conditional graph edge bypasses rerank node entirely when all top chunks are high-confidence — saves ~2s per query in clear-cut cases
- **Embedding cache**: single-query embeddings are LRU-cached (512 entries) so repeated/similar queries skip the Azure embedding API call entirely (~1.5s savings)
- **Compact routing catalog**: merges synonyms/disambiguation into trigger_patterns, truncates covers lists to top-5 per artifact — reduces router prompt token count for faster LLM response
- **Progressive status updates**: pipeline stages show real-time progress via `st.status()` ("Routing query..." → "Retrieving chunks..." → "Grading relevance..." → "Generating answer...") using LangGraph's `stream()` mode instead of a static spinner
- **Streaming answer generation**: answer tokens stream to the Streamlit UI in real time via `st.write_stream()` instead of waiting for the full response
- **Follow-up questions**: conversation history (last 5 Q&A turns) is passed to the router and generator so the pipeline can resolve references like "tell me more" or "compare that with X"; the router rewrites follow-ups into self-contained queries for retrieval
- **Q&A caching**: session-scoped cache for pipeline results; all standalone queries are cached regardless of conversation position — repeated identical questions return instantly; vague follow-ups ("tell me more") are detected and excluded from cache; invalidated automatically on artifact upload, re-ingest, or prompt edits; cache stats shown in sidebar
- **Persistent chat history**: conversations stored as JSON files keyed by session ID (blob or local); session ID persisted in URL query params so history survives page reloads; "New conversation" button starts a fresh session

## What this PoC covers (and does not)

**Covers:** artifact parsing, validation, one-item-per-chunk chunking,
embedding, persistent Chroma ingest, metadata pre-filtering (usecase +
component_type + review_status), General-layer OR-merge, LLM query routing
with self-query metadata filter extraction, multi-query expansion (3
variations per query), LLM relevance grading, intent-aware reranking, chunk
deduplication, soft fallback retrieval (⚠️ disclaimer answers when no
high-confidence chunks found), grounded answer generation with citations,
output validation (citation existence + LLM grounding check), user feedback
(👍/👎) with cache invalidation loop, graceful refusal on out-of-domain
queries, smart re-ingestion with version tracking, Azure Blob Storage
integration (versioned artifacts + prompt templates), auto-ingest on startup,
LLM retry logic, blob-backed prompt management, streaming answer generation,
conversation history with follow-up support, session-scoped Q&A caching, and
Streamlit demo UI with artifact upload, prompt editor, and feedback buttons.

**Does not cover:** NL-to-SQL, summarization, visualization (downstream nodes),
encryption-at-rest, CI/CD image delivery, self-containment hardening. Those
are later phases per the PoC plan.

## Changelog

### v0.7 — Working copy for Release 7 (2026-06-30)

- **Soft fallback retrieval** — when no chunks clear `COSINE_THRESHOLD` (0.15), `rerank_node` returns `fallback_chunks` (best available low-score chunks) instead of an empty result; `generate_node` uses them with a ⚠️ disclaimer prefix (`is_fallback=True`) rather than hard-declining; Streamlit chunks expander shows "Fallback context" header
- **Multi-query retrieval** — new `query_expander_node` generates 3 semantically diverse query variations via LLM; `retrieval_node` loops over all queries and merges results by `chunk_id` (deduplication), increasing the candidate pool from ~10 to ~30–40 unique chunks before reranking; skips when `FakeChatLLM` (offline mode)
- **Self-query metadata filtering** — router LLM now outputs a `chroma_filter` dict (e.g. `{"component_type": "playbook", "review_status": "approved"}`); `retrieval_node` uses it directly as the Chroma pre-filter when present, bypassing the default usecase/component_type logic; keys validated against known metadata fields before use
- **Output validation** — new `validation_node` runs after `generate_node` in `build_agent_graph()` (used by `chat.py`): (1) citation existence check — flags `[artifact::chunk]` refs not in effective context (no LLM cost); (2) LLM grounding check — flags claims unsupported by provided chunks; sets `validation_result = {"passed": bool, "issues": list}`
- **Feedback agent** — new `FeedbackStore` class (same blob/local pattern as `ChatHistoryStore`) persists thumbs up/down ratings per message; `should_invalidate_cache(query)` returns True after 3 negative ratings on the same query, triggering `QACache.invalidate()`; 👍/👎 buttons rendered in Streamlit for current and historical messages
- **LangGraph topology updated** — full graph now: `route → expand_queries → retrieve → rerank → generate → validation`; `build_pre_generate_graph()` (Streamlit streaming): `route → expand_queries → retrieve → rerank`
- **Routing debug improvements** — Streamlit routing expander now shows `chroma_filter` (when set) and expanded query variations per response
- **3 new modules** — `ciathena_kb/query_expander_node.py`, `ciathena_kb/validation_node.py`, `ciathena_kb/feedback_store.py`

### v0.6 — Working copy for Release 6 (2026-06-28)

- **Intent-aware reranking** — router's intent (definition/how-to/advisory/comparison) boosts matching component types via `INTENT_BOOST` (0.03); definition queries surface concept/methodology chunks, advisory queries surface playbook/process_flow chunks
- **Chunk deduplication** — limits to max 2 chunks per `artifact_id` across all rerank paths (including skip-rerank bypass) to ensure diverse context in the generation window
- **Cleaner output** — removed timing display (`⏱️`) from chat responses for a more business-oriented presentation
- **Unified finalization** — all rerank code paths (LLM grading, high-confidence bypass, score-gap bypass, skip-rerank) now apply the same intent-aware sorting and dedup logic via shared `_finalize()` helper

### v0.5 — Working copy for Release 5 (2026-06-28)

- **Persistent chat history** — conversations survive page reloads; messages stored in JSON files keyed by session ID (blob or local `.chroma/chat_history/`); session ID persisted in URL query params (`?session=...`); "New conversation" button in sidebar starts a fresh session; MAX_MESSAGES=200 with FIFO truncation
- **Response time optimizations** — conditional rerank bypass via LangGraph conditional edge (skips rerank node entirely when all top-k chunks have cosine >= 0.7); adaptive score-gap reranking (skips LLM call when gap between top-k and next chunk exceeds 0.1); embedding LRU cache (512 entries, avoids redundant Azure API calls); compact routing catalog (merged triggers, truncated covers for fewer prompt tokens)
- **Progressive status updates** — replaced static spinner with `st.status()` showing per-stage labels ("Routing query..." → "Retrieving chunks..." → "Grading relevance..." → "Generating answer...") using LangGraph's `graph.stream(stream_mode="updates")`
- **Intent-aware reranking** — router's intent (definition/how-to/advisory/comparison) boosts matching component types so definition queries surface concept chunks and advisory queries surface playbook chunks
- **Chunk deduplication** — limits to 2 chunks per artifact to ensure the generation context window has diverse information instead of redundant content from the same source
- **Cleaner output** — removed timing display from chat responses for a more business-oriented presentation

### v0.4 — Working copy for Release 4 (2026-06-26)

- **Batch rerank optimization** — rerank step uses a single batched LLM call instead of one-per-chunk, reducing LLM calls from ~8 to 1; chunks with cosine score >= 0.7 skip LLM grading entirely
- **Streaming answer generation** — answer tokens stream to the Streamlit UI in real time via `st.write_stream()` instead of waiting for the full response
- **Follow-up question support** — conversation history (last 5 Q&A turns) injected into router and generator; router rewrites follow-ups into self-contained queries for retrieval
- **Session-scoped Q&A cache** — standalone queries cached regardless of conversation position; vague follow-ups detected by `is_followup_query()` heuristic and excluded; generation-based invalidation on artifact upload, re-ingest, or prompt edits; FIFO eviction at 100 entries; cache stats shown in sidebar
- **Cache bug fixes** — fixed cache never serving hits after the first turn; fixed only first-turn queries being cached
- **`dataset_catalog` component type** — new chunked artifact type for dataset navigation; body key `datasets`, items chunked by `dataset_ref`, embeds display name, role, grain, vendor, known gaps with business aliases and disambiguation triggers as recall boosters
- **Persistent chat history** — conversations survive page reloads; messages stored in JSON files keyed by session ID (blob or local `.chroma/chat_history/`); session ID persisted in URL query params (`?session=...`); "New conversation" button in sidebar starts a fresh session; MAX_MESSAGES=200 with FIFO truncation
- **Response time optimizations** — conditional rerank bypass (skips LLM call when top-k chunks are all high-confidence or score gap is clear), adaptive score-gap reranking, embedding LRU cache (512 entries), compact routing catalog (fewer prompt tokens), progressive `st.status()` updates replacing static spinner

### v0.3 — Working copy for Release 3

- Azure Blob Storage integration (versioned artifacts + prompt templates)
- Blob-backed prompt management (edit router/rerank/generate prompts via Streamlit sidebar)
- Auto-ingest on startup from blob
- Smart re-ingestion with version tracking and content hashing
- Ingestion log with expandable entries in sidebar
- LLM retry logic (3 attempts, exponential backoff for transient Azure errors)
- Temperature auto-strip for reasoning/o-series models
- Dynamic model label from deployment name
- Artifact validation with skip-and-warn for invalid files
- Streamlit Cloud in-memory ChromaDB fallback

### v0.2 — Working copy for Release 2

- LLM-powered query routing with routing catalog
- Metadata pre-filtering (usecase + component_type + review_status)
- General-layer OR-merge retrieval
- LLM relevance grading (rerank node)
- Grounded answer generation with citations
- Graceful refusal on out-of-domain queries
- Streamlit demo UI with chat interface

### v0.1 — Working copy for Release 1

- Artifact YAML parsing and validation (envelope + body)
- Universal chunking rule (one list-item = one chunk)
- Azure OpenAI embedding with offline fake fallback
- ChromaDB vector store with persistent storage
- Basic retrieval-only pipeline (`demo.py`)
