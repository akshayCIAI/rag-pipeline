# ciATHENA Knowledge Spine — Domain Intelligence Agent (Scenario B)

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
    rerank_node.py          # post-retrieval relevance grading + threshold
    generate_node.py        # grounded answer generation with citations
    agent_graph.py          # assembles the full LangGraph: route → retrieve → rerank → generate
    ingestion_log.py        # version-aware ingestion tracking (skip unchanged artifacts)
    blob_client.py          # Azure Blob Storage client for artifacts + prompts (optional, versioned)
    prompt_manager.py       # prompt template manager (blob-backed with built-in defaults)
    qa_cache.py             # session-scoped Q&A result cache (generation-based invalidation)
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
   → in_domain? · usecase · component_type(s) · intent · rewritten_query
    │
    ▼
2. RETRIEVE (persistent Chroma)
   metadata pre-filter: usecase ∈ {chosen, General}   (OR-merge)
                        review_status = "approved"     (governance)
                        component_type ∈ chosen        (soft)
    │
    ▼
3. RERANK / DOCUMENT-FILTER
   high-confidence chunks (cosine ≥ 0.7) auto-pass
   borderline chunks batch-graded in 1 LLM call → top-k
    │
    ▼
4. GENERATE (LLM, streamed)
   grounded answer + citations [artifact_id::chunk_id]
   tokens stream to UI in real time; refuses when no approved context survives filtering
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
- **Chat interface** — ask questions, see answers streamed in real time with citations, routing details, and retrieved chunks; supports follow-up questions with conversation history (last 5 turns)
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
- **Batch rerank optimization**: rerank step uses a single batched LLM call instead of one per chunk, reducing rerank from ~8 calls to 1; chunks with cosine score >= 0.7 skip LLM grading entirely
- **Streaming answer generation**: answer tokens stream to the Streamlit UI in real time via `st.write_stream()` instead of waiting for the full response; route/retrieve/rerank run first with a spinner, then the answer appears token-by-token
- **Follow-up questions**: conversation history (last 5 Q&A turns) is passed to the router and generator so the pipeline can resolve references like "tell me more" or "compare that with X"; the router rewrites follow-ups into self-contained queries for retrieval
- **Q&A caching**: session-scoped cache for pipeline results; all standalone queries are cached regardless of conversation position — repeated identical questions return instantly; vague follow-ups ("tell me more") are detected and excluded from cache; invalidated automatically on artifact upload, re-ingest, or prompt edits; cache stats shown in sidebar

## What this PoC covers (and does not)

**Covers:** artifact parsing, validation, one-item-per-chunk chunking,
embedding, persistent Chroma ingest, metadata pre-filtering (usecase +
component_type + review_status), General-layer OR-merge, LLM query routing,
LLM relevance grading, grounded answer generation with citations, graceful
refusal on out-of-domain or no-context queries, smart re-ingestion with
version tracking, Azure Blob Storage integration (versioned artifacts +
prompt templates), auto-ingest on startup, LLM retry logic, blob-backed
prompt management, streaming answer generation, conversation history with
follow-up support, session-scoped Q&A caching, and Streamlit demo UI with
artifact upload and prompt editor.

**Does not cover:** NL-to-SQL, summarization, visualization (downstream nodes),
encryption-at-rest, CI/CD image delivery, self-containment hardening. Those
are later phases per the PoC plan.
