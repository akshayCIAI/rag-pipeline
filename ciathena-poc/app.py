"""
app.py -- Streamlit demo for the ciATHENA Domain Intelligence Agent.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pathlib

from ciathena_kb import (
    load_artifacts,
    load_artifact,
    load_artifact_from_bytes,
    chunk_artifact,
    chunk_all,
    get_embedder,
    get_chat_llm,
    KnowledgeStore,
    build_agent_graph,
    build_pre_generate_graph,
    make_stream_generate,
    validate_answer,
    IngestionLog,
    ArtifactError,
    get_blob_client,
    PromptManager,
    QACache,
    is_followup_query,
    ChatHistoryStore,
    FeedbackStore,
)
from ciathena_kb.llm import FakeChatLLM
from ciathena_kb.embedder import FakeHashEmbedder
from ciathena_kb.ingestion_log import _bytes_hash
from ciathena_kb.generate_node import BASE_MODEL_DISCLAIMER

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"
MAX_HISTORY_TURNS = int(os.environ.get("HISTORY_MAX_TURNS", "5"))


# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------
def get_embedder_cached():
    if "embedder" not in st.session_state:
        st.session_state.embedder = get_embedder()
    return st.session_state.embedder


def get_store_cached():
    if "store" not in st.session_state:
        st.session_state.store = KnowledgeStore(embedder=get_embedder_cached())
    return st.session_state.store


def get_llm_cached():
    if "llm" not in st.session_state:
        st.session_state.llm = get_chat_llm()
    return st.session_state.llm


def get_log():
    if "ingestion_log" not in st.session_state:
        st.session_state.ingestion_log = IngestionLog()
    return st.session_state.ingestion_log


def get_blob():
    if "blob_client" not in st.session_state:
        st.session_state.blob_client = get_blob_client()
    return st.session_state.blob_client


def get_prompt_manager():
    if "prompt_manager" not in st.session_state:
        st.session_state.prompt_manager = PromptManager(blob_client=get_blob())
    return st.session_state.prompt_manager


def get_qa_cache():
    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = QACache(max_entries=100)
    return st.session_state.qa_cache


def _get_session_id() -> str:
    """Get or create a session ID persisted in URL query params."""
    params = st.query_params
    sid = params.get("session")
    if not sid:
        sid = uuid.uuid4().hex[:12]
        st.query_params["session"] = sid
    return sid


def get_history_store() -> ChatHistoryStore:
    """Get or create the persistent chat history store for this session."""
    sid = _get_session_id()
    cache_key = f"history_store_{sid}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = ChatHistoryStore(
            session_id=sid, blob_client=get_blob(),
        )
    return st.session_state[cache_key]


def get_feedback_store() -> FeedbackStore:
    """Get or create the persistent feedback store for this session."""
    sid = _get_session_id()
    cache_key = f"feedback_store_{sid}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = FeedbackStore(
            session_id=sid, blob_client=get_blob(),
        )
    return st.session_state[cache_key]


def _render_feedback_buttons(msg: dict, feedback_store: FeedbackStore) -> None:
    """Render thumbs up/down buttons for an assistant message."""
    mid = msg.get("message_id", "")
    if not mid:
        return

    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}
    given_rating = st.session_state.feedback_given.get(mid)

    if given_rating is not None:
        icon = "👍" if given_rating > 0 else "👎"
        st.caption(f"Feedback recorded {icon}")
        return

    col1, col2, _ = st.columns([1, 1, 10])
    with col1:
        if st.button("👍", key=f"thumb_up_{mid}", help="Mark as helpful"):
            feedback_store.append(
                message_id=mid,
                query=msg.get("user_query", ""),
                answer=msg.get("content", ""),
                rating=1,
                artifacts_cited=msg.get("citations", []),
            )
            st.session_state.feedback_given[mid] = 1
            st.rerun()
    with col2:
        if st.button("👎", key=f"thumb_down_{mid}", help="Mark as not helpful"):
            feedback_store.append(
                message_id=mid,
                query=msg.get("user_query", ""),
                answer=msg.get("content", ""),
                rating=-1,
                artifacts_cited=msg.get("citations", []),
            )
            st.session_state.feedback_given[mid] = -1
            if feedback_store.should_invalidate_cache(msg.get("user_query", "")):
                get_qa_cache().invalidate()
            st.rerun()


def _build_history(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, str]]:
    """Extract last N Q&A pairs from session messages for LLM context."""
    pairs = []
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            content = msg.get("content", "")
            if content and not content.startswith("**Azure OpenAI is temporarily unavailable"):
                pairs.append({"role": msg["role"], "content": content})
    return pairs[-(max_turns * 2):]


def _smart_ingest_artifact(artifact, data_bytes, store, log, embedder, file_hash=None):
    """Ingest a single artifact only if it changed. Returns (ingested: bool, chunks: int)."""
    content_hash = file_hash or (_bytes_hash(data_bytes) if data_bytes else None)
    needs, reason = log.needs_reingest(artifact, current_hash=content_hash)

    if not needs:
        return False, 0

    chunks = chunk_artifact(artifact)
    if chunks:
        store.ingest(chunks)
    log.record(artifact, chunk_count=len(chunks),
               embedding_model=embedder.model_name, file_hash=content_hash)
    return True, len(chunks)


def load_and_ensure_ingested():
    """Load artifacts from blob/local and smart-ingest on startup."""
    blob = get_blob()
    embedder = get_embedder_cached()
    store = get_store_cached()
    log = get_log()
    pm = get_prompt_manager()

    skipped_files: list[tuple[str, str]] = []

    if blob:
        blob_names = blob.list_artifacts()
        artifacts = []
        ingested_count = 0
        skipped_count = 0
        total_chunks = 0

        for name in blob_names:
            data = blob.download(name)
            uri = f"blob://{blob.container_name}/artifacts/{name}"
            try:
                artifact = load_artifact_from_bytes(data, source_name=uri)
            except Exception as e:
                print(f"  Skipping {name}: {e}")
                skipped_files.append((name, str(e)))
                continue

            artifacts.append(artifact)

            ingested, n_chunks = _smart_ingest_artifact(
                artifact, data, store, log, embedder,
                file_hash=_bytes_hash(data),
            )
            if ingested:
                ingested_count += 1
                total_chunks += n_chunks
            else:
                skipped_count += 1

        if ingested_count > 0:
            print(f"  Auto-ingest: {ingested_count} new/changed, {skipped_count} unchanged, {total_chunks} chunks embedded")
        elif blob_names:
            print(f"  Auto-ingest: all {skipped_count} artifacts unchanged, 0 embeddings needed")
    else:
        artifacts = load_artifacts(ARTIFACTS_DIR)
        if store.count() == 0:
            for a in artifacts:
                chunks = chunk_artifact(a)
                if chunks:
                    store.ingest(chunks)
                log.record(a, chunk_count=len(chunks), embedding_model=embedder.model_name)

    st.session_state.skipped_artifacts = skipped_files

    prompts = {key: pm.get(key) for key in pm.all_keys}
    llm = get_llm_cached()
    pre_graph = build_pre_generate_graph(store=store, artifacts=artifacts, llm=llm, prompts=prompts)
    stream_gen = make_stream_generate(
        llm, system_prompt=prompts.get("generate_system"),
        base_model_prompt=prompts.get("base_model_system"))
    return artifacts, pre_graph, stream_gen


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ciATHENA Knowledge Spine",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
artifacts, pre_graph, stream_gen = load_and_ensure_ingested()
embedder = get_embedder_cached()
store = get_store_cached()
llm = get_llm_cached()
log = get_log()
blob = get_blob()
pm = get_prompt_manager()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 ciATHENA")
    st.caption("Domain Intelligence Agent")

    st.divider()

    # Connection status
    st.subheader("Status")
    is_real_embedder = not isinstance(embedder, FakeHashEmbedder)
    is_real_llm = not isinstance(llm, FakeChatLLM)

    col1, col2, col3 = st.columns(3)
    with col1:
        if is_real_embedder:
            st.success("Embedder", icon="✅")
        else:
            st.warning("Embedder", icon="⚠️")
    with col2:
        if is_real_llm:
            st.success("Chat LLM", icon="✅")
        else:
            st.warning("Chat LLM", icon="⚠️")
    with col3:
        if blob:
            st.success("Blob", icon="✅")
        else:
            st.info("Local", icon="📁")

    st.caption(f"Embedding: `{embedder.model_name}`")
    st.caption(f"Chat: `{llm.model_name}`")
    st.caption(f"Storage: `{'Azure Blob' if blob else 'local'}`")
    st.caption(f"Chunks in store: **{store.count()}**")
    _cache = get_qa_cache()
    _cs = _cache.stats
    st.caption(f"Cache: **{_cs['entries']}** entries, **{_cs['hits']}** hits / **{_cs['misses']}** misses")
    st.caption(f"Session: `{_get_session_id()}`")

    st.divider()

    # ---- ANSWERING MODE ----
    st.subheader("Answering mode")
    base_model_enabled = st.toggle(
        "🧠 Base-model fallback",
        value=st.session_state.get("base_model_enabled", True),
        key="base_model_enabled",
        help=(
            "When ON: if a question is in the pharma domain but no approved "
            "knowledge is found, answer from the base model's general knowledge "
            "(clearly labelled, no citations). When OFF: the assistant declines "
            "instead of guessing."
        ),
    )
    if base_model_enabled:
        st.caption("🧠 Ungrounded pharma questions → base model (labelled).")
    else:
        st.caption("🔒 Strict: ungrounded questions are declined.")

    if st.button("New conversation", use_container_width=True, type="primary"):
        old_sid = _get_session_id()
        new_sid = uuid.uuid4().hex[:12]
        st.query_params["session"] = new_sid
        old_key = f"history_store_{old_sid}"
        if old_key in st.session_state:
            del st.session_state[old_key]
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ---- UPLOAD ARTIFACTS ----
    st.subheader("Upload Artifacts")
    uploaded_files = st.file_uploader(
        "Drop .yml / .yaml files",
        type=["yml", "yaml"],
        accept_multiple_files=True,
        key="artifact_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            raw_bytes = uf.getvalue()

            try:
                if blob:
                    blob_uri = blob.upload(uf.name, raw_bytes)
                    artifact = load_artifact_from_bytes(raw_bytes, source_name=blob_uri)
                    content_hash = _bytes_hash(raw_bytes)
                else:
                    dest = ARTIFACTS_DIR / uf.name
                    dest.write_bytes(raw_bytes)
                    artifact = load_artifact(dest)
                    content_hash = None

                ingested, n_chunks = _smart_ingest_artifact(
                    artifact, raw_bytes, store, log, embedder,
                    file_hash=content_hash,
                )

                version = artifact.envelope.get("content_version", "?")
                storage_label = "blob + Chroma" if blob else "Chroma"
                if ingested:
                    st.success(
                        f"**{artifact.artifact_id}** v{version} "
                        f"— {n_chunks} chunks ingested ({storage_label})",
                        icon="✅",
                    )
                else:
                    st.info(
                        f"**{artifact.artifact_id}** v{version} "
                        f"— unchanged, skipped embedding ({storage_label})",
                        icon="ℹ️",
                    )

                if blob:
                    versions = blob.list_versions(uf.name)
                    if versions:
                        st.caption(f"  Version history: {len(versions)} snapshot(s)")

            except ArtifactError as e:
                st.error(f"**Validation failed:** {e}", icon="❌")
                if not blob:
                    dest = ARTIFACTS_DIR / uf.name
                    dest.unlink(missing_ok=True)
            except Exception as e:
                st.error(f"**Error:** {e}", icon="❌")

        get_qa_cache().invalidate()
        st.cache_resource.clear()
        st.rerun()

    # ---- RE-INGEST ALL ----
    if st.button("Re-ingest all artifacts", use_container_width=True):
        store.clear()
        log.clear()
        get_qa_cache().invalidate()

        reingest_skipped = []
        if blob:
            blob_names = blob.list_artifacts()
            fresh_artifacts = []
            for name in blob_names:
                data = blob.download(name)
                uri = f"blob://{blob.container_name}/artifacts/{name}"
                try:
                    a = load_artifact_from_bytes(data, source_name=uri)
                except Exception as e:
                    reingest_skipped.append((name, str(e)))
                    continue
                fresh_artifacts.append(a)
                chunks = chunk_artifact(a)
                if chunks:
                    store.ingest(chunks)
                log.record(a, chunk_count=len(chunks),
                           embedding_model=embedder.model_name, file_hash=_bytes_hash(data))
        else:
            fresh_artifacts = load_artifacts(ARTIFACTS_DIR)
            for a in fresh_artifacts:
                chunks = chunk_artifact(a)
                if chunks:
                    store.ingest(chunks)
                log.record(a, chunk_count=len(chunks), embedding_model=embedder.model_name)

        st.session_state.skipped_artifacts = reingest_skipped
        src = "blob" if blob else "local"
        st.success(f"Re-ingested {len(fresh_artifacts)} artifacts ({store.count()} chunks) from {src}", icon="🔄")
        if reingest_skipped:
            st.warning(f"Skipped {len(reingest_skipped)} invalid file(s) — see sidebar for details.", icon="⚠️")
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # ---- INGESTION LOG ----
    st.subheader("Ingestion Log")
    log_entries = log.get_all()
    if log_entries:
        for entry in log_entries:
            status_icon = "✅" if entry.get("review_status") == "approved" else "⏳"
            source_icon = "☁️" if entry.get("source_path", "").startswith("blob://") else "📁"
            with st.expander(f"{status_icon} {source_icon} {entry['artifact_id']}"):
                st.markdown(f"**Title:** {entry.get('title', '')}")
                st.markdown(f"**Version:** `{entry.get('content_version', '')}`")
                st.markdown(f"**Type:** `{entry.get('component_type', '')}`")
                st.markdown(f"**Usecase:** `{entry.get('usecase', '')}`")
                st.markdown(f"**Layer:** `{entry.get('layer', '')}`")
                st.markdown(f"**Status:** `{entry.get('review_status', '')}`")
                st.markdown(f"**Chunks:** {entry.get('chunk_count', 0)}")
                st.markdown(f"**Embedding:** `{entry.get('embedding_model', '')}`")
                st.markdown(f"**Source:** `{entry.get('source_path', '')}`")
                st.markdown(f"**Ingested:** {entry.get('ingested_at', '')[:19]}")
                st.markdown(f"**File hash:** `{entry.get('file_hash', '')}`")
    else:
        st.caption("No artifacts ingested yet.")

    # ---- SKIPPED FILES ----
    skipped = st.session_state.get("skipped_artifacts", [])
    if skipped:
        st.divider()
        st.subheader("Skipped Files")
        st.caption(f"{len(skipped)} file(s) failed validation and were not ingested.")
        for fname, reason in skipped:
            with st.expander(f"⚠️ {fname}"):
                st.markdown(f"**Reason:** {reason}")

    st.divider()

    # ---- PROMPT MANAGEMENT ----
    st.subheader("Prompt Management")
    if blob:
        st.caption("Prompts stored in Azure Blob — edit and save without redeploying.")
    else:
        st.caption("Blob not configured — edits apply to current session only.")

    for key in pm.all_keys:
        with st.expander(f"📝 {pm.label_for(key)}"):
            current_val = pm.get(key)
            default_val = pm.default_for(key)
            edited = st.text_area(
                f"Edit {pm.label_for(key)}",
                value=current_val,
                height=200,
                key=f"prompt_{key}",
                label_visibility="collapsed",
            )
            col_save, col_reset = st.columns(2)
            with col_save:
                if st.button("Save", key=f"save_{key}", use_container_width=True):
                    pm.save(key, edited)
                    get_qa_cache().invalidate()
                    st.success("Saved! Pipeline will use the updated prompt.", icon="✅")
                    st.cache_resource.clear()
                    st.rerun()
            with col_reset:
                if st.button("Reset to default", key=f"reset_{key}", use_container_width=True):
                    pm.save(key, default_val)
                    get_qa_cache().invalidate()
                    st.success("Reset to default.", icon="🔄")
                    st.cache_resource.clear()
                    st.rerun()

    st.divider()

    # ---- SAMPLE QUERIES ----
    st.subheader("Try these")
    sample_queries = [
        "What is gross to net?",
        "What is MMM?",
        "Best metric for launch brand adoption?",
        "Where should I move my budget across channels?",
        "Is this channel saturated?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"sample_{sq}", use_container_width=True):
            st.session_state["pending_query"] = sq


# ---------------------------------------------------------------------------
# Chat history (persistent)
# ---------------------------------------------------------------------------
history_store = get_history_store()
if "messages" not in st.session_state:
    st.session_state.messages = history_store.messages

# Header
st.title("🧬 ciATHENA Knowledge Spine")
st.caption("Ask any question about pharma life-sciences commercial analytics")

# Render chat history
_fb_store = get_feedback_store()
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route_expander"):
            with st.expander("🔍 How it routed", expanded=False):
                st.markdown(msg["route_expander"])
        if msg.get("chunks_expander"):
            with st.expander("📦 Retrieved chunks", expanded=False):
                st.markdown(msg["chunks_expander"])
        if msg["role"] == "assistant":
            _render_feedback_buttons(msg, _fb_store)


# ---------------------------------------------------------------------------
# Handle input
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask a question...")
query = pending or user_input

if query:
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)
    history_store.append(user_msg)
    with st.chat_message("user"):
        st.markdown(query)

    qa_cache = get_qa_cache()
    history = _build_history(st.session_state.messages[:-1])

    cached = qa_cache.get(query)

    with st.chat_message("assistant"):
        start = time.time()

        fallback_chunks: list = []
        expanded_queries_display: list = []
        is_base_model = False
        if cached:
            route = cached.route
            graded = cached.graded_chunks
            answer = cached.answer
            citations = cached.citations
            # Base-model answers are unambiguously prefixed with the disclaimer.
            is_base_model = isinstance(answer, str) and answer.startswith(BASE_MODEL_DISCLAIMER)
            st.markdown(answer)
            elapsed = time.time() - start
        else:
            _STATUS_LABELS = {
                "route": "Routing query...",
                "expand_queries": "Expanding query variations...",
                "retrieve": "Retrieving chunks...",
                "rerank": "Grading relevance...",
                "skip_rerank": "High-confidence chunks — skipping rerank...",
                "decline": "Out of domain.",
            }
            try:
                with st.status("Thinking...", expanded=False) as status:
                    result = {}
                    for event in pre_graph.stream(
                        {"user_query": query, "conversation_history": history},
                        stream_mode="updates",
                    ):
                        for node_name in event:
                            label = _STATUS_LABELS.get(node_name, node_name)
                            status.update(label=label)
                            result.update(event[node_name])
                    status.update(label="Generating answer...", state="running")
            except Exception as e:
                elapsed = time.time() - start
                error_msg = (
                    f"**Azure OpenAI is temporarily unavailable.** "
                    f"The service returned an error after retrying: `{type(e).__name__}: {e}`\n\n"
                    f"Please try again in a few minutes."
                )
                st.error(error_msg, icon="⚠️")
                assistant_msg = {
                    "role": "assistant",
                    "content": error_msg,
                    "route_expander": None,
                    "chunks_expander": None,
                }
                st.session_state.messages.append(assistant_msg)
                history_store.append({"role": "assistant", "content": error_msg})
                st.stop()

            route = result.get("route", {})
            graded = result.get("graded_chunks", [])
            fallback_chunks = result.get("fallback_chunks", [])
            expanded_queries_display = result.get("expanded_queries", [])

            try:
                answer = st.write_stream(
                    stream_gen(
                        route, graded, query,
                        conversation_history=history,
                        base_model_enabled=st.session_state.get("base_model_enabled", True),
                    )
                )
            except Exception as e:
                elapsed = time.time() - start
                error_msg = (
                    f"**Azure OpenAI is temporarily unavailable.** "
                    f"The service returned an error during answer generation: "
                    f"`{type(e).__name__}: {e}`\n\n"
                    f"Please try again in a few minutes."
                )
                st.error(error_msg, icon="⚠️")
                assistant_msg = {
                    "role": "assistant",
                    "content": error_msg,
                    "route_expander": None,
                    "chunks_expander": None,
                }
                st.session_state.messages.append(assistant_msg)
                history_store.append({"role": "assistant", "content": error_msg})
                st.stop()

            elapsed = time.time() - start
            # Base-model fallback fired iff the streamed answer carries the disclaimer.
            is_base_model = isinstance(answer, str) and answer.startswith(BASE_MODEL_DISCLAIMER)
            effective_chunks = graded  # base-model answers are uncited by design
            citations = sorted({c.get("chunk_id", "") for c in effective_chunks if c.get("chunk_id")})

            # Intent-aware validation — skip for base-model answers and offline mode
            if not is_base_model and not isinstance(llm, FakeChatLLM) and effective_chunks:
                _val = validate_answer(
                    llm, route, effective_chunks, answer,
                    system_prompt=pm.get("validation_grounding"),
                )
                _verdict = _val.get("verdict", "pass")
                if _verdict == "warn":
                    st.warning(
                        f"Quality check: {_val.get('reason', '')}. {_val.get('suggestion', '')}",
                        icon="⚠️",
                    )
                elif _verdict == "fail":
                    st.error(
                        f"Quality check: {_val.get('reason', '')}. {_val.get('suggestion', '')}",
                        icon="❌",
                    )

            if not is_followup_query(query):
                qa_cache.put(query, route, graded, answer, citations)

        if is_base_model:
            st.caption("🧠 Source: **base model** (general knowledge — not governed/approved artifacts)")
        elif citations:
            citation_text = ", ".join(f"`{c}`" for c in citations)
            st.caption(f"📎 Citations: {citation_text}")

        route_md = ""
        if route:
            in_domain = route.get("in_domain", True)
            route_md += f"**In domain:** {'Yes' if in_domain else 'No'}\n\n"
            route_md += f"**Usecase:** `{route.get('usecase', 'General')}`\n\n"
            ct = route.get("component_types", [])
            route_md += f"**Component types:** `{', '.join(ct) if ct else '(all)'}` (soft)\n\n"
            route_md += f"**Intent:** `{route.get('intent', 'definition')}`\n\n"
            rq = route.get("rewritten_query", "")
            if rq:
                route_md += f"**Rewritten query:** {rq}\n\n"
            cf = route.get("chroma_filter")
            if cf:
                route_md += f"**Metadata filter:** `{cf}`\n\n"
            if expanded_queries_display:
                eq_list = "\n".join(f"- {q}" for q in expanded_queries_display)
                route_md += f"**Expanded queries:**\n{eq_list}\n\n"

        if route_md:
            with st.expander("🔍 How it routed", expanded=False):
                st.markdown(route_md)

        chunks_md = ""
        # For base-model answers, show the closest retrieved chunks as diagnostics
        # (they were below the relevance threshold and NOT used in the answer).
        if graded:
            display_chunks = graded
        elif is_base_model and not cached:
            display_chunks = fallback_chunks
        else:
            display_chunks = []
        if display_chunks:
            if not graded:
                chunks_md = ("🧠 **Answered from base model** — the chunks below were "
                             "retrieved but fell below the relevance threshold and were "
                             "NOT used in the answer:\n\n")
            for i, c in enumerate(display_chunks, 1):
                score = c.get("score", 0)
                cid = c.get("chunk_id", "")
                ct = c.get("component_type", "")
                uc = c.get("usecase", "")
                chunks_md += f"**{i}.** `[{score:.3f}]` `{cid}` — {ct} / {uc}\n\n"
                text_preview = c.get("text", "")[:200]
                if text_preview:
                    chunks_md += f"> {text_preview}...\n\n"

        if chunks_md:
            with st.expander("📦 Retrieved chunks", expanded=False):
                st.markdown(chunks_md)

        _msg_id = uuid.uuid4().hex[:12]
        _render_feedback_buttons(
            {"message_id": _msg_id, "user_query": query, "content": answer, "citations": citations},
            get_feedback_store(),
        )

        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "route_expander": route_md if route_md else None,
            "chunks_expander": chunks_md if chunks_md else None,
            "message_id": _msg_id,
            "user_query": query,
            "citations": citations,
        }
        st.session_state.messages.append(assistant_msg)
        history_store.append({"role": "assistant", "content": answer})
