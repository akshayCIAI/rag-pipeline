"""
app.py -- Streamlit demo for the ciATHENA Domain Intelligence Agent.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time

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
    IngestionLog,
    ArtifactError,
    get_blob_client,
    PromptManager,
)
from ciathena_kb.llm import FakeChatLLM
from ciathena_kb.embedder import FakeHashEmbedder
from ciathena_kb.ingestion_log import _bytes_hash

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"


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

    if blob:
        blob_names = blob.list_artifacts()
        artifacts = []
        ingested_count = 0
        skipped_count = 0
        total_chunks = 0

        for name in blob_names:
            data = blob.download(name)
            uri = f"blob://{blob.container_name}/artifacts/{name}"
            artifact = load_artifact_from_bytes(data, source_name=uri)
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

    prompts = {key: pm.get(key) for key in pm.all_keys}
    llm = get_llm_cached()
    graph = build_agent_graph(store=store, artifacts=artifacts, llm=llm, prompts=prompts)
    return artifacts, graph


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
artifacts, graph = load_and_ensure_ingested()
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

        st.cache_resource.clear()
        st.rerun()

    # ---- RE-INGEST ALL ----
    if st.button("Re-ingest all artifacts", use_container_width=True):
        store.clear()
        log.clear()

        if blob:
            blob_names = blob.list_artifacts()
            fresh_artifacts = []
            for name in blob_names:
                data = blob.download(name)
                uri = f"blob://{blob.container_name}/artifacts/{name}"
                a = load_artifact_from_bytes(data, source_name=uri)
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

        src = "blob" if blob else "local"
        st.success(f"Re-ingested {len(fresh_artifacts)} artifacts ({store.count()} chunks) from {src}", icon="🔄")
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
                    st.success("Saved! Pipeline will use the updated prompt.", icon="✅")
                    st.cache_resource.clear()
                    st.rerun()
            with col_reset:
                if st.button("Reset to default", key=f"reset_{key}", use_container_width=True):
                    pm.save(key, default_val)
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
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🧬 ciATHENA Knowledge Spine")
st.caption("Ask any question about pharma life-sciences commercial analytics")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route_expander"):
            with st.expander("🔍 How it routed", expanded=False):
                st.markdown(msg["route_expander"])
        if msg.get("chunks_expander"):
            with st.expander("📦 Retrieved chunks", expanded=False):
                st.markdown(msg["chunks_expander"])


# ---------------------------------------------------------------------------
# Handle input
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask a question...")
query = pending or user_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            try:
                result = graph.invoke({"user_query": query})
            except Exception as e:
                elapsed = time.time() - start
                error_msg = (
                    f"**Azure OpenAI is temporarily unavailable.** "
                    f"The service returned an error after retrying: `{type(e).__name__}: {e}`\n\n"
                    f"Please try again in a few minutes."
                )
                st.error(error_msg, icon="⚠️")
                st.caption(f"⏱️ {elapsed:.1f}s")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "route_expander": None,
                    "chunks_expander": None,
                })
                st.stop()
            elapsed = time.time() - start

        route = result.get("route", {})
        graded = result.get("graded_chunks", [])
        answer = result.get("answer", "")
        citations = result.get("citations", [])

        st.markdown(answer)

        if citations:
            citation_text = ", ".join(f"`{c}`" for c in citations)
            st.caption(f"📎 Citations: {citation_text}")

        st.caption(f"⏱️ {elapsed:.1f}s")

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

        if route_md:
            with st.expander("🔍 How it routed", expanded=False):
                st.markdown(route_md)

        chunks_md = ""
        if graded:
            for i, c in enumerate(graded, 1):
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

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "route_expander": route_md if route_md else None,
            "chunks_expander": chunks_md if chunks_md else None,
        })
