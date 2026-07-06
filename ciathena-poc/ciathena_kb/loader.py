"""
ciathena_kb.loader
------------------
Loads ciATHENA knowledge artifacts (YAML) from disk, validates the shared
envelope, and returns parsed artifact objects. Pure-Python, no LLM needed.

An artifact file is a single YAML document with two logical parts:
  * the ENVELOPE (identity / retrieval-filtering / governance fields)
  * the BODY (the type-specific content)

Because our artifacts are authored as one YAML doc (envelope keys at the top
level, body keys alongside), we split them by a known set of envelope keys.
"""

from __future__ import annotations

import pathlib
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import yaml

# Envelope keys defined by the artifact standard (section 2 of the templates doc).
ENVELOPE_KEYS = {
    "artifact_id", "title", "artifact_type", "schema_version", "content_version",
    "usecase", "layer", "component_type", "personas", "client_agnostic",
    "inherits_from", "depends_on", "owner", "review_status", "last_reviewed",
    "embedding_model", "scope_table",
}

# Controlled vocabularies (kept closed per the standard).
VALID_LAYERS = {"general", "usecase", "udm", "client_facing"}
VALID_COMPONENT_TYPES = {
    "concept", "methodology", "sttm_mapping", "dq_rule", "process_flow",
    "playbook", "anomaly", "dataset_catalog", "metadata_contract",
    "agent_contract", "sql_generation_contract",
}
VALID_REVIEW_STATUS = {"draft", "sme_review", "approved", "deprecated"}

# component_type -> the body list key whose items each become one chunk.
# Whole-doc types (config/interface) are not chunked for retrieval.
LIST_KEY_BY_TYPE = {
    "concept": "concepts",
    "methodology": "steps",
    "process_flow": "steps",
    "sttm_mapping": "mappings",
    "dq_rule": "rules",
    "playbook": "scenarios",
    "anomaly": "signals",
    "dataset_catalog": "datasets",
}
WHOLE_DOC_TYPES = {"metadata_contract", "agent_contract", "sql_generation_contract"}


class ArtifactError(ValueError):
    """Raised when an artifact fails envelope validation."""


@dataclass
class Artifact:
    envelope: dict[str, Any]
    body: dict[str, Any]
    source_path: str = ""
    # convenience accessors
    @property
    def artifact_id(self) -> str:
        return self.envelope["artifact_id"]

    @property
    def component_type(self) -> str:
        return self.envelope["component_type"]

    @property
    def list_key(self) -> str | None:
        return LIST_KEY_BY_TYPE.get(self.component_type)

    @property
    def is_whole_doc(self) -> bool:
        return self.component_type in WHOLE_DOC_TYPES


def _split_envelope_body(doc: dict[str, Any]) -> tuple[dict, dict]:
    envelope = {k: v for k, v in doc.items() if k in ENVELOPE_KEYS}
    body = {k: v for k, v in doc.items() if k not in ENVELOPE_KEYS}
    return envelope, body


# Only these two are structurally required — everything downstream reads the
# rest via .get(...) with defaults. Newer artifact generations (e.g. the L1
# knowledge files) legitimately omit fields like `usecase` / `embedding_model`
# and introduce new `layer` / `component_type` values, so we accept-and-warn
# instead of hard-rejecting. This keeps validation agnostic to the artifacts'
# exact envelope keys.
HARD_REQUIRED_KEYS = ("artifact_id", "component_type")
RECOMMENDED_KEYS = (
    "title", "usecase", "layer", "schema_version",
    "content_version", "review_status", "embedding_model",
)


def _validate_envelope(env: dict[str, Any], path: str) -> None:
    missing = [k for k in HARD_REQUIRED_KEYS if k not in env]
    if missing:
        raise ArtifactError(f"{path}: missing required envelope keys: {missing}")

    soft_missing = [k for k in RECOMMENDED_KEYS if k not in env]
    if soft_missing:
        warnings.warn(
            f"{path}: missing recommended envelope keys {soft_missing} "
            f"(accepted; defaults will be used)"
        )

    # Controlled vocabularies are advisory, not gates: unknown values are
    # accepted so new layers / component_types don't block ingestion.
    for key, vocab in (
        ("layer", VALID_LAYERS),
        ("component_type", VALID_COMPONENT_TYPES),
        ("review_status", VALID_REVIEW_STATUS),
    ):
        val = env.get(key)
        if val is not None and val not in vocab:
            warnings.warn(f"{path}: unrecognized {key} '{val}' (accepted anyway)")


# ── YAML resilience ──────────────────────────────────────────────────────────
# A common hand-authoring defect is a block-mapping colon with no following
# space, e.g.  key:"value"  instead of  key: "value"  — PyYAML rejects this as a
# scanner error. We do NOT rewrite well-formed files: _load_docs only attempts a
# conservative, line-oriented repair AFTER a normal parse has already failed,
# then retries once. Valid YAML is never touched; genuinely broken files still
# raise and get reported as skipped.

_MISSING_COLON_SPACE = re.compile(r'^(\s*)("?[^\s#][^:]*?"?):(?=\S)')
_BLOCK_SCALAR_OPENER = re.compile(r':\s*[|>][+-]?\d*\s*(#.*)?$')


def _repair_yaml_text(text: str) -> str:
    """Insert the missing space after a *mapping* colon (`key:val` -> `key: val`).
    Only the first colon on a line is a candidate. Comments, blank lines, and the
    interior of literal/folded (`|` / `>`) block scalars are left byte-for-byte so
    indentation-sensitive content is never altered."""
    out: list[str] = []
    block_indent: int | None = None
    for raw in text.split("\n"):
        stripped = raw.lstrip(" ")
        indent = len(raw) - len(stripped)
        if block_indent is not None:  # inside a literal/folded block scalar
            if stripped == "" or indent > block_indent:
                out.append(raw)
                continue
            block_indent = None  # dedented -> block scalar ended
        if stripped == "" or stripped.startswith("#"):
            out.append(raw)
            continue
        fixed = _MISSING_COLON_SPACE.sub(r'\1\2: ', raw)
        out.append(fixed)
        if _BLOCK_SCALAR_OPENER.search(fixed):
            block_indent = indent
    return "\n".join(out)


def _load_docs(text: str, source: str) -> list[dict]:
    """Parse a YAML stream into dict docs, with one repair-and-retry on error.

    Artifacts may use a '---' line to visually separate envelope from body; YAML
    treats that as a document separator, so callers merge the returned docs into
    one mapping (envelope keys + body keys coexist).
    """
    try:
        raw_docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        raw_docs = list(yaml.safe_load_all(_repair_yaml_text(text)))
    docs = [d for d in raw_docs if isinstance(d, dict)]
    if not docs:
        raise ArtifactError(f"{source}: no YAML mapping found")
    return docs


def load_artifact(path: str | pathlib.Path) -> Artifact:
    """Load and validate a single artifact YAML file."""
    path = pathlib.Path(path)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    docs = _load_docs(text, str(path))
    doc: dict[str, Any] = {}
    for d in docs:
        doc.update(d)
    env, body = _split_envelope_body(doc)
    _validate_envelope(env, str(path))
    return Artifact(envelope=env, body=body, source_path=str(path))


def load_artifact_from_bytes(data: bytes, source_name: str) -> Artifact:
    """Parse and validate an artifact from raw YAML bytes (e.g. from blob storage)."""
    text = data.decode("utf-8")
    docs = _load_docs(text, source_name)
    doc: dict[str, Any] = {}
    for d in docs:
        doc.update(d)
    env, body = _split_envelope_body(doc)
    _validate_envelope(env, source_name)
    return Artifact(envelope=env, body=body, source_path=source_name)


def load_artifacts(directory: str | pathlib.Path) -> list[Artifact]:
    """Load every .yml / .yaml artifact under a directory (recursively)."""
    directory = pathlib.Path(directory)
    files = sorted([*directory.rglob("*.yml"), *directory.rglob("*.yaml")])
    artifacts: list[Artifact] = []
    for f in files:
        artifacts.append(load_artifact(f))
    return artifacts
