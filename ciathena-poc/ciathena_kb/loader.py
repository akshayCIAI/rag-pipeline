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


def _validate_envelope(env: dict[str, Any], path: str) -> None:
    required = [
        "artifact_id", "title", "component_type", "usecase", "layer",
        "schema_version", "content_version", "review_status", "embedding_model",
    ]
    missing = [k for k in required if k not in env]
    if missing:
        raise ArtifactError(f"{path}: missing envelope keys: {missing}")
    if env["layer"] not in VALID_LAYERS:
        raise ArtifactError(f"{path}: invalid layer '{env['layer']}'")
    if env["component_type"] not in VALID_COMPONENT_TYPES:
        raise ArtifactError(f"{path}: invalid component_type '{env['component_type']}'")
    if env["review_status"] not in VALID_REVIEW_STATUS:
        raise ArtifactError(f"{path}: invalid review_status '{env['review_status']}'")


def load_artifact(path: str | pathlib.Path) -> Artifact:
    """Load and validate a single artifact YAML file."""
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as fh:
        # Artifacts may use a '---' line to visually separate envelope from
        # body. YAML treats that as a document separator, so merge all docs
        # in the stream into one mapping (envelope keys + body keys coexist).
        docs = [d for d in yaml.safe_load_all(fh) if isinstance(d, dict)]
    if not docs:
        raise ArtifactError(f"{path}: no YAML mapping found")
    doc: dict[str, Any] = {}
    for d in docs:
        doc.update(d)
    env, body = _split_envelope_body(doc)
    _validate_envelope(env, str(path))
    return Artifact(envelope=env, body=body, source_path=str(path))


def load_artifact_from_bytes(data: bytes, source_name: str) -> Artifact:
    """Parse and validate an artifact from raw YAML bytes (e.g. from blob storage)."""
    text = data.decode("utf-8")
    docs = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    if not docs:
        raise ArtifactError(f"{source_name}: no YAML mapping found")
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
