"""
ciathena_kb.chunker
-------------------
Turns parsed artifacts into retrieval chunks following the universal chunking
rule from the standard:

  * Each item under the artifact's top-level list key = exactly one chunk.
  * Whole-doc types (metadata_contract, agent_contract, sql_generation_contract)
    are config/interface and are NOT chunked for retrieval.
  * Each chunk carries:
      - text:     the embedded text (human fields + triggers/synonyms for recall)
      - metadata: the filterable envelope fields + the item's own id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loader import Artifact

# Per component_type: which item fields are human-text (embedded) and which
# fields are recall boosters (triggers/synonyms) to append to the text.
# Anything not listed is still stored in metadata if scalar.
TEXT_FIELDS = {
    "concept": ["name", "definition", "what_it_is_not", "ai_routing_note"],
    "methodology": ["name", "purpose", "logic", "sme_notes"],
    "process_flow": ["name", "purpose", "logic", "sme_notes"],
    "sttm_mapping": ["udm_concept", "transformation", "keep_drop_rationale", "grain"],
    "dq_rule": ["rule_type", "expression", "rationale", "remediation"],
    "playbook": ["primary_analysis", "analysis_logic", "narrative_template"],
    "anomaly": ["highlight_text"],
    "dataset_catalog": ["display_name", "canonical_name", "use_case_role", "grain", "vendor", "known_gaps"],
}
RECALL_FIELDS = {
    "concept": ["synonyms", "disambiguation_triggers", "adjacent_concepts"],
    "methodology": [],
    "process_flow": [],
    "sttm_mapping": ["synonyms", "disambiguation_triggers"],
    "dq_rule": [],
    "playbook": ["trigger_patterns"],
    "anomaly": [],
    "dataset_catalog": ["business_aliases", "disambiguation_triggers", "source_of_truth_for"],
}
# The per-item id field name, used to build a stable chunk id.
ITEM_ID_FIELD = {
    "concept": "concept_id",
    "methodology": "step_id",
    "process_flow": "step_id",
    "sttm_mapping": "udm_concept",
    "dq_rule": "rule_id",
    "playbook": "scenario_id",
    "anomaly": "signal_id",
    "dataset_catalog": "dataset_ref",
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _flatten(value: Any) -> str:
    """Render a YAML value (str / list / dict) into embeddable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(_flatten(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_flatten(v)}" for k, v in value.items())
    return str(value)


def _scalar_meta(envelope: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool. Coerce lists to CSV."""
    meta: dict[str, Any] = {}
    for k, v in envelope.items():
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, (list, tuple)):
            meta[k] = ",".join(str(x) for x in v) if v else ""
        # dicts in the envelope (rare) are skipped from metadata
    return meta


# Conventional list-key names, tried in order when a component_type is not in
# LIST_KEY_BY_TYPE (new artifact generations may use a type we don't map yet).
_PREFERRED_LIST_KEYS = (
    "concepts", "steps", "mappings", "rules", "scenarios", "signals", "datasets",
)


def _detect_list_key(body: dict[str, Any]) -> str | None:
    """Fallback list-key resolution: the body key holding a non-empty list of
    dict items (preferring the conventional names above, else the largest)."""
    candidates = {
        k: v for k, v in body.items()
        if isinstance(v, list) and v and any(isinstance(i, dict) for i in v)
    }
    for name in _PREFERRED_LIST_KEYS:
        if name in candidates:
            return name
    if candidates:
        return max(candidates, key=lambda k: len(candidates[k]))
    return None


def _detect_id_field(ctype: str, item: dict[str, Any]) -> str:
    if ctype in ITEM_ID_FIELD:
        return ITEM_ID_FIELD[ctype]
    for k in item:  # first `*_id` field wins, else fall back to positional index
        if k.endswith("_id"):
            return k
    return "id"


def chunk_artifact(artifact: Artifact) -> list[Chunk]:
    ctype = artifact.component_type
    if artifact.is_whole_doc:
        return []  # not embedded for retrieval

    # Known types use their explicit list key; unknown types auto-detect.
    list_key = artifact.list_key or _detect_list_key(artifact.body)
    if not list_key or list_key not in artifact.body:
        return []
    items = artifact.body[list_key] or []

    base_meta = _scalar_meta(artifact.envelope)
    known_type = ctype in TEXT_FIELDS
    text_fields = TEXT_FIELDS.get(ctype, [])
    recall_fields = RECALL_FIELDS.get(ctype, [])

    chunks: list[Chunk] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get(_detect_id_field(ctype, item), idx))
        chunk_id = f"{artifact.artifact_id}::{item_id}"

        parts: list[str] = []
        if known_type:
            # curated field order (human text first, then recall boosters)
            for f in text_fields + recall_fields:
                txt = _flatten(item.get(f))
                if txt:
                    parts.append(txt)
        else:
            # Unknown component_type: embed every non-id field so the item stays
            # retrievable without needing a per-type field map.
            for k, v in item.items():
                if k.endswith("_id"):
                    continue
                txt = _flatten(v)
                if txt:
                    parts.append(txt)
        text = "\n".join(parts).strip()
        if not text:
            continue

        meta = dict(base_meta)
        meta["chunk_id"] = chunk_id
        meta["item_id"] = item_id
        # carry the item's own scope_table / phase / severity if present (filterable)
        for extra in ("phase", "severity", "table", "metric"):
            if extra in item and isinstance(item[extra], (str, int, float, bool)):
                meta[extra] = item[extra]

        chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta))
    return chunks


def chunk_all(artifacts: list[Artifact]) -> list[Chunk]:
    out: list[Chunk] = []
    for a in artifacts:
        out.extend(chunk_artifact(a))
    return out
