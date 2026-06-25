"""
ciathena_kb.catalog
-------------------
Builds a routing catalog from loaded artifacts. The catalog is a compact text
block injected into the router prompt so the LLM can map a raw user question
onto the right usecase / component_type filters without guessing.

It extracts: distinct usecases, component_types, and per-artifact routing
signals (trigger_patterns, disambiguation_triggers, ai_routing_note).
"""

from __future__ import annotations

from typing import Any

from .loader import Artifact


def _collect(items: list[dict[str, Any]], field: str) -> list[str]:
    out: list[str] = []
    for item in items:
        val = item.get(field)
        if isinstance(val, list):
            out.extend(str(v) for v in val if v)
        elif isinstance(val, str) and val.strip():
            out.append(val.strip())
    return out


def build_routing_catalog(artifacts: list[Artifact]) -> str:
    """Return a structured text block describing the available knowledge corpus,
    suitable for injection into the router system prompt."""
    usecases: set[str] = set()
    component_types: set[str] = set()
    entries: list[str] = []

    for a in artifacts:
        uc = a.envelope.get("usecase", "")
        ct = a.component_type
        usecases.add(uc)
        component_types.add(ct)

        list_key = a.list_key
        items = a.body.get(list_key, []) if list_key else []
        if not isinstance(items, list):
            items = []

        triggers = _collect(items, "trigger_patterns")
        disambig = _collect(items, "disambiguation_triggers")
        routing_notes = _collect(items, "ai_routing_note")
        synonyms = _collect(items, "synonyms")

        item_names = (_collect(items, "name") or _collect(items, "title")
                      or _collect(items, "phase") or _collect(items, "scenario_id"))

        entry = f"- artifact: {a.artifact_id}\n"
        entry += f"  title: {a.envelope.get('title', '')}\n"
        entry += f"  usecase: {uc}\n"
        entry += f"  component_type: {ct}\n"
        entry += f"  layer: {a.envelope.get('layer', '')}\n"
        if item_names:
            entry += f"  covers: {item_names}\n"
        if triggers:
            entry += f"  trigger_patterns: {triggers}\n"
        if disambig:
            entry += f"  disambiguation_triggers: {disambig}\n"
        if routing_notes:
            entry += f"  routing_notes: {routing_notes}\n"
        if synonyms:
            entry += f"  synonyms: {synonyms}\n"

        entries.append(entry)

    header = (
        "AVAILABLE KNOWLEDGE CORPUS\n"
        f"Usecases: {sorted(usecases)}\n"
        f"Component types: {sorted(component_types)}\n"
        f"Total artifacts: {len(artifacts)}\n\n"
        "ARTIFACT DETAILS:\n"
    )
    return header + "\n".join(entries)
