from __future__ import annotations

from rag_qa_builder.models import Concept
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import normalize_text, similarity


def canonicalize_concepts(concepts: list[Concept]) -> list[Concept]:
    merged: list[Concept] = []
    for concept in sorted(concepts, key=lambda item: item.importance, reverse=True):
        matched = None
        for existing in merged:
            if normalize_text(concept.canonical_name) == normalize_text(existing.canonical_name):
                matched = existing
                break
            if similarity(concept.canonical_name, existing.canonical_name) >= 0.9:
                matched = existing
                break
        if not matched:
            concept.concept_id = stable_id("concept", normalize_text(concept.canonical_name))
            merged.append(concept)
            continue
        aliases = set(matched.aliases)
        aliases.add(concept.canonical_name)
        aliases.update(concept.aliases)
        matched.aliases = sorted(alias for alias in aliases if normalize_text(alias) != normalize_text(matched.canonical_name))
        matched.source_section_ids = sorted(set(matched.source_section_ids + concept.source_section_ids))
        matched.importance = max(matched.importance, concept.importance)
        matched.definition = matched.definition or concept.definition
    return merged

