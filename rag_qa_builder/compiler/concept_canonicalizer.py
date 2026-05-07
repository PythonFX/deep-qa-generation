from __future__ import annotations

import re

from rag_qa_builder.models import Concept
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import normalize_text, similarity

CANONICAL_BLACKLIST = {
    "the", "a", "an", "this", "that", "these", "those", "provided", "listing", "equal",
    "experiments", "experiment", "english", "french", "englishto-german", "jakob",
}

DOMAIN_ALLOWLIST = {
    "transformer", "attention", "encoder", "decoder", "bleu", "embedding", "network",
    "normalization", "residual", "feed-forward", "gpu", "gpus", "wmt", "architecture",
    "position", "representation", "training", "sequence", "translation", "head", "layer",
}


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
    merged = _collapse_near_duplicate_concepts(merged)
    return [concept for concept in merged if _keep_canonical_concept(concept)]


def _collapse_near_duplicate_concepts(concepts: list[Concept]) -> list[Concept]:
    kept: list[Concept] = []
    for concept in sorted(concepts, key=lambda item: (item.importance, len(item.canonical_name)), reverse=True):
        matched = None
        concept_norm = normalize_text(concept.canonical_name)
        for existing in kept:
            existing_norm = normalize_text(existing.canonical_name)
            if concept_norm == existing_norm:
                matched = existing
                break
            if concept_norm in existing_norm and len(concept_norm) <= len(existing_norm):
                matched = existing
                break
        if not matched:
            kept.append(concept)
            continue
        aliases = set(matched.aliases)
        aliases.add(concept.canonical_name)
        aliases.update(concept.aliases)
        matched.aliases = sorted(aliases)
        matched.source_section_ids = sorted(set(matched.source_section_ids + concept.source_section_ids))
        matched.importance = max(matched.importance, concept.importance)
        matched.definition = matched.definition or concept.definition
    return kept


def _keep_canonical_concept(concept: Concept) -> bool:
    name = concept.canonical_name.strip()
    norm = normalize_text(name)
    if not norm or norm in CANONICAL_BLACKLIST:
        return False
    if _looks_like_person_name(name):
        return False
    if re.fullmatch(r"[A-Z][a-z]+", name) and norm not in DOMAIN_ALLOWLIST:
        return False
    if re.fullmatch(r"[A-Za-z]+", name) and norm not in DOMAIN_ALLOWLIST and concept.importance < 0.8:
        return False
    if re.fullmatch(r"[A-Z]{2,}", name):
        return True
    if any(token in norm for token in DOMAIN_ALLOWLIST):
        return True
    if " " in name and concept.importance >= 0.7:
        return True
    return concept.importance >= 0.9


def _looks_like_person_name(name: str) -> bool:
    tokens = [token for token in re.split(r"\s+", name.strip()) if token]
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    for token in tokens:
        pure = token.replace(".", "")
        if not pure:
            return False
        if not (pure[:1].isupper() and pure[1:].islower()):
            return False
    return True
