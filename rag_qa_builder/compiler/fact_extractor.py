from __future__ import annotations

import re

from rag_qa_builder.config import AppConfig
from rag_qa_builder.llm.prompt_runner import PromptRunner
from rag_qa_builder.models import Concept, DocumentSection, Evidence, Fact
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import normalize_text, split_sentences


def extract_facts(
    concepts: list[Concept],
    sections: list[DocumentSection],
    config: AppConfig,
    prompt_runner: PromptRunner,
) -> tuple[list[Fact], list[Evidence]]:
    section_lookup = {section.section_id: section for section in sections}
    facts: list[Fact] = []
    evidences: list[Evidence] = []

    for concept in concepts:
        concept_sections = [section_lookup[section_id] for section_id in concept.source_section_ids if section_id in section_lookup]
        used = 0
        for section in concept_sections:
            for sentence in split_sentences(section.text):
                if used >= config.fact_extraction.max_facts_per_concept:
                    break
                if normalize_text(concept.canonical_name) not in normalize_text(sentence):
                    continue
                evidence = _make_evidence(section, sentence)
                fact = _make_fact(concept, sentence, evidence.evidence_id)
                if fact.confidence >= config.fact_extraction.min_confidence:
                    evidences.append(evidence)
                    facts.append(fact)
                    used += 1
        llm_result = prompt_runner.maybe_run_json(
            "extract_facts",
            "Extract atomic facts about the target concept. Return JSON with top-level 'facts'. Every fact must be directly supported by evidence. No external knowledge.",
            {
                "concept": concept.model_dump(mode="json"),
                "sections": [section.model_dump(mode="json") for section in concept_sections[:10]],
            },
        )
        for item in (llm_result or {}).get("facts", []):
            statement = item.get("statement")
            evidence_text = item.get("evidence")
            if not statement or not evidence_text or not concept_sections:
                continue
            section = concept_sections[0]
            evidence = _make_evidence(section, evidence_text)
            fact = Fact(
                fact_id=stable_id("fact", f"{concept.concept_id}:{statement}"),
                fact_type=item.get("fact_type", "claim"),
                subject_concept_id=concept.concept_id,
                related_concept_ids=item.get("related_concept_ids", []),
                statement=statement,
                structured=item.get("structured", {}),
                qualifiers=item.get("qualifiers", {}),
                confidence=float(item.get("confidence", 0.7)),
                importance=float(item.get("importance", concept.importance)),
                evidence_ids=[evidence.evidence_id],
                metadata={"source": "llm"},
            )
            if fact.confidence >= config.fact_extraction.min_confidence:
                evidences.append(evidence)
                facts.append(fact)
    return facts, evidences


def _make_fact(concept: Concept, sentence: str, evidence_id: str) -> Fact:
    fact_type = _infer_fact_type(sentence)
    return Fact(
        fact_id=stable_id("fact", f"{concept.concept_id}:{sentence}"),
        fact_type=fact_type,
        subject_concept_id=concept.concept_id,
        related_concept_ids=[],
        statement=sentence,
        structured={},
        qualifiers={},
        confidence=_estimate_confidence(sentence),
        importance=concept.importance,
        evidence_ids=[evidence_id],
        metadata={"source": "heuristic"},
    )


def _make_evidence(section: DocumentSection, text: str) -> Evidence:
    rel = section.text.find(text)
    char_start = section.char_start + rel if rel >= 0 else None
    char_end = char_start + len(text) if char_start is not None else None
    return Evidence(
        evidence_id=stable_id("ev", f"{section.section_id}:{text}"),
        doc_id=section.doc_id,
        section_id=section.section_id,
        section_path=section.section_path,
        text=text,
        char_start=char_start,
        char_end=char_end,
        source_hint=section.title,
    )


def _infer_fact_type(sentence: str) -> str:
    lowered = normalize_text(sentence)
    if re.search(r"\d", sentence):
        return "numeric"
    if any(word in lowered for word in ["如果", "when", "if"]):
        return "condition"
    if any(word in lowered for word in ["因为", "导致", "therefore", "because", "so that"]):
        return "cause_effect"
    if any(word in lowered for word in ["必须", "不能", "should", "must", "require"]):
        return "constraint"
    if any(word in lowered for word in ["步骤", "首先", "然后", "最后", "step"]):
        return "procedure"
    if any(word in lowered for word in ["相比", "比", "different", "compared"]):
        return "comparison"
    if any(word in lowered for word in ["例如", "for example", "比如"]):
        return "example"
    return "claim"


def _estimate_confidence(sentence: str) -> float:
    value = 0.6
    if len(sentence) >= 20:
        value += 0.1
    if re.search(r"\d", sentence):
        value += 0.1
    if any(token in sentence for token in ["必须", "不能", "因为", "导致", "例如", "包括", "is", "are"]):
        value += 0.1
    return min(value, 0.95)

