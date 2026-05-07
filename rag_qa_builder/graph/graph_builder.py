from __future__ import annotations

from rag_qa_builder.models import Concept, ConceptFactRelation, Fact
from rag_qa_builder.utils.ids import stable_id


def build_concept_fact_relations(concepts: list[Concept], facts: list[Fact]) -> list[ConceptFactRelation]:
    concept_map = {concept.concept_id: concept for concept in concepts}
    relations: list[ConceptFactRelation] = []
    for fact in facts:
        if fact.subject_concept_id and fact.subject_concept_id in concept_map:
            relation_type = _relation_type_for_fact(fact.fact_type)
            relations.append(
                ConceptFactRelation(
                    relation_id=stable_id("rel", f"{fact.subject_concept_id}:{fact.fact_id}:{relation_type}"),
                    concept_id=fact.subject_concept_id,
                    fact_id=fact.fact_id,
                    relation_type=relation_type,
                    confidence=fact.confidence,
                )
            )
            concept_map[fact.subject_concept_id].related_fact_ids.append(fact.fact_id)
    return relations


def _relation_type_for_fact(fact_type: str) -> str:
    mapping = {
        "definition": "defines",
        "constraint": "constrains",
        "cause_effect": "causes",
        "comparison": "compares",
        "procedure": "explains",
        "config": "configures",
        "example": "example_of",
    }
    return mapping.get(fact_type, "mentioned_by")

