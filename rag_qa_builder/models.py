from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    file_path: str
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    section_id: str
    doc_id: str
    title: str | None = None
    level: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    char_start: int
    char_end: int
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)


class Concept(BaseModel):
    concept_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    concept_type: str
    definition: str | None = None
    importance: float = 0.0
    source_section_ids: list[str] = Field(default_factory=list)
    related_fact_ids: list[str] = Field(default_factory=list)
    related_concept_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    evidence_id: str
    doc_id: str
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    char_start: int | None = None
    char_end: int | None = None
    source_hint: str | None = None


class Fact(BaseModel):
    fact_id: str
    fact_type: str
    subject_concept_id: str | None = None
    related_concept_ids: list[str] = Field(default_factory=list)
    statement: str
    structured: dict[str, Any] = Field(default_factory=dict)
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    importance: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    depends_on_fact_ids: list[str] = Field(default_factory=list)
    contrasts_with_fact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConceptFactRelation(BaseModel):
    relation_id: str
    concept_id: str
    fact_id: str
    relation_type: str
    confidence: float = 0.0


class FactCombination(BaseModel):
    combination_id: str
    fact_ids: list[str]
    concept_ids: list[str]
    pattern: str
    rationale: str
    expected_question_type: str
    expected_answer_points: list[str]
    difficulty: str
    score: float


class QuestionBlueprint(BaseModel):
    blueprint_id: str
    source_combination_id: str | None = None
    pattern: str
    fact_ids: list[str]
    concept_ids: list[str]
    intended_question: str
    expected_answer_points: list[str]
    difficulty: str
    question_type: str
    answer_requirements: list[str] = Field(default_factory=list)
    unsupported_answer_patterns: list[str] = Field(default_factory=list)


class QAPair(BaseModel):
    qa_id: str
    question: str
    reference_answer: str
    concept_ids: list[str]
    fact_ids: list[str]
    evidence_ids: list[str]
    question_type: str
    difficulty: str
    answer_requirements: list[str] = Field(default_factory=list)
    unsupported_answer_patterns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAValidationResult(BaseModel):
    qa_id: str
    is_answerable: bool
    is_faithful: bool
    is_ambiguous: bool
    requires_external_knowledge: bool
    has_hallucination: bool
    scores: dict[str, float]
    issues: list[str] = Field(default_factory=list)
    passed: bool


class DatasetEntry(BaseModel):
    id: str
    question: str
    reference_answer: str
    reference_context: list[dict[str, Any]]
    concept_ids: list[str]
    fact_ids: list[str]
    evidence_ids: list[str]
    question_type: str
    difficulty: str
    answer_requirements: list[str] = Field(default_factory=list)
    unsupported_answer_patterns: list[str] = Field(default_factory=list)

