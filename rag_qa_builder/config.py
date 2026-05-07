from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "rag_qa_dataset"
    language: str = "zh"


class InputConfig(BaseModel):
    file_types: list[str] = Field(default_factory=lambda: [".md", ".markdown", ".txt"])
    encoding: str = "utf-8"


class LLMConfig(BaseModel):
    enabled: bool = True
    profile: str = "kimi"
    model: str = "minimax-m1"
    config_path: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    retry_times: int = 3
    request_timeout_seconds: int = 120
    prompt_version: str = "v1"


class StructureConfig(BaseModel):
    markdown_heading_as_section: bool = True
    txt_section_detection: bool = True
    min_section_chars: int = 80
    max_section_chars_for_single_llm_call: int = 50000


class ConceptExtractionConfig(BaseModel):
    max_concepts_per_doc: int = 80
    min_importance: float = 0.4
    include_aliases: bool = True
    merge_similar_concepts: bool = True


class FactExtractionConfig(BaseModel):
    max_facts_per_concept: int = 20
    min_confidence: float = 0.65
    allowed_fact_types: list[str] = Field(default_factory=list)


class CombinationConfig(BaseModel):
    max_facts_per_combination: int = 4
    min_combination_score: float = 0.7
    enabled_patterns: list[str] = Field(default_factory=list)


class QAGenerationConfig(BaseModel):
    target_size: int = 200
    question_language: str = "zh"
    avoid_phrases: list[str] = Field(default_factory=list)
    distribution: dict[str, float] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    enabled: bool = True
    min_overall_score: float = 4.0
    require_answerability: bool = True
    require_faithfulness: bool = True
    reject_ambiguous_question: bool = True
    reject_external_knowledge: bool = True
    deduplicate_questions: bool = True
    duplicate_similarity_threshold: float = 0.88


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    concept_extraction: ConceptExtractionConfig = Field(default_factory=ConceptExtractionConfig)
    fact_extraction: FactExtractionConfig = Field(default_factory=FactExtractionConfig)
    combination: CombinationConfig = Field(default_factory=CombinationConfig)
    qa_generation: QAGenerationConfig = Field(default_factory=QAGenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


def load_config(config_path: str | Path | None) -> AppConfig:
    if not config_path:
        return AppConfig()
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)

