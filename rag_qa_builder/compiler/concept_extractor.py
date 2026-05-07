from __future__ import annotations

import re
from collections import defaultdict

from rag_qa_builder.config import AppConfig
from rag_qa_builder.llm.prompt_runner import PromptRunner
from rag_qa_builder.models import Concept, Document, DocumentSection
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import contains_cjk, keywords, normalize_text


def extract_concepts(
    documents: list[Document],
    sections: list[DocumentSection],
    config: AppConfig,
    prompt_runner: PromptRunner,
) -> list[Concept]:
    section_map = defaultdict(list)
    for section in sections:
        section_map[section.doc_id].append(section)

    raw: list[Concept] = []
    for document in documents:
        heuristics = _extract_heuristic_concepts(document, section_map[document.doc_id], config)
        raw.extend(heuristics)
        llm_payload = {
            "document": {
                "doc_id": document.doc_id,
                "file_name": document.file_name,
                "text": document.text[:20000],
            },
            "sections": [section.model_dump(mode="json") for section in section_map[document.doc_id][:20]],
        }
        llm_result = prompt_runner.maybe_run_json(
            "extract_concepts",
            "Extract core concepts from the document. Return JSON with a top-level 'concepts' array only. Do not use external knowledge.",
            llm_payload,
        )
        for item in (llm_result or {}).get("concepts", []):
            name = item.get("name") or item.get("canonical_name")
            if not name:
                continue
            raw.append(
                Concept(
                    concept_id=stable_id("concept", f"{document.doc_id}:{name}"),
                    canonical_name=name,
                    aliases=item.get("aliases", []),
                    concept_type=item.get("concept_type", "other"),
                    definition=item.get("definition"),
                    importance=float(item.get("importance", 0.6)),
                    source_section_ids=item.get("source_section_ids", []),
                )
            )
    return raw


def _extract_heuristic_concepts(document: Document, sections: list[DocumentSection], config: AppConfig) -> list[Concept]:
    candidates: dict[str, Concept] = {}
    for section in sections:
        for token in keywords(f"{section.title or ''} {section.text}", limit=20):
            if len(token) < 2:
                continue
            concept_type = _infer_concept_type(token, section.text)
            identifier = stable_id("concept", f"{document.doc_id}:{normalize_text(token)}")
            if identifier not in candidates:
                candidates[identifier] = Concept(
                    concept_id=identifier,
                    canonical_name=token,
                    aliases=[],
                    concept_type=concept_type,
                    definition=_find_definition(token, section.text),
                    importance=min(0.95, 0.4 + len(token) / 20),
                    source_section_ids=[section.section_id],
                    metadata={"source": "heuristic"},
                )
            elif section.section_id not in candidates[identifier].source_section_ids:
                candidates[identifier].source_section_ids.append(section.section_id)
    concepts = [concept for concept in candidates.values() if concept.importance >= config.concept_extraction.min_importance]
    return sorted(concepts, key=lambda item: item.importance, reverse=True)[: config.concept_extraction.max_concepts_per_doc]


def _infer_concept_type(name: str, text: str) -> str:
    lowered = normalize_text(f"{name} {text}")
    if any(word in lowered for word in ["流程", "步骤", "pipeline", "process"]):
        return "process"
    if any(word in lowered for word in ["配置", "config", "参数"]):
        return "config"
    if any(word in lowered for word in ["约束", "限制", "constraint"]):
        return "constraint"
    if any(word in lowered for word in ["错误", "异常", "error"]):
        return "error"
    if any(word in lowered for word in ["方法", "strategy", "approach"]):
        return "method"
    return "term" if contains_cjk(name) or re.search(r"[A-Za-z]", name) else "other"


def _find_definition(name: str, text: str) -> str | None:
    patterns = [
        rf"{re.escape(name)}[：:]\s*([^\n。]+)",
        rf"{re.escape(name)}\s*是([^\n。]+)",
        rf"{re.escape(name)}\s+refers to\s+([^\n\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

