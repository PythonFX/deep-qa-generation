from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag_qa_builder.config import AppConfig
from rag_qa_builder.exporters.json_exporter import export_json
from rag_qa_builder.exporters.jsonl_exporter import export_jsonl
from rag_qa_builder.llm.prompt_runner import PromptRunner
from rag_qa_builder.models import DocumentSection
from rag_qa_builder.readers import read_documents
from rag_qa_builder.compiler.structure_mapper import map_documents_to_sections
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import keywords, normalize_text, similarity, split_sentences, unwrap_line_breaks


QUESTION_WORTHY_TYPES = {
    "core_claim",
    "mechanism",
    "cause_effect",
    "condition_boundary",
    "contrast_tradeoff",
    "implication",
    "evidence_result",
    "procedure_logic",
}

VAGUE_QUESTION_PATTERNS = {
    "综合多个事实",
    "回答真实问题",
    "这些概念",
    "这些事实",
    "这种结果",
    "这个流程",
    "这里有哪些",
    "上述",
    "前文",
    "该方法",
    "该系统",
    "该模型",
}

MAX_UNIT_CHARS = 9000
MAX_CARDS_PER_UNIT = 8
MAX_CARDS_FOR_PLANNING = 120
MAX_CARDS_PER_PLAN_CALL = 18


class EvidenceSpan(BaseModel):
    evidence_id: str
    doc_id: str
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    char_start: int | None = None
    char_end: int | None = None
    source_hint: str | None = None


class EvidenceCard(BaseModel):
    card_id: str
    doc_id: str
    section_id: str
    section_path: list[str] = Field(default_factory=list)
    card_type: str
    subject: str
    event_or_topic: str
    claim: str
    mechanism: str | None = None
    conditions: list[str] = Field(default_factory=list)
    contrasts: list[str] = Field(default_factory=list)
    implications: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    reasoning_hooks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    centrality: float = 0.0
    reasoning_depth: float = 0.0
    evidence_density: float = 0.0
    novelty: float = 0.0
    synthesis_potential: float = 0.0
    quality_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuestionPlan(BaseModel):
    plan_id: str
    card_ids: list[str]
    question: str
    question_type: str
    reasoning_task: str
    expected_answer_points: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    quality_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeepQAPair(BaseModel):
    qa_id: str
    question: str
    reference_answer: str
    answer_points: list[str] = Field(default_factory=list)
    card_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    question_type: str
    difficulty: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeepQAValidationResult(BaseModel):
    qa_id: str
    is_self_contained_question: bool
    is_answerable: bool
    is_evidence_grounded: bool
    has_reasoning_value: bool
    is_duplicate: bool
    scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    passed: bool = False


class DeepDatasetEntry(BaseModel):
    id: str
    question: str
    reference_answer: str
    reference_context: list[dict[str, Any]]
    question_type: str
    difficulty: str
    answer_points: list[str] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class DeepQAPipeline:
    """Generate RAG benchmark QA from question-worthy evidence cards.

    This pipeline intentionally avoids the older concept/fact/relation graph. It
    treats the source document as a set of grounded reasoning opportunities:
    claims, mechanisms, boundaries, tradeoffs, implications, and result evidence.
    """

    def __init__(self, input_path: str | Path, output_dir: str | Path, config: AppConfig, dry_run: bool = False) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.dry_run = dry_run
        if dry_run:
            self.config.llm.enabled = False
        self.prompt_runner = PromptRunner(config, self.output_dir)

    def generate_all(self) -> dict[str, Any]:
        documents, sections = self.build_structure()
        cards, evidence = self.build_evidence_cards(sections)
        plans = self.build_question_plans(cards, evidence)
        qa_candidates = self.generate_qa_pairs(plans, cards, evidence)
        validations, passed = self.validate_qa_pairs(qa_candidates, plans, cards, evidence)
        dataset = self.export_final(passed, evidence)
        return {
            "documents": documents,
            "sections": sections,
            "evidence_cards": cards,
            "evidence": evidence,
            "question_plans": plans,
            "qa_candidates": qa_candidates,
            "qa_validated": validations,
            "dataset": dataset,
        }

    def build_structure(self) -> tuple[list, list[DocumentSection]]:
        documents, read_errors = read_documents(self.input_path, self.config.input.file_types, self.config.input.encoding)
        sections = map_documents_to_sections(documents)
        export_json(self.output_dir, "documents.json", documents)
        export_json(self.output_dir, "document_structure.json", sections)
        if read_errors:
            export_jsonl(self.output_dir, "errors.jsonl", read_errors)
        return documents, sections

    def build_evidence_cards(self, sections: list[DocumentSection]) -> tuple[list[EvidenceCard], list[EvidenceSpan]]:
        cards: list[EvidenceCard] = []
        evidence: list[EvidenceSpan] = []
        for unit in _build_card_units(sections):
            unit = _clean_section(unit)
            llm_payload = self.prompt_runner.maybe_run_json(
                "deep_extract_evidence_cards",
                _evidence_card_system_prompt(),
                {
                    "section": _section_payload(unit),
                    "instructions": {
                        "goal": "Extract only question-worthy, evidence-grounded reasoning units.",
                        "max_cards": MAX_CARDS_PER_UNIT,
                        "allowed_card_types": sorted(QUESTION_WORTHY_TYPES),
                        "self_contained_future_questions": (
                            "Every future question must mention the concrete subject and event/topic. "
                            "Do not create abstract prompts such as '综合多个事实回答真实问题'."
                        ),
                    },
                    "schema_hint": {
                        "cards": [
                            {
                                "card_type": "mechanism",
                                "subject": "specific method/entity/event from the text",
                                "event_or_topic": "specific action, result, design choice, or claim",
                                "claim": "one complete statement grounded in this section",
                                "mechanism": "why or how it works, if stated",
                                "conditions": ["conditions or boundaries, if stated"],
                                "contrasts": ["tradeoffs or comparisons, if stated"],
                                "implications": ["what this claim implies, if stated"],
                                "caveats": ["limitations, if stated"],
                                "reasoning_hooks": ["why this would make a valuable non-trivial QA item"],
                                "evidence_spans": ["short exact source snippets supporting the card"],
                                "centrality": 0.0,
                                "reasoning_depth": 0.0,
                                "evidence_density": 0.0,
                                "novelty": 0.0,
                                "synthesis_potential": 0.0,
                            }
                        ]
                    },
                },
            )
            parsed_cards, parsed_evidence = _parse_evidence_cards(unit, llm_payload)
            if not parsed_cards:
                parsed_cards, parsed_evidence = _heuristic_evidence_cards(unit)
            cards.extend(parsed_cards)
            evidence.extend(parsed_evidence)

        cards = _deduplicate_cards(cards)
        evidence = _filter_evidence_for_cards(evidence, cards)
        cards.sort(key=lambda item: (item.quality_score, item.reasoning_depth, item.centrality), reverse=True)
        export_json(self.output_dir, "evidence_cards.json", cards)
        export_json(self.output_dir, "evidence_spans.json", evidence)
        return cards, evidence

    def build_question_plans(self, cards: list[EvidenceCard], evidence: list[EvidenceSpan]) -> list[QuestionPlan]:
        selected_cards = cards[:MAX_CARDS_FOR_PLANNING]
        plans: list[QuestionPlan] = []
        for batch in _chunk(selected_cards, MAX_CARDS_PER_PLAN_CALL):
            llm_payload = self.prompt_runner.maybe_run_json(
                "deep_plan_questions",
                _question_plan_system_prompt(),
                {
                    "cards": [_card_payload(card) for card in batch],
                    "requirements": {
                        "target": "Create difficult but answerable RAG benchmark question plans.",
                        "self_contained_question": (
                            "The question must explicitly include the concrete subject and event/topic. "
                            "It must still make sense if shown alone to a RAG system."
                        ),
                        "avoid": sorted(VAGUE_QUESTION_PATTERNS),
                        "preferred_question_types": [
                            "why_mechanism",
                            "cause_effect",
                            "compare_tradeoff",
                            "condition_boundary",
                            "result_interpretation",
                            "multi_step_synthesis",
                        ],
                    },
                    "schema_hint": {
                        "plans": [
                            {
                                "card_ids": ["card id(s) used"],
                                "question": "standalone question with explicit subject and event/topic",
                                "question_type": "why_mechanism",
                                "reasoning_task": "what reasoning the answer must perform",
                                "expected_answer_points": ["grounded answer point"],
                                "difficulty": "medium|hard",
                                "quality_score": 0.0,
                            }
                        ]
                    },
                },
            )
            batch_plans = _parse_question_plans(llm_payload, {card.card_id: card for card in batch})
            if not batch_plans:
                batch_plans = _heuristic_question_plans(batch)
            plans.extend(batch_plans)

        plans.extend(_cross_card_synthesis_plans(selected_cards))
        plans = _deduplicate_plans(plans, {card.card_id: card for card in selected_cards})
        plans = plans[: self.config.qa_generation.target_size * 3]
        export_json(self.output_dir, "question_plans.deep.json", plans)
        return plans

    def generate_qa_pairs(
        self,
        plans: list[QuestionPlan],
        cards: list[EvidenceCard],
        evidence: list[EvidenceSpan],
    ) -> list[DeepQAPair]:
        card_lookup = {card.card_id: card for card in cards}
        evidence_lookup = {item.evidence_id: item for item in evidence}
        qa_pairs: list[DeepQAPair] = []
        for plan in plans:
            plan_cards = [card_lookup[card_id] for card_id in plan.card_ids if card_id in card_lookup]
            if not plan_cards or not _question_is_self_contained(plan.question, plan_cards):
                continue
            plan_evidence = [
                evidence_lookup[evidence_id]
                for card in plan_cards
                for evidence_id in card.evidence_ids
                if evidence_id in evidence_lookup
            ]
            llm_payload = self.prompt_runner.maybe_run_json(
                "deep_generate_grounded_answer",
                _answer_system_prompt(),
                {
                    "question_plan": plan.model_dump(mode="json"),
                    "evidence_cards": [_card_payload(card) for card in plan_cards],
                    "evidence_spans": [item.model_dump(mode="json") for item in plan_evidence],
                    "requirements": {
                        "grounding": "Every answer point must be supported by the provided evidence spans.",
                        "no_external_knowledge": "Do not add knowledge not stated or directly implied by the source.",
                        "answer_style": "Answer directly and analytically, not as a quote dump.",
                    },
                    "schema_hint": {
                        "reference_answer": "complete answer grounded in the article",
                        "answer_points": ["one grounded answer point"],
                        "cited_evidence_ids": ["evidence ids used"],
                    },
                },
            )
            qa = _parse_qa_pair(plan, plan_cards, plan_evidence, llm_payload)
            if qa is None:
                qa = _heuristic_qa_pair(plan, plan_cards, plan_evidence)
            if qa and qa.reference_answer:
                qa_pairs.append(qa)
            if len(qa_pairs) >= self.config.qa_generation.target_size * 2:
                break
        export_jsonl(self.output_dir, "qa_candidates.deep.jsonl", qa_pairs)
        return qa_pairs

    def validate_qa_pairs(
        self,
        qa_pairs: list[DeepQAPair],
        plans: list[QuestionPlan],
        cards: list[EvidenceCard],
        evidence: list[EvidenceSpan],
    ) -> tuple[list[DeepQAValidationResult], list[DeepQAPair]]:
        card_lookup = {card.card_id: card for card in cards}
        evidence_lookup = {item.evidence_id: item for item in evidence}
        seen_questions: list[str] = []
        validations: list[DeepQAValidationResult] = []
        passed: list[DeepQAPair] = []
        for qa in qa_pairs:
            qa_cards = [card_lookup[card_id] for card_id in qa.card_ids if card_id in card_lookup]
            qa_evidence = [evidence_lookup[evidence_id] for evidence_id in qa.evidence_ids if evidence_id in evidence_lookup]
            self_contained = _question_is_self_contained(qa.question, qa_cards)
            answerable = bool(qa_cards and qa_evidence and qa.reference_answer.strip())
            grounded = _answer_is_grounded(qa.reference_answer, qa_evidence, qa_cards)
            reasoning_value = _has_reasoning_value(qa, qa_cards)
            duplicate = any(similarity(qa.question, seen) >= self.config.validation.duplicate_similarity_threshold for seen in seen_questions)
            scores = {
                "self_contained_question": 5.0 if self_contained else 1.0,
                "answerability": 5.0 if answerable else 0.0,
                "evidence_grounding": 5.0 if grounded else 2.0,
                "reasoning_value": 5.0 if reasoning_value else 2.0,
                "duplication": 1.0 if duplicate else 5.0,
            }
            issues: list[str] = []
            if not self_contained:
                issues.append("question_not_self_contained")
            if not answerable:
                issues.append("not_answerable_from_bound_evidence")
            if not grounded:
                issues.append("answer_not_grounded_in_evidence")
            if not reasoning_value:
                issues.append("low_reasoning_value")
            if duplicate:
                issues.append("duplicate_question")
            validation = DeepQAValidationResult(
                qa_id=qa.qa_id,
                is_self_contained_question=self_contained,
                is_answerable=answerable,
                is_evidence_grounded=grounded,
                has_reasoning_value=reasoning_value,
                is_duplicate=duplicate,
                scores=scores,
                issues=issues,
                passed=not issues and sum(scores.values()) / len(scores) >= self.config.validation.min_overall_score,
            )
            validations.append(validation)
            if validation.passed:
                seen_questions.append(qa.question)
                passed.append(qa)
            if len(passed) >= self.config.qa_generation.target_size:
                break
        export_jsonl(self.output_dir, "qa_validated.deep.jsonl", validations)
        export_jsonl(self.output_dir, "qa_rejected.deep.jsonl", [qa for qa in qa_pairs if qa.qa_id not in {item.qa_id for item in passed}])
        return validations, passed

    def export_final(self, qa_pairs: list[DeepQAPair], evidence: list[EvidenceSpan]) -> list[DeepDatasetEntry]:
        evidence_lookup = {item.evidence_id: item for item in evidence}
        dataset: list[DeepDatasetEntry] = []
        for qa in qa_pairs:
            dataset.append(
                DeepDatasetEntry(
                    id=qa.qa_id,
                    question=qa.question,
                    reference_answer=qa.reference_answer,
                    reference_context=[
                        evidence_lookup[evidence_id].model_dump(mode="json")
                        for evidence_id in qa.evidence_ids
                        if evidence_id in evidence_lookup
                    ],
                    question_type=qa.question_type,
                    difficulty=qa.difficulty,
                    answer_points=qa.answer_points,
                    evidence_card_ids=qa.card_ids,
                    evidence_ids=qa.evidence_ids,
                )
            )
        export_jsonl(self.output_dir, "dataset.deep.final.jsonl", dataset)
        return dataset


def _evidence_card_system_prompt() -> str:
    return (
        "You are building a high-quality RAG QA benchmark from a source section. "
        "Do not extract generic concepts or isolated atomic facts. Extract evidence cards: "
        "claims, mechanisms, conditions, tradeoffs, implications, caveats, and results that can support difficult questions. "
        "A good card preserves the concrete subject and event/topic, contains exact evidence spans, and explains why it is question-worthy. "
        "Reject metadata, author names, references, filler definitions, and anything not useful for deep comprehension QA. "
        "Return JSON only."
    )


def _question_plan_system_prompt() -> str:
    return (
        "You design RAG benchmark questions from evidence cards. "
        "Each question must be self-contained: it must name the concrete subject and the concrete event, design choice, result, or claim. "
        "Never ask abstract questions like '综合多个事实回答真实问题' or '这些概念有什么区别'. "
        "Favor questions requiring explanation, causal reasoning, condition/boundary analysis, tradeoff comparison, result interpretation, or multi-card synthesis. "
        "Every question must be answerable using only the supplied cards. Return JSON only."
    )


def _answer_system_prompt() -> str:
    return (
        "You write reference answers for RAG evaluation. "
        "Use only the supplied evidence cards and evidence spans. "
        "The answer should synthesize and explain, but every answer point must be grounded in the provided source evidence. "
        "Do not introduce external knowledge. Return JSON only."
    )


def _build_card_units(sections: list[DocumentSection]) -> list[DocumentSection]:
    units: list[DocumentSection] = []
    for section in sections:
        if len(section.text) <= MAX_UNIT_CHARS:
            units.append(section)
            continue
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", section.text) if item.strip()]
        current = ""
        cursor = section.char_start
        index = 1
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if current and len(candidate) > MAX_UNIT_CHARS:
                units.append(_section_part(section, current, index, cursor))
                cursor += len(current)
                index += 1
                current = paragraph
            else:
                current = candidate
        if current:
            units.append(_section_part(section, current, index, cursor))
    return units


def _section_part(section: DocumentSection, text: str, index: int, cursor: int) -> DocumentSection:
    title = section.title or (section.section_path[-1] if section.section_path else "section")
    return DocumentSection(
        section_id=stable_id("deep_sec", f"{section.section_id}:{index}:{cursor}"),
        doc_id=section.doc_id,
        title=f"{title} [{index}]",
        level=section.level,
        section_path=section.section_path,
        text=text,
        char_start=cursor,
        char_end=cursor + len(text),
        summary=text[:200].strip(),
        keywords=section.keywords,
    )


def _clean_section(section: DocumentSection) -> DocumentSection:
    cleaned = unwrap_line_breaks(section.text)
    lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if "@" in stripped:
            continue
        if any(token in lowered for token in ["copyright", "permission to", "references", "bibliography"]):
            continue
        lines.append(line)
    text = "\n".join(lines).strip() or section.text
    return DocumentSection(
        section_id=section.section_id,
        doc_id=section.doc_id,
        title=section.title,
        level=section.level,
        section_path=section.section_path,
        text=text,
        char_start=section.char_start,
        char_end=section.char_start + len(text),
        summary=text[:200].strip(),
        keywords=section.keywords or keywords(text),
    )


def _section_payload(section: DocumentSection) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "doc_id": section.doc_id,
        "title": section.title,
        "section_path": section.section_path,
        "text": section.text,
    }


def _parse_evidence_cards(section: DocumentSection, payload: dict[str, Any] | None) -> tuple[list[EvidenceCard], list[EvidenceSpan]]:
    if not payload:
        return [], []
    raw_cards = payload.get("cards", []) if isinstance(payload, dict) else []
    if isinstance(raw_cards, dict):
        raw_cards = raw_cards.get("items", [])
    if not isinstance(raw_cards, list):
        return [], []

    cards: list[EvidenceCard] = []
    evidence: list[EvidenceSpan] = []
    for index, item in enumerate(raw_cards[:MAX_CARDS_PER_UNIT], start=1):
        if not isinstance(item, dict):
            continue
        subject = _clean_short_text(item.get("subject", ""))
        event = _clean_short_text(item.get("event_or_topic", ""))
        claim = _clean_long_text(item.get("claim", ""))
        card_type = _normalize_card_type(item.get("card_type", "core_claim"))
        spans = _normalize_string_list(item.get("evidence_spans", []))
        if not subject or not event or not claim or not spans:
            continue
        if not _specific_enough(subject) or not _specific_enough(event):
            continue
        evidence_ids: list[str] = []
        for span_index, span in enumerate(spans[:4], start=1):
            span_text = _best_source_span(section.text, span)
            if not span_text:
                continue
            start = section.text.find(span_text)
            evidence_id = stable_id("deep_ev", f"{section.section_id}:{index}:{span_index}:{span_text}")
            evidence.append(
                EvidenceSpan(
                    evidence_id=evidence_id,
                    doc_id=section.doc_id,
                    section_id=section.section_id,
                    section_path=section.section_path,
                    text=span_text,
                    char_start=section.char_start + start if start >= 0 else None,
                    char_end=section.char_start + start + len(span_text) if start >= 0 else None,
                    source_hint=section.title,
                )
            )
            evidence_ids.append(evidence_id)
        if not evidence_ids:
            continue
        scores = _card_scores(item)
        card_id = stable_id("deep_card", f"{section.section_id}:{subject}:{event}:{claim}")
        cards.append(
            EvidenceCard(
                card_id=card_id,
                doc_id=section.doc_id,
                section_id=section.section_id,
                section_path=section.section_path,
                card_type=card_type,
                subject=subject,
                event_or_topic=event,
                claim=claim,
                mechanism=_clean_long_text(item.get("mechanism", "")) or None,
                conditions=_normalize_string_list(item.get("conditions", [])),
                contrasts=_normalize_string_list(item.get("contrasts", [])),
                implications=_normalize_string_list(item.get("implications", [])),
                caveats=_normalize_string_list(item.get("caveats", [])),
                reasoning_hooks=_normalize_string_list(item.get("reasoning_hooks", [])),
                evidence_ids=evidence_ids,
                centrality=scores["centrality"],
                reasoning_depth=scores["reasoning_depth"],
                evidence_density=scores["evidence_density"],
                novelty=scores["novelty"],
                synthesis_potential=scores["synthesis_potential"],
                quality_score=scores["quality_score"],
                metadata={"source": "llm"},
            )
        )
    return cards, evidence


def _heuristic_evidence_cards(section: DocumentSection) -> tuple[list[EvidenceCard], list[EvidenceSpan]]:
    sentences = [sentence for sentence in split_sentences(section.text) if _sentence_is_card_candidate(sentence)]
    title_subject = _section_subject(section)
    section_keywords = section.keywords or keywords(section.text, limit=6)
    cards: list[EvidenceCard] = []
    evidence: list[EvidenceSpan] = []
    for sentence in sentences:
        card_type = _infer_card_type(sentence)
        if card_type == "core_claim" and len(cards) >= 2:
            continue
        subject = _infer_subject(sentence, title_subject, section_keywords)
        event = _infer_event(sentence, subject)
        if not subject or not event:
            continue
        evidence_id = stable_id("deep_ev", f"{section.section_id}:{sentence}")
        start = section.text.find(sentence)
        evidence.append(
            EvidenceSpan(
                evidence_id=evidence_id,
                doc_id=section.doc_id,
                section_id=section.section_id,
                section_path=section.section_path,
                text=sentence,
                char_start=section.char_start + start if start >= 0 else None,
                char_end=section.char_start + start + len(sentence) if start >= 0 else None,
                source_hint=section.title,
            )
        )
        score = _heuristic_card_quality(sentence, card_type)
        cards.append(
            EvidenceCard(
                card_id=stable_id("deep_card", f"{section.section_id}:{subject}:{event}:{sentence}"),
                doc_id=section.doc_id,
                section_id=section.section_id,
                section_path=section.section_path,
                card_type=card_type,
                subject=subject,
                event_or_topic=event,
                claim=sentence,
                mechanism=sentence if card_type in {"mechanism", "cause_effect"} else None,
                conditions=[sentence] if card_type == "condition_boundary" else [],
                contrasts=[sentence] if card_type == "contrast_tradeoff" else [],
                implications=[sentence] if card_type == "implication" else [],
                reasoning_hooks=[_reasoning_hook_for_type(card_type)],
                evidence_ids=[evidence_id],
                centrality=score,
                reasoning_depth=score,
                evidence_density=0.75,
                novelty=score,
                synthesis_potential=score,
                quality_score=score,
                metadata={"source": "heuristic"},
            )
        )
        if len(cards) >= MAX_CARDS_PER_UNIT:
            break
    return cards, evidence


def _parse_question_plans(payload: dict[str, Any] | None, card_lookup: dict[str, EvidenceCard]) -> list[QuestionPlan]:
    if not payload:
        return []
    raw_plans = payload.get("plans", []) if isinstance(payload, dict) else []
    if isinstance(raw_plans, dict):
        raw_plans = raw_plans.get("items", [])
    if not isinstance(raw_plans, list):
        return []
    plans: list[QuestionPlan] = []
    for item in raw_plans:
        if not isinstance(item, dict):
            continue
        card_ids = [card_id for card_id in _normalize_string_list(item.get("card_ids", [])) if card_id in card_lookup]
        question = _clean_question(item.get("question", ""))
        if not card_ids or not question:
            continue
        plan_cards = [card_lookup[card_id] for card_id in card_ids]
        if not _question_is_self_contained(question, plan_cards):
            continue
        plan_id = stable_id("deep_plan", f"{question}:{'|'.join(card_ids)}")
        plans.append(
            QuestionPlan(
                plan_id=plan_id,
                card_ids=card_ids,
                question=question,
                question_type=_clean_short_text(item.get("question_type", "multi_step_synthesis")) or "multi_step_synthesis",
                reasoning_task=_clean_long_text(item.get("reasoning_task", "")) or "基于文章证据进行解释和综合。",
                expected_answer_points=_normalize_string_list(item.get("expected_answer_points", [])),
                difficulty=_normalize_difficulty(item.get("difficulty", "medium")),
                quality_score=_safe_float(item.get("quality_score", 0.8), 0.8),
                metadata={"source": "llm"},
            )
        )
    return plans


def _heuristic_question_plans(cards: list[EvidenceCard]) -> list[QuestionPlan]:
    plans: list[QuestionPlan] = []
    for card in cards:
        question_type, question = _question_for_card(card)
        if not _question_is_self_contained(question, [card]):
            continue
        plans.append(
            QuestionPlan(
                plan_id=stable_id("deep_plan", f"{card.card_id}:{question}"),
                card_ids=[card.card_id],
                question=question,
                question_type=question_type,
                reasoning_task=_reasoning_task_for_type(question_type),
                expected_answer_points=[card.claim],
                difficulty="hard" if card.reasoning_depth >= 0.75 else "medium",
                quality_score=card.quality_score,
                metadata={"source": "heuristic"},
            )
        )
    return plans


def _cross_card_synthesis_plans(cards: list[EvidenceCard]) -> list[QuestionPlan]:
    by_section: dict[str, list[EvidenceCard]] = defaultdict(list)
    for card in cards:
        by_section[card.section_id].append(card)
    plans: list[QuestionPlan] = []
    for section_cards in by_section.values():
        ranked = sorted(section_cards, key=lambda item: item.quality_score, reverse=True)[:6]
        for left_index, left in enumerate(ranked):
            for right in ranked[left_index + 1 :]:
                if not _cards_can_synthesize(left, right):
                    continue
                question = (
                    f"文章中{left.subject}关于{left.event_or_topic}的论述，"
                    f"和{right.subject}关于{right.event_or_topic}的论述之间有什么因果、取舍或边界关系？"
                )
                if not _question_is_self_contained(question, [left, right]):
                    continue
                plans.append(
                    QuestionPlan(
                        plan_id=stable_id("deep_plan", f"{left.card_id}:{right.card_id}:{question}"),
                        card_ids=[left.card_id, right.card_id],
                        question=question,
                        question_type="multi_step_synthesis",
                        reasoning_task="综合两张证据卡，解释它们之间的因果、取舍或边界关系。",
                        expected_answer_points=[left.claim, right.claim],
                        difficulty="hard",
                        quality_score=round((left.quality_score + right.quality_score) / 2 + 0.1, 3),
                        metadata={"source": "heuristic_cross_card"},
                    )
                )
                break
    return plans


def _parse_qa_pair(
    plan: QuestionPlan,
    cards: list[EvidenceCard],
    evidence: list[EvidenceSpan],
    payload: dict[str, Any] | None,
) -> DeepQAPair | None:
    if not payload:
        return None
    answer = _clean_long_text(payload.get("reference_answer", ""))
    if not answer:
        return None
    evidence_ids = _normalize_string_list(payload.get("cited_evidence_ids", []))
    valid_evidence_ids = {item.evidence_id for item in evidence}
    evidence_ids = [evidence_id for evidence_id in evidence_ids if evidence_id in valid_evidence_ids]
    if not evidence_ids:
        evidence_ids = sorted({evidence_id for card in cards for evidence_id in card.evidence_ids})
    return DeepQAPair(
        qa_id=stable_id("deep_qa", plan.plan_id),
        question=plan.question,
        reference_answer=answer,
        answer_points=_normalize_string_list(payload.get("answer_points", [])) or plan.expected_answer_points,
        card_ids=plan.card_ids,
        evidence_ids=evidence_ids,
        question_type=plan.question_type,
        difficulty=plan.difficulty,
        metadata={"source": "llm", "plan_id": plan.plan_id},
    )


def _heuristic_qa_pair(plan: QuestionPlan, cards: list[EvidenceCard], evidence: list[EvidenceSpan]) -> DeepQAPair:
    answer_points: list[str] = []
    for card in cards:
        parts = [card.claim]
        if card.mechanism and card.mechanism not in parts:
            parts.append(card.mechanism)
        parts.extend(item for item in card.conditions[:2] if item not in parts)
        parts.extend(item for item in card.contrasts[:2] if item not in parts)
        parts.extend(item for item in card.implications[:2] if item not in parts)
        answer_points.extend(parts)
    answer_points = _dedupe_strings(answer_points)[:6]
    answer = " ".join(answer_points)
    return DeepQAPair(
        qa_id=stable_id("deep_qa", plan.plan_id),
        question=plan.question,
        reference_answer=answer,
        answer_points=answer_points,
        card_ids=plan.card_ids,
        evidence_ids=sorted({item.evidence_id for item in evidence}),
        question_type=plan.question_type,
        difficulty=plan.difficulty,
        metadata={"source": "heuristic", "plan_id": plan.plan_id},
    )


def _card_payload(card: EvidenceCard) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "card_type": card.card_type,
        "subject": card.subject,
        "event_or_topic": card.event_or_topic,
        "claim": card.claim,
        "mechanism": card.mechanism,
        "conditions": card.conditions,
        "contrasts": card.contrasts,
        "implications": card.implications,
        "caveats": card.caveats,
        "reasoning_hooks": card.reasoning_hooks,
        "evidence_ids": card.evidence_ids,
        "scores": {
            "centrality": card.centrality,
            "reasoning_depth": card.reasoning_depth,
            "evidence_density": card.evidence_density,
            "novelty": card.novelty,
            "synthesis_potential": card.synthesis_potential,
            "quality_score": card.quality_score,
        },
    }


def _question_for_card(card: EvidenceCard) -> tuple[str, str]:
    subject = card.subject
    topic = card.event_or_topic
    if card.card_type in {"mechanism", "cause_effect", "procedure_logic"}:
        return "why_mechanism", f"文章如何解释{subject}在{topic}中的作用机制或因果逻辑？"
    if card.card_type == "contrast_tradeoff":
        return "compare_tradeoff", f"文章如何分析{subject}在{topic}上的取舍，关键差异会带来什么影响？"
    if card.card_type == "condition_boundary":
        return "condition_boundary", f"文章指出{subject}在{topic}上有哪些条件或边界限制，这些限制为什么重要？"
    if card.card_type in {"implication", "evidence_result"}:
        return "result_interpretation", f"文章中{subject}关于{topic}的结果或结论意味着什么，依据是什么？"
    return "multi_step_synthesis", f"文章中{subject}关于{topic}的核心论述是什么，它依赖哪些证据或理由？"


def _question_is_self_contained(question: str, cards: list[EvidenceCard]) -> bool:
    normalized = normalize_text(question)
    if len(question.strip()) < 18:
        return False
    if any(pattern in question for pattern in VAGUE_QUESTION_PATTERNS):
        return False
    if not cards:
        return False
    subject_hits = 0
    topic_hits = 0
    for card in cards:
        subject = normalize_text(card.subject)
        topic = normalize_text(card.event_or_topic)
        if subject and subject in normalized:
            subject_hits += 1
        if topic and (topic in normalized or _loose_overlap(topic, normalized)):
            topic_hits += 1
    return subject_hits >= 1 and topic_hits >= 1


def _answer_is_grounded(answer: str, evidence: list[EvidenceSpan], cards: list[EvidenceCard]) -> bool:
    if not answer.strip() or not evidence:
        return False
    evidence_text = " ".join(item.text for item in evidence)
    overlap = _token_overlap(answer, evidence_text)
    card_claim_overlap = max((_token_overlap(answer, card.claim) for card in cards), default=0.0)
    return overlap >= 0.18 or card_claim_overlap >= 0.35


def _has_reasoning_value(qa: DeepQAPair, cards: list[EvidenceCard]) -> bool:
    if len(cards) >= 2:
        return True
    if not cards:
        return False
    card = cards[0]
    if card.card_type in {"mechanism", "cause_effect", "condition_boundary", "contrast_tradeoff", "implication"}:
        return True
    reasoning_terms = ["为什么", "如何解释", "取舍", "边界", "意味着", "因果", "依据", "机制", "影响"]
    return any(term in qa.question for term in reasoning_terms)


def _deduplicate_cards(cards: list[EvidenceCard]) -> list[EvidenceCard]:
    unique: list[EvidenceCard] = []
    for card in cards:
        if any(similarity(card.claim, existing.claim) > 0.9 for existing in unique):
            continue
        unique.append(card)
    return unique


def _filter_evidence_for_cards(evidence: list[EvidenceSpan], cards: list[EvidenceCard]) -> list[EvidenceSpan]:
    used = {evidence_id for card in cards for evidence_id in card.evidence_ids}
    deduped: dict[str, EvidenceSpan] = {}
    for item in evidence:
        if item.evidence_id in used:
            deduped[item.evidence_id] = item
    return list(deduped.values())


def _deduplicate_plans(plans: list[QuestionPlan], card_lookup: dict[str, EvidenceCard]) -> list[QuestionPlan]:
    unique: list[QuestionPlan] = []
    for plan in sorted(plans, key=lambda item: item.quality_score, reverse=True):
        cards = [card_lookup[card_id] for card_id in plan.card_ids if card_id in card_lookup]
        if not _question_is_self_contained(plan.question, cards):
            continue
        if any(similarity(plan.question, existing.question) > 0.86 for existing in unique):
            continue
        unique.append(plan)
    return unique


def _cards_can_synthesize(left: EvidenceCard, right: EvidenceCard) -> bool:
    if left.card_id == right.card_id:
        return False
    if left.subject == right.subject:
        return True
    if left.card_type != right.card_type and (left.synthesis_potential + right.synthesis_potential) >= 1.2:
        return True
    return _token_overlap(left.claim, right.claim) >= 0.18


def _normalize_card_type(value: Any) -> str:
    text = _clean_short_text(value).lower()
    return text if text in QUESTION_WORTHY_TYPES else "core_claim"


def _normalize_difficulty(value: Any) -> str:
    text = _clean_short_text(value).lower()
    return text if text in {"easy", "medium", "hard"} else "medium"


def _card_scores(item: dict[str, Any]) -> dict[str, float]:
    centrality = _safe_float(item.get("centrality", 0.7), 0.7)
    reasoning_depth = _safe_float(item.get("reasoning_depth", 0.7), 0.7)
    evidence_density = _safe_float(item.get("evidence_density", 0.7), 0.7)
    novelty = _safe_float(item.get("novelty", 0.65), 0.65)
    synthesis = _safe_float(item.get("synthesis_potential", 0.65), 0.65)
    quality = round((centrality * 0.25) + (reasoning_depth * 0.3) + (evidence_density * 0.2) + (novelty * 0.1) + (synthesis * 0.15), 3)
    return {
        "centrality": centrality,
        "reasoning_depth": reasoning_depth,
        "evidence_density": evidence_density,
        "novelty": novelty,
        "synthesis_potential": synthesis,
        "quality_score": quality,
    }


def _heuristic_card_quality(sentence: str, card_type: str) -> float:
    base = 0.55
    if card_type in {"mechanism", "cause_effect", "contrast_tradeoff", "condition_boundary"}:
        base += 0.2
    if any(term in sentence.lower() for term in ["because", "therefore", "however", "while", "although", "so that"]):
        base += 0.1
    if any(term in sentence for term in ["因为", "因此", "然而", "虽然", "导致", "取决于", "限制", "相比"]):
        base += 0.1
    if len(sentence) > 120:
        base += 0.05
    return round(min(base, 0.95), 3)


def _infer_card_type(sentence: str) -> str:
    lowered = sentence.lower()
    if any(term in sentence for term in ["因为", "因此", "导致", "使得", "原因"]) or any(term in lowered for term in ["because", "therefore", "leads to", "resulting"]):
        return "cause_effect"
    if any(term in sentence for term in ["通过", "机制", "过程", "步骤"]) or any(term in lowered for term in ["mechanism", "process", "by using"]):
        return "mechanism"
    if any(term in sentence for term in ["相比", "不同", "取舍", "然而", "但"]) or any(term in lowered for term in ["however", "whereas", "while", "tradeoff", "unlike"]):
        return "contrast_tradeoff"
    if any(term in sentence for term in ["如果", "当", "只有", "限制", "条件", "必须", "不能"]) or any(term in lowered for term in ["if", "when", "only", "must", "cannot", "constraint", "limitation"]):
        return "condition_boundary"
    if any(term in sentence for term in ["意味着", "说明", "表明", "容易", "可以"]) or any(term in lowered for term in ["implies", "suggests", "shows", "can", "could"]):
        return "implication"
    if any(term in sentence for term in ["结果", "实验", "性能"]) or any(term in lowered for term in ["result", "experiment", "performance"]):
        return "evidence_result"
    return "core_claim"


def _infer_subject(sentence: str, title_subject: str, section_keywords: list[str]) -> str:
    if re.search(r"[\u4e00-\u9fff]", sentence) and _specific_enough(title_subject):
        return title_subject
    candidates = re.findall(r"[A-Z][A-Za-z0-9_\-]{2,}(?:\s+[A-Z][A-Za-z0-9_\-]{2,})*|[\u4e00-\u9fffA-Za-z0-9_\-]{2,12}", sentence)
    for candidate in candidates:
        cleaned = _clean_short_text(candidate)
        if _specific_enough(cleaned):
            return cleaned
    for keyword in section_keywords:
        if _specific_enough(keyword):
            return keyword
    return title_subject if _specific_enough(title_subject) else ""


def _infer_event(sentence: str, subject: str) -> str:
    compact = _compact_event_label(sentence)
    if compact:
        return compact
    cleaned = sentence.replace(subject, "", 1).strip(" ，,。.;；") if subject in sentence else sentence.strip(" ，,。.;；")
    if not cleaned:
        return ""
    if len(cleaned) <= 60:
        return cleaned
    return cleaned[:60].rstrip(" ，,。.;；")


def _reasoning_hook_for_type(card_type: str) -> str:
    mapping = {
        "mechanism": "需要解释机制，而不是复述定义。",
        "cause_effect": "需要串联原因和结果。",
        "condition_boundary": "需要识别适用条件和限制。",
        "contrast_tradeoff": "需要比较取舍及其影响。",
        "implication": "需要解释结论意味着什么。",
        "evidence_result": "需要把结果和依据联系起来。",
        "procedure_logic": "需要解释步骤顺序和作用。",
    }
    return mapping.get(card_type, "需要基于证据综合文章的核心论述。")


def _reasoning_task_for_type(question_type: str) -> str:
    mapping = {
        "why_mechanism": "解释文章中给出的机制、原因和影响。",
        "cause_effect": "串联文章中的原因、过程和结果。",
        "compare_tradeoff": "比较文章中明确提出的差异、取舍和影响。",
        "condition_boundary": "识别文章中的条件、限制和适用边界。",
        "result_interpretation": "解释文章结果或结论的含义及依据。",
    }
    return mapping.get(question_type, "综合文章证据回答非表层问题。")


def _best_source_span(source: str, span: str) -> str:
    span = _clean_long_text(span)
    if not span:
        return ""
    if span in source:
        return span
    normalized_span = normalize_text(span)
    best = ""
    best_score = 0.0
    for sentence in split_sentences(source):
        score = similarity(normalized_span, normalize_text(sentence))
        if score > best_score:
            best = sentence
            best_score = score
    return best if best_score >= 0.55 else span[:500]


def _clean_question(value: Any) -> str:
    text = _clean_long_text(value)
    if text and text[-1] not in "？?":
        text += "？"
    return text


def _clean_short_text(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:120].strip(" ，,。.;；")


def _clean_long_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_clean_long_text(item) for item in value if _clean_long_text(item)]


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if not any(similarity(value, existing) > 0.9 for existing in deduped):
            deduped.append(value)
    return deduped


def _specific_enough(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return False
    generic = {
        "this",
        "that",
        "these",
        "those",
        "the",
        "and",
        "or",
        "its",
        "it",
        "reads",
        "maps",
        "extracts",
        "generates",
        "method",
        "model",
        "system",
        "result",
        "concept",
        "fact",
        "问题",
        "方法",
        "背景",
        "约束",
        "过程",
        "模型",
        "系统",
        "结果",
        "概念",
        "事实",
        "这里",
        "这些",
        "每个",
    }
    return normalized not in generic


def _section_subject(section: DocumentSection) -> str:
    for value in section.section_path:
        cleaned = _clean_short_text(value)
        if _specific_enough(cleaned):
            return cleaned
    title = _clean_short_text(section.title or "")
    return title if _specific_enough(title) else ""


def _compact_event_label(sentence: str) -> str:
    lowered = sentence.lower()
    if "chunk" in lowered and ("生成问题" in sentence or "question" in lowered):
        return "chunk 级问题生成"
    if "concept" in lowered and "fact" in lowered and ("graph" in lowered or "图" in sentence):
        return "概念-事实图构建与事实组合"
    if "fact" in lowered and "evidence" in lowered and ("不能" in sentence or "must" in lowered or "cannot" in lowered):
        return "fact 的 evidence 支持约束"
    if "pipeline" in lowered and "qa" in lowered:
        return "structure mapping, extraction, and QA generation"
    if "文档外知识" in sentence:
        return "文档外知识限制"
    return ""


def _sentence_is_card_candidate(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) > 500:
        return False
    if re.search(r"[\u4e00-\u9fff]", stripped):
        return len(stripped) >= 16
    return len(stripped) >= 35


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _loose_overlap(topic: str, normalized_question: str) -> bool:
    tokens = [token for token in _tokens(topic) if len(token) >= 2]
    if not tokens:
        return False
    return sum(1 for token in tokens if token in normalized_question) >= max(1, len(tokens) // 2)


def _tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", normalized)


def _chunk(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
