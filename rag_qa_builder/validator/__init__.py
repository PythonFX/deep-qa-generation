from __future__ import annotations

from rag_qa_builder.config import AppConfig
from rag_qa_builder.models import Evidence, QAPair, QAValidationResult
from rag_qa_builder.validator.ambiguity_checker import is_ambiguous
from rag_qa_builder.validator.answerability_checker import is_answerable
from rag_qa_builder.validator.duplicate_checker import is_duplicate
from rag_qa_builder.validator.evidence_checker import has_evidence_text
from rag_qa_builder.validator.hallucination_checker import has_hallucination


def validate_qa_pairs(qa_pairs: list[QAPair], evidence_lookup: dict[str, Evidence], config: AppConfig) -> tuple[list[QAValidationResult], list[QAPair]]:
    validations: list[QAValidationResult] = []
    passed_qas: list[QAPair] = []
    seen_questions: list[str] = []
    for qa in qa_pairs:
        evidences = [evidence_lookup[evidence_id] for evidence_id in qa.evidence_ids if evidence_id in evidence_lookup]
        answerable = is_answerable(qa, evidences)
        faithful = has_evidence_text(qa.reference_answer, evidences)
        ambiguous = is_ambiguous(qa)
        hallucination = has_hallucination(qa, evidences)
        external = False
        duplicate = is_duplicate(qa.question, seen_questions, config.validation.duplicate_similarity_threshold)
        scores = {
            "answerability": 5.0 if answerable else 0.0,
            "faithfulness": 5.0 if faithful else 2.0,
            "ambiguity": 1.0 if ambiguous else 5.0,
            "hallucination": 1.0 if hallucination else 5.0,
            "duplication": 1.0 if duplicate else 5.0,
        }
        issues: list[str] = []
        if not answerable:
            issues.append("insufficient_evidence")
        if not faithful:
            issues.append("answer_not_grounded")
        if ambiguous:
            issues.append("ambiguous_question")
        if hallucination:
            issues.append("hallucination_risk")
        if duplicate:
            issues.append("duplicate_question")
        overall = sum(scores.values()) / len(scores)
        passed = overall >= config.validation.min_overall_score and not issues
        validations.append(
            QAValidationResult(
                qa_id=qa.qa_id,
                is_answerable=answerable,
                is_faithful=faithful,
                is_ambiguous=ambiguous,
                requires_external_knowledge=external,
                has_hallucination=hallucination,
                scores=scores,
                issues=issues,
                passed=passed,
            )
        )
        if passed:
            seen_questions.append(qa.question)
            passed_qas.append(qa)
    return validations, passed_qas

