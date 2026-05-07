from __future__ import annotations

from rag_qa_builder.models import Evidence


def has_evidence_text(answer: str, evidences: list[Evidence]) -> bool:
    answer_lower = answer.lower()
    return any(evidence.text[:20].lower() in answer_lower for evidence in evidences if evidence.text)

