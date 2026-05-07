from __future__ import annotations

from rag_qa_builder.models import Evidence, QAPair


def is_answerable(qa: QAPair, evidences: list[Evidence]) -> bool:
    return bool(qa.question.strip() and qa.reference_answer.strip() and evidences)

