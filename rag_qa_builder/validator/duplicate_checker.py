from __future__ import annotations

from rag_qa_builder.utils.text_utils import similarity


def is_duplicate(question: str, existing_questions: list[str], threshold: float) -> bool:
    return any(similarity(question, existing) >= threshold for existing in existing_questions)

