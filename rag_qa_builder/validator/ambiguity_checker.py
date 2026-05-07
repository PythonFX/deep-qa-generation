from __future__ import annotations

from rag_qa_builder.models import QAPair


def is_ambiguous(qa: QAPair) -> bool:
    vague_terms = ["这个", "那个", "它们", "something", "things"]
    return any(term in qa.question for term in vague_terms) and len(qa.concept_ids) > 1

