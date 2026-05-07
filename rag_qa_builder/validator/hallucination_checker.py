from __future__ import annotations

from rag_qa_builder.models import Evidence, QAPair


def has_hallucination(qa: QAPair, evidences: list[Evidence]) -> bool:
    corpus = " ".join(evidence.text for evidence in evidences)
    tokens = [token for token in qa.reference_answer.split() if len(token) > 5]
    if not tokens:
        return False
    misses = sum(1 for token in tokens if token not in corpus)
    return misses > max(2, len(tokens) // 2)

