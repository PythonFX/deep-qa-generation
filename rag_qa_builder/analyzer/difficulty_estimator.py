from __future__ import annotations

from rag_qa_builder.models import Fact


def estimate_difficulty(facts: list[Fact]) -> str:
    if len(facts) >= 4:
        return "hard"
    if len(facts) >= 2:
        return "medium"
    return "easy"

