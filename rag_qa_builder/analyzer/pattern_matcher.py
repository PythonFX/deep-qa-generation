from __future__ import annotations

from rag_qa_builder.models import Fact


def infer_pattern(facts: list[Fact]) -> str:
    types = {fact.fact_type for fact in facts}
    if "comparison" in types:
        return "concept_comparison"
    if "cause_effect" in types:
        return "cause_effect_chain"
    if "constraint" in types:
        return "constraint_reasoning"
    if "condition" in types:
        return "condition_action"
    if "procedure" in types:
        return "procedure_chain"
    return "multi_fact_synthesis"

