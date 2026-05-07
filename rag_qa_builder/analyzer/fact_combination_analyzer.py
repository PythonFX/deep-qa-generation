from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from rag_qa_builder.analyzer.difficulty_estimator import estimate_difficulty
from rag_qa_builder.analyzer.pattern_matcher import infer_pattern
from rag_qa_builder.config import AppConfig
from rag_qa_builder.models import Fact, FactCombination
from rag_qa_builder.utils.ids import stable_id


def analyze_fact_combinations(facts: list[Fact], config: AppConfig) -> list[FactCombination]:
    by_concept: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.subject_concept_id:
            by_concept[fact.subject_concept_id].append(fact)

    combinations_found: list[FactCombination] = []
    for concept_id, concept_facts in by_concept.items():
        for size in range(1, min(len(concept_facts), config.combination.max_facts_per_combination) + 1):
            for group in combinations(concept_facts, size):
                pattern = infer_pattern(list(group))
                score = round(sum(fact.confidence for fact in group) / len(group), 3)
                if score < config.combination.min_combination_score:
                    continue
                concept_ids = sorted({fact.subject_concept_id for fact in group if fact.subject_concept_id})
                rationale = "；".join(fact.statement[:60] for fact in group)
                combinations_found.append(
                    FactCombination(
                        combination_id=stable_id("comb", f"{concept_id}:{'|'.join(f.fact_id for f in group)}"),
                        fact_ids=[fact.fact_id for fact in group],
                        concept_ids=concept_ids,
                        pattern=pattern,
                        rationale=rationale,
                        expected_question_type=_expected_question_type(pattern),
                        expected_answer_points=[fact.statement for fact in group],
                        difficulty=estimate_difficulty(list(group)),
                        score=score,
                    )
                )
    combinations_found.sort(key=lambda item: (-item.score, -len(item.fact_ids)))
    return combinations_found


def _expected_question_type(pattern: str) -> str:
    mapping = {
        "concept_comparison": "comparison",
        "cause_effect_chain": "cause_effect",
        "constraint_reasoning": "constraint",
        "condition_action": "scenario",
        "procedure_chain": "procedure",
    }
    return mapping.get(pattern, "multi_fact_synthesis")

