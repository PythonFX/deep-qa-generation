from __future__ import annotations

from rag_qa_builder.config import AppConfig
from rag_qa_builder.generator.answer_generator import compose_reference_answer
from rag_qa_builder.models import Evidence, Fact, QAPair, QuestionBlueprint
from rag_qa_builder.utils.ids import stable_id


def generate_qa_candidates(
    blueprints: list[QuestionBlueprint],
    facts: list[Fact],
    evidences: list[Evidence],
    config: AppConfig,
) -> list[QAPair]:
    fact_lookup = {fact.fact_id: fact for fact in facts}
    qa_pairs: list[QAPair] = []
    for blueprint in blueprints[: config.qa_generation.target_size]:
        blueprint_facts = [fact_lookup[fact_id] for fact_id in blueprint.fact_ids if fact_id in fact_lookup]
        evidence_ids = sorted({evidence_id for fact in blueprint_facts for evidence_id in fact.evidence_ids})
        answer = compose_reference_answer(blueprint_facts, evidences)
        if not answer:
            continue
        question = _render_question(blueprint)
        qa_pairs.append(
            QAPair(
                qa_id=stable_id("qa", blueprint.blueprint_id),
                question=question,
                reference_answer=answer,
                concept_ids=blueprint.concept_ids,
                fact_ids=blueprint.fact_ids,
                evidence_ids=evidence_ids,
                question_type=blueprint.question_type,
                difficulty=blueprint.difficulty,
                answer_requirements=blueprint.answer_requirements,
                unsupported_answer_patterns=blueprint.unsupported_answer_patterns,
                metadata={"pattern": blueprint.pattern},
            )
        )
    return qa_pairs


def _render_question(blueprint: QuestionBlueprint) -> str:
    if blueprint.question_type == "comparison":
        return "这些概念或方法之间的主要差异和共同点是什么？"
    if blueprint.question_type == "cause_effect":
        return "为什么会出现这种结果，它会带来什么影响？"
    if blueprint.question_type == "procedure":
        return "这个流程应该按什么顺序执行，各步骤分别起什么作用？"
    if blueprint.question_type == "constraint":
        return "这里有哪些关键约束，这些约束为什么重要？"
    if blueprint.question_type == "scenario":
        return "在满足这些条件的情况下，应该如何处理？"
    return blueprint.intended_question.rstrip("。") + "？"

