from __future__ import annotations

from rag_qa_builder.models import FactCombination, QuestionBlueprint
from rag_qa_builder.utils.ids import stable_id


def build_question_blueprints(combinations: list[FactCombination]) -> list[QuestionBlueprint]:
    blueprints: list[QuestionBlueprint] = []
    for combination in combinations:
        blueprints.append(
            QuestionBlueprint(
                blueprint_id=stable_id("qb", combination.combination_id),
                source_combination_id=combination.combination_id,
                pattern=combination.pattern,
                fact_ids=combination.fact_ids,
                concept_ids=combination.concept_ids,
                intended_question=_intended_question(combination),
                expected_answer_points=combination.expected_answer_points,
                difficulty=combination.difficulty,
                question_type=combination.expected_question_type,
                answer_requirements=combination.expected_answer_points,
                unsupported_answer_patterns=["引入文档未出现的外部知识", "忽略关键条件或限制"],
            )
        )
    return blueprints


def _intended_question(combination: FactCombination) -> str:
    mapping = {
        "comparison": "比较相关概念或方法的异同与适用场景",
        "cause_effect": "说明原因、机制与影响",
        "constraint": "解释约束、原因和使用边界",
        "procedure": "梳理流程步骤与先后关系",
        "scenario": "结合条件判断该怎么做",
        "multi_fact_synthesis": "综合多个事实回答真实问题",
    }
    return mapping.get(combination.expected_question_type, "综合多个事实回答真实问题")

