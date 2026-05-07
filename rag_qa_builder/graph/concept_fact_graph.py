from __future__ import annotations

from pydantic import BaseModel, Field

from rag_qa_builder.models import Concept, ConceptFactRelation, Fact


class ConceptFactGraph(BaseModel):
    nodes: dict[str, list] = Field(default_factory=dict)
    relations: list[ConceptFactRelation] = Field(default_factory=list)


def build_graph_payload(concepts: list[Concept], facts: list[Fact], relations: list[ConceptFactRelation]) -> ConceptFactGraph:
    return ConceptFactGraph(nodes={"concepts": concepts, "facts": facts}, relations=relations)

