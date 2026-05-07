from __future__ import annotations

from rag_qa_builder.models import Evidence, Fact


def bind_and_deduplicate_evidence(facts: list[Fact], evidences: list[Evidence]) -> tuple[list[Fact], list[Evidence]]:
    seen_evidence: dict[tuple[str, str], Evidence] = {}
    evidence_id_map: dict[str, str] = {}
    for evidence in evidences:
        key = (evidence.section_id or "", evidence.text.strip())
        if key not in seen_evidence:
            seen_evidence[key] = evidence
        evidence_id_map[evidence.evidence_id] = seen_evidence[key].evidence_id
    bound_facts: list[Fact] = []
    for fact in facts:
        resolved = []
        for evidence_id in fact.evidence_ids:
            if evidence_id in evidence_id_map:
                resolved.append(evidence_id_map[evidence_id])
        fact.evidence_ids = sorted(set(resolved))
        if fact.evidence_ids:
            bound_facts.append(fact)
    return bound_facts, list(seen_evidence.values())

