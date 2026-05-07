from __future__ import annotations

from rag_qa_builder.models import Evidence, Fact


def compose_reference_answer(facts: list[Fact], evidences: list[Evidence]) -> str:
    if not facts:
        return ""
    evidence_lookup = {evidence.evidence_id: evidence for evidence in evidences}
    lines: list[str] = []
    for fact in facts:
        text = fact.statement.strip()
        if text not in lines:
            lines.append(text)
        for evidence_id in fact.evidence_ids:
            evidence = evidence_lookup.get(evidence_id)
            if evidence and evidence.text.strip() not in lines and len(lines) < len(facts) + 2:
                lines.append(evidence.text.strip())
    return " ".join(lines)

