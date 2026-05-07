from __future__ import annotations

import re
from dataclasses import dataclass

from rag_qa_builder.config import AppConfig
from rag_qa_builder.llm.prompt_runner import PromptRunner
from rag_qa_builder.models import Concept, DocumentSection, Evidence, Fact
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import normalize_text, similarity, split_sentences, unwrap_line_breaks

PRONOUN_BLACKLIST = {
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "theirs",
    "we", "our", "ours", "you", "your", "yours", "i", "me", "my", "mine",
    "each", "all", "both", "either", "neither", "another", "other", "others",
}

GENERIC_BLACKLIST = {
    "paper", "work", "approach", "method", "model", "models", "system", "result", "results",
    "thing", "things", "part", "section", "example", "examples", "table", "figure",
}

AFFILIATION_BLACKLIST = {
    "google", "brain", "research", "university", "toronto", "provided", "permission",
    "journalistic", "scholarly", "works", "email", "gmail",
}

DOMAIN_KEYWORDS = {
    "attention", "transformer", "encoder", "decoder", "embedding", "head", "layer",
    "normalization", "bleu", "network", "architecture", "representation", "translation",
    "sequence", "recurrent", "convolutional", "softmax", "dropout", "training",
    "position", "positional", "feed-forward", "feed", "forward", "model",
}

MAX_SECTION_TEXT = 12000
MAX_CONCEPTS_PER_SECTION = 12
MAX_FACTS_PER_SECTION = 24


@dataclass
class SectionKnowledgeBundle:
    concepts: list[Concept]
    facts: list[Fact]
    evidences: list[Evidence]


def extract_section_knowledge(
    sections: list[DocumentSection],
    config: AppConfig,
    prompt_runner: PromptRunner,
) -> SectionKnowledgeBundle:
    all_concepts: list[Concept] = []
    all_facts: list[Fact] = []
    all_evidences: list[Evidence] = []

    extraction_units = _build_extraction_units(sections, config)
    for unit in extraction_units:
        cleaned_unit = _clean_section_for_extraction(unit)
        heuristics = _heuristic_section_knowledge(cleaned_unit)
        llm_result = prompt_runner.maybe_run_json(
            "extract_section_knowledge",
            (
                "You are compiling a document knowledge layer from one section of a source document. "
                "You need to extract important concepts and facts from the document. "
                "Return JSON only with keys: concepts, facts, fact_relations. "
                "Each concept must be a real domain concept, not a pronoun or generic filler word. "
                "Exclude author names, affiliations, emails, paper metadata, and copyright text. "
                "Each fact must be atomic, grounded in the text, and linked to one subject concept. "
                "Each fact should include short evidence copied from the source text. "
                "For fact_relations, only include the strongest 1-3 local relations and avoid speculation."
            ),
            {
                "section": {
                    "section_id": cleaned_unit.section_id,
                    "title": cleaned_unit.title,
                    "section_path": cleaned_unit.section_path,
                    "text": cleaned_unit.text[:MAX_SECTION_TEXT],
                },
                "limits": {
                    "max_concepts": MAX_CONCEPTS_PER_SECTION,
                    "max_facts": MAX_FACTS_PER_SECTION,
                    "max_fact_relations": 12,
                },
                "schema_hint": {
                    "concepts": [
                        {
                            "name": "Transformer",
                            "aliases": ["The Transformer"],
                            "concept_type": "method",
                            "definition": "A sequence transduction architecture based entirely on attention.",
                            "importance": 0.95,
                        }
                    ],
                    "facts": [
                        {
                            "statement": "The Transformer relies entirely on attention mechanisms and removes recurrence.",
                            "fact_type": "definition",
                            "subject_concept": "Transformer",
                            "related_concepts": ["attention mechanisms", "recurrence"],
                            "confidence": 0.92,
                            "importance": 0.9,
                            "evidence": "The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism...",
                        }
                    ],
                    "fact_relations": [
                        {
                            "source_fact": "The Transformer relies entirely on attention mechanisms and removes recurrence.",
                            "target_fact": "The model allows significantly more parallelization.",
                            "relation_type": "causes",
                        }
                    ],
                },
            },
        )
        parsed = _parse_llm_section_knowledge(cleaned_unit, llm_result)
        bundle = _merge_section_results(heuristics, parsed)
        all_concepts.extend(bundle.concepts)
        all_facts.extend(bundle.facts)
        all_evidences.extend(bundle.evidences)

    return SectionKnowledgeBundle(
        concepts=all_concepts,
        facts=all_facts,
        evidences=all_evidences,
    )


def _build_extraction_units(sections: list[DocumentSection], config: AppConfig) -> list[DocumentSection]:
    units: list[DocumentSection] = []
    max_chars = max(2000, config.structure.max_section_chars_for_single_llm_call // 4)
    for section in sections:
        if len(section.text) <= max_chars:
            units.append(section)
            continue
        parts = _split_large_section(section.text, max_chars=max_chars)
        cursor = section.char_start
        for index, part in enumerate(parts, start=1):
            title = section.title or (section.section_path[-1] if section.section_path else "section")
            unit_id = stable_id("sec", f"{section.section_id}:{index}:{cursor}")
            units.append(
                DocumentSection(
                    section_id=unit_id,
                    doc_id=section.doc_id,
                    title=f"{title} [{index}]",
                    level=section.level,
                    section_path=section.section_path,
                    text=part,
                    char_start=cursor,
                    char_end=cursor + len(part),
                    summary=part[:200].strip(),
                    keywords=section.keywords,
                )
            )
            cursor += len(part)
    return units


def _split_large_section(text: str, max_chars: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if not paragraphs:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _clean_section_for_extraction(section: DocumentSection) -> DocumentSection:
    text = section.text
    if "Abstract" in text[:2000]:
        abstract_pos = text.find("Abstract")
        if abstract_pos > 0:
            text = text[abstract_pos:]

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            cleaned_lines.append("")
            continue
        if "@" in stripped:
            continue
        if lowered.startswith("provided proper attribution"):
            continue
        if any(token in lowered for token in ["copyright", "permission to", "journalistic", "scholarly works"]):
            continue
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines).strip() or section.text
    cleaned_text = unwrap_line_breaks(cleaned_text)
    return DocumentSection(
        section_id=section.section_id,
        doc_id=section.doc_id,
        title=section.title,
        level=section.level,
        section_path=section.section_path,
        text=cleaned_text,
        char_start=section.char_start,
        char_end=section.char_start + len(cleaned_text),
        summary=cleaned_text[:200].strip(),
        keywords=section.keywords,
    )


def _heuristic_section_knowledge(section: DocumentSection) -> SectionKnowledgeBundle:
    concepts: list[Concept] = []
    facts: list[Fact] = []
    evidences: list[Evidence] = []

    for name in _candidate_concepts_from_text(section):
        concepts.append(
            Concept(
                concept_id=stable_id("concept_raw", f"{section.section_id}:{name}"),
                canonical_name=name,
                aliases=[],
                concept_type=_infer_concept_type(name),
                definition=_find_definition(name, section.text),
                importance=0.55 if len(name.split()) > 1 else 0.45,
                source_section_ids=[section.section_id],
                metadata={"source": "heuristic", "section_id": section.section_id},
            )
        )

    concept_names = [concept.canonical_name for concept in concepts]
    for sentence in split_sentences(section.text)[:MAX_FACTS_PER_SECTION]:
        subject = _best_subject_for_sentence(sentence, concept_names)
        if not subject:
            continue
        evidence = _make_evidence(section, sentence)
        facts.append(
            Fact(
                fact_id=stable_id("fact_raw", f"{section.section_id}:{sentence}"),
                fact_type=_infer_fact_type(sentence),
                subject_concept_id=stable_id("concept_raw", f"{section.section_id}:{subject}"),
                related_concept_ids=[],
                statement=sentence,
                structured={},
                qualifiers={},
                confidence=_estimate_confidence(sentence),
                importance=0.55,
                evidence_ids=[evidence.evidence_id],
                metadata={
                    "source": "heuristic",
                    "section_id": section.section_id,
                    "subject_concept_name": subject,
                },
            )
        )
        evidences.append(evidence)

    return SectionKnowledgeBundle(concepts=concepts, facts=facts, evidences=evidences)


def _parse_llm_section_knowledge(section: DocumentSection, llm_result: dict | None) -> SectionKnowledgeBundle:
    if not isinstance(llm_result, dict):
        return SectionKnowledgeBundle(concepts=[], facts=[], evidences=[])

    concepts: list[Concept] = []
    facts: list[Fact] = []
    evidences: list[Evidence] = []
    local_fact_ids_by_text: dict[str, str] = {}

    for item in _normalize_concept_items(llm_result.get("concepts", [])):
        name = item.get("name", "").strip()
        if not _is_valid_concept_name(name):
            continue
        aliases = _normalize_string_list(item.get("aliases", []))
        concepts.append(
            Concept(
                concept_id=stable_id("concept_raw", f"{section.section_id}:{name}"),
                canonical_name=name,
                aliases=aliases,
                concept_type=item.get("concept_type", _infer_concept_type(name)),
                definition=item.get("definition"),
                importance=_bounded_float(item.get("importance"), fallback=0.75),
                source_section_ids=[section.section_id],
                metadata={"source": "llm", "section_id": section.section_id},
            )
        )

    for item in _normalize_fact_items(llm_result.get("facts", [])):
        statement = item.get("statement", "").strip()
        evidence_text = item.get("evidence", "").strip()
        subject_name = item.get("subject_concept", "").strip()
        if not statement or not evidence_text or not _is_valid_concept_name(subject_name):
            continue
        evidence = _make_evidence(section, evidence_text)
        fact_id = stable_id("fact_raw", f"{section.section_id}:{statement}")
        local_fact_ids_by_text[normalize_text(statement)] = fact_id
        facts.append(
            Fact(
                fact_id=fact_id,
                fact_type=item.get("fact_type", _infer_fact_type(statement)),
                subject_concept_id=stable_id("concept_raw", f"{section.section_id}:{subject_name}"),
                related_concept_ids=[],
                statement=statement,
                structured=item.get("structured", {}) if isinstance(item.get("structured"), dict) else {},
                qualifiers=item.get("qualifiers", {}) if isinstance(item.get("qualifiers"), dict) else {},
                confidence=_bounded_float(item.get("confidence"), fallback=0.8),
                importance=_bounded_float(item.get("importance"), fallback=0.75),
                evidence_ids=[evidence.evidence_id],
                metadata={
                    "source": "llm",
                    "section_id": section.section_id,
                    "subject_concept_name": subject_name,
                    "related_concept_names": _normalize_string_list(item.get("related_concepts", [])),
                },
            )
        )
        evidences.append(evidence)

    for rel in _normalize_relation_items(llm_result.get("fact_relations", [])):
        source_id = _match_fact_id(rel.get("source_fact", ""), local_fact_ids_by_text)
        target_id = _match_fact_id(rel.get("target_fact", ""), local_fact_ids_by_text)
        if not source_id or not target_id or source_id == target_id:
            continue
        for fact in facts:
            if fact.fact_id != source_id:
                continue
            relation_type = rel.get("relation_type", "related")
            if relation_type in {"causes", "depends_on", "supports", "requires"}:
                fact.depends_on_fact_ids = sorted(set(fact.depends_on_fact_ids + [target_id]))
            elif relation_type in {"contrasts", "compares", "tradeoff"}:
                fact.contrasts_with_fact_ids = sorted(set(fact.contrasts_with_fact_ids + [target_id]))
            break

    return SectionKnowledgeBundle(concepts=concepts, facts=facts, evidences=evidences)


def _merge_section_results(primary: SectionKnowledgeBundle, secondary: SectionKnowledgeBundle) -> SectionKnowledgeBundle:
    concepts = {concept.concept_id: concept for concept in primary.concepts}
    for concept in secondary.concepts:
        concepts[concept.concept_id] = concept

    evidences = {evidence.evidence_id: evidence for evidence in primary.evidences}
    for evidence in secondary.evidences:
        evidences[evidence.evidence_id] = evidence

    primary_facts = primary.facts
    if secondary.facts:
        llm_signatures = {normalize_text(fact.statement) for fact in secondary.facts}
        primary_facts = [
            fact
            for fact in primary.facts
            if _keep_heuristic_fact(fact, llm_signatures)
        ]

    facts = {fact.fact_id: fact for fact in primary_facts}
    for fact in secondary.facts:
        facts[fact.fact_id] = fact

    return SectionKnowledgeBundle(
        concepts=list(concepts.values()),
        facts=list(facts.values()),
        evidences=list(evidences.values()),
    )


def _keep_heuristic_fact(fact: Fact, llm_signatures: set[str]) -> bool:
    statement = fact.statement.strip()
    if fact.metadata.get("source") != "heuristic":
        return True
    if len(statement) < 60:
        return False
    if statement.endswith(","):
        return False
    if not re.search(r"[.!?。！？]$", statement):
        return False
    norm = normalize_text(statement)
    if any(similarity(norm, signature) >= 0.8 for signature in llm_signatures):
        return False
    return True


def _candidate_concepts_from_text(section: DocumentSection) -> list[str]:
    patterns = [
        r"\b[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+){0,4}\b",
        r"\b[a-z]+-[a-z]+(?:-[a-z]+)*\b",
        r"[\u4e00-\u9fff]{2,12}",
    ]
    matches: list[str] = []
    seen: set[str] = set()
    title_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", section.title or "")
    for token in title_tokens:
        if _is_valid_concept_name(token):
            seen.add(normalize_text(token))
            matches.append(token)
    for pattern in patterns:
        for match in re.findall(pattern, section.text):
            value = match.strip()
            norm = normalize_text(value)
            if norm in seen or not _is_valid_concept_name(value):
                continue
            seen.add(norm)
            matches.append(value)
            if len(matches) >= MAX_CONCEPTS_PER_SECTION:
                return matches
    return matches


def _best_subject_for_sentence(sentence: str, concept_names: list[str]) -> str | None:
    scored: list[tuple[float, str]] = []
    sentence_norm = normalize_text(sentence)
    for name in concept_names:
        norm = normalize_text(name)
        if norm and norm in sentence_norm:
            scored.append((len(norm), name))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _normalize_concept_items(items: object) -> list[dict]:
    if isinstance(items, dict):
        items = items.get("items", []) or items.get("concepts", [])
    if not isinstance(items, list):
        return []
    normalized: list[dict] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            if "concept" in item and "name" not in item:
                item = {**item, "name": item["concept"]}
            if "canonical_name" in item and "name" not in item:
                item = {**item, "name": item["canonical_name"]}
            normalized.append(item)
    return normalized


def _normalize_fact_items(items: object) -> list[dict]:
    if isinstance(items, dict):
        items = items.get("items", []) or items.get("facts", [])
    if not isinstance(items, list):
        return []
    normalized: list[dict] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"statement": item, "evidence": item})
        elif isinstance(item, dict):
            if "fact" in item and "statement" not in item:
                item = {**item, "statement": item["fact"]}
            if "text" in item and "statement" not in item:
                item = {**item, "statement": item["text"]}
            if "evidence" not in item:
                item = {**item, "evidence": item.get("quote") or item.get("evidence_text") or item.get("statement")}
            normalized.append(item)
    return normalized


def _normalize_relation_items(items: object) -> list[dict]:
    if isinstance(items, dict):
        items = items.get("items", []) or items.get("relations", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _normalize_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _bounded_float(value: object, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, result))


def _is_valid_concept_name(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    norm = normalize_text(name)
    if not norm or norm in PRONOUN_BLACKLIST or norm in GENERIC_BLACKLIST or norm in AFFILIATION_BLACKLIST:
        return False
    parts = [part for part in re.split(r"[\s\-]+", name) if part]
    lowered_parts = [normalize_text(part) for part in parts]
    if any(part in AFFILIATION_BLACKLIST for part in lowered_parts):
        return False
    if len(norm) <= 2 and not re.search(r"[A-Z]{2,}", name):
        return False
    if re.fullmatch(r"[a-z]+", name) and len(name) <= 4:
        return False
    if re.fullmatch(r"[a-z]+", name) and norm not in {"bert", "gpt", "lstm"}:
        return False
    if _looks_like_person_name(name) and not _contains_domain_keyword(name):
        return False
    if _looks_like_metadata_phrase(name):
        return False
    return True


def _contains_domain_keyword(name: str) -> bool:
    norm = normalize_text(name)
    return any(keyword in norm for keyword in DOMAIN_KEYWORDS)


def _looks_like_person_name(name: str) -> bool:
    tokens = [token for token in re.split(r"\s+", name.strip()) if token]
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    def is_person_token(token: str) -> bool:
        pure = token.replace(".", "")
        return pure[:1].isupper() and pure[1:].islower()
    return all(is_person_token(token) for token in tokens)


def _looks_like_metadata_phrase(name: str) -> bool:
    lowered = name.lower().strip()
    if lowered in {"abstract", "references", "appendix"}:
        return True
    if lowered.startswith("figure ") or lowered.startswith("table "):
        return True
    return False


def _infer_concept_type(name: str) -> str:
    lowered = normalize_text(name)
    if any(token in lowered for token in ["attention", "transformer", "architecture", "network", "mechanism"]):
        return "method"
    if any(token in lowered for token in ["encoder", "decoder", "layer", "embedding", "head"]):
        return "component"
    if any(token in lowered for token in ["bleu", "score", "metric"]):
        return "metric"
    return "term"


def _find_definition(name: str, text: str) -> str | None:
    patterns = [
        rf"{re.escape(name)}\s+is\s+([^\n\.]+)",
        rf"{re.escape(name)}\s+are\s+([^\n\.]+)",
        rf"{re.escape(name)}[：:]\s*([^\n。]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _make_evidence(section: DocumentSection, text: str) -> Evidence:
    rel = section.text.find(text)
    char_start = section.char_start + rel if rel >= 0 else None
    char_end = char_start + len(text) if char_start is not None else None
    return Evidence(
        evidence_id=stable_id("ev", f"{section.section_id}:{text}"),
        doc_id=section.doc_id,
        section_id=section.section_id,
        section_path=section.section_path,
        text=text,
        char_start=char_start,
        char_end=char_end,
        source_hint=section.title,
    )


def _infer_fact_type(sentence: str) -> str:
    lowered = normalize_text(sentence)
    if re.search(r"\d", sentence):
        return "numeric"
    if any(word in lowered for word in ["if", "when", "under", "unless", "如果"]):
        return "condition"
    if any(word in lowered for word in ["because", "therefore", "lead", "allows", "enable", "because", "因此"]):
        return "cause_effect"
    if any(word in lowered for word in ["must", "should", "cannot", "require", "need", "不能", "必须"]):
        return "constraint"
    if any(word in lowered for word in ["first", "then", "finally", "step", "followed by"]):
        return "procedure"
    if any(word in lowered for word in ["compared", "whereas", "while", "unlike", "vs"]):
        return "comparison"
    if any(word in lowered for word in ["for example", "such as", "例如", "比如"]):
        return "example"
    if any(word in lowered for word in ["is", "are", "refers to", "defined as"]):
        return "definition"
    return "claim"


def _estimate_confidence(sentence: str) -> float:
    value = 0.55
    if len(sentence) >= 40:
        value += 0.1
    if re.search(r"\d", sentence):
        value += 0.1
    if any(token in sentence for token in [" is ", " are ", " because ", " allows ", " enables ", " must "]):
        value += 0.1
    return min(value, 0.9)


def _match_fact_id(text: str, local_fact_ids_by_text: dict[str, str]) -> str | None:
    norm = normalize_text(text)
    if not norm:
        return None
    if norm in local_fact_ids_by_text:
        return local_fact_ids_by_text[norm]
    best: tuple[float, str | None] = (0.0, None)
    for candidate_text, fact_id in local_fact_ids_by_text.items():
        score = similarity(norm, candidate_text)
        if score > best[0]:
            best = (score, fact_id)
    return best[1] if best[0] >= 0.7 else None
