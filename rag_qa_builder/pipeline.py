from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_qa_builder.analyzer.fact_combination_analyzer import analyze_fact_combinations
from rag_qa_builder.compiler.concept_canonicalizer import canonicalize_concepts
from rag_qa_builder.compiler.evidence_binder import bind_and_deduplicate_evidence
from rag_qa_builder.compiler.section_knowledge_extractor import SectionKnowledgeBundle, extract_section_knowledge
from rag_qa_builder.compiler.structure_mapper import map_documents_to_sections
from rag_qa_builder.config import AppConfig
from rag_qa_builder.exporters.json_exporter import export_json
from rag_qa_builder.exporters.jsonl_exporter import export_jsonl
from rag_qa_builder.generator.qa_generator import generate_qa_candidates
from rag_qa_builder.generator.question_blueprint_generator import build_question_blueprints
from rag_qa_builder.graph.concept_fact_graph import build_graph_payload
from rag_qa_builder.graph.graph_builder import build_concept_fact_relations
from rag_qa_builder.llm.prompt_runner import PromptRunner
from rag_qa_builder.models import Concept, DatasetEntry, Evidence, Fact, QAPair
from rag_qa_builder.readers import read_documents
from rag_qa_builder.utils.json_utils import dump_jsonl
from rag_qa_builder.utils.text_utils import normalize_text
from rag_qa_builder.validator import validate_qa_pairs


class Pipeline:
    def __init__(self, input_path: str | Path, output_dir: str | Path, config: AppConfig, dry_run: bool = False) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.dry_run = dry_run
        if dry_run:
            self.config.llm.enabled = False
        self.prompt_runner = PromptRunner(config, self.output_dir)
        self.errors: list[dict[str, Any]] = []
        self._section_knowledge_bundle: SectionKnowledgeBundle | None = None
        self._raw_concepts: list[Concept] = []

    def build_structure(self) -> tuple[list, list]:
        documents, read_errors = read_documents(self.input_path, self.config.input.file_types, self.config.input.encoding)
        self.errors.extend(read_errors)
        sections = map_documents_to_sections(documents)
        export_json(self.output_dir, "documents.json", documents)
        export_json(self.output_dir, "document_structure.json", sections)
        self._flush_errors()
        return documents, sections

    def extract_concepts(self, documents: list, sections: list) -> list:
        bundle = self._get_section_knowledge(sections)
        self._raw_concepts = bundle.concepts
        canonicalized = canonicalize_concepts(bundle.concepts)
        export_json(self.output_dir, "concepts.raw.json", bundle.concepts)
        export_json(self.output_dir, "concepts.json", canonicalized)
        return canonicalized

    def extract_facts(self, concepts: list, sections: list) -> tuple[list, list]:
        bundle = self._get_section_knowledge(sections)
        self._raw_concepts = self._raw_concepts or bundle.concepts
        facts_raw = self._remap_facts_to_canonical_concepts(bundle.facts, self._raw_concepts, concepts)
        evidence_raw = bundle.evidences
        facts, evidence = bind_and_deduplicate_evidence(facts_raw, evidence_raw)
        export_json(self.output_dir, "facts.raw.json", facts_raw)
        export_json(self.output_dir, "evidence.raw.json", evidence_raw)
        export_json(self.output_dir, "facts.json", facts)
        export_json(self.output_dir, "evidence.json", evidence)
        return facts, evidence

    def build_graph(self, concepts: list, facts: list) -> tuple[list, Any]:
        relations = build_concept_fact_relations(concepts, facts)
        graph = build_graph_payload(concepts, facts, relations)
        export_json(self.output_dir, "concept_fact_graph.json", graph)
        return relations, graph

    def analyze_combinations(self, facts: list) -> list:
        combos = analyze_fact_combinations(facts, self.config)
        export_json(self.output_dir, "fact_combinations.json", combos)
        return combos

    def generate_qa(self, combinations: list, facts: list, evidence: list[Evidence]) -> tuple[list, list]:
        blueprints = build_question_blueprints(combinations)
        qas = generate_qa_candidates(blueprints, facts, evidence, self.config)
        export_json(self.output_dir, "question_blueprints.json", blueprints)
        export_jsonl(self.output_dir, "qa_candidates.jsonl", qas)
        return blueprints, qas

    def validate_qa(self, qa_pairs: list[QAPair], evidence: list[Evidence]) -> tuple[list, list]:
        evidence_lookup = {item.evidence_id: item for item in evidence}
        validations, passed = validate_qa_pairs(qa_pairs, evidence_lookup, self.config)
        rejected_ids = {item.qa_id for item in validations if not item.passed}
        rejected = [qa for qa in qa_pairs if qa.qa_id in rejected_ids]
        export_jsonl(self.output_dir, "qa_validated.jsonl", validations)
        export_jsonl(self.output_dir, "qa_rejected.jsonl", rejected)
        return validations, passed

    def export_final(self, qa_pairs: list[QAPair], evidence: list[Evidence]) -> list[DatasetEntry]:
        evidence_lookup = {item.evidence_id: item for item in evidence}
        dataset: list[DatasetEntry] = []
        for qa in qa_pairs:
            dataset.append(
                DatasetEntry(
                    id=qa.qa_id,
                    question=qa.question,
                    reference_answer=qa.reference_answer,
                    reference_context=[
                        evidence_lookup[evidence_id].model_dump(mode="json")
                        for evidence_id in qa.evidence_ids
                        if evidence_id in evidence_lookup
                    ],
                    concept_ids=qa.concept_ids,
                    fact_ids=qa.fact_ids,
                    evidence_ids=qa.evidence_ids,
                    question_type=qa.question_type,
                    difficulty=qa.difficulty,
                    answer_requirements=qa.answer_requirements,
                    unsupported_answer_patterns=qa.unsupported_answer_patterns,
                )
            )
        export_jsonl(self.output_dir, "dataset.final.jsonl", dataset)
        return dataset

    def generate_all(self) -> dict[str, Any]:
        documents, sections = self.build_structure()
        concepts = self.extract_concepts(documents, sections)
        facts, evidence = self.extract_facts(concepts, sections)
        _, graph = self.build_graph(concepts, facts)
        combinations = self.analyze_combinations(facts)
        blueprints, qas = self.generate_qa(combinations, facts, evidence)
        validations, passed = self.validate_qa(qas, evidence)
        dataset = self.export_final(passed, evidence)
        self._flush_errors()
        return {
            "documents": documents,
            "sections": sections,
            "concepts": concepts,
            "facts": facts,
            "evidence": evidence,
            "graph": graph,
            "combinations": combinations,
            "blueprints": blueprints,
            "qa_candidates": qas,
            "qa_validated": validations,
            "dataset": dataset,
        }

    def _flush_errors(self) -> None:
        if self.errors:
            dump_jsonl(self.output_dir / "errors.jsonl", self.errors)

    def _get_section_knowledge(self, sections: list) -> SectionKnowledgeBundle:
        if self._section_knowledge_bundle is None:
            self._section_knowledge_bundle = extract_section_knowledge(sections, self.config, self.prompt_runner)
        return self._section_knowledge_bundle

    def _remap_facts_to_canonical_concepts(
        self,
        facts: list[Fact],
        raw_concepts: list[Concept],
        canonical_concepts: list[Concept],
    ) -> list[Fact]:
        raw_id_to_name = {concept.concept_id: concept.canonical_name for concept in raw_concepts}
        name_to_canonical_id: dict[str, str] = {}
        for concept in canonical_concepts:
            name_to_canonical_id[normalize_text(concept.canonical_name)] = concept.concept_id
            for alias in concept.aliases:
                name_to_canonical_id[normalize_text(alias)] = concept.concept_id

        remapped: list[Fact] = []
        for fact in facts:
            subject_name = fact.metadata.get("subject_concept_name") or raw_id_to_name.get(fact.subject_concept_id or "", "")
            subject_id = name_to_canonical_id.get(normalize_text(subject_name))
            if not subject_id:
                continue
            fact.subject_concept_id = subject_id
            related_names = fact.metadata.get("related_concept_names", [])
            if isinstance(related_names, str):
                related_names = [related_names]
            fact.related_concept_ids = sorted(
                {
                    name_to_canonical_id[normalize_text(name)]
                    for name in related_names
                    if normalize_text(name) in name_to_canonical_id and name_to_canonical_id[normalize_text(name)] != subject_id
                }
            )
            remapped.append(fact)
        return remapped
