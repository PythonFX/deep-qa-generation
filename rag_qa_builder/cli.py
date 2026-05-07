from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_qa_builder.config import load_config
from rag_qa_builder.deep_qa_pipeline import DeepQAPipeline
from rag_qa_builder.pipeline import Pipeline
from rag_qa_builder.utils.logging import console


def _make_pipeline(input_path: str, output: str, config_path: str | None, dry_run: bool) -> Pipeline:
    config = load_config(config_path)
    return Pipeline(input_path=input_path, output_dir=output, config=config, dry_run=dry_run)


def generate(
    input_path: str,
    output: str,
    config_path: str | None = None,
    language: str | None = None,
    target_size: int | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path, output, config_path, dry_run)
    if language:
        pipeline.config.project.language = language
        pipeline.config.qa_generation.question_language = language
    if target_size:
        pipeline.config.qa_generation.target_size = target_size
    _maybe_prepare_output(output, force, resume)
    result = pipeline.generate_all()
    console.print(f"[green]Done[/green] docs={len(result['documents'])} concepts={len(result['concepts'])} facts={len(result['facts'])} dataset={len(result['dataset'])}")


def generate_deep(
    input_path: str,
    output: str,
    config_path: str | None = None,
    language: str | None = None,
    target_size: int | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    config = load_config(config_path)
    if language:
        config.project.language = language
        config.qa_generation.question_language = language
    if target_size:
        config.qa_generation.target_size = target_size
    _maybe_prepare_output(output, force, resume)
    pipeline = DeepQAPipeline(input_path=input_path, output_dir=output, config=config, dry_run=dry_run)
    result = pipeline.generate_all()
    console.print(
        "[green]Done[/green] "
        f"docs={len(result['documents'])} "
        f"evidence_cards={len(result['evidence_cards'])} "
        f"question_plans={len(result['question_plans'])} "
        f"dataset={len(result['dataset'])}"
    )


def build_structure(
    input_path: str,
    output: str,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    console.print(f"[green]Done[/green] docs={len(documents)} sections={len(sections)}")


def extract_concepts_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    console.print(f"[green]Done[/green] concepts={len(concepts)}")


def extract_facts_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    facts, evidence = pipeline.extract_facts(concepts, sections)
    console.print(f"[green]Done[/green] facts={len(facts)} evidence={len(evidence)}")


def build_graph_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    facts, _ = pipeline.extract_facts(concepts, sections)
    relations, _ = pipeline.build_graph(concepts, facts)
    console.print(f"[green]Done[/green] relations={len(relations)}")


def analyze_combinations_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    facts, _ = pipeline.extract_facts(concepts, sections)
    combos = pipeline.analyze_combinations(facts)
    console.print(f"[green]Done[/green] combinations={len(combos)}")


def generate_qa_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    facts, evidence = pipeline.extract_facts(concepts, sections)
    combos = pipeline.analyze_combinations(facts)
    _, qas = pipeline.generate_qa(combos, facts, evidence)
    console.print(f"[green]Done[/green] qa_candidates={len(qas)}")


def validate_qa_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    facts, evidence = pipeline.extract_facts(concepts, sections)
    combos = pipeline.analyze_combinations(facts)
    _, qas = pipeline.generate_qa(combos, facts, evidence)
    validations, passed = pipeline.validate_qa(qas, evidence)
    console.print(f"[green]Done[/green] validated={len(validations)} passed={len(passed)}")


def export_final_cmd(
    output: str,
    input_path: str | None = None,
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    pipeline = _make_pipeline(input_path or output, output, config_path, dry_run)
    documents, sections = pipeline.build_structure()
    concepts = pipeline.extract_concepts(documents, sections)
    facts, evidence = pipeline.extract_facts(concepts, sections)
    combos = pipeline.analyze_combinations(facts)
    _, qas = pipeline.generate_qa(combos, facts, evidence)
    _, passed = pipeline.validate_qa(qas, evidence)
    dataset = pipeline.export_final(passed, evidence)
    console.print(f"[green]Done[/green] dataset={len(dataset)}")


def _maybe_prepare_output(output: str, force: bool, resume: bool) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    if force and not resume:
        for child in output_path.iterdir():
            if child.is_file():
                child.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-qa-builder", description="Build QA benchmark datasets from markdown and text documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command: argparse.ArgumentParser, require_input: bool = True) -> None:
        command.add_argument("--input", required=require_input)
        command.add_argument("--output", required=True)
        command.add_argument("--config")
        command.add_argument("--dry-run", action="store_true")

    generate_parser = subparsers.add_parser("generate")
    add_common_arguments(generate_parser)
    generate_parser.add_argument("--language")
    generate_parser.add_argument("--target-size", type=int)
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.add_argument("--resume", action="store_true")

    generate_deep_parser = subparsers.add_parser("generate-deep")
    add_common_arguments(generate_deep_parser)
    generate_deep_parser.add_argument("--language")
    generate_deep_parser.add_argument("--target-size", type=int)
    generate_deep_parser.add_argument("--force", action="store_true")
    generate_deep_parser.add_argument("--resume", action="store_true")

    for name in [
        "build-structure",
        "extract-concepts",
        "extract-facts",
        "build-graph",
        "analyze-combinations",
        "generate-qa",
        "validate-qa",
        "export-final",
    ]:
        add_common_arguments(subparsers.add_parser(name))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "generate":
        generate(
            input_path=args.input,
            output=args.output,
            config_path=args.config,
            language=args.language,
            target_size=args.target_size,
            force=args.force,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    elif command == "generate-deep":
        generate_deep(
            input_path=args.input,
            output=args.output,
            config_path=args.config,
            language=args.language,
            target_size=args.target_size,
            force=args.force,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    elif command == "build-structure":
        build_structure(args.input, args.output, args.config, args.dry_run)
    elif command == "extract-concepts":
        extract_concepts_cmd(args.output, args.input, args.config, args.dry_run)
    elif command == "extract-facts":
        extract_facts_cmd(args.output, args.input, args.config, args.dry_run)
    elif command == "build-graph":
        build_graph_cmd(args.output, args.input, args.config, args.dry_run)
    elif command == "analyze-combinations":
        analyze_combinations_cmd(args.output, args.input, args.config, args.dry_run)
    elif command == "generate-qa":
        generate_qa_cmd(args.output, args.input, args.config, args.dry_run)
    elif command == "validate-qa":
        validate_qa_cmd(args.output, args.input, args.config, args.dry_run)
    elif command == "export-final":
        export_final_cmd(args.output, args.input, args.config, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
