from pathlib import Path

from rag_qa_builder.compiler.structure_mapper import map_documents_to_sections
from rag_qa_builder.readers import read_documents


def test_structure_mapper_builds_sections() -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    documents, _ = read_documents(fixture_dir, [".md", ".txt", ".markdown"])
    sections = map_documents_to_sections(documents)
    assert sections
    assert any(section.section_path for section in sections)
    assert any(section.title == "背景" for section in sections)

