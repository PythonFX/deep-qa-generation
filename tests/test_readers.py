from pathlib import Path

from rag_qa_builder.readers import read_documents


def test_read_documents_reads_markdown_and_text() -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    documents, errors = read_documents(fixture_dir, [".md", ".txt", ".markdown"])
    assert not errors
    assert len(documents) == 2
    assert {doc.file_type for doc in documents} == {".md", ".txt"}

