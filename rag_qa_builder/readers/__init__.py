from __future__ import annotations

from pathlib import Path

from rag_qa_builder.models import Document
from rag_qa_builder.readers.markdown_reader import read_markdown
from rag_qa_builder.readers.text_reader import read_text


def read_documents(input_path: str | Path, allowed_types: list[str], encoding: str = "utf-8") -> tuple[list[Document], list[dict]]:
    root = Path(input_path)
    documents: list[Document] = []
    errors: list[dict] = []
    paths: list[Path]
    if root.is_file():
        paths = [root]
    else:
        paths = [path for path in root.rglob("*") if path.is_file()]

    for path in paths:
        if path.name.startswith(".") or path.stat().st_size == 0:
            continue
        suffix = path.suffix.lower()
        if suffix not in allowed_types:
            continue
        try:
            if suffix in {".md", ".markdown"}:
                documents.append(read_markdown(path, encoding=encoding))
            else:
                documents.append(read_text(path, encoding=encoding))
        except Exception as exc:
            errors.append({
                "stage": "read_documents",
                "source_id": str(path),
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            })
    return documents, errors

