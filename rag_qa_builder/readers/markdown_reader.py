from __future__ import annotations

from pathlib import Path

from rag_qa_builder.models import Document
from rag_qa_builder.utils.ids import stable_id


def read_markdown(path: str | Path, encoding: str = "utf-8") -> Document:
    file_path = Path(path)
    text = file_path.read_text(encoding=encoding)
    return Document(
        doc_id=stable_id("doc", str(file_path.resolve())),
        file_path=str(file_path.resolve()),
        file_name=file_path.name,
        file_type=file_path.suffix.lower(),
        text=text,
    )

