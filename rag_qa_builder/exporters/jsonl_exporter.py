from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from rag_qa_builder.utils.json_utils import dump_jsonl


def export_jsonl(output_dir: str | Path, file_name: str, rows: Iterable[Any]) -> Path:
    target = Path(output_dir) / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(target, rows)
    return target

