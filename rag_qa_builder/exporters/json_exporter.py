from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_qa_builder.utils.json_utils import dump_json


def export_json(output_dir: str | Path, file_name: str, data: Any) -> Path:
    target = Path(output_dir) / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    dump_json(target, data)
    return target

