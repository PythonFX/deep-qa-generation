from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import orjson
from pydantic import BaseModel


def dump_json(path: str | Path, data: Any) -> None:
    payload = _normalize(data)
    Path(path).write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def load_json(path: str | Path) -> Any:
    return orjson.loads(Path(path).read_bytes())


def dump_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = _normalize(row)
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def load_jsonl(path: str | Path) -> list[Any]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[Any] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, list):
        return [_normalize(item) for item in data]
    if isinstance(data, dict):
        return {key: _normalize(value) for key, value in data.items()}
    return data
