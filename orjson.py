from __future__ import annotations

import json
from typing import Any

OPT_INDENT_2 = 1
OPT_SORT_KEYS = 2


def dumps(data: Any, option: int = 0) -> bytes:
    indent = 2 if option & OPT_INDENT_2 else None
    sort_keys = bool(option & OPT_SORT_KEYS)
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=sort_keys).encode("utf-8")


def loads(data: bytes | bytearray | str) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return json.loads(data.decode("utf-8"))
    return json.loads(data)

