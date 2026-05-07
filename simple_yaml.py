from __future__ import annotations

from typing import Any, TextIO


def safe_load(text: str | TextIO) -> Any:
    if hasattr(text, "read"):
        text = text.read()
    if text is None:
        return {}
    if not isinstance(text, str):
        text = str(text)
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            value = _parse_scalar(stripped[2:])
            if not isinstance(parent, list):
                raise ValueError("Invalid YAML structure near list item")
            parent.append(value)
            continue

        key, raw_value = stripped.split(":", 1)
        raw_value = raw_value.strip()
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        next_indent = len(next_line) - len(next_line.lstrip(" ")) if next_line else -1
        next_stripped = next_line.strip() if next_line else ""

        if raw_value:
            parent[key] = _parse_scalar(raw_value)
            continue

        container: Any
        if next_line and next_indent > indent and next_stripped.startswith("- "):
            container = []
        else:
            container = {}
        parent[key] = container
        stack.append((indent, container))

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
