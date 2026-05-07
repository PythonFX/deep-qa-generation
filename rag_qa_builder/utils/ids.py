from __future__ import annotations

import hashlib


def stable_id(prefix: str, value: str, length: int = 10) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"

