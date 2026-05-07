from __future__ import annotations

import re
from difflib import SequenceMatcher


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "on", "by", "is", "are",
    "的", "了", "和", "与", "或", "及", "在", "是", "对", "通过", "用于", "一个",
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().lower()
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", cleaned)
    return cleaned


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=normalize_text(left), b=normalize_text(right)).ratio()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def keywords(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text)
    scored: dict[str, int] = {}
    for token in tokens:
        norm = normalize_text(token)
        if not norm or norm in STOPWORDS:
            continue
        scored[norm] = scored.get(norm, 0) + 1
    return [token for token, _ in sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))

