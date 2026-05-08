from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from rag_qa_builder.config import AppConfig
from rag_qa_builder.models import Document, DocumentSection
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import keywords, normalize_text


DEFAULT_MIN_SECTION_CHARS = 1200
DEFAULT_MAX_SECTION_CHARS = 6500
DEFAULT_TARGET_SECTION_CHARS = 3800
DEFAULT_TARGET_SECTIONS_PER_40K_CHARS = 10
DEFAULT_OVERLAP_SENTENCES = 1


@dataclass
class SectioningSettings:
    min_section_chars: int
    target_section_chars: int
    max_section_chars: int
    target_sections_per_40k_chars: int
    target_count_tolerance: float
    overlap_sentences: int


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int
    tokens: list[str]


@dataclass
class BoundaryCandidate:
    index: int
    score: float
    semantic_distance: float
    heading_signal: float
    keyword_shift: float


def map_documents_to_semantic_sections(documents: list[Document], config: AppConfig) -> tuple[list[Document], list[DocumentSection]]:
    settings = _settings_from_config(config)
    cleaned_documents: list[Document] = []
    sections: list[DocumentSection] = []
    for document in documents:
        cleaned_text = clean_pdf_like_text(document.text)
        cleaned_document = Document(
            doc_id=document.doc_id,
            file_path=document.file_path,
            file_name=document.file_name,
            file_type=document.file_type,
            text=cleaned_text,
            metadata={**document.metadata, "sectioning_text": "pdf_cleaned"},
        )
        cleaned_documents.append(cleaned_document)
        sections.extend(_semantic_sections_for_document(cleaned_document, settings))
    return cleaned_documents, sections


def _settings_from_config(config: AppConfig) -> SectioningSettings:
    sectioning = getattr(config, "semantic_sectioning", None)
    return SectioningSettings(
        min_section_chars=max(200, int(getattr(sectioning, "min_section_chars", DEFAULT_MIN_SECTION_CHARS))),
        target_section_chars=max(500, int(getattr(sectioning, "target_section_chars", DEFAULT_TARGET_SECTION_CHARS))),
        max_section_chars=max(1000, int(getattr(sectioning, "max_section_chars", DEFAULT_MAX_SECTION_CHARS))),
        target_sections_per_40k_chars=max(1, int(getattr(sectioning, "target_sections_per_40k_chars", DEFAULT_TARGET_SECTIONS_PER_40K_CHARS))),
        target_count_tolerance=max(0.05, min(0.9, float(getattr(sectioning, "target_count_tolerance", 0.35)))),
        overlap_sentences=max(0, int(getattr(sectioning, "overlap_sentences", DEFAULT_OVERLAP_SENTENCES))),
    )


def clean_pdf_like_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = _drop_repeated_page_noise(text.splitlines())

    paragraphs: list[str] = []
    current = ""
    for raw_line in lines:
        line = _normalize_line(raw_line)
        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            continue
        if _looks_like_pdf_noise_line(line):
            continue
        if _looks_like_standalone_heading(line):
            if current:
                paragraphs.append(current.strip())
            paragraphs.append(line)
            current = ""
            continue
        if not current:
            current = line
            continue
        if current.endswith("-") and _starts_lowercase_or_cjk(line):
            current = current[:-1] + line
        elif _should_join_pdf_line(current, line):
            separator = "" if _cjk_boundary(current, line) else " "
            current = f"{current}{separator}{line}"
        else:
            paragraphs.append(current.strip())
            current = line
    if current:
        paragraphs.append(current.strip())

    cleaned = "\n\n".join(paragraph for paragraph in paragraphs if paragraph)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = _trim_front_matter_and_references(cleaned)
    return cleaned.strip()


def _semantic_sections_for_document(document: Document, settings: SectioningSettings) -> list[DocumentSection]:
    sentences = _sentence_spans(document.text)
    if not sentences:
        return [_make_section(document, 1, 0, document.text, 0, len(document.text), [])]
    if len(document.text) <= settings.max_section_chars:
        return [_make_section(document, 1, 0, document.text, 0, len(document.text), sentences)]

    boundaries = _score_boundaries(sentences)
    target_min, target_max = _target_section_count_range(len(document.text), settings)
    best_sections = _split_with_adaptive_boundary_threshold(document, sentences, boundaries, target_min, target_max, settings)
    return best_sections


def _split_with_adaptive_boundary_threshold(
    document: Document,
    sentences: list[SentenceSpan],
    boundaries: list[BoundaryCandidate],
    target_min: int,
    target_max: int,
    settings: SectioningSettings,
) -> list[DocumentSection]:
    if not boundaries:
        return _force_split_by_size(document, sentences, settings)

    scores = sorted(boundary.score for boundary in boundaries)
    percentiles = [0.82, 0.76, 0.70, 0.64, 0.58, 0.52, 0.46, 0.40, 0.34, 0.28]
    attempts: list[list[DocumentSection]] = []
    for percentile in percentiles:
        threshold = _percentile(scores, percentile)
        section_ranges = _ranges_for_threshold(sentences, boundaries, threshold, settings)
        sections = _sections_from_sentence_ranges(document, sentences, section_ranges)
        sections = _merge_too_small_sections(document, sections, settings)
        sections = _split_too_large_sections(document, sections, settings)
        attempts.append(sections)
        if target_min <= len(sections) <= target_max:
            return sections

    return min(attempts, key=lambda items: _section_count_distance(len(items), target_min, target_max))


def _ranges_for_threshold(
    sentences: list[SentenceSpan],
    boundaries: list[BoundaryCandidate],
    threshold: float,
    settings: SectioningSettings,
) -> list[tuple[int, int]]:
    boundary_lookup = {boundary.index: boundary for boundary in boundaries}
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(sentences)):
        candidate = boundary_lookup.get(index)
        current_chars = sentences[index - 1].end - sentences[start].start
        next_chars = sentences[index].end - sentences[start].start
        can_cut = current_chars >= settings.min_section_chars
        should_cut = bool(candidate and candidate.score >= threshold and can_cut)
        must_cut = next_chars >= settings.max_section_chars
        if should_cut or must_cut:
            cut_index = index
            if must_cut and not should_cut:
                cut_index = _best_cut_before(sentences, boundaries, start, index, settings)
            if cut_index <= start:
                cut_index = index
            ranges.append((start, cut_index))
            start = max(start + 1, cut_index - settings.overlap_sentences)
    if start < len(sentences):
        ranges.append((start, len(sentences)))
    return ranges


def _sections_from_sentence_ranges(
    document: Document,
    sentences: list[SentenceSpan],
    ranges: list[tuple[int, int]],
) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    for ordinal, (start_index, end_index) in enumerate(ranges, start=1):
        selected = sentences[start_index:end_index]
        if not selected:
            continue
        char_start = selected[0].start
        char_end = selected[-1].end
        text = document.text[char_start:char_end].strip()
        if text:
            sections.append(_make_section(document, ordinal, start_index, text, char_start, char_end, selected))
    return sections


def _merge_too_small_sections(document: Document, sections: list[DocumentSection], settings: SectioningSettings) -> list[DocumentSection]:
    if len(sections) <= 1:
        return sections
    merged: list[DocumentSection] = []
    for section in sections:
        if merged and len(section.text) < settings.min_section_chars:
            prev = merged.pop()
            char_start = min(prev.char_start, section.char_start)
            char_end = max(prev.char_end, section.char_end)
            text = document.text[char_start:char_end].strip()
            ordinal = len(merged) + 1
            merged.append(_make_section(document, ordinal, 0, text, char_start, char_end, []))
        else:
            merged.append(section)
    return [_renumber_section(document, section, index) for index, section in enumerate(merged, start=1)]


def _split_too_large_sections(document: Document, sections: list[DocumentSection], settings: SectioningSettings) -> list[DocumentSection]:
    result: list[DocumentSection] = []
    for section in sections:
        if len(section.text) <= settings.max_section_chars:
            result.append(section)
            continue
        local_doc = Document(
            doc_id=document.doc_id,
            file_path=document.file_path,
            file_name=document.file_name,
            file_type=document.file_type,
            text=section.text,
            metadata=document.metadata,
        )
        local_sentences = _sentence_spans(section.text)
        for local in _force_split_by_size(local_doc, local_sentences, settings):
            char_start = section.char_start + local.char_start
            char_end = section.char_start + local.char_end
            text = document.text[char_start:char_end].strip()
            result.append(_make_section(document, len(result) + 1, 0, text, char_start, char_end, []))
    return [_renumber_section(document, section, index) for index, section in enumerate(result, start=1)]


def _force_split_by_size(document: Document, sentences: list[SentenceSpan], settings: SectioningSettings) -> list[DocumentSection]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(sentences):
        end = start + 1
        best_end = end
        while end < len(sentences):
            chars = sentences[end].end - sentences[start].start
            if chars > settings.max_section_chars:
                break
            if chars >= settings.target_section_chars:
                best_end = end + 1
                break
            best_end = end + 1
            end += 1
        ranges.append((start, best_end))
        start = best_end
    return _sections_from_sentence_ranges(document, sentences, ranges)


def _score_boundaries(sentences: list[SentenceSpan]) -> list[BoundaryCandidate]:
    idf = _idf(sentences)
    vectors = [_sentence_vector(sentence, idf) for sentence in sentences]
    boundaries: list[BoundaryCandidate] = []
    for index in range(1, len(sentences)):
        left_vector = _average_vectors(vectors[max(0, index - 3) : index])
        right_vector = _average_vectors(vectors[index : min(len(vectors), index + 3)])
        semantic_distance = 1.0 - _cosine(left_vector, right_vector)
        keyword_shift = _keyword_shift(sentences[max(0, index - 3) : index], sentences[index : min(len(sentences), index + 3)])
        heading_signal = 1.0 if _looks_like_standalone_heading(sentences[index].text) else 0.0
        score = (semantic_distance * 0.62) + (heading_signal * 0.2) + (keyword_shift * 0.18)
        boundaries.append(
            BoundaryCandidate(
                index=index,
                score=round(score, 4),
                semantic_distance=round(semantic_distance, 4),
                heading_signal=heading_signal,
                keyword_shift=round(keyword_shift, 4),
            )
        )
    return boundaries


def _sentence_spans(text: str) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    pattern = re.compile(r".+?(?:[。！？!?]|(?<!\b[A-Z])\.(?=\s+[A-Z0-9])|$)", re.S)
    for match in pattern.finditer(text):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip()
        if not sentence:
            continue
        start = match.start()
        end = match.end()
        if len(sentence) < 8 and not _looks_like_standalone_heading(sentence):
            continue
        spans.append(SentenceSpan(text=sentence, start=start, end=end, tokens=_tokens(sentence)))
    return spans


def _idf(sentences: list[SentenceSpan]) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    for sentence in sentences:
        for token in set(sentence.tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total = max(1, len(sentences))
    return {token: math.log((1 + total) / (1 + freq)) + 1.0 for token, freq in doc_freq.items()}


def _sentence_vector(sentence: SentenceSpan, idf: dict[str, float]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in sentence.tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return {token: count * idf.get(token, 1.0) for token, count in counts.items()}


def _average_vectors(vectors: list[dict[str, float]]) -> dict[str, float]:
    if not vectors:
        return {}
    total: dict[str, float] = {}
    for vector in vectors:
        for token, value in vector.items():
            total[token] = total.get(token, 0.0) + value
    return {token: value / len(vectors) for token, value in total.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_shift(left: list[SentenceSpan], right: list[SentenceSpan]) -> float:
    left_tokens = {token for sentence in left for token in sentence.tokens if len(token) >= 4}
    right_tokens = {token for sentence in right for token in sentence.tokens if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return 1.0 - (overlap / union if union else 0.0)


def _best_cut_before(
    sentences: list[SentenceSpan],
    boundaries: list[BoundaryCandidate],
    start: int,
    end: int,
    settings: SectioningSettings,
) -> int:
    valid = [
        boundary
        for boundary in boundaries
        if start < boundary.index <= end and sentences[boundary.index - 1].end - sentences[start].start >= settings.min_section_chars
    ]
    if valid:
        return max(valid, key=lambda item: item.score).index
    return max(start + 1, end)


def _target_section_count_range(char_count: int, settings: SectioningSettings) -> tuple[int, int]:
    target = max(1, round(char_count / 40000 * settings.target_sections_per_40k_chars))
    target = max(target, math.ceil(char_count / settings.max_section_chars))
    lower = max(1, math.floor(target * (1 - settings.target_count_tolerance)))
    upper = max(lower, math.ceil(target * (1 + settings.target_count_tolerance)))
    return lower, upper


def _section_count_distance(count: int, target_min: int, target_max: int) -> int:
    if target_min <= count <= target_max:
        return 0
    if count < target_min:
        return target_min - count
    return count - target_max


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return values[index]


def _make_section(
    document: Document,
    ordinal: int,
    sentence_start_index: int,
    text: str,
    char_start: int,
    char_end: int,
    sentences: list[SentenceSpan],
) -> DocumentSection:
    title = _section_title(text, ordinal)
    identifier = stable_id("semsec", f"{document.doc_id}:{char_start}:{char_end}:{ordinal}")
    return DocumentSection(
        section_id=identifier,
        doc_id=document.doc_id,
        title=title,
        level=1,
        section_path=[title],
        text=text,
        char_start=char_start,
        char_end=char_end,
        summary=text[:200].strip(),
        keywords=keywords(text),
        metadata={
            "sectioner": "semantic",
            "ordinal": ordinal,
            "sentence_start_index": sentence_start_index,
            "sentence_count": len(sentences),
        },
    )


def _renumber_section(document: Document, section: DocumentSection, ordinal: int) -> DocumentSection:
    return _make_section(document, ordinal, 0, section.text, section.char_start, section.char_end, [])


def _section_title(text: str, ordinal: int) -> str:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if _looks_like_standalone_heading(first_line):
        return first_line[:80]
    first_sentence = re.split(r"[。！？!?\.]\s+", text.strip(), maxsplit=1)[0].strip()
    compact = first_sentence[:72].strip(" ,，.;；。")
    return compact or f"Semantic section {ordinal}"


def _drop_repeated_page_noise(lines: list[str]) -> list[str]:
    stripped_lines = [_normalize_line(line) for line in lines]
    counts: dict[str, int] = {}
    for line in stripped_lines:
        if line and len(line) <= 90:
            counts[line] = counts.get(line, 0) + 1
    repeat_threshold = max(3, len(lines) // 80)
    cleaned: list[str] = []
    for line, stripped in zip(lines, stripped_lines):
        if _looks_like_page_number(stripped):
            continue
        if stripped and counts.get(stripped, 0) >= repeat_threshold and not _looks_like_standalone_heading(stripped):
            continue
        cleaned.append(line)
    return cleaned


def _trim_front_matter_and_references(text: str) -> str:
    abstract_match = re.search(r"(?i)\babstract\b", text[:6000])
    if abstract_match and abstract_match.start() > 0:
        text = text[abstract_match.start() :]

    references_match = re.search(r"(?im)(?:^|\n)\s*(references|bibliography)\s*(?:\n|$)", text)
    if not references_match:
        references_match = re.search(r"(?i)\breferences\b", text)
    if references_match and references_match.start() > max(2500, len(text) * 0.45):
        text = text[: references_match.start()]
    return text


def _looks_like_pdf_noise_line(line: str) -> bool:
    lowered = line.lower()
    if "@" in line:
        return True
    noise_terms = [
        "provided proper attribution",
        "permission to make digital or hard copies",
        "copyright",
        "journalistic or scholarly works",
        "google hereby grants permission",
    ]
    return any(term in lowered for term in noise_terms)


def _normalize_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _should_join_pdf_line(previous: str, current: str) -> bool:
    if _looks_like_standalone_heading(current):
        return False
    if not previous:
        return False
    if not re.search(r"[。！？!?\.:\)]$", previous):
        return True
    if _starts_lowercase_or_cjk(current) and not _looks_like_standalone_heading(current):
        return True
    return False


def _looks_like_page_number(line: str) -> bool:
    return bool(re.match(r"^(?:page\s*)?\d{1,4}$", line.lower())) or bool(re.match(r"^-\s*\d{1,4}\s*-$", line))


def _looks_like_standalone_heading(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*|[A-Z])[\.\)]\s+\S+", line):
        return True
    if line.startswith("#"):
        return True
    if line.isupper() and len(line.split()) <= 12 and any(char.isalpha() for char in line):
        return True
    return False


def _starts_lowercase_or_cjk(line: str) -> bool:
    return bool(line and (line[0].islower() or re.match(r"[\u4e00-\u9fff]", line[0])))


def _cjk_boundary(previous: str, current: str) -> bool:
    return bool(previous and current and re.search(r"[\u4e00-\u9fff]$", previous) and re.match(r"[\u4e00-\u9fff]", current))


def _tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z][a-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if re.search(r"[\u4e00-\u9fff]", token) and len(token) > 2:
            tokens.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return [token for token in tokens if token not in _STOPWORDS]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "has",
    "have",
    "their",
    "than",
    "then",
    "there",
    "which",
    "的",
    "了",
    "和",
    "与",
    "在",
    "是",
    "对",
    "为",
    "中",
}
