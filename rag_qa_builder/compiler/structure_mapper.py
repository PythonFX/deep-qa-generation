from __future__ import annotations

import re

from rag_qa_builder.models import Document, DocumentSection
from rag_qa_builder.utils.ids import stable_id
from rag_qa_builder.utils.text_utils import keywords


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def map_documents_to_sections(documents: list[Document]) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    for document in documents:
        if document.file_type in {".md", ".markdown"}:
            mapped = _map_markdown(document)
        else:
            mapped = _map_text(document)
        sections.extend(mapped)
    return sections


def _map_markdown(document: Document) -> list[DocumentSection]:
    lines = document.text.splitlines(keepends=True)
    matches: list[tuple[int, int, str]] = []
    offset = 0
    for line in lines:
        match = HEADING_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            matches.append((offset, level, title))
        offset += len(line)
    if not matches:
        return [_make_section(document, None, None, [], document.text, 0, len(document.text))]

    sections: list[DocumentSection] = []
    path_stack: list[tuple[int, str]] = []
    for index, (start, level, title) in enumerate(matches):
        content_start = document.text.find("\n", start)
        content_start = len(document.text) if content_start == -1 else content_start + 1
        end = matches[index + 1][0] if index + 1 < len(matches) else len(document.text)
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))
        body = document.text[content_start:end].strip()
        if not body:
            continue
        section_path = [item[1] for item in path_stack]
        sections.append(_make_section(document, title, level, section_path, body, content_start, end))
    return sections or [_make_section(document, None, None, [], document.text, 0, len(document.text))]


def _map_text(document: Document) -> list[DocumentSection]:
    lines = document.text.splitlines()
    headings: list[tuple[int, str]] = []
    cursor = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _looks_like_text_heading(lines, index):
            headings.append((cursor, stripped))
        cursor += len(line) + 1
    if not headings:
        return [_make_section(document, None, None, [], document.text, 0, len(document.text))]
    sections: list[DocumentSection] = []
    for index, (start, title) in enumerate(headings):
        body_start = document.text.find("\n", start)
        body_start = len(document.text) if body_start == -1 else body_start + 1
        end = headings[index + 1][0] if index + 1 < len(headings) else len(document.text)
        body = document.text[body_start:end].strip()
        if body:
            sections.append(_make_section(document, title, 1, [title], body, body_start, end))
    return sections or [_make_section(document, None, None, [], document.text, 0, len(document.text))]


def _looks_like_text_heading(lines: list[str], index: int) -> bool:
    line = lines[index].strip()
    if not line or len(line) > 80:
        return False
    prev_blank = index == 0 or not lines[index - 1].strip()
    next_blank = index == len(lines) - 1 or not lines[index + 1].strip()
    numbered = bool(re.match(r"^\d+(\.\d+)*[\.\)]?\s+\S+", line))
    uppercase = line.isupper() and any(ch.isalpha() for ch in line)
    titlecase = line == line.title() and len(line.split()) <= 10
    return prev_blank and (next_blank or numbered or uppercase or titlecase)


def _make_section(
    document: Document,
    title: str | None,
    level: int | None,
    section_path: list[str],
    text: str,
    char_start: int,
    char_end: int,
) -> DocumentSection:
    identifier = stable_id("sec", f"{document.doc_id}:{char_start}:{char_end}:{title or 'root'}")
    return DocumentSection(
        section_id=identifier,
        doc_id=document.doc_id,
        title=title,
        level=level,
        section_path=section_path,
        text=text,
        char_start=char_start,
        char_end=char_end,
        summary=text[:200].strip(),
        keywords=keywords(text),
    )

