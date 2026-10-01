"""Split a parsed Article into overlapping, self-contained chunks for embedding.

Splits on ##/### headings first, then by size within any section that's
still too long. The size split only ever breaks between paragraphs/list
items/headings (the blank-line-separated "units" parser.py already produces),
so it never cuts mid-sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from faq_rag.parser import Article

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 200

_HEADING_RE = re.compile(r"^#{2,3}\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    url: str
    title: str
    section: str
    category: str
    text: str


def chunk_article(
    article: Article,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Turn one Article into a list of standalone, prefixed Chunks."""
    slug = urlparse(article.url).path.strip("/")
    chunks: list[Chunk] = []
    counter = 0

    for section_title, body in _split_into_sections(article.text):
        for piece in _split_by_size(body, max_chars, overlap_chars):
            text = f"{article.title} > {section_title}\n\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=f"{slug}::{counter:03d}",
                    url=article.url,
                    title=article.title,
                    section=section_title,
                    category=article.category,
                    text=text,
                )
            )
            counter += 1

    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split article text on ## / ### heading lines. Content before the first
    heading (if any) becomes a section titled "Overview"."""
    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_body: list[str] = []

    for unit in text.split("\n\n"):
        match = _HEADING_RE.match(unit)
        if match:
            if current_body:
                sections.append((current_title, current_body))
            current_title = match.group(1).strip()
            current_body = []
        else:
            current_body.append(unit)
    if current_body:
        sections.append((current_title, current_body))

    return [(title, "\n\n".join(body)) for title, body in sections]


def _split_by_size(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack paragraph/list-item units into ~max_chars pieces, carrying
    the trailing ~overlap_chars of units into the start of the next piece."""
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]

    units = body.split("\n\n")
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit) + 2  # +2 for the "\n\n" this unit will be joined with
        if current and current_len + unit_len > max_chars:
            pieces.append("\n\n".join(current))
            current, current_len = _carry_overlap(current, overlap_chars)
        current.append(unit)
        current_len += unit_len

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _carry_overlap(units: list[str], overlap_chars: int) -> tuple[list[str], int]:
    """Take trailing units from the end of `units` totalling ~overlap_chars,
    to seed the start of the next piece."""
    carried: list[str] = []
    carried_len = 0
    for unit in reversed(units):
        if carried_len >= overlap_chars:
            break
        carried.insert(0, unit)
        carried_len += len(unit) + 2
    return carried, carried_len
