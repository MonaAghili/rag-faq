"""Turn a raw Dropbox help-center article page into a clean, structured Article.

Pure functions only (no I/O) so this is easy to unit-test against saved HTML
fixtures. Selectors below were confirmed by hand against live article pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PreformattedString, Tag

# Bump whenever a parser change alters article text: `crawl` then re-fetches
# pages saved by an older version instead of skipping them as already done.
# Version 2 added tables and text outside paragraphs.
PARSER_VERSION = 2

TITLE_CLASS = "hp01-header-updated-date__headline"
UPDATED_DATE_CLASS = "hp01-header-updated-date__updated-date"
BREADCRUMB_ITEM_CLASS = "hp03-breadcrumb__item"
GRID_CLASS = "aem-Grid"
BODY_CLASS = "responsivegrid"
EXCLUDED_BODY_CLASSES = ("hp04-helpfulness", "hp13-related-articles", "hp14-community-answers")

_HEADING_MARKERS = {"h2": "##", "h3": "###", "h4": "####"}
_INLINE_TAGS = {
    "a", "abbr", "b", "br", "cite", "code", "del", "em", "font", "i", "img", "ins",
    "kbd", "label", "mark", "q", "s", "small", "span", "strong", "sub", "sup", "time", "u",
}


@dataclass
class Article:
    url: str
    title: str
    category: str
    updated_date: str | None
    text: str
    parser_version: int = 1  # records saved before versioning existed are version 1


def parse_article(html: str, url: str) -> Article | None:
    """Extract title/category/date/body text, or None if this isn't a real
    article page (e.g. a redirect or template page missing the usual header)."""
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1", class_=TITLE_CLASS)
    if h1 is None:
        return None
    title = h1.get_text(strip=True)

    body = _find_body(h1)
    if body is None:
        return None

    _strip_excluded(body)
    text = _render_body(body)
    if not text:
        return None

    return Article(
        url=url,
        title=title,
        category=_extract_category(soup),
        updated_date=_extract_updated_date(soup),
        text=text,
        parser_version=PARSER_VERSION,
    )


def _find_body(h1: Tag) -> Tag | None:
    # The body is the .responsivegrid that is a *direct* child of the
    # .aem-Grid containing the h1 - each page also has an unrelated
    # .responsivegrid elsewhere (header/footer chrome), so a bare
    # class selector would grab the wrong element.
    grid = h1.find_parent(class_=GRID_CLASS)
    if grid is None:
        return None
    return grid.find("div", class_=BODY_CLASS, recursive=False)


def _strip_excluded(body: Tag) -> None:
    for cls in EXCLUDED_BODY_CLASSES:
        for el in body.find_all(class_=cls):
            el.decompose()
    for tag in body.find_all(["script", "svg"]):
        tag.decompose()


def _extract_category(soup: BeautifulSoup) -> str:
    crumbs = soup.find_all(class_=BREADCRUMB_ITEM_CLASS)
    if len(crumbs) >= 2:
        return crumbs[1].get_text(strip=True)
    return "Uncategorized"


def _extract_updated_date(soup: BeautifulSoup) -> str | None:
    el = soup.find(class_=UPDATED_DATE_CLASS)
    if el is None:
        return None
    # Some articles show an audience badge (e.g. "Admins") right before the
    # date in the same container, so look for "Updated ..." rather than
    # assuming the date text starts at the beginning.
    text = el.get_text(" ", strip=True)
    match = re.search(r"Updated\s+(.+)", text)
    return match.group(1).strip() if match else (text or None)


def _render_inline(node: Tag | NavigableString) -> str:
    """Render inline content to text: keep **bold**, drop link hrefs (the
    citation already happens at the chunk level so we don't need inline URLs)."""
    if isinstance(node, PreformattedString):  # HTML comments, CDATA, doctype
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in ("strong", "b"):
        raw = "".join(_render_inline(c) for c in node.children)
        inner = raw.strip()
        if not inner:
            return raw
        # The page often puts the separating space *inside* the tag
        # (<strong>Admin console </strong>in), so move it outside the markers.
        leading = raw[: len(raw) - len(raw.lstrip())]
        trailing = raw[len(raw.rstrip()) :]
        return f"{leading}**{inner}**{trailing}"
    if node.name == "br":
        return "\n"
    return "".join(_render_inline(c) for c in node.children)


def _render_block(el: Tag) -> list[str]:
    """Render one block-level element (and, for generic containers, recurse
    into its children) to a list of markdown-ish lines."""
    if el.name in _HEADING_MARKERS:
        text = _render_inline(el).strip()
        return [f"{_HEADING_MARKERS[el.name]} {text}"] if text else []

    if el.name == "ul":
        lines = []
        for li in el.find_all("li", recursive=False):
            text = _render_inline(li).strip()
            if text:
                lines.append(f"- {text}")
        return lines

    if el.name == "ol":
        lines = []
        for i, li in enumerate(el.find_all("li", recursive=False), start=1):
            text = _render_inline(li).strip()
            if text:
                lines.append(f"{i}. {text}")
        return lines

    if el.name == "p":
        text = _render_inline(el).strip()
        return [text] if text else []

    if el.name == "table":
        return _render_table(el)

    return _render_container(el)


def _render_container(el: Tag) -> list[str]:
    """Render a generic container. Text sitting directly in it, or in inline
    tags like <a>/<strong>, becomes a paragraph of its own: some pages keep
    their content in bare <div>s and table cells rather than <p>s."""
    lines: list[str] = []
    run: list[Tag | NavigableString] = []

    def flush() -> None:
        text = "".join(_render_inline(node) for node in run).strip()
        if text:
            lines.append(text)
        run.clear()

    for child in el.children:
        if isinstance(child, NavigableString) or child.name in _INLINE_TAGS:
            run.append(child)
        else:
            flush()
            lines.extend(_render_block(child))
    flush()
    return lines


def _render_table(table: Tag) -> list[str]:
    """One bullet per row, each self-contained ("Header: value; ..."), so a
    row still makes sense if the chunker splits a long table."""
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        texts = [" ".join(" ".join(_render_container(cell)).split()) for cell in cells]
        if any(texts):
            rows.append((cells, texts))
    if not rows:
        return []

    header: list[str] = []
    if all(cell.name == "th" for cell in rows[0][0]):
        header = rows[0][1]
        rows = rows[1:]

    lines = []
    for _, texts in rows:
        if header and len(header) == len(texts):
            lines.append("- " + "; ".join(f"{h}: {v}" if h else v for h, v in zip(header, texts) if v))
        else:
            lines.append("- " + " | ".join(t for t in texts if t))
    return lines


def _render_body(body: Tag) -> str:
    text = "\n\n".join(_render_container(body))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
