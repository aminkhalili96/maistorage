from __future__ import annotations

import hashlib
import re
from typing import Iterable

from bs4 import BeautifulSoup

from app.models import ChunkRecord, DocumentSource
from app.services.providers import tokenize

# ---------------------------------------------------------------------------
# Generic page titles that should fall back to section_path
# ---------------------------------------------------------------------------
GENERIC_TITLES = {"Overview", "Documentation", "Home", "Index", "Contents"}

# ---------------------------------------------------------------------------
# Navigation / boilerplate chunk filter
# ---------------------------------------------------------------------------

# Signals that a chunk is structural (TOC, release-notes list, footer, etc.)
_NAV_PATTERN = re.compile(
    r"("
    r"Release \d+\.\d+|"          # NCCL release notes numbering
    r"NCCL Release \d|"            # explicit NCCL release header
    r"Corporate Info|"             # footer boilerplate
    r"NVIDIA\.com Home|"           # footer nav
    r"About NVIDIA|"               # footer nav
    r"Developer Home|"             # footer nav
    r"Developer Program|"          # footer nav
    r"Privacy Policy|"             # footer legal
    r"Terms of Service|"           # footer legal
    r"Copyright \u00a9 \d{4}|"     # copyright line
    r"Contact Us|"                 # footer nav
    r"Previous.{0,10}Next|"        # pagination
    r"Table of Contents"           # TOC header
    r")",
    re.IGNORECASE,
)

# Chunks shorter than this are almost always headings, labels, or index entries
_MIN_CONTENT_TOKENS = 25

# If a chunk has this many or more nav signal matches it is structural noise
_NAV_SIGNAL_THRESHOLD = 3


def _is_navigation_chunk(text: str) -> bool:
    """Return True for TOC pages, release-notes lists, footers, and other
    structural noise that should NOT be stored in the search index."""
    if len(tokenize(text)) < _MIN_CONTENT_TOKENS:
        return True
    matches = _NAV_PATTERN.findall(text)
    return len(matches) >= _NAV_SIGNAL_THRESHOLD


def _find_sentence_boundary(text: str, pos: int, scan_back: int = 100) -> int:
    """Scan backward from *pos* up to *scan_back* chars for a sentence boundary.

    Returns the best split position (end of sentence). Falls back to *pos*
    if no boundary is found within the scan range.
    """
    search_start = max(pos - scan_back, 0)
    window = text[search_start:pos]
    # Prefer paragraph breaks, then sentence-ending punctuation followed by space
    for sep in ("\n\n", ". ", ".\n", "! ", "? "):
        idx = window.rfind(sep)
        if idx != -1:
            return search_start + idx + len(sep)
    return pos


def _extend_past_table(text: str, pos: int) -> int:
    """If *pos* is inside a markdown table row, extend forward to the end of the table."""
    # Check if current line starts with '|' (table row)
    line_start = text.rfind("\n", 0, pos) + 1
    if pos < len(text) and text[line_start:pos].lstrip().startswith("|"):
        # Scan forward past table rows
        cursor = pos
        while cursor < len(text):
            nl = text.find("\n", cursor)
            if nl == -1:
                return len(text)
            next_line = text[nl + 1: nl + 2]
            if next_line != "|":
                return nl + 1
            cursor = nl + 1
    return pos


def _extend_past_code_block(text: str, pos: int) -> int:
    """If *pos* is inside a fenced code block, extend forward to the closing fence."""
    # Count ``` occurrences before pos — odd count means we're inside a code block
    fence_count = text[:pos].count("```")
    if fence_count % 2 == 1:
        close = text.find("```", pos)
        if close != -1:
            # Move past the closing fence and its trailing newline
            end = close + 3
            if end < len(text) and text[end] == "\n":
                end += 1
            return end
    return pos


def split_with_overlap(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _find_sentence_boundary(text, end)
            # If inside a markdown table, extend to end of table
            end = _extend_past_table(text, end)
            # If inside a fenced code block, extend to closing ```
            end = _extend_past_code_block(text, end)
            # Ensure we make forward progress
            if end <= start:
                end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, 0)
    return chunks


def extract_html_sections(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_buffer: list[str] = []

    for node in body.find_all(["h1", "h2", "h3", "p", "li"]):
        text = " ".join(node.stripped_strings)
        if not text:
            continue
        if node.name in {"h1", "h2", "h3"}:
            if current_buffer:
                sections.append((current_heading, " ".join(current_buffer)))
            current_heading = text
            current_buffer = []
        else:
            current_buffer.append(text)

    if current_buffer:
        sections.append((current_heading, " ".join(current_buffer)))
    return sections or [("Overview", " ".join(body.stripped_strings))]


def build_chunk_id(source_id: str, section_path: str, chunk_text: str) -> tuple[str, str]:
    section_slug = re.sub(r"[^a-z0-9]+", "-", section_path.lower()).strip("-") or "overview"
    content_hash = hashlib.sha256(f"{section_path}\n{chunk_text}".encode("utf-8")).hexdigest()[:16]
    return f"{source_id}-{section_slug[:36]}-{content_hash}", content_hash


def chunk_sections(
    source: DocumentSource,
    url: str,
    title: str,
    sections: Iterable[tuple[str, str]],
    updated_at: str | None = None,
    retrieved_at: str | None = None,
    doc_version: str | None = None,
    snapshot_id: str | None = None,
    source_kind: str = "corpus",
    max_chars: int = 900,
    overlap: int = 120,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for section_path, content in sections:
        # P6: strip trailing '#' and whitespace from section headings
        section_path = section_path.rstrip("# ").strip()
        # P7: fall back to last segment of section_path when title is generic
        effective_title = title
        if title in GENERIC_TITLES and section_path:
            effective_title = section_path.split(" > ")[-1].strip() if " > " in section_path else section_path.strip()
        for chunk_text in split_with_overlap(content, max_chars=max_chars, overlap=overlap):
            if _is_navigation_chunk(chunk_text):
                continue  # skip TOC / release-note lists / footer boilerplate
            chunk_id, content_hash = build_chunk_id(source.id, section_path, chunk_text)
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    source_id=source.id,
                    title=effective_title,
                    url=url,
                    section_path=section_path,
                    doc_family=source.doc_family,
                    doc_type=source.doc_type,
                    product_tags=source.product_tags,
                    updated_at=updated_at,
                    retrieved_at=retrieved_at,
                    content_hash=content_hash,
                    doc_version=doc_version or source.doc_version,
                    snapshot_id=snapshot_id or source.snapshot_id,
                    source_kind=source_kind,
                    content=chunk_text,
                    sparse_terms=list(dict.fromkeys(tokenize(f"{effective_title} {section_path} {chunk_text}"))),
                )
            )
    return records


def chunk_html_document(
    source: DocumentSource,
    url: str,
    html: str,
    updated_at: str | None = None,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[ChunkRecord]:
    sections = extract_html_sections(html)
    title = next((heading for heading, _ in sections if heading), source.title)
    return chunk_sections(
        source,
        url,
        title,
        sections,
        updated_at=updated_at,
        retrieved_at=source.retrieved_at,
        doc_version=source.doc_version,
        snapshot_id=source.snapshot_id,
        source_kind=source.source_kind,
        max_chars=max_chars,
        overlap=overlap,
    )


def extract_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown by ## headings into (heading, content) tuples.

    Uses the same (heading, content) tuple format as extract_html_sections()
    so the output plugs straight into chunk_sections().
    """
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_buffer: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        # Match ## or # headings (but not code blocks starting with #!)
        if stripped.startswith("#") and not stripped.startswith("#!"):
            # Extract heading text by stripping leading # characters
            heading_text = stripped.lstrip("#").strip()
            if heading_text:
                if current_buffer:
                    sections.append((current_heading, " ".join(current_buffer)))
                current_heading = heading_text
                current_buffer = []
                continue
        # Skip empty lines but keep content
        if stripped:
            current_buffer.append(stripped)

    if current_buffer:
        sections.append((current_heading, " ".join(current_buffer)))

    return sections or [("Overview", text.strip())]


def chunk_markdown_document(
    source: DocumentSource,
    url: str,
    text: str,
    updated_at: str | None = None,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[ChunkRecord]:
    """Chunk a markdown document into ChunkRecords, reusing chunk_sections()."""
    sections = extract_markdown_sections(text)
    title = next((heading for heading, _ in sections if heading), source.title)
    return chunk_sections(
        source,
        url,
        title,
        sections,
        updated_at=updated_at,
        retrieved_at=source.retrieved_at,
        doc_version=source.doc_version,
        snapshot_id=source.snapshot_id,
        source_kind=source.source_kind,
        max_chars=max_chars,
        overlap=overlap,
    )


def chunk_pdf_document(
    source: DocumentSource,
    url: str,
    text: str,
    updated_at: str | None = None,
    title: str | None = None,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[ChunkRecord]:
    normalized = re.sub(r"\s+", " ", text).strip()
    sections = [("Overview", normalized)] if normalized else []
    return chunk_sections(
        source,
        url,
        title or source.title,
        sections,
        updated_at=updated_at,
        retrieved_at=source.retrieved_at,
        doc_version=source.doc_version,
        snapshot_id=source.snapshot_id,
        source_kind=source.source_kind,
        max_chars=max_chars,
        overlap=overlap,
    )
