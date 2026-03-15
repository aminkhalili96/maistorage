from __future__ import annotations

import hashlib
import re
from typing import Iterable

from bs4 import BeautifulSoup

from app.models import ChunkRecord, DocumentSource
from app.services.providers import tokenize


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
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
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
        for chunk_text in split_with_overlap(content, max_chars=max_chars, overlap=overlap):
            chunk_id, content_hash = build_chunk_id(source.id, section_path, chunk_text)
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    source_id=source.id,
                    title=title,
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
                    sparse_terms=tokenize(f"{title} {section_path} {chunk_text}"),
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
