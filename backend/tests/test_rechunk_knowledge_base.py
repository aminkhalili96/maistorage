"""Re-chunk all raw HTML/PDF sources using the current chunking pipeline,
saving fresh .jsonl files to data/knowledge_base/normalized/.

Run from the backend/ directory:
    PYTHONPATH=. pytest tests/test_rechunk_knowledge_base.py -s -q
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.knowledge_base import load_sources
from app.services.chunking import chunk_html_document, chunk_pdf_document
from app.models import DocumentSource


def _rechunk_source(source: DocumentSource, settings, manifest: dict) -> int:
    source_html_root = settings.raw_html_root / source.id
    source_pdf_root = settings.raw_pdf_root / source.id
    normalized_path = settings.normalized_doc_root / f"{source.id}.jsonl"

    source_payload = manifest.get("sources", {}).get(source.id, {})
    retrieved_at = source_payload.get("retrieved_at") or manifest.get("retrieved_at")
    snapshot_id = source_payload.get("snapshot_id") or manifest.get("snapshot_id")
    doc_version = source_payload.get("doc_version")

    bound_source = source.model_copy(update={
        "retrieved_at": retrieved_at,
        "snapshot_id": snapshot_id,
        "doc_version": doc_version,
    })

    records = []
    if source_html_root.exists():
        for html_path in sorted(source_html_root.glob("*.html")):
            html = html_path.read_text()
            page_url = source_payload.get("local_url_map", {}).get(html_path.name, source.url)
            records.extend(chunk_html_document(bound_source, page_url, html, updated_at=retrieved_at))

    if source_pdf_root.exists():
        try:
            from pypdf import PdfReader
        except ImportError:
            PdfReader = None  # type: ignore
        if PdfReader:
            for pdf_path in sorted(source_pdf_root.glob("*.pdf")):
                reader = PdfReader(str(pdf_path))
                pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
                records.extend(chunk_pdf_document(
                    bound_source,
                    source_payload.get("pdf_url") or source.pdf_url or source.url,
                    pdf_text,
                    updated_at=retrieved_at,
                    title=f"{source.title} PDF",
                ))

    if records:
        lines = [r.model_dump_json() for r in records]
        normalized_path.write_text("\n".join(lines))

    return len(records)


def test_rechunk_all_sources():
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    manifest = json.loads(settings.knowledge_base_manifest_path.read_text()) if settings.knowledge_base_manifest_path.exists() else {}
    settings.normalized_doc_root.mkdir(parents=True, exist_ok=True)

    total_before = 0
    total_after = 0

    for source in sources:
        html_root = settings.raw_html_root / source.id
        pdf_root = settings.raw_pdf_root / source.id
        if not html_root.exists() and not pdf_root.exists():
            continue

        # Count existing lines if file existed before
        existing_path = settings.normalized_doc_root / f"{source.id}.jsonl"

        count = _rechunk_source(source, settings, manifest)
        print(f"  {source.id}: {count} chunks after filter")
        total_after += count

    print(f"\nTotal re-chunked: {total_after} substantive chunks written")
    assert total_after > 0, "No chunks were produced — check raw HTML paths"
    print("Re-chunking complete.")
