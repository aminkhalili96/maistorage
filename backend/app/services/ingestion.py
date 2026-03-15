from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pypdf import PdfReader

from app.config import Settings
from app.corpus import load_corpus_manifest, load_normalized_chunks
from app.models import ChunkRecord, DocumentSource, IngestRequest, IngestionStatus
from app.services.chunking import chunk_html_document, chunk_pdf_document
from app.services.indexes import SearchIndex


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        index: SearchIndex,
        sources: list[DocumentSource],
        demo_chunks: list[ChunkRecord],
    ) -> None:
        self.settings = settings
        self.index = index
        self.sources = sources
        self.demo_chunks = demo_chunks
        self.manifest = load_corpus_manifest(settings.corpus_manifest_path)
        self.status = IngestionStatus(snapshot_id=self.manifest.get("snapshot_id"))

    def bootstrap_local_corpus(self) -> None:
        if self.index.count() > 0:
            return
        normalized_chunks = load_normalized_chunks(self.settings.normalized_doc_root)
        if normalized_chunks:
            self.index.upsert(normalized_chunks)
            self.status.loaded_demo_corpus = False
            self.status.chunk_counts["normalized"] = len(normalized_chunks)
            self.status.updated_at = datetime.now(UTC).isoformat()
            self.status.last_refresh_at = self.manifest.get("retrieved_at")
            return

        if self.settings.raw_html_root.exists() or self.settings.raw_pdf_root.exists():
            all_chunks: list[ChunkRecord] = []
            for source in self.sources:
                all_chunks.extend(self._normalize_local_source(source))
            if all_chunks:
                self.index.upsert(all_chunks)
                self.status.loaded_demo_corpus = False
                self.status.chunk_counts["normalized"] = len(all_chunks)
                self.status.updated_at = datetime.now(UTC).isoformat()
                self.status.last_refresh_at = self.manifest.get("retrieved_at")
                return

        if not self.settings.is_assessment_mode:
            self.bootstrap_demo_corpus()

    def bootstrap_demo_corpus(self) -> None:
        if self.index.count() > 0:
            return
        self.index.upsert(self.demo_chunks)
        self.status.loaded_demo_corpus = True
        self.status.chunk_counts["demo"] = len(self.demo_chunks)
        self.status.updated_at = datetime.now(UTC).isoformat()

    def get_status(self) -> IngestionStatus:
        return self.status

    def prepare_job(self) -> str:
        job_id = f"job-{int(datetime.now(UTC).timestamp())}"
        self.status.last_job_id = job_id
        self.status.active = True
        self.status.errors = []
        self.status.updated_at = datetime.now(UTC).isoformat()
        return job_id

    def run_job(self, job_id: str, request: IngestRequest) -> None:
        self.settings.raw_html_root.mkdir(parents=True, exist_ok=True)
        self.settings.raw_pdf_root.mkdir(parents=True, exist_ok=True)
        self.settings.normalized_doc_root.mkdir(parents=True, exist_ok=True)

        selected_families = set(request.families or [])
        source_counts: dict[str, int] = defaultdict(int)
        chunk_counts: dict[str, int] = defaultdict(int)
        changed_sources: list[str] = []
        errors: list[str] = []

        try:
            for source in self.sources:
                if selected_families and source.doc_family not in selected_families:
                    continue
                try:
                    chunks = self._ingest_source(source, force_refresh=request.force_refresh)
                    if chunks:
                        self.index.upsert(chunks)
                        changed_sources.append(source.id)
                    source_counts[source.doc_family] += 1
                    chunk_counts[source.id] += len(chunks)
                except Exception as exc:  # pragma: no cover
                    errors.append(f"{source.id}: {exc}")
        finally:
            self.status.active = False
            self.status.last_job_id = job_id
            self.status.snapshot_id = self.manifest.get("snapshot_id", self.status.snapshot_id)
            self.status.source_counts = dict(source_counts)
            self.status.chunk_counts = dict(chunk_counts)
            self.status.changed_sources = changed_sources
            self.status.errors = errors
            self.status.updated_at = datetime.now(UTC).isoformat()
            self.status.last_refresh_at = datetime.now(UTC).isoformat()

    def _ingest_source(self, source: DocumentSource, force_refresh: bool = False) -> list[ChunkRecord]:
        if force_refresh:
            self._fetch_source_to_local(source)
        local_chunks = self._normalize_local_source(source)
        if local_chunks:
            return local_chunks
        if force_refresh:
            return []
        return local_chunks

    def _normalize_local_source(self, source: DocumentSource) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        source_html_root = self.settings.raw_html_root / source.id
        source_pdf_root = self.settings.raw_pdf_root / source.id
        normalized_path = self.settings.normalized_doc_root / f"{source.id}.jsonl"
        lines: list[str] = []

        source_payload = self.manifest.get("sources", {}).get(source.id, {})
        retrieved_at = source_payload.get("retrieved_at") or self.manifest.get("retrieved_at")
        snapshot_id = source_payload.get("snapshot_id") or self.manifest.get("snapshot_id")
        doc_version = source_payload.get("doc_version")

        bound_source = source.model_copy(
            update={
                "retrieved_at": retrieved_at,
                "snapshot_id": snapshot_id,
                "doc_version": doc_version,
            }
        )

        if source_html_root.exists():
            for html_path in sorted(source_html_root.glob("*.html")):
                html = html_path.read_text()
                page_url = source_payload.get("local_url_map", {}).get(html_path.name, source.url)
                records.extend(
                    chunk_html_document(
                        bound_source,
                        page_url,
                        html,
                        updated_at=retrieved_at,
                    )
                )

        if source_pdf_root.exists():
            for pdf_path in sorted(source_pdf_root.glob("*.pdf")):
                reader = PdfReader(str(pdf_path))
                pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
                records.extend(
                    chunk_pdf_document(
                        bound_source,
                        source_payload.get("pdf_url") or source.pdf_url or source.url,
                        pdf_text,
                        updated_at=retrieved_at,
                        title=f"{source.title} PDF",
                    )
                )

        if records:
            lines = [record.model_dump_json() for record in records]
            normalized_path.write_text("\n".join(lines))
        return records

    def _fetch_source_to_local(self, source: DocumentSource) -> None:
        client = httpx.Client(timeout=20.0, follow_redirects=True)
        source_html_root = self.settings.raw_html_root / source.id
        source_html_root.mkdir(parents=True, exist_ok=True)
        source_pdf_root = self.settings.raw_pdf_root / source.id
        source_pdf_root.mkdir(parents=True, exist_ok=True)

        response = client.get(source.url)
        response.raise_for_status()
        html_path = source_html_root / "root.html"
        html_path.write_text(response.text)

        if source.pdf_url:
            pdf_response = client.get(source.pdf_url)
            pdf_response.raise_for_status()
            (source_pdf_root / "source.pdf").write_bytes(pdf_response.content)

        snapshot_id = self.manifest.get("snapshot_id") or datetime.now(UTC).strftime("%Y%m%d")
        content_hash = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
        self.manifest.setdefault("sources", {})
        self.manifest["sources"][source.id] = {
            "retrieved_at": datetime.now(UTC).isoformat(),
            "snapshot_id": snapshot_id,
            "content_hash": content_hash,
            "doc_version": source.doc_version,
            "local_url_map": {"root.html": source.url},
            "pdf_url": source.pdf_url,
        }
        self.manifest["snapshot_id"] = snapshot_id
        self.manifest["retrieved_at"] = datetime.now(UTC).isoformat()
        self.settings.corpus_manifest_path.write_text(json.dumps(self.manifest, indent=2))
