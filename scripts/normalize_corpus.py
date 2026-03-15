from __future__ import annotations

from app.config import get_settings
from app.corpus import load_demo_chunks, load_sources
from app.models import IngestRequest
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder


def main() -> None:
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    job_id = ingestion.prepare_job()
    ingestion.run_job(job_id, request=IngestRequest(families=[], force_refresh=False))


if __name__ == "__main__":
    main()
