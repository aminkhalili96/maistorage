# Corpus Refresh and Versioning

## Purpose

This project uses an offline NVIDIA documentation snapshot for assessment reliability, but it is designed to support refresh and incremental re-indexing when the upstream docs change.

## Source Manifest

The source registry lives in:

- `data/sources/nvidia_sources.json`

The snapshot manifest lives in:

- `data/corpus/manifest.json`

Each source entry records:

- canonical source URL
- optional PDF URL
- `retrieved_at`
- `snapshot_id`
- `content_hash`
- optional `pdf_hash`
- optional `doc_version`
- local file mapping for HTML snapshots

## Local Corpus Layout

- `data/corpus/raw/html/<source-id>/`
- `data/corpus/raw/pdfs/<source-id>/`
- `data/corpus/normalized/<source-id>.jsonl`

The raw layer is the reproducible snapshot. The normalized layer is the RAG-ready artifact consumed by indexing and evaluation.

## Refresh Flow

1. Pull the source list from `data/sources/nvidia_sources.json`.
2. Re-download each root page and selected child pages.
3. Re-download PDFs where `pdf_url` is available.
4. Recompute content hashes.
5. Update `data/corpus/manifest.json`.
6. Rebuild only the changed normalized files.
7. Upsert changed chunks into Pinecone.
8. Re-run retrieval and answer-level evaluation before promotion.

## Incremental Re-indexing

Chunk IDs are derived from:

- `source_id`
- `section_path`
- `content_hash`

That makes incremental updates predictable:

- unchanged chunks keep the same ID
- changed chunks receive a new ID
- stale chunk IDs can be deleted safely

## Promotion Strategy

Recommended production strategy:

- build into a candidate namespace or candidate index
- run retrieval benchmarks and citation checks
- run RAGAS if credentials are available
- switch the app to the candidate only when evaluation passes

For the assessment, a single local snapshot is enough. The candidate/prod split is documented to show how the same design scales beyond the demo.

## Failure Handling

- if refresh fails, keep the last successful snapshot
- never overwrite the production-ready snapshot until evaluation passes
- log changed sources and refresh timestamps
- surface snapshot and last refresh time in the UI

## Commands

- Download or refresh the bundled corpus:

```bash
backend/.venv/bin/python scripts/download_corpus.py
```

- Rebuild normalized JSONL chunks from the bundled raw corpus:

```bash
PYTHONPATH=backend backend/.venv/bin/python scripts/normalize_corpus.py
```

## Assessment Positioning

Suggested explanation in the presentation:

> The demo uses a versioned local snapshot for reliability. In a production setup, the same source manifest and normalization pipeline can refresh changed documentation automatically, re-index incrementally, and promote only after retrieval and grounding checks pass.
