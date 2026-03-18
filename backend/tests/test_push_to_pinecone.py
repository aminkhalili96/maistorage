"""Upload all freshly re-chunked normalized JSONL files to Pinecone.

Run from backend/:
    PYTHONPATH=. .venv/bin/pytest tests/test_push_to_pinecone.py -s -q

NOTE: Fail-Safe mode for Gemini Free Tier (Strict 2,000 TPM limit).
Uses 2-chunk batches with a 10s delay.
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from collections import defaultdict
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import httpx

from app.config import get_settings
from app.models import ChunkRecord
from app.services.indexes import _sparse_vector, _sanitize_pinecone_metadata
from app.services.providers import OpenAIEmbedder


def _dedup_sparse(sparse: dict) -> dict:
    """Merge values for colliding hash bucket indices."""
    merged: dict[int, float] = defaultdict(float)
    for idx, val in zip(sparse["indices"], sparse["values"]):
        merged[idx] += val
    indices = sorted(merged.keys())
    values = [merged[i] for i in indices]
    return {"indices": indices, "values": values}


def test_push_normalized_to_pinecone():
    settings = get_settings()
    normalized_root = settings.normalized_doc_root

    # Prioritize hardware docs (H100, H200, etc.)
    priority_stems = ["h100", "h200", "a100", "l40s", "dgx-basepod"]
    
    files = sorted(normalized_root.glob("*.jsonl"))
    priority_files = [f for f in files if f.stem in priority_stems]
    other_files = [f for f in files if f.stem not in priority_stems]
    
    ordered_files = priority_files + other_files

    all_records: list[ChunkRecord] = []
    print(f"\nLoading chunks in priority order: {[f.stem for f in ordered_files]}")
    for jsonl_path in ordered_files:
        if not jsonl_path.exists(): continue
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                all_records.append(ChunkRecord.model_validate_json(line))

    print(f"Loaded {len(all_records)} total chunks.")
    assert len(all_records) > 0, "No chunks found"

    embedder = OpenAIEmbedder(settings)

    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Fail-Safe: 2 chunks @ ~200 tokens = ~400 tokens / call.
    # 6 calls per minute = 2,400 TPM. 
    # Let's do 1 chunk per call @ 10s delay = 6 chunks per minute = ~1200 TPM (Very safe).
    EMBED_BATCH = 1
    UPSERT_BATCH = 20

    @retry(
        wait=wait_exponential(multiplier=2, min=15, max=120),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=lambda rs: print(f"    [Retry {rs.attempt_number}] Rate limited. Backing off for {rs.next_action.sleep}s...")
    )
    def _embed_with_retry(texts: list[str]) -> list[list[float]]:
        try:
            return embedder.embed_documents(texts)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise
            raise

    print(f"Pushing to Pinecone '{settings.pinecone_index_name}' in namespace '{settings.pinecone_namespace}'...")
    upsert_queue = []

    for start in range(0, len(all_records), EMBED_BATCH):
        batch = all_records[start : start + EMBED_BATCH]
        current_source = batch[0].metadata.get("source_id", "unknown")
        
        try:
             vectors = _embed_with_retry([c.content for c in batch])
        except Exception as e:
             print(f"\nFATAL: Failed to embed batch at {start}. Error: {e}")
             break

        for chunk, vector in zip(batch, vectors):
            meta = _sanitize_pinecone_metadata(chunk.model_dump())
            meta["content"] = chunk.content
            upsert_queue.append(
                {
                    "id": chunk.id,
                    "values": vector,
                    "sparse_values": _dedup_sparse(_sparse_vector(chunk.content)),
                    "metadata": meta,
                }
            )
            if len(upsert_queue) >= UPSERT_BATCH:
                index.upsert(vectors=upsert_queue, namespace=settings.pinecone_namespace)
                upsert_queue.clear()

        done = min(start + EMBED_BATCH, len(all_records))
        print(f"  {done}/{len(all_records)} chunks processed (Current: {current_source})")
        
        # 10 second delay for ultimate safety on Free Tier
        time.sleep(10)

    if upsert_queue:
        index.upsert(vectors=upsert_queue, namespace=settings.pinecone_namespace)

    print("\nUpload complete.")
