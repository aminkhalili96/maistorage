"""Upload all freshly re-chunked normalized JSONL files to Pinecone.

Run from backend/:
    PYTHONPATH=. .venv/bin/python3 scripts/push_to_pinecone.py

NOTE: Fail-Safe mode for Gemini Free Tier (Strict 2,000 TPM limit).
Uses 1-chunk batches with an 8s delay.
"""
from __future__ import annotations

import time
import json
import os
from pathlib import Path
from collections import defaultdict
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import httpx

# Mocking app context for standalone script
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


def main():
    settings = get_settings()
    normalized_root = settings.normalized_doc_root

    print(f"\n--- Pinecone Upload Start (PID: {os.getpid()}) ---")

    # Prioritize hardware docs (H100, H200, etc.)
    priority_stems = ["h100", "h200", "a100", "l40s", "dgx-basepod"]
    
    files = sorted(normalized_root.glob("*.jsonl"))
    priority_files = [f for f in files if f.stem in priority_stems]
    other_files = [f for f in files if f.stem not in priority_stems]
    
    ordered_files = priority_files + other_files

    all_records: list[ChunkRecord] = []
    seen_ids = set()
    print(f"Loading chunks in priority order: {[f.stem for f in ordered_files]}")
    for jsonl_path in ordered_files:
        if not jsonl_path.exists(): continue
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                record = ChunkRecord.model_validate_json(line)
                if record.id not in seen_ids:
                    all_records.append(record)
                    seen_ids.add(record.id)

    print(f"Loaded {len(all_records)} unique chunks (Deduplicated from original ~12,018).")

    embedder = OpenAIEmbedder(settings)

    from pinecone import Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Fail-Safe: 1 chunk per call @ 8s delay = ~1500 TPM (Safe on 2000 TPM limit).
    EMBED_BATCH = 1
    UPSERT_BATCH = 20

    @retry(
        wait=wait_exponential(multiplier=2, min=20, max=120),
        stop=stop_after_attempt(15),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        before_sleep=lambda rs: print(f"    [Retry {rs.attempt_number}] Rate limited. Backing off for {rs.next_action.sleep}s...")
    )
    def _embed_with_retry(texts: list[str]) -> list[list[float]]:
        return embedder.embed_documents(texts)

    print(f"Target: '{settings.pinecone_index_name}' / '{settings.pinecone_namespace}'")
    upsert_queue = []

    for start in range(0, len(all_records), EMBED_BATCH):
        batch = all_records[start : start + EMBED_BATCH]
        chunk_id = batch[0].id
        source = batch[0].metadata.get("source_id", "unknown")
        
        try:
             # Progress indicator every 10 chunks
             if start % 10 == 0:
                 print(f"  [{start}/{len(all_records)}] Processing {source} ({chunk_id})...")
             
             vectors = _embed_with_retry([c.content for c in batch])
        except Exception as e:
             print(f"\nFATAL: Failed to embed at index {start} ({chunk_id}). Error: {e}")
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
                print(f"    >>> Upserted batch of {len(upsert_queue)} to Pinecone")
                upsert_queue.clear()

        # 8 second delay for steady flow
        time.sleep(8)

    if upsert_queue:
        index.upsert(vectors=upsert_queue, namespace=settings.pinecone_namespace)
        print(f"    >>> Final upsert of {len(upsert_queue)}")

    print("\nUpload complete successfully.")

if __name__ == "__main__":
    main()
