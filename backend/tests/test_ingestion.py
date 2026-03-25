from __future__ import annotations

import json
from pathlib import Path

from app.models import DocumentSource
from app.services.chunking import split_with_overlap
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder


def _write_fixture_manifest(path: Path, source_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "snapshot_id": "fixture-snapshot",
                "retrieved_at": "2026-03-15T00:00:00Z",
                "sources": {
                    source_id: {
                        "retrieved_at": "2026-03-15T00:00:00Z",
                        "snapshot_id": "fixture-snapshot",
                        "doc_version": "v1",
                        "local_url_map": {"root.html": "https://example.com/docs"},
                    }
                },
            }
        )
    )


def test_ingestion_fixture_produces_expected_chunk_count_and_metadata(dev_settings, tmp_path):
    source = DocumentSource(
        id="fixture-doc",
        title="Fixture Doc",
        url="https://example.com/docs",
        doc_family="core",
        doc_type="html",
        crawl_prefix="https://example.com/docs",
        product_tags=["gpu", "training"],
    )
    html_root = dev_settings.raw_html_root / source.id
    html_root.mkdir(parents=True, exist_ok=True)
    dev_settings.normalized_doc_root.mkdir(parents=True, exist_ok=True)
    html = """
    <html><body>
      <h1>Overview</h1>
      <p>Mixed precision training improves throughput on Tensor Cores by using FP16 for
      compute and FP32 for weight accumulation, reducing memory footprint while maintaining
      numerical stability across the full training run on NVIDIA GPUs.</p>
      <h2>Scaling</h2>
      <p>NCCL all-reduce can become the communication bottleneck when scaling beyond four GPUs,
      as the cross-GPU bandwidth saturates the PCIe or NVLink interconnect during gradient
      synchronization across nodes in a distributed training cluster.</p>
    </body></html>
    """
    (html_root / "root.html").write_text(html)
    _write_fixture_manifest(dev_settings.knowledge_base_manifest_path, source.id)

    service = IngestionService(dev_settings, InMemoryHybridIndex(KeywordEmbedder()), [source], [])
    chunks = service._normalize_local_source(source)

    assert len(chunks) >= 1
    assert all(chunk.source_id == source.id for chunk in chunks)
    assert all(chunk.snapshot_id == "fixture-snapshot" for chunk in chunks)
    assert all(chunk.retrieved_at == "2026-03-15T00:00:00Z" for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)
    normalized = (dev_settings.normalized_doc_root / f"{source.id}.jsonl").read_text().splitlines()
    assert len(normalized) >= 1


def test_chunk_fixture_respects_size_limits():
    content = "abcdefghijklmnopqrstuvwxyz" * 40
    chunks = split_with_overlap(content, max_chars=150, overlap=30)

    assert len(chunks) > 1
    assert all(len(chunk) <= 150 for chunk in chunks)
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert left[-30:] == right[:30]
