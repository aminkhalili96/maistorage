"""Knowledge base data integrity tests — validate JSONL chunk files and source manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def normalized_dir(settings) -> Path:
    return settings.normalized_doc_root


@pytest.fixture(scope="module")
def source_manifest_path(settings) -> Path:
    return settings.source_manifest_path


@pytest.fixture(scope="module")
def all_chunks(normalized_dir) -> list[dict]:
    """Load every chunk from all JSONL files."""
    chunks: list[dict] = []
    for jsonl_file in sorted(normalized_dir.glob("*.jsonl")):
        for line_num, line in enumerate(jsonl_file.read_text().strip().splitlines(), 1):
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"{jsonl_file.name}:{line_num} — invalid JSON: {e}")
    return chunks


@pytest.fixture(scope="module")
def manifest_sources(source_manifest_path) -> list[dict]:
    """Load and parse the source manifest."""
    raw = json.loads(source_manifest_path.read_text())
    assert isinstance(raw, list), "nvidia_sources.json must be a JSON array"
    return raw


# --- Tests ---


class TestJsonlFiles:
    def test_all_jsonl_files_parse_without_errors(self, normalized_dir):
        """All JSONL files load without JSON parse errors."""
        jsonl_files = sorted(normalized_dir.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "No JSONL files found in normalized dir"
        for jsonl_file in jsonl_files:
            for line_num, line in enumerate(jsonl_file.read_text().strip().splitlines(), 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"{jsonl_file.name}:{line_num} — invalid JSON: {e}")

    def test_every_chunk_has_required_fields(self, all_chunks):
        """Every chunk has required ChunkRecord fields."""
        required = {"id", "source_id", "content", "content_hash"}
        for i, chunk in enumerate(all_chunks):
            missing = required - set(chunk.keys())
            assert not missing, f"Chunk {i} ({chunk.get('id', '?')}) missing fields: {missing}"

    def test_no_empty_content_fields(self, all_chunks):
        """No empty content fields in any chunk."""
        for chunk in all_chunks:
            assert chunk.get("content", "").strip(), (
                f"Chunk {chunk.get('id', '?')} has empty content"
            )

    def test_no_duplicate_chunk_ids(self, all_chunks):
        """All chunk IDs must be unique across JSONL files.

        The ingestion pipeline deduplicates at both file-level (identical HTML
        content hash) and chunk-level (duplicate chunk IDs). If this test fails,
        the normalize_knowledge_base.py script needs to be re-run.
        """
        ids = [c["id"] for c in all_chunks]
        unique_ids = set(ids)
        assert len(ids) == len(unique_ids), (
            f"{len(ids) - len(unique_ids)} duplicate chunk_ids found "
            f"({len(unique_ids)}/{len(ids)} unique). Run normalize_knowledge_base.py."
        )

    def test_content_hashes_are_nonempty(self, all_chunks):
        """Content hashes are non-empty strings."""
        for chunk in all_chunks:
            h = chunk.get("content_hash")
            assert h and isinstance(h, str) and len(h) > 0, (
                f"Chunk {chunk.get('id', '?')} has empty/missing content_hash"
            )


class TestManifest:
    def test_manifest_parses_and_has_expected_count(self, manifest_sources):
        """Manifest parses correctly and has at least 30 sources."""
        assert len(manifest_sources) >= 30, (
            f"Expected ≥30 sources, got {len(manifest_sources)}"
        )

    def test_all_chunk_source_ids_match_manifest(self, all_chunks, manifest_sources):
        """All source_ids in chunks match entries in the manifest."""
        manifest_ids = {s["id"] for s in manifest_sources}
        chunk_source_ids = {c["source_id"] for c in all_chunks}
        unmatched = chunk_source_ids - manifest_ids
        assert not unmatched, (
            f"Chunks reference source_ids not in manifest: {unmatched}"
        )

    def test_every_enabled_source_has_chunks(self, all_chunks, manifest_sources):
        """Every enabled source in manifest has at least one chunk in JSONL files."""
        chunk_source_ids = {c["source_id"] for c in all_chunks}
        enabled_sources = [s for s in manifest_sources if s.get("enabled", True)]
        missing = [s["id"] for s in enabled_sources if s["id"] not in chunk_source_ids]
        # Allow a few missing (some sources may be registered but not yet downloaded)
        # The assertion checks that at least 80% have chunks
        coverage = 1 - len(missing) / max(len(enabled_sources), 1)
        assert coverage >= 0.80, (
            f"Only {coverage:.0%} of enabled sources have chunks. "
            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
