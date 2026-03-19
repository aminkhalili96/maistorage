"""Tests for the normalize/chunk pipeline functions in app.services.chunking."""

from __future__ import annotations

import pytest

from app.models import ChunkRecord, DocumentSource
from app.services.chunking import (
    _is_navigation_chunk,
    build_chunk_id,
    chunk_markdown_document,
    extract_markdown_sections,
    split_with_overlap,
)

# ---------------------------------------------------------------------------
# Fixtures & sample data
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """\
# GPU Architecture

## Memory Hierarchy
The H100 features 80GB of HBM3 memory with 3.35 TB/s bandwidth.
This represents a significant improvement over the A100 generation.
The memory subsystem supports ECC protection and uses a 5120-bit bus
interface to deliver sustained throughput for large model training
workloads that require constant data movement between compute and memory.

## Compute Units
The H100 has 132 streaming multiprocessors (SMs) arranged in GPCs.
Each SM contains 128 CUDA cores for FP32 operations and 64 FP64 cores.
The fourth-generation Tensor Cores deliver significant speedups for
mixed precision matrix operations commonly used in transformer training
and inference workloads across enterprise AI deployments.

## NVLink
NVLink 4.0 provides 900 GB/s total bidirectional bandwidth per GPU.
This high-speed interconnect enables efficient multi-GPU communication
for collective operations such as all-reduce and all-gather that are
critical for data-parallel and tensor-parallel distributed training
strategies used in modern large language model development pipelines.
"""

SAMPLE_MARKDOWN_NO_HEADINGS = (
    "This is a plain text document with no markdown headings. "
    "It contains information about NVIDIA GPUs and their memory bandwidth "
    "specifications across multiple product generations."
)


@pytest.fixture
def test_source() -> DocumentSource:
    return DocumentSource(
        id="test-source",
        title="Test Doc",
        url="https://example.com",
        doc_family="test",
        doc_type="markdown",
        crawl_prefix="https://example.com",
        product_tags=["h100", "gpu"],
    )


# ---------------------------------------------------------------------------
# extract_markdown_sections
# ---------------------------------------------------------------------------


class TestExtractMarkdownSections:
    def test_splits_by_headings(self):
        sections = extract_markdown_sections(SAMPLE_MARKDOWN)

        headings = [h for h, _ in sections]
        assert "Memory Hierarchy" in headings
        assert "Compute Units" in headings
        assert "NVLink" in headings

        # Verify content is associated with the correct heading
        mem_section = next(c for h, c in sections if h == "Memory Hierarchy")
        assert "80GB" in mem_section
        assert "HBM3" in mem_section

        compute_section = next(c for h, c in sections if h == "Compute Units")
        assert "streaming multiprocessors" in compute_section
        assert "CUDA cores" in compute_section

    def test_no_headings_returns_overview(self):
        sections = extract_markdown_sections(SAMPLE_MARKDOWN_NO_HEADINGS)

        assert len(sections) == 1
        heading, content = sections[0]
        assert heading == "Overview"
        assert "NVIDIA GPUs" in content


# ---------------------------------------------------------------------------
# split_with_overlap
# ---------------------------------------------------------------------------


class TestSplitWithOverlap:
    def test_chunks_within_max_chars(self):
        # Build a long string that must be split
        long_text = "NVIDIA GPU performance analysis. " * 100  # ~3300 chars
        max_chars = 500
        chunks = split_with_overlap(long_text, max_chars=max_chars)

        assert len(chunks) > 1
        for chunk in chunks:
            # Allow some tolerance because sentence boundary search may
            # overshoot by up to ~100 chars in the worst case.
            assert len(chunk) <= max_chars + 150, (
                f"Chunk length {len(chunk)} exceeds max_chars {max_chars} + tolerance"
            )

    def test_overlap_between_consecutive_chunks(self):
        long_text = " ".join(f"word{i}" for i in range(300))
        max_chars = 400
        overlap = 80
        chunks = split_with_overlap(long_text, max_chars=max_chars, overlap=overlap)

        assert len(chunks) >= 2
        # Each consecutive pair should share some text (the overlap region)
        for i in range(len(chunks) - 1):
            tail_of_current = chunks[i][-overlap:]
            # The start of the next chunk should contain some of the tail
            # of the current chunk (overlap guarantees shared content).
            shared = set(tail_of_current.split()) & set(chunks[i + 1].split())
            assert len(shared) > 0, (
                f"Chunks {i} and {i+1} share no words despite overlap={overlap}"
            )

    def test_short_text_returns_single_chunk(self):
        short = "A brief note about GPUs."
        chunks = split_with_overlap(short, max_chars=900)
        assert chunks == [short]

    def test_empty_text_returns_empty_list(self):
        assert split_with_overlap("") == []
        assert split_with_overlap("   ") == []


# ---------------------------------------------------------------------------
# chunk_markdown_document
# ---------------------------------------------------------------------------


class TestChunkMarkdownDocument:
    def test_preserves_source_metadata(self, test_source):
        records = chunk_markdown_document(
            source=test_source,
            url="https://example.com/gpu",
            text=SAMPLE_MARKDOWN,
        )

        assert len(records) > 0
        for rec in records:
            assert isinstance(rec, ChunkRecord)
            assert rec.source_id == "test-source"
            assert rec.doc_family == "test"
            assert rec.doc_type == "markdown"
            assert rec.url == "https://example.com/gpu"
            assert rec.product_tags == ["h100", "gpu"]
            assert rec.content  # non-empty content
            assert rec.sparse_terms  # tokenized terms populated
            assert rec.id  # chunk ID generated
            assert rec.content_hash  # hash generated


# ---------------------------------------------------------------------------
# build_chunk_id
# ---------------------------------------------------------------------------


class TestBuildChunkId:
    def test_deterministic(self):
        id1, hash1 = build_chunk_id("src-1", "Memory Hierarchy", "80GB HBM3 memory")
        id2, hash2 = build_chunk_id("src-1", "Memory Hierarchy", "80GB HBM3 memory")
        assert id1 == id2
        assert hash1 == hash2

    def test_different_inputs_produce_different_ids(self):
        id_a, _ = build_chunk_id("src-1", "Memory", "80GB HBM3")
        id_b, _ = build_chunk_id("src-1", "Compute", "132 SMs")
        assert id_a != id_b

    def test_source_id_in_chunk_id(self):
        chunk_id, _ = build_chunk_id("my-source", "Overview", "some text")
        assert chunk_id.startswith("my-source-")


# ---------------------------------------------------------------------------
# _is_navigation_chunk
# ---------------------------------------------------------------------------


class TestIsNavigationChunk:
    def test_short_text_is_navigation(self):
        # Fewer than _MIN_CONTENT_TOKENS (25) words → navigation
        assert _is_navigation_chunk("Next Previous") is True
        assert _is_navigation_chunk("Home > Docs") is True

    def test_substantive_content_passes(self):
        content = (
            "The NVIDIA H100 Tensor Core GPU delivers exceptional performance "
            "for large language model training and inference workloads. "
            "It features 80GB of HBM3 memory with 3.35 TB/s bandwidth, "
            "providing significant improvements over the previous generation."
        )
        assert _is_navigation_chunk(content) is False

    def test_nav_signals_detected(self):
        # A chunk with multiple navigation signals should be flagged
        nav_text = (
            "Corporate Info About NVIDIA Developer Home Privacy Policy "
            "Terms of Service Contact Us Developer Program "
            "NVIDIA.com Home Copyright \u00a9 2025 NVIDIA Corporation. "
            "All rights reserved. Some additional padding text here to "
            "make this chunk exceed the minimum token count threshold."
        )
        assert _is_navigation_chunk(nav_text) is True
