from app.models import DocumentSource
from app.services.chunking import (
    build_chunk_id,
    chunk_html_document,
    chunk_markdown_document,
    chunk_pdf_document,
    extract_html_sections,
    extract_markdown_sections,
    split_with_overlap,
)
from app.services.chunking import _is_navigation_chunk  # noqa: PLC2701 — private but tested explicitly


def test_extract_html_sections_preserves_headings():
    html = """
    <html><body>
      <h1>Overview</h1>
      <p>GPU jobs can be compute-bound.</p>
      <h2>Memory</h2>
      <p>Some layers are memory-bound.</p>
    </body></html>
    """
    sections = extract_html_sections(html)
    assert sections[0][0] == "Overview"
    assert sections[1][0] == "Memory"


def test_chunk_html_document_carries_source_metadata():
    source = DocumentSource(
        id="demo",
        title="Demo Source",
        url="https://example.com",
        doc_family="core",
        doc_type="html",
        crawl_prefix="https://example.com",
        product_tags=["gpu"],
    )
    html = (
        "<html><body><h1>Overview</h1><p>"
        "Mixed precision training improves throughput on Tensor Cores by using FP16 for "
        "compute operations and FP32 for weight accumulation, reducing memory footprint "
        "while maintaining numerical stability across the entire training run."
        "</p></body></html>"
    )
    chunks = chunk_html_document(source, source.url, html)
    assert chunks
    assert chunks[0].source_id == "demo"
    assert chunks[0].doc_family == "core"
    assert chunks[0].content_hash


def test_chunk_html_document_accepts_custom_chunk_limits():
    source = DocumentSource(
        id="demo",
        title="Demo Source",
        url="https://example.com",
        doc_family="core",
        doc_type="html",
        crawl_prefix="https://example.com",
    )
    # Content needs enough tokens per chunk (>= 25) after splitting, so use max_chars=300
    html = (
        "<html><body><h1>Overview</h1><p>"
        + ("Mixed precision training on Tensor Cores improves throughput by using FP16 for "
           "compute operations while FP32 handles weight accumulation and gradient updates. " * 4)
        + "</p></body></html>"
    )

    chunks = chunk_html_document(source, source.url, html, max_chars=300, overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 300 for chunk in chunks)


def test_build_chunk_id_is_stable_for_same_content():
    left_id, left_hash = build_chunk_id("cuda-guide", "Overview", "Tensor Cores improve throughput.")
    right_id, right_hash = build_chunk_id("cuda-guide", "Overview", "Tensor Cores improve throughput.")
    assert left_id == right_id
    assert left_hash == right_hash


def test_split_with_overlap_respects_max_chars():
    text = "0123456789 " * 120
    chunks = split_with_overlap(text, max_chars=120, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert chunks[0][-20:].strip()


# ---------------------------------------------------------------------------
# Navigation filter tests
# ---------------------------------------------------------------------------

def test_is_navigation_chunk_filters_short_text():
    """Text with fewer than 25 tokens should be classified as navigation noise."""
    short_text = "Overview GPU training"
    assert _is_navigation_chunk(short_text) is True


def test_is_navigation_chunk_filters_nav_patterns():
    """Text with 3+ nav pattern signals should be classified as navigation noise."""
    nav_text = (
        "Table of Contents | Privacy Policy | Terms of Service | "
        "NVIDIA.com Home | About NVIDIA | Contact Us | Developer Home | "
        "Corporate Info | Copyright © 2024 NVIDIA Corporation. All rights reserved."
    )
    assert _is_navigation_chunk(nav_text) is True


def test_is_navigation_chunk_keeps_real_content():
    """A real technical paragraph should NOT be classified as navigation noise."""
    technical_text = (
        "NCCL implements multi-GPU and multi-node collective communication primitives "
        "that are performance-optimized for NVIDIA GPUs and Networking. It is designed "
        "to fit in the MPI communication model, with both collective communication and "
        "point-to-point send/receive primitives. NCCL provides the following collective "
        "communication primitives: AllReduce, Broadcast, Reduce, AllGather, and ReduceScatter."
    )
    assert _is_navigation_chunk(technical_text) is False


# ---------------------------------------------------------------------------
# PDF chunking tests
# ---------------------------------------------------------------------------

def test_chunk_pdf_document_produces_valid_records():
    """PDF text input should produce ChunkRecord objects with correct metadata."""
    source = DocumentSource(
        id="h100-spec",
        title="H100 Datasheet",
        url="https://nvidia.com/h100.pdf",
        doc_family="hardware",
        doc_type="pdf",
        crawl_prefix="https://nvidia.com",
        product_tags=["h100"],
    )
    pdf_text = (
        "The NVIDIA H100 Tensor Core GPU delivers unprecedented acceleration for AI and HPC workloads. "
        "Built on the NVIDIA Hopper architecture, H100 provides up to 9x faster AI training and up to "
        "30x faster AI inference compared to the prior generation A100. The H100 features 80GB HBM3 "
        "memory with 3.35 TB/s memory bandwidth, and 700 GB/s NVLink 4.0 bandwidth per GPU. "
        "The H100 SXM5 variant delivers 3958 TFLOPS of FP16 performance with sparsity. "
        "NVSwitch 3.0 enables all-to-all GPU communication at 900 GB/s per GPU in the DGX H100 system."
    )
    chunks = chunk_pdf_document(source, source.url, pdf_text)

    assert chunks
    assert all(chunk.source_id == "h100-spec" for chunk in chunks)
    assert all(chunk.doc_family == "hardware" for chunk in chunks)
    assert all(chunk.doc_type == "pdf" for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)
    assert all(chunk.content.strip() for chunk in chunks)


# ---------------------------------------------------------------------------
# Overlap correctness test
# ---------------------------------------------------------------------------

def test_split_overlap_correctness():
    """The tail of chunk[0] should appear at the start of chunk[1]."""
    text = "word " * 300  # ~1500 chars
    overlap = 50
    chunks = split_with_overlap(text, max_chars=200, overlap=overlap)

    assert len(chunks) >= 2
    tail = chunks[0][-overlap:].strip()
    head = chunks[1][:overlap].strip()
    # The overlapping region should share tokens
    tail_tokens = set(tail.split())
    head_tokens = set(head.split())
    assert tail_tokens & head_tokens, f"No overlap found between tail={tail!r} and head={head!r}"


# ---------------------------------------------------------------------------
# Markdown chunking tests
# ---------------------------------------------------------------------------

def test_extract_markdown_sections_splits_on_headings():
    """Markdown text should be split into (heading, content) tuples by ## headings."""
    md = """# Overview

This is the overview section with enough content to pass filters.

## Architecture

The system architecture uses a client-server model with distributed components.

## Performance

Performance tuning involves adjusting batch sizes and learning rates.
"""
    sections = extract_markdown_sections(md)
    assert len(sections) == 3
    assert sections[0][0] == "Overview"
    assert "overview section" in sections[0][1]
    assert sections[1][0] == "Architecture"
    assert "client-server" in sections[1][1]
    assert sections[2][0] == "Performance"


def test_extract_markdown_sections_handles_no_headings():
    """Markdown without headings should produce a single 'Overview' section."""
    md = "This is plain text content without any headings at all."
    sections = extract_markdown_sections(md)
    assert len(sections) == 1
    assert sections[0][0] == "Overview"
    assert "plain text" in sections[0][1]


def test_chunk_markdown_document_produces_valid_records():
    """Markdown input should produce ChunkRecord objects with correct metadata."""
    source = DocumentSource(
        id="test-md",
        title="Test Markdown Source",
        url="https://example.com/docs",
        doc_family="infrastructure",
        doc_type="markdown",
        crawl_prefix="https://example.com",
        product_tags=["test"],
    )
    md_text = (
        "# Slurm Workload Manager\n\n"
        "Slurm is an open-source cluster management and job scheduling system "
        "for Linux clusters. It provides a framework for starting, executing, "
        "and monitoring work on a set of allocated nodes. Slurm manages a queue "
        "of pending jobs and allocates resources based on configured policies "
        "including fair-share scheduling, backfill, and preemption rules.\n\n"
        "## GPU Scheduling\n\n"
        "Slurm supports GPU scheduling through the Generic Resources (GRES) "
        "framework. Users request GPUs via the --gres flag, for example "
        "--gres=gpu:a100:4 requests four A100 GPUs. The GRES configuration "
        "is defined in gres.conf on each compute node, specifying the available "
        "GPU types, counts, and device paths for proper resource isolation."
    )
    chunks = chunk_markdown_document(source, source.url, md_text)

    assert chunks
    assert all(chunk.source_id == "test-md" for chunk in chunks)
    assert all(chunk.doc_family == "infrastructure" for chunk in chunks)
    assert all(chunk.doc_type == "markdown" for chunk in chunks)
    assert all(chunk.content.strip() for chunk in chunks)


def test_chunk_markdown_document_skips_nav_chunks():
    """Short or navigation-like markdown content should be filtered out."""
    source = DocumentSource(
        id="nav-md",
        title="Nav Test",
        url="https://example.com",
        doc_family="core",
        doc_type="markdown",
        crawl_prefix="https://example.com",
    )
    # Heading with very short content that should be filtered
    md_text = (
        "# Title\n\n"
        "Short.\n\n"
        "## Real Section\n\n"
        "NCCL implements multi-GPU and multi-node collective communication primitives "
        "that are performance-optimized for NVIDIA GPUs and Networking. It is designed "
        "to fit in the MPI communication model, with both collective communication and "
        "point-to-point send/receive primitives."
    )
    chunks = chunk_markdown_document(source, source.url, md_text)
    # The short "Short." section should be filtered, only real content remains
    assert all(len(chunk.content.split()) >= 25 for chunk in chunks)
