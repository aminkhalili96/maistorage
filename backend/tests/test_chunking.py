from app.models import DocumentSource
from app.services.chunking import build_chunk_id, chunk_html_document, extract_html_sections, split_with_overlap


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
    html = "<html><body><h1>Overview</h1><p>Mixed precision improves throughput.</p></body></html>"
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
    html = "<html><body><h1>Overview</h1><p>" + ("Mixed precision improves throughput. " * 12) + "</p></body></html>"

    chunks = chunk_html_document(source, source.url, html, max_chars=60, overlap=12)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 60 for chunk in chunks)


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
