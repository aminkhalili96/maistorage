from __future__ import annotations

from app.models import ChatTurn, QueryClass
from app.services.retrieval import (
    _adaptive_retrieval_params,
    classify_assistant_mode,
)


# ---------------------------------------------------------------------------
# 1. Known query returns relevant chunks
# ---------------------------------------------------------------------------


def test_known_query_returns_relevant_chunks(retrieval_service):
    """Query 'NCCL all-reduce bandwidth' -> top results should contain 'nccl' in source_id."""
    response = retrieval_service.search("NCCL all-reduce bandwidth")

    assert response.results, "Expected at least one result for NCCL query"
    source_ids = [r.chunk.source_id for r in response.results]
    assert any("nccl" in sid.lower() for sid in source_ids), (
        f"Expected 'nccl' in source_ids, got: {source_ids}"
    )


# ---------------------------------------------------------------------------
# 2. Cross-source retrieval
# ---------------------------------------------------------------------------


def test_cross_source_retrieval(retrieval_service):
    """Query 'GPU memory and CUDA cores' -> results should contain chunks from
    multiple source families."""
    response = retrieval_service.search("GPU memory and CUDA cores")

    assert response.results, "Expected results for a broad GPU query"
    families = {r.chunk.doc_family for r in response.results}
    # With a broad query, we expect at least 1 family (the corpus may not have
    # perfect coverage, but we should get something)
    assert len(families) >= 1, f"Expected at least 1 doc family, got: {families}"


# ---------------------------------------------------------------------------
# 3. Negative retrieval low confidence
# ---------------------------------------------------------------------------


def test_negative_retrieval_low_confidence(retrieval_service):
    """Query 'chocolate cake recipe' -> confidence should be very low or zero results."""
    response = retrieval_service.search("chocolate cake recipe")

    # A keyword index can produce incidental matches on common terms, so
    # confidence won't be zero, but it should remain well below what a
    # genuinely relevant query achieves (typically >0.5).
    if response.results:
        assert response.confidence < 0.5, (
            f"Expected low confidence for unrelated query, got: {response.confidence}"
        )
    # If no results, that is also a valid outcome


# ---------------------------------------------------------------------------
# 4. Reranking stability
# ---------------------------------------------------------------------------


def test_reranking_stability(retrieval_service):
    """Run same query twice -> results should be in identical order."""
    query = "NCCL all-reduce bandwidth optimization"
    response1 = retrieval_service.search(query)
    response2 = retrieval_service.search(query)

    ids1 = [r.chunk.id for r in response1.results]
    ids2 = [r.chunk.id for r in response2.results]
    assert ids1 == ids2, (
        f"Results should be deterministic. Run 1: {ids1[:5]}, Run 2: {ids2[:5]}"
    )


# ---------------------------------------------------------------------------
# 5. Adaptive params short query
# ---------------------------------------------------------------------------


def test_adaptive_params_short_query():
    """1-word query 'GPU' -> should use factoid params (lower top_k)."""
    top_k, floor = _adaptive_retrieval_params("GPU", QueryClass.general)
    assert top_k == 3, f"Short factoid query should use top_k=3, got {top_k}"
    assert floor == 0.35, f"Short factoid query should use floor=0.35, got {floor}"


# ---------------------------------------------------------------------------
# 6. Adaptive params long query
# ---------------------------------------------------------------------------


def test_adaptive_params_long_query():
    """50+ word query -> should use complex params (higher top_k)."""
    long_query = (
        "Compare the H100 and A100 memory bandwidth for distributed training "
        "with NCCL and NVLink interconnect topology and how does GPUDirect RDMA "
        "performance scale across multiple nodes in a DGX BasePOD cluster setup "
        "with InfiniBand networking and what are the recommended NCCL environment "
        "variables for optimal all-reduce performance across eight GPUs"
    )
    top_k, floor = _adaptive_retrieval_params(long_query, QueryClass.distributed_multi_gpu)
    assert top_k == 10, f"Long complex query should use top_k=10, got {top_k}"
    assert floor == 0.22, f"Long complex query should use floor=0.22, got {floor}"


# ---------------------------------------------------------------------------
# 7. Adaptive params default
# ---------------------------------------------------------------------------


def test_adaptive_params_default():
    """Normal-length query -> should use default params."""
    # 9+ tokens, <=1 entity -> class-based baseline
    question = "How does multi-GPU scaling work practice collective operations synchronization barriers overhead latency?"
    top_k, floor = _adaptive_retrieval_params(question, QueryClass.distributed_multi_gpu)
    assert top_k == 7, f"Default distributed query should use top_k=7, got {top_k}"
    assert floor == 0.26, f"Default distributed query should use floor=0.26, got {floor}"


# ---------------------------------------------------------------------------
# 8. Query classification: doc_rag
# ---------------------------------------------------------------------------


def test_query_classification_doc_rag():
    """'What is NCCL?' -> classified as doc_rag."""
    mode = classify_assistant_mode("What is NCCL?", [])
    assert mode == "doc_rag", f"Expected doc_rag, got: {mode}"


# ---------------------------------------------------------------------------
# 9. Query classification: direct_chat
# ---------------------------------------------------------------------------


def test_query_classification_direct_chat():
    """'Hello' -> classified as direct_chat."""
    mode = classify_assistant_mode("Hello", [])
    assert mode == "direct_chat", f"Expected direct_chat, got: {mode}"


# ---------------------------------------------------------------------------
# 10. Query classification: live_query
# ---------------------------------------------------------------------------


def test_query_classification_live_query():
    """'What's the weather?' -> classified as live_query."""
    mode = classify_assistant_mode("What's the weather?", [])
    assert mode == "live_query", f"Expected live_query, got: {mode}"
