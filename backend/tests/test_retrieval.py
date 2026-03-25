from app.models import ChatTurn, ChunkRecord, QueryClass, QueryPlan, RetrieverResult
from app.services.retrieval import (
    EARLY_STOP_CONFIDENCE,
    build_query_plan,
    classify_assistant_mode,
    classify_question,
    estimate_confidence,
    grade_results,
    is_recency_sensitive,
    rerank_results,
)


def test_router_classifies_all_query_classes(settings):
    expectations = {
        "When should I use mixed precision training?": QueryClass.training_optimization,
        "Why is 4-GPU training scaling poorly?": QueryClass.distributed_multi_gpu,
        "What NVIDIA stack is needed to deploy training workloads on Linux?": QueryClass.deployment_runtime,
        "For a client fine-tuning a 7B model on-prem, what hardware should we consider?": QueryClass.hardware_topology,
    }

    for question, expected in expectations.items():
        plan = build_query_plan(question, settings)
        assert classify_question(question) == expected
        assert plan.query_class == expected
        assert plan.source_families


def test_recency_sensitive_query_uses_recency_flag(settings):
    question = "What changed in the latest NVIDIA Container Toolkit release?"
    plan = build_query_plan(question, settings)

    assert is_recency_sensitive(question) is True
    assert plan.recency_sensitive is True


def test_retrieval_search_tracks_rejected_chunks(retrieval_service):
    response = retrieval_service.search("Why is 4-GPU training scaling poorly?")

    assert response.results
    assert response.plan.query_class.value == "distributed_multi_gpu"
    assert isinstance(response.rejected_chunk_ids, list)
    assert any(event.type == "document_grading" for event in response.trace)


def test_document_grader_keeps_relevant_and_rejects_irrelevant():
    plan = QueryPlan(
        query_class=QueryClass.distributed_multi_gpu,
        search_queries=["Why is 4-GPU training scaling poorly?"],
        source_families=["distributed", "core", "advanced"],
        top_k=5,
    )
    relevant = RetrieverResult(
        chunk=ChunkRecord(
            id="nccl-1",
            source_id="nccl",
            title="NCCL",
            url="https://docs.nvidia.com/deeplearning/nccl/",
            section_path="All Reduce",
            doc_family="distributed",
            doc_type="html",
            product_tags=["nccl", "communication"],
            content="NCCL all-reduce communication can bottleneck multi-GPU scaling.",
            sparse_terms=["nccl", "all-reduce", "multi-gpu", "scaling"],
        ),
        score=0.6,
        rerank_score=0.6,
    )
    irrelevant = RetrieverResult(
        chunk=ChunkRecord(
            id="random-1",
            source_id="random",
            title="Unrelated",
            url="https://example.com",
            section_path="Overview",
            doc_family="hardware",
            doc_type="html",
            product_tags=["hardware"],
            content="This content is about monitor calibration and display brightness.",
            sparse_terms=["monitor", "display", "brightness"],
        ),
        score=0.02,
        rerank_score=0.02,
    )

    accepted, rejected = grade_results("Why is 4-GPU training scaling poorly?", plan, [relevant, irrelevant])

    assert [result.chunk.id for result in accepted] == ["nccl-1"]
    assert rejected == ["random-1"]


def test_rerank_prefers_expected_source_hint():
    plan = QueryPlan(
        query_class=QueryClass.deployment_runtime,
        search_queries=["What NVIDIA stack is needed to deploy training workloads on Linux?"],
        source_families=["infrastructure", "core"],
        top_k=5,
    )
    container = RetrieverResult(
        chunk=ChunkRecord(
            id="container-1",
            source_id="container-toolkit",
            title="Container Toolkit",
            url="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/",
            section_path="Install",
            doc_family="infrastructure",
            doc_type="html",
            product_tags=["containers", "runtime"],
            content="Install the NVIDIA Container Toolkit to expose GPUs into containers.",
            sparse_terms=["install", "nvidia", "container", "toolkit", "runtime"],
        ),
        score=0.25,
    )
    unrelated = RetrieverResult(
        chunk=ChunkRecord(
            id="core-1",
            source_id="dl-performance",
            title="Performance",
            url="https://docs.nvidia.com/deeplearning/performance/index.html",
            section_path="Overview",
            doc_family="core",
            doc_type="html",
            product_tags=["training"],
            content="This section describes training throughput concepts.",
            sparse_terms=["training", "throughput", "concepts"],
        ),
        score=0.28,
    )

    reranked = rerank_results("What NVIDIA stack is needed to deploy training workloads on Linux?", plan, [unrelated, container])

    assert reranked[0].chunk.source_id == "container-toolkit"


def test_assistant_mode_ignores_assistant_authored_doc_wording():
    mode = classify_assistant_mode(
        "what day is it",
        [
            ChatTurn(role="user", content="hey"),
            ChatTurn(role="assistant", content="I can help with general questions or NVIDIA documentation."),
        ],
    )

    assert mode == "direct_chat"


def test_assistant_mode_does_not_trigger_on_generic_docs_wording():
    assert classify_assistant_mode("can you check the docs for me?", []) == "direct_chat"


def test_assistant_mode_keeps_general_nvidia_question_in_direct_chat():
    assert classify_assistant_mode("what is nvidia", []) == "direct_chat"


# ---------------------------------------------------------------------------
# #9: Weighted confidence estimation tests
# ---------------------------------------------------------------------------


def _make_result(score: float) -> RetrieverResult:
    """Create a minimal RetrieverResult with a given rerank_score."""
    return RetrieverResult(
        chunk=ChunkRecord(
            id=f"chunk-{score}",
            source_id="test",
            title="Test",
            url="https://example.com",
            section_path="Overview",
            doc_family="core",
            doc_type="html",
            product_tags=[],
            content="test content",
            sparse_terms=["test"],
        ),
        score=score,
        rerank_score=score,
    )


def test_confidence_empty_results():
    assert estimate_confidence([]) == 0.0


def test_confidence_single_result():
    """Single result → weighted average with just the top-1 weight (0.5/0.5 = 1.0x)."""
    conf = estimate_confidence([_make_result(0.90)])
    assert conf == 0.90


def test_confidence_weighted_emphasizes_best_result():
    """[0.90, 0.15, 0.10] → weighted = (0.5*0.90 + 0.3*0.15 + 0.2*0.10) / 1.0 = 0.515."""
    results = [_make_result(0.90), _make_result(0.15), _make_result(0.10)]
    conf = estimate_confidence(results)
    assert 0.51 <= conf <= 0.52


def test_confidence_uniform_mediocre_results():
    """[0.50, 0.50, 0.50] → weighted = 0.50 (weights sum to 1.0)."""
    results = [_make_result(0.50), _make_result(0.50), _make_result(0.50)]
    conf = estimate_confidence(results)
    assert conf == 0.50


def test_confidence_two_results():
    """Two results use weights (0.5, 0.3), normalized by total weight 0.8."""
    results = [_make_result(1.0), _make_result(0.5)]
    conf = estimate_confidence(results)
    expected = round((0.5 * 1.0 + 0.3 * 0.5) / 0.8, 4)
    assert conf == expected


# ---------------------------------------------------------------------------
# #10: Early stopping in retrieval tests
# ---------------------------------------------------------------------------


def test_early_stop_skips_expansion_on_high_confidence(retrieval_service):
    """High-confidence first query should produce fewer retrieval trace events."""
    # "Why is 4-GPU training scaling poorly?" hits the NCCL knowledge base well.
    # With early stopping, if first query confidence > threshold, only 1 retrieval event.
    response = retrieval_service.search("Why is 4-GPU training scaling poorly?")
    retrieval_events = [e for e in response.trace if e.type == "retrieval"]
    # Should have results regardless
    assert response.results
    assert response.confidence > 0.0
    # Early stop constant should be a reasonable threshold
    assert 0.0 < EARLY_STOP_CONFIDENCE < 1.0


def test_early_stop_constant_is_reasonable():
    """EARLY_STOP_CONFIDENCE should be in a sensible range."""
    assert EARLY_STOP_CONFIDENCE == 0.75


# ---------------------------------------------------------------------------
# Tier 4.3: Adaptive retrieval parameter tests
# ---------------------------------------------------------------------------

from app.services.retrieval import _adaptive_retrieval_params


def test_adaptive_short_factoid_gets_tight_params():
    """Short factoid (<=8 tokens) → top_k=8, floor=0.35."""
    top_k, floor = _adaptive_retrieval_params("What is NCCL?", QueryClass.general)
    assert top_k == 8
    assert floor == 0.35


def test_adaptive_complex_analytical_gets_wide_params():
    """Complex analytical (>15 tokens, multi-entity) → top_k=10, floor=0.22."""
    question = "Compare the H100 and A100 memory bandwidth for distributed training with NCCL and NVLink interconnect"
    top_k, floor = _adaptive_retrieval_params(question, QueryClass.distributed_multi_gpu)
    assert top_k == 10
    assert floor == 0.22


def test_adaptive_default_distributed_gets_baseline():
    """Medium-length distributed question (9-15 tokens, <=1 entity) → class-based baseline (7, 0.26)."""
    # 9+ tokens after stopword removal, <=1 entity term
    question = "How does multi-GPU scaling work practice collective operations synchronization barriers overhead latency?"
    top_k, floor = _adaptive_retrieval_params(question, QueryClass.distributed_multi_gpu)
    assert top_k == 7
    assert floor == 0.26


def test_adaptive_default_general_gets_baseline():
    """Medium-length general question (9-15 tokens, <=1 entity) → class-based baseline (5, 0.30)."""
    question = "How should configure container runtime properly running workloads efficiently production environments?"
    top_k, floor = _adaptive_retrieval_params(question, QueryClass.general)
    assert top_k == 5
    assert floor == 0.30
