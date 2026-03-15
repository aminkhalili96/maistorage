from app.models import ChunkRecord, QueryClass, QueryPlan, RetrieverResult
from app.services.retrieval import (
    build_query_plan,
    classify_question,
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
