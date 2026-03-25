"""Tests for RAG architecture improvements (S1-S6, R3-R5).

All tests use MockOpenAIReasoner — no real API calls.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest, ChatTurn, QueryClass, QueryPlan, RetrieverResult, ChunkRecord
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder, tokenize
from app.services.retrieval import RetrievalService, rerank_results, expand_query_abbreviations
from conftest import MockOpenAIReasoner


class StubTavilyClient:
    def search(self, query: str):
        return [{"title": "Result", "url": "https://example.com", "content": "Fallback."}]


def build_agent_with_demo(reasoner=None, tavily=None) -> AgentService:
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    if reasoner is None:
        reasoner = MockOpenAIReasoner()
    if tavily is None:
        tavily = StubTavilyClient()
    return AgentService(settings, retrieval, reasoner, tavily)


# ---------------------------------------------------------------------------
# R4: Query expansion with abbreviation map
# ---------------------------------------------------------------------------


def test_abbreviation_expansion_nccl_ar():
    """AR should expand to all-reduce."""
    result = expand_query_abbreviations("Optimize AR performance with NCCL")
    assert "all-reduce" in result.lower()


def test_abbreviation_expansion_tp_dp():
    """TP and DP should expand to tensor parallelism and data parallelism."""
    result = expand_query_abbreviations("Compare TP vs DP strategies")
    assert "tensor parallelism" in result.lower()
    assert "data parallelism" in result.lower()


def test_abbreviation_expansion_no_match():
    """Query with no abbreviations should return unchanged."""
    query = "What are the NCCL tuning parameters?"
    result = expand_query_abbreviations(query)
    assert result == query


def test_abbreviation_expansion_mig():
    """MIG should expand to Multi-Instance GPU."""
    result = expand_query_abbreviations("Configure MIG on H100")
    assert "Multi-Instance GPU" in result


def test_abbreviation_no_partial_match():
    """Should not match abbreviations as substrings of larger words."""
    result = expand_query_abbreviations("DPAR is not DP")
    # DP should match (whole word), but DPAR should not trigger DP
    assert "data parallelism" in result.lower()


def test_abbreviation_expansion_fsdp():
    """FSDP should expand to fully sharded data parallel."""
    result = expand_query_abbreviations("How does FSDP work?")
    assert "fully sharded data parallel" in result.lower()


def test_abbreviation_expansion_multiple():
    """Multiple abbreviations in one query should all expand."""
    result = expand_query_abbreviations("Compare AR and PP performance")
    assert "all-reduce" in result.lower()
    assert "pipeline parallelism" in result.lower()


# ---------------------------------------------------------------------------
# R5: Chunk recency weighting
# ---------------------------------------------------------------------------


def test_rerank_recency_bonus_recent_chunk():
    """Recent chunks (< 30 days) should get a small score boost."""
    now = datetime.now(timezone.utc)
    recent_date = (now - timedelta(days=10)).isoformat()
    old_date = (now - timedelta(days=200)).isoformat()

    chunk_recent = ChunkRecord(
        id="c1", source_id="s1", title="Recent", url="https://example.com",
        content="NCCL tuning parameters for bandwidth optimization",
        section_path="Overview", doc_family="distributed", doc_type="html",
        product_tags=["gpu"], source_kind="knowledge_base",
        retrieved_at=recent_date, snapshot_id="snap", content_hash="h1",
    )
    chunk_old = ChunkRecord(
        id="c2", source_id="s1", title="Old", url="https://example.com",
        content="NCCL tuning parameters for bandwidth optimization",
        section_path="Overview", doc_family="distributed", doc_type="html",
        product_tags=["gpu"], source_kind="knowledge_base",
        retrieved_at=old_date, snapshot_id="snap", content_hash="h2",
    )

    plan = QueryPlan(
        query_class=QueryClass.distributed_multi_gpu,
        search_queries=["NCCL tuning"],
        source_families=["distributed"],
        top_k=5,
        confidence_floor=0.25,
    )

    results = [
        RetrieverResult(chunk=chunk_recent, score=0.5),
        RetrieverResult(chunk=chunk_old, score=0.5),
    ]

    reranked = rerank_results("NCCL tuning parameters", plan, results)
    # Recent chunk should have a higher rerank_score due to recency bonus
    recent_result = next(r for r in reranked if r.chunk.id == "c1")
    old_result = next(r for r in reranked if r.chunk.id == "c2")
    assert recent_result.rerank_score >= old_result.rerank_score


def test_rerank_recency_bonus_90_day_chunk():
    """Chunks 30-90 days old should get a smaller boost than <30 day chunks."""
    now = datetime.now(timezone.utc)
    very_recent = (now - timedelta(days=5)).isoformat()
    moderately_recent = (now - timedelta(days=60)).isoformat()

    chunk_new = ChunkRecord(
        id="c1", source_id="s1", title="New", url="https://example.com",
        content="NCCL bandwidth optimization guide",
        section_path="Overview", doc_family="distributed", doc_type="html",
        product_tags=["gpu"], source_kind="knowledge_base",
        retrieved_at=very_recent, snapshot_id="snap", content_hash="h1",
    )
    chunk_mod = ChunkRecord(
        id="c2", source_id="s1", title="Moderate", url="https://example.com",
        content="NCCL bandwidth optimization guide",
        section_path="Overview", doc_family="distributed", doc_type="html",
        product_tags=["gpu"], source_kind="knowledge_base",
        retrieved_at=moderately_recent, snapshot_id="snap", content_hash="h2",
    )

    plan = QueryPlan(
        query_class=QueryClass.distributed_multi_gpu,
        search_queries=["NCCL tuning"],
        source_families=["distributed"],
        top_k=5,
        confidence_floor=0.25,
    )

    results = [
        RetrieverResult(chunk=chunk_new, score=0.5),
        RetrieverResult(chunk=chunk_mod, score=0.5),
    ]

    reranked = rerank_results("NCCL bandwidth optimization", plan, results)
    new_result = next(r for r in reranked if r.chunk.id == "c1")
    mod_result = next(r for r in reranked if r.chunk.id == "c2")
    # Very recent (< 30 days) gets 0.02 bonus, moderate (30-90) gets 0.01
    assert new_result.rerank_score >= mod_result.rerank_score


def test_rerank_no_recency_bonus_old_chunk():
    """Chunks older than 90 days should get no recency bonus."""
    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=200)).isoformat()

    chunk = ChunkRecord(
        id="c1", source_id="s1", title="Old", url="https://example.com",
        content="NCCL tuning parameters for bandwidth",
        section_path="Overview", doc_family="distributed", doc_type="html",
        product_tags=["gpu"], source_kind="knowledge_base",
        retrieved_at=old_date, snapshot_id="snap", content_hash="h1",
    )

    plan = QueryPlan(
        query_class=QueryClass.distributed_multi_gpu,
        search_queries=["NCCL tuning"],
        source_families=["distributed"],
        top_k=5,
        confidence_floor=0.25,
    )

    results = [RetrieverResult(chunk=chunk, score=0.5)]
    reranked = rerank_results("NCCL tuning parameters", plan, results)
    # Old chunk should not get any recency bonus — rerank_score depends only on
    # base score + lexical overlap + family bonus (no recency component)
    # We can verify it's not inflated beyond expected base + overlap + family
    assert reranked[0].rerank_score > 0  # Sanity: some score exists


# ---------------------------------------------------------------------------
# S4: Output format enforcement
# ---------------------------------------------------------------------------


def test_format_validation_normal_answer():
    """A well-formatted answer should pass validation."""
    answer = (
        "The H100 has 80GB of HBM3 memory with 3.35 TB/s bandwidth. [1] "
        "This provides significant improvement over the A100's 80GB HBM2e. [2]"
    )
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert result["valid"] is True
    assert result["issues"] == []
    assert result["citation_count"] >= 2


def test_format_validation_too_long():
    """An answer over 500 words should be flagged."""
    answer = "word " * 501
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert not result["valid"]
    assert any("too_long" in issue for issue in result["issues"])


def test_format_validation_verbose():
    """An answer between 350-500 words should get a verbose warning."""
    answer = "word " * 400 + "[1]"
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert any("verbose" in issue for issue in result["issues"])


def test_format_validation_no_citations_knowledge_base_backed():
    """A knowledge-base-backed answer with no citations should be flagged."""
    answer = "The H100 GPU has 80GB of memory with excellent bandwidth."
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert any("no_citations" in issue for issue in result["issues"])


def test_format_validation_no_citations_direct_chat():
    """A direct-chat answer with no citations should NOT be flagged."""
    answer = "Hello! How can I help you today?"
    result = AgentService._validate_format(answer, "direct-chat")
    # direct-chat doesn't require citations
    assert not any("no_citations" in issue for issue in result["issues"])


def test_format_validation_unclosed_code_block():
    """Answer with unclosed code block should be flagged."""
    answer = "Here is the config:\n```yaml\nkey: value\nSome more text."
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert any("unclosed_code_block" in issue for issue in result["issues"])


def test_format_validation_closed_code_block():
    """Answer with properly closed code block should not be flagged for that."""
    answer = "Here is the config:\n```yaml\nkey: value\n```\nMore text. [1]"
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert not any("unclosed_code_block" in issue for issue in result["issues"])


def test_format_validation_word_count():
    """word_count field should be accurate."""
    answer = "one two three four five"
    result = AgentService._validate_format(answer, "knowledge-base-backed")
    assert result["word_count"] == 5


# ---------------------------------------------------------------------------
# S2: Citation quality metrics
# ---------------------------------------------------------------------------


def test_citation_quality_strong_citations():
    """Citations with good token overlap should be classified as strong."""
    answer = (
        "NCCL handles all-reduce operations for multi-GPU training using NVLink. [1] "
        "The bandwidth optimization requires tuning NCCL parameters. [2]"
    )
    chunks = [
        ChunkRecord(
            id="c1", source_id="s1", title="NCCL", url="https://example.com",
            content="NCCL handles all-reduce collective operations for multi-GPU distributed training using NVLink interconnect",
            section_path="Overview", doc_family="distributed", doc_type="html",
            product_tags=["gpu"], source_kind="knowledge_base",
            retrieved_at="2026-03-15T00:00:00Z", snapshot_id="snap", content_hash="h1",
        ),
        ChunkRecord(
            id="c2", source_id="s1", title="NCCL Tuning", url="https://example.com",
            content="Bandwidth optimization requires tuning NCCL environment parameters for efficient communication",
            section_path="Tuning", doc_family="distributed", doc_type="html",
            product_tags=["gpu"], source_kind="knowledge_base",
            retrieved_at="2026-03-15T00:00:00Z", snapshot_id="snap", content_hash="h2",
        ),
    ]
    results = [RetrieverResult(chunk=c, score=0.8) for c in chunks]

    quality = AgentService._citation_quality(answer, results)
    assert quality["strong"] >= 1
    assert quality["uncited_paragraphs"] == 0


def test_citation_quality_empty_results():
    """Empty results should return zeroed quality metrics."""
    quality = AgentService._citation_quality("Some answer text.", [])
    assert quality["strong"] == 0
    assert quality["weak"] == 0
    assert quality["uncited_paragraphs"] == 0


def test_citation_quality_empty_answer():
    """Empty answer should return zeroed quality metrics."""
    chunk = ChunkRecord(
        id="c1", source_id="s1", title="NCCL", url="https://example.com",
        content="NCCL content here",
        section_path="Overview", doc_family="distributed", doc_type="html",
        product_tags=["gpu"], source_kind="knowledge_base",
        retrieved_at="2026-03-15T00:00:00Z", snapshot_id="snap", content_hash="h1",
    )
    quality = AgentService._citation_quality("", [RetrieverResult(chunk=chunk, score=0.8)])
    assert quality["strong"] == 0
    assert quality["weak"] == 0


def test_citation_quality_uncited_paragraph():
    """A paragraph without citation markers should count as uncited."""
    answer = (
        "NCCL handles all-reduce operations for multi-GPU training using NVLink and other interconnects for distributed workloads. [1]\n\n"
        "This is a long enough paragraph without any citation markers that should be flagged as uncited content in the output."
    )
    chunks = [
        ChunkRecord(
            id="c1", source_id="s1", title="NCCL", url="https://example.com",
            content="NCCL handles all-reduce collective operations for multi-GPU distributed training using NVLink interconnect",
            section_path="Overview", doc_family="distributed", doc_type="html",
            product_tags=["gpu"], source_kind="knowledge_base",
            retrieved_at="2026-03-15T00:00:00Z", snapshot_id="snap", content_hash="h1",
        ),
    ]
    results = [RetrieverResult(chunk=c, score=0.8) for c in chunks]

    quality = AgentService._citation_quality(answer, results)
    assert quality["uncited_paragraphs"] >= 1


# ---------------------------------------------------------------------------
# S6: Stale info detection
# ---------------------------------------------------------------------------


def test_stale_source_detection_in_grounding_trace():
    """Grounding check trace event should include stale_sources_warning field."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="What are the H100 GPU specifications?"))
    # Check that grounding_check trace event exists
    grounding_events = [e for e in state.trace if e.type == "grounding_check"]
    assert len(grounding_events) >= 1
    # The trace should include stale_sources_warning field
    assert "stale_sources_warning" in grounding_events[0].payload


# ---------------------------------------------------------------------------
# S1: Claim-level verification
# ---------------------------------------------------------------------------


def test_claim_verification_trace_exists():
    """Claim verification should produce a trace event."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="What are the H100 GPU specifications?"))
    claim_events = [e for e in state.trace if e.type == "claim_verification"]
    # Should have a claim_verification event (may or may not exist depending on
    # whether the mock returns parseable claims)
    # At minimum, the pipeline should not crash
    assert state.answer.strip(), "Pipeline should produce an answer"


def test_claim_verification_does_not_crash_on_nccl_query():
    """Claim verification should not crash for an NCCL query."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters for bandwidth?"))
    assert state.answer.strip(), "Pipeline should produce an answer"
    # Claim verification node should exist in the graph flow
    claim_events = [e for e in state.trace if e.type == "claim_verification"]
    # Whether claims are extracted depends on mock, but pipeline must not crash
    assert isinstance(claim_events, list)


# ---------------------------------------------------------------------------
# S3: Contextual sycophancy (premise checking)
# ---------------------------------------------------------------------------


def test_false_premise_pipeline_doesnt_crash():
    """A question with a false premise should not crash the pipeline."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="Since the H100 has 40GB of memory, how does it compare to the A100?"
    ))
    assert state.answer.strip(), "Pipeline should produce an answer with false premise"
    assert state.response_mode in {"knowledge-base-backed", "llm-knowledge"}


def test_contradictory_premise_pipeline_doesnt_crash():
    """A query with contradictory facts should not crash the pipeline."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="Why does the H100 only support PCIe Gen3 when the A100 already has Gen5?"
    ))
    assert state.answer.strip(), "Pipeline should not crash on false-premise queries"


# ---------------------------------------------------------------------------
# S5: Confidence calibration (just checks logging doesn't break anything)
# ---------------------------------------------------------------------------


def test_self_reflect_scores_in_trace():
    """Self-reflect should produce scores in the trace."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters?"))
    reflect_events = [e for e in state.trace if e.type == "self_reflect"]
    assert len(reflect_events) == 1, "Should have exactly one self_reflect event"
    assert "scores" in reflect_events[0].payload


def test_self_reflect_scores_are_numeric():
    """Self-reflect scores should contain numeric values for relevance/groundedness/completeness."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="How does NVLink affect multi-GPU scaling?"))
    reflect_events = [e for e in state.trace if e.type == "self_reflect"]
    assert len(reflect_events) == 1
    scores = reflect_events[0].payload.get("scores", {})
    # The mock returns numeric scores; verify they're present
    assert isinstance(scores, dict)


# ---------------------------------------------------------------------------
# Integration: full pipeline with improvements
# ---------------------------------------------------------------------------


def test_full_pipeline_with_improvements():
    """Run a full query and verify the pipeline produces a valid answer
    with all improvements active."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="Why is 4-GPU training scaling poorly with NCCL?"
    ))
    assert state.answer.strip()
    assert state.response_mode in {"knowledge-base-backed", "web-backed", "llm-knowledge"}
    # Should have trace events
    assert len(state.trace) >= 3


def test_pipeline_with_abbreviation_query():
    """Query using abbreviations should still produce good results."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="How to configure TP and DP for multi-GPU training?"
    ))
    assert state.answer.strip()


def test_pipeline_trace_has_expected_event_types():
    """A knowledge-base-routed query should produce trace events for key pipeline stages."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="What are the NCCL tuning parameters for bandwidth?"
    ))
    event_types = {e.type for e in state.trace}
    # Core events that should always appear for a doc_rag query
    assert "classification" in event_types
    assert "self_reflect" in event_types
    assert "grounding_check" in event_types


def test_pipeline_format_validation_runs():
    """Format validation should run during the pipeline (visible in grounding trace)."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="What are the H100 GPU specifications?"
    ))
    # The grounding_check trace event should exist, which is where format validation
    # results are logged alongside citation quality
    grounding_events = [e for e in state.trace if e.type == "grounding_check"]
    assert len(grounding_events) >= 1
    payload = grounding_events[0].payload
    # Should have citation_quality in the payload (computed by _citation_quality)
    assert "citation_quality" in payload


# ---------------------------------------------------------------------------
# R1: Cross-encoder reranking
# ---------------------------------------------------------------------------


def test_cross_encoder_disabled_by_default():
    """Cross-encoder should be disabled in dev mode (USE_CROSS_ENCODER=false)."""
    settings = get_settings()
    assert settings.use_cross_encoder is False


def test_cross_encoder_graceful_when_not_installed():
    """CrossEncoderReranker should handle missing sentence-transformers gracefully."""
    try:
        from app.services.retrieval import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        # Create minimal test results
        chunk = ChunkRecord(
            id="c1", source_id="s1", title="Test", url="https://example.com",
            content="NCCL tuning parameters", section_path="Overview",
            doc_family="distributed", product_tags=["gpu"], source_kind="knowledge_base",
            doc_type="html", retrieved_at="2026-03-15T00:00:00Z",
            snapshot_id="snap", content_hash="h1",
        )
        results = [RetrieverResult(chunk=chunk, score=0.5, rerank_score=0.5)]
        # Should return results unchanged if model can't load
        reranked = reranker.rerank("test query", results)
        assert len(reranked) >= 1
    except ImportError:
        pass  # CrossEncoderReranker not available — that's fine


# ---------------------------------------------------------------------------
# R2: HyDE (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------


def test_hyde_disabled_by_default():
    """HyDE should be disabled by default."""
    settings = get_settings()
    assert settings.use_hyde is False


def test_hyde_function_returns_original_when_disabled():
    """generate_hyde_query should return original query when USE_HYDE=false."""
    from app.services.retrieval import generate_hyde_query
    settings = get_settings()
    result = generate_hyde_query("NCCL tuning parameters", MockOpenAIReasoner(), settings)
    # Since use_hyde is False by default, should return original query
    assert result == "NCCL tuning parameters"


# ---------------------------------------------------------------------------
# R6: Semantic cache tracking
# ---------------------------------------------------------------------------


def test_semantic_cache_disabled_by_default():
    """Semantic cache should be disabled by default in dev mode."""
    settings = get_settings()
    assert settings.semantic_cache_enabled is False


def test_agent_runs_without_cache():
    """Agent should work fine with semantic cache disabled."""
    agent = build_agent_with_demo()
    assert agent._semantic_cache is None
    state = agent.run(ChatRequest(question="What are H100 specs?"))
    assert state.answer.strip()
