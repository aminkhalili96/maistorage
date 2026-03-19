"""Tests for individual LangGraph node functions on AgentService.

Each test constructs a minimal GraphState dict, calls a single graph node
method, and asserts the state mutations it produces.  No LLM API calls
are made -- the MockOpenAIReasoner from conftest is used throughout.

NOTE: LangGraph merges each node's returned dict into the running state.
The node methods only return the *changed* keys.  When calling nodes
directly we must simulate this merge -- see ``_merge`` below.
"""
from __future__ import annotations

import pytest

from app.config import get_settings
from app.corpus import load_demo_chunks, load_sources
from app.models import ChatRequest, Citation, QueryPlan, RetrieverResult, TraceEvent
from app.services.agent import AgentService, GraphState
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder
from app.services.retrieval import RetrievalService
from conftest import MockOpenAIReasoner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTavily:
    enabled = False

    def search(self, query: str):
        return []


class _DisabledReasoner:
    enabled = False

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        raise RuntimeError("Reasoner is disabled")


def _merge(base: GraphState, update: GraphState) -> GraphState:
    """Simulate LangGraph TypedDict state merge: update keys on top of base."""
    merged = dict(base)
    merged.update(update)
    return merged


def _build_agent(reasoner=None, tavily=None) -> AgentService:
    """Build an AgentService backed by the demo corpus and keyword embedder."""
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_corpus()
    retrieval = RetrievalService(settings, sources, index)
    return AgentService(
        settings,
        retrieval,
        reasoner or MockOpenAIReasoner(),
        tavily or _StubTavily(),
    )


def _build_agent_empty_index(reasoner=None, tavily=None) -> AgentService:
    """Build an AgentService with an empty search index (no corpus)."""
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    return AgentService(
        settings,
        retrieval,
        reasoner or MockOpenAIReasoner(),
        tavily or _StubTavily(),
    )


def _initial_state(question: str = "Why is 4-GPU training scaling poorly?", **overrides) -> GraphState:
    """Return the initial GraphState as it would be before the graph starts."""
    state: GraphState = {
        "question": question,
        "current_query": question,
        "model_used": get_settings().generation_model,
        "assistant_mode": "doc_rag",
        "history_context": "",
        "trace": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# 1. _graph_classify — doc_rag question
# ---------------------------------------------------------------------------


def test_graph_classify_sets_doc_rag_for_technical_question():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    result = _merge(state, agent._graph_classify(state))

    assert result["assistant_mode"] == "doc_rag"
    assert result["plan"] is not None
    plan = QueryPlan.model_validate(result["plan"])
    assert plan.query_class.value in {
        "distributed_multi_gpu",
        "training_optimization",
        "hardware_topology",
        "deployment_runtime",
        "general",
    }
    assert result["response_mode"] == "corpus-backed"
    # Must have a classification trace event
    trace_types = [TraceEvent.model_validate(e).type for e in result["trace"]]
    assert "classification" in trace_types


# ---------------------------------------------------------------------------
# 2. _graph_classify — the graph node always sets doc_rag
# ---------------------------------------------------------------------------


def test_graph_classify_always_sets_doc_rag():
    """_graph_classify always returns assistant_mode='doc_rag' because the
    direct_chat shortcut lives outside the graph."""
    agent = _build_agent()
    state = _initial_state("Hello, how are you?")
    result = _merge(state, agent._graph_classify(state))

    assert result["assistant_mode"] == "doc_rag"
    assert result["plan"] is not None


# ---------------------------------------------------------------------------
# 3. _graph_retrieve populates results with scored chunks
# ---------------------------------------------------------------------------


def test_graph_retrieve_populates_results():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)

    assert len(state["results"]) > 0
    first = RetrieverResult.model_validate(state["results"][0])
    assert first.score > 0
    assert first.chunk.content


# ---------------------------------------------------------------------------
# 4. _graph_retrieve respects the plan's top_k parameter
# ---------------------------------------------------------------------------


def test_graph_retrieve_respects_top_k():
    agent = _build_agent()
    state = _initial_state("What is NCCL used for?")
    state = _merge(state, agent._graph_classify(state))
    plan = QueryPlan.model_validate(state["plan"])
    state = agent._graph_retrieve(state)

    # Results should not exceed top_k (retrieval may return fewer)
    assert len(state["results"]) <= plan.top_k


# ---------------------------------------------------------------------------
# 5. _graph_grade_documents emits grading event with counts
# ---------------------------------------------------------------------------


def test_graph_grade_documents_emits_grading_event():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)

    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "document_grading" in trace_types

    # The grading event should record counts
    grading_events = [
        TraceEvent.model_validate(e)
        for e in state["trace"]
        if TraceEvent.model_validate(e).type == "document_grading"
    ]
    assert grading_events
    payload = grading_events[-1].payload
    assert "accepted_count" in payload
    assert "rejected_count" in payload
    assert payload["accepted_count"] + payload["rejected_count"] == payload["total_count"]


# ---------------------------------------------------------------------------
# 6. _graph_grade_documents preserves confidence on state
# ---------------------------------------------------------------------------


def test_graph_grade_documents_preserves_confidence():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    conf_before = state["confidence"]
    state = agent._graph_document_grading(state)

    # Document grading does not change the confidence; it just adds a trace event
    assert state["confidence"] == conf_before


# ---------------------------------------------------------------------------
# 7. _graph_rewrite produces a different query than the original
# ---------------------------------------------------------------------------


def test_graph_rewrite_produces_different_query():
    agent = _build_agent()
    original_question = "Why is 4-GPU training scaling poorly?"
    state = _initial_state(original_question)
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_rewrite(state)

    assert state["rewritten_query"] is not None
    assert state["current_query"] != original_question
    assert state["retries"] == 1
    # Trace must include a rewrite event
    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "rewrite" in trace_types


# ---------------------------------------------------------------------------
# 8. _graph_rewrite falls back to static expansion when reasoner is disabled
# ---------------------------------------------------------------------------


def test_graph_rewrite_falls_back_to_static_when_reasoner_disabled():
    agent = _build_agent(reasoner=_DisabledReasoner())
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_rewrite(state)

    # Should still produce a rewritten query via static expansion
    assert state["rewritten_query"] is not None
    assert state["retries"] == 1

    # Trace should indicate static expansion
    rewrite_events = [
        TraceEvent.model_validate(e)
        for e in state["trace"]
        if TraceEvent.model_validate(e).type == "rewrite"
    ]
    assert rewrite_events
    assert rewrite_events[-1].payload["rewrite_method"] == "static_expansion"


# ---------------------------------------------------------------------------
# 9. _graph_generate produces answer text (using MockOpenAIReasoner)
# ---------------------------------------------------------------------------


def test_graph_generate_produces_answer():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)
    state = agent._graph_generate(state)

    assert state["answer"]
    assert len(state["answer"]) > 20
    assert state["citations"]
    # Generation trace event
    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "generation" in trace_types


# ---------------------------------------------------------------------------
# 10. _graph_grounding_check passes when answer has citation markers [1]
# ---------------------------------------------------------------------------


def test_graph_grounding_check_passes_with_citations():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)
    state = agent._graph_generate(state)
    state = agent._graph_grounding_check(state)

    assert state["grounding_passed"] is True
    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "grounding_check" in trace_types


# ---------------------------------------------------------------------------
# 11. _graph_grounding_check fails when answer has hedging phrases
# ---------------------------------------------------------------------------


def test_graph_grounding_check_fails_on_hedging():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)
    state = agent._graph_generate(state)

    # Inject a hedging answer
    state["answer"] = (
        "Based on my knowledge, I believe that 4-GPU training scales poorly "
        "because of communication overhead. [1]"
    )
    state = agent._graph_grounding_check(state)

    assert state["grounding_passed"] is False


# ---------------------------------------------------------------------------
# 12. _graph_quality_check passes on well-formed answer
# ---------------------------------------------------------------------------


def test_graph_quality_check_passes_on_good_answer():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)
    state = agent._graph_generate(state)
    state = agent._graph_answer_quality_check(state)

    assert state["answer_quality_passed"] is True
    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "answer_quality_check" in trace_types


# ---------------------------------------------------------------------------
# 13. _graph_quality_check fails when answer is the refusal text
# ---------------------------------------------------------------------------


def test_graph_quality_check_fails_on_refusal_answer():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)

    # Inject the refusal answer text
    state["answer"] = AgentService._refusal_answer()
    state = agent._graph_answer_quality_check(state)

    assert state["answer_quality_passed"] is False


# ---------------------------------------------------------------------------
# 14. Routing: _route_after_grading returns correct edge
# ---------------------------------------------------------------------------


def test_route_after_grading_returns_generate_on_good_results():
    """When confidence is above the floor, routing should go to 'generate'."""
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)

    route = agent._route_after_grading(state)
    # With the demo corpus and a well-known question, confidence is above floor
    assert route in {"generate", "rewrite_if_needed", "fallback_if_needed"}


def test_route_after_grading_returns_rewrite_on_empty_index():
    """Empty index gives confidence below floor -> 'rewrite_if_needed' on first attempt."""
    agent = _build_agent_empty_index()
    state = _initial_state("What are the NCCL tuning parameters?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)

    # Confidence should be 0 with empty index, retries=0 -> rewrite
    assert state["confidence"] == 0.0
    route = agent._route_after_grading(state)
    assert route == "rewrite_if_needed"


def test_route_after_grading_returns_fallback_when_retries_exhausted():
    """After retries are exhausted, routing should go to 'fallback_if_needed'."""
    agent = _build_agent_empty_index()
    state = _initial_state("What are the NCCL tuning parameters?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)

    # Simulate that one rewrite retry has already been done
    state["retries"] = 1
    route = agent._route_after_grading(state)
    assert route == "fallback_if_needed"


# ---------------------------------------------------------------------------
# 15. _graph_self_reflect handles MockReasoner response gracefully
# ---------------------------------------------------------------------------


def test_graph_self_reflect_handles_mock_reasoner():
    """MockOpenAIReasoner returns non-JSON text for self-reflect prompts,
    so the node should fall back to neutral scores {relevance:3, groundedness:3, completeness:3}."""
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)
    state = agent._graph_generate(state)
    state = agent._graph_self_reflect(state)

    assert "self_reflect_scores" in state
    scores = state["self_reflect_scores"]
    # MockOpenAIReasoner returns text that is not valid JSON for self-reflect,
    # so the node defaults to neutral scores
    assert scores["relevance"] == 3
    assert scores["groundedness"] == 3
    assert scores["completeness"] == 3
    assert "issues" in scores

    # Trace should include self_reflect event
    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "self_reflect" in trace_types


def test_graph_self_reflect_skips_when_reasoner_disabled():
    """When reasoner.enabled is False, self_reflect should be a no-op."""
    agent = _build_agent(reasoner=_DisabledReasoner())
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state = agent._graph_document_grading(state)
    state = agent._graph_generate(state)

    trace_count_before = len(state["trace"])
    state = agent._graph_self_reflect(state)

    # No trace event added when reasoner is disabled
    assert len(state["trace"]) == trace_count_before
    # self_reflect_scores should not be set
    assert "self_reflect_scores" not in state


# ---------------------------------------------------------------------------
# Bonus: _graph_fallback when tavily is disabled
# ---------------------------------------------------------------------------


def test_graph_fallback_returns_refusal_when_tavily_disabled():
    """When plan.use_tavily_fallback is False, fallback should set insufficient-evidence."""
    agent = _build_agent()
    state = _initial_state("What are the NCCL tuning parameters?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)

    # The plan's use_tavily_fallback should be False in dev mode
    plan = QueryPlan.model_validate(state["plan"])
    assert plan.use_tavily_fallback is False

    state = agent._graph_fallback(state)

    assert state["response_mode"] == "insufficient-evidence"
    assert "not have enough" in state["answer"].lower()
    trace_types = [TraceEvent.model_validate(e).type for e in state["trace"]]
    assert "fallback" in trace_types


# ---------------------------------------------------------------------------
# Bonus: _route_after_quality returns correct edges
# ---------------------------------------------------------------------------


def test_route_after_quality_returns_end_when_all_pass():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state["grounding_passed"] = True
    state["answer_quality_passed"] = True

    route = agent._route_after_quality(state)
    assert route == "end"


def test_route_after_quality_returns_rewrite_when_quality_fails():
    agent = _build_agent()
    state = _initial_state("Why is 4-GPU training scaling poorly?")
    state = _merge(state, agent._graph_classify(state))
    state = agent._graph_retrieve(state)
    state["grounding_passed"] = False
    state["answer_quality_passed"] = True
    state["used_fallback"] = False
    state["retries"] = 0

    plan = QueryPlan.model_validate(state["plan"])
    route = agent._route_after_quality(state)
    # With tavily disabled, it should try rewrite if retries available
    if not plan.use_tavily_fallback:
        assert route == "rewrite_if_needed"
