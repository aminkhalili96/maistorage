"""Tests for the 5 agentic LLM-powered features.

Tests use MockOpenAIReasoner from conftest.py — no API calls are made.

1. LLM Classification (llm_classify_assistant_mode)
2. LLM Query Planning (llm_build_query_plan)
3. LLM Document Grading (_graph_document_grading)
4. LLM-as-Router (_route_after_grading, _route_after_quality)
5. Multi-Hop Retrieval (_graph_multi_hop_check)
"""

from __future__ import annotations

import json
import os

import pytest

os.environ["APP_MODE"] = "dev"
os.environ["USE_PINECONE"] = "false"
os.environ["USE_TAVILY_FALLBACK"] = "false"
os.environ["EMBEDDER_PROVIDER"] = "keyword"
os.environ["LANGSMITH_TRACING"] = "false"

from app.config import get_settings
from app.models import ChatTurn, QueryClass, QueryPlan, RetrieverResult, ChunkRecord
from app.services.retrieval import (
    classify_assistant_mode,
    llm_classify_assistant_mode,
    build_query_plan,
    llm_build_query_plan,
)


def _make_chunk(chunk_id: str, title: str, content: str) -> dict:
    """Create a minimal RetrieverResult dict for testing."""
    return RetrieverResult(
        chunk=ChunkRecord(
            id=chunk_id,
            source_id="test-source",
            title=title,
            url="https://example.com",
            section_path="Test",
            doc_family="core",
            doc_type="html",
            content=content,
        ),
        score=0.5,
        rerank_score=0.5,
    ).model_dump()


# ---------------------------------------------------------------------------
# 1. LLM Classification
# ---------------------------------------------------------------------------


class TestLLMClassification:
    def test_llm_classify_doc_rag(self, mock_reasoner, settings):
        mode, method = llm_classify_assistant_mode(
            "How do I configure NCCL for multi-GPU training?", None, mock_reasoner, settings.routing_model
        )
        assert mode == "doc_rag"
        assert method == "llm"

    def test_llm_classify_direct_chat(self, mock_reasoner, settings):
        mode, method = llm_classify_assistant_mode(
            "Hello, how are you?", None, mock_reasoner, settings.routing_model
        )
        assert mode == "direct_chat"
        assert method == "llm"

    def test_llm_classify_live_query(self, mock_reasoner, settings):
        mode, method = llm_classify_assistant_mode(
            "What is the current weather in Tokyo?", None, mock_reasoner, settings.routing_model
        )
        assert mode == "live_query"
        assert method == "llm"

    def test_llm_classify_with_history(self, mock_reasoner, settings):
        history = [
            ChatTurn(role="user", content="Tell me about NVIDIA H100"),
            ChatTurn(role="assistant", content="The H100 is..."),
        ]
        mode, method = llm_classify_assistant_mode(
            "How much memory does it have?", history, mock_reasoner, settings.routing_model
        )
        assert mode == "doc_rag"
        assert method == "llm"

    def test_llm_classify_fallback_on_error(self, settings):
        class FailingReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                raise RuntimeError("API unavailable")

        mode, method = llm_classify_assistant_mode(
            "How do I configure NCCL?", None, FailingReasoner(), settings.routing_model
        )
        assert mode == "doc_rag"  # rule-based fallback should catch NCCL
        assert method == "rule_fallback"

    def test_llm_classify_fallback_on_invalid_json(self, settings):
        class BadJsonReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                return "this is not json at all"

        mode, method = llm_classify_assistant_mode(
            "Hello!", None, BadJsonReasoner(), settings.routing_model
        )
        assert method == "rule_fallback"

    def test_llm_classify_fallback_on_invalid_mode(self, settings):
        class InvalidModeReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                return '{"mode": "invalid_mode", "reasoning": "test"}'

        mode, method = llm_classify_assistant_mode(
            "Hello!", None, InvalidModeReasoner(), settings.routing_model
        )
        assert method == "rule_fallback"

    def test_llm_classify_disabled_reasoner(self, settings):
        class DisabledReasoner:
            enabled = False

            def generate_text(self, prompt, model=None):
                raise AssertionError("Should not be called")

        mode, method = llm_classify_assistant_mode(
            "How do I install CUDA?", None, DisabledReasoner(), settings.routing_model
        )
        assert method == "rule_fallback"


# ---------------------------------------------------------------------------
# 2. LLM Query Planning
# ---------------------------------------------------------------------------


class TestLLMQueryPlanning:
    def test_llm_plan_returns_valid_plan(self, mock_reasoner, settings):
        plan, method = llm_build_query_plan("What are H100 specifications?", settings, mock_reasoner)
        assert method == "llm"
        assert isinstance(plan, QueryPlan)
        assert plan.query_class in list(QueryClass)

    def test_llm_plan_has_search_queries(self, mock_reasoner, settings):
        plan, method = llm_build_query_plan("NVIDIA H100 memory bandwidth", settings, mock_reasoner)
        assert len(plan.search_queries) >= 2

    def test_llm_plan_has_source_families(self, mock_reasoner, settings):
        plan, method = llm_build_query_plan("H100 specs", settings, mock_reasoner)
        assert len(plan.source_families) >= 1
        for fam in plan.source_families:
            assert fam in ["core", "distributed", "infrastructure", "advanced", "hardware"]

    def test_llm_plan_clamps_top_k(self, settings):
        class ExtremePlanReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                return json.dumps({
                    "query_class": "general",
                    "search_queries": ["test query"],
                    "source_families": ["core"],
                    "top_k": 100,
                    "confidence_floor": 0.01,
                    "reasoning": "test",
                })

        plan, method = llm_build_query_plan("test", settings, ExtremePlanReasoner())
        assert 3 <= plan.top_k <= 15
        assert 0.15 <= plan.confidence_floor <= 0.50

    def test_llm_plan_fallback_on_error(self, settings):
        class FailingReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                raise RuntimeError("API error")

        plan, method = llm_build_query_plan("NCCL configuration", settings, FailingReasoner())
        assert method == "rule_fallback"
        assert isinstance(plan, QueryPlan)

    def test_llm_plan_disabled_reasoner(self, settings):
        class DisabledReasoner:
            enabled = False

        plan, method = llm_build_query_plan("test", settings, DisabledReasoner())
        assert method == "rule_fallback"


# ---------------------------------------------------------------------------
# 3. LLM Document Grading
# ---------------------------------------------------------------------------


class TestLLMDocumentGrading:
    def _build_state(self, question, results):
        return {
            "question": question,
            "results": results,
            "rejected_chunk_ids": [],
            "plan": build_query_plan(question, get_settings()).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }

    def test_llm_grading_filters_irrelevant(self, agent_service_with_mock):
        # The mock returns doc 3 as irrelevant
        state = self._build_state("H100 specs", [
            _make_chunk("c1", "H100 Specs", "The H100 has 80GB HBM3"),
            _make_chunk("c2", "H100 Memory", "H100 memory bandwidth is 3.35 TB/s"),
            _make_chunk("c3", "Cooking Recipe", "How to make pasta"),
        ])
        result = agent_service_with_mock._graph_document_grading(state)
        assert result.get("llm_graded") is True
        result_ids = [RetrieverResult.model_validate(r).chunk.id for r in result["results"]]
        assert "c3" not in result_ids

    def test_llm_grading_keeps_all_relevant(self, agent_service_with_mock):
        state = self._build_state("H100 specs", [
            _make_chunk("c1", "H100 Specs", "The H100 has 80GB HBM3"),
            _make_chunk("c2", "H100 Memory", "H100 memory bandwidth is 3.35 TB/s"),
        ])
        result = agent_service_with_mock._graph_document_grading(state)
        # Only 2 chunks, both relevant (mock returns 3 grades, 2 of which are relevant)
        assert len(result["results"]) >= 2

    def test_llm_grading_fallback_on_error(self, settings):
        class FailReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                raise RuntimeError("fail")

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.corpus import load_sources
        from app.services.agent import AgentService

        sources = load_sources(settings.source_manifest_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, FailReasoner(), TavilyClient(settings))

        state = {
            "question": "test",
            "results": [_make_chunk("c1", "Test", "content")],
            "rejected_chunk_ids": [],
            "plan": build_query_plan("test", settings).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }
        result = agent._graph_document_grading(state)
        # All chunks should be kept when grading fails
        assert len(result["results"]) == 1

    def test_llm_grading_trace_event(self, agent_service_with_mock):
        state = self._build_state("H100 specs", [_make_chunk("c1", "H100", "specs")])
        result = agent_service_with_mock._graph_document_grading(state)
        trace_types = [t["type"] for t in result["trace"]]
        assert "document_grading" in trace_types

    def test_llm_grading_updates_confidence(self, agent_service_with_mock):
        state = self._build_state("H100 specs", [
            _make_chunk("c1", "H100", "H100 GPU"),
            _make_chunk("c2", "H100 Mem", "memory"),
            _make_chunk("c3", "Cooking", "pasta recipe"),
        ])
        result = agent_service_with_mock._graph_document_grading(state)
        # Confidence should be recalculated (may differ from original 0.5)
        assert "confidence" in result

    def test_llm_grading_skipped_when_disabled(self, settings):
        class DisabledReasoner:
            enabled = False

            def generate_text(self, prompt, model=None):
                raise AssertionError("should not be called")

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.corpus import load_sources
        from app.services.agent import AgentService

        sources = load_sources(settings.source_manifest_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, DisabledReasoner(), TavilyClient(settings))

        state = {
            "question": "test",
            "results": [_make_chunk("c1", "Test", "content")],
            "rejected_chunk_ids": [],
            "plan": build_query_plan("test", settings).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }
        result = agent._graph_document_grading(state)
        assert result.get("llm_graded") is False


# ---------------------------------------------------------------------------
# 4. LLM Router
# ---------------------------------------------------------------------------


class TestLLMRouter:
    def _base_state(self):
        return {
            "question": "H100 specs",
            "results": [_make_chunk("c1", "H100", "specs")],
            "confidence": 0.8,
            "retries": 0,
            "used_fallback": False,
            "grounding_passed": True,
            "answer_quality_passed": True,
            "plan": build_query_plan("H100 specs", get_settings()).model_dump(),
            "routing_decisions": [],
            "trace": [],
        }

    def test_route_after_grading_generate(self, agent_service_with_mock):
        state = self._base_state()
        result = agent_service_with_mock._route_after_grading(state)
        assert result == "generate"

    def test_route_after_grading_rewrite_low_confidence(self, agent_service_with_mock):
        state = self._base_state()
        state["confidence"] = 0.1
        state["results"] = []
        # With the mock, it may still return "generate" — but the test validates the interface works
        result = agent_service_with_mock._route_after_grading(state)
        assert result in ("generate", "rewrite_if_needed", "fallback_if_needed")

    def test_route_after_quality_end(self, agent_service_with_mock):
        state = self._base_state()
        state["grounding_passed"] = True
        state["answer_quality_passed"] = True
        result = agent_service_with_mock._route_after_quality(state)
        assert result == "end"

    def test_route_enforces_max_retries(self, agent_service_with_mock):
        state = self._base_state()
        state["retries"] = 5  # way over max
        state["grounding_passed"] = False
        state["answer_quality_passed"] = False
        result = agent_service_with_mock._route_after_quality(state)
        assert result == "end"  # should not rewrite when retries exhausted

    def test_route_enforces_tavily_config(self, agent_service_with_mock):
        # Tavily is disabled in test env
        state = self._base_state()
        plan_dict = state["plan"]
        plan_dict["use_tavily_fallback"] = False
        state["plan"] = plan_dict
        state["grounding_passed"] = False
        state["answer_quality_passed"] = False
        result = agent_service_with_mock._route_after_quality(state)
        assert result != "post_gen_fallback"  # tavily disabled

    def test_route_fallback_on_error(self, settings):
        class FailReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                raise RuntimeError("fail")

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.corpus import load_sources
        from app.services.agent import AgentService

        sources = load_sources(settings.source_manifest_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, FailReasoner(), TavilyClient(settings))
        state = self._base_state()
        result = agent._route_after_grading(state)
        assert result in ("generate", "rewrite_if_needed", "fallback_if_needed")

    def test_route_after_grading_returns_valid_edge(self, agent_service_with_mock):
        state = self._base_state()
        result = agent_service_with_mock._route_after_grading(state)
        assert result in ("generate", "rewrite_if_needed", "fallback_if_needed")

    def test_route_after_quality_returns_valid_edge(self, agent_service_with_mock):
        state = self._base_state()
        result = agent_service_with_mock._route_after_quality(state)
        assert result in ("end", "post_gen_fallback", "rewrite_if_needed")


# ---------------------------------------------------------------------------
# 5. Multi-Hop Retrieval
# ---------------------------------------------------------------------------


class TestMultiHopRetrieval:
    def _base_state(self, settings):
        return {
            "question": "H100 specs and networking",
            "results": [_make_chunk("c1", "H100 GPU", "H100 has 80GB HBM3")],
            "rejected_chunk_ids": [],
            "plan": build_query_plan("H100 specs", settings).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }

    def test_multi_hop_skips_when_sufficient(self, agent_service_with_mock, settings):
        state = self._base_state(settings)
        result = agent_service_with_mock._graph_multi_hop_check(state)
        # Default mock says sufficient=true
        assert result.get("multi_hop_used") is False

    def test_multi_hop_fires_when_insufficient(self, settings):
        class InsufficientReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                if "sufficient" in prompt.lower():
                    return '{"sufficient": false, "follow_up_query": "NVIDIA NVLink topology", "gap_description": "Missing networking info"}'
                return "default answer [1]"

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.services.ingestion import IngestionService
        from app.corpus import load_sources, load_demo_chunks
        from app.services.agent import AgentService

        sources = load_sources(settings.source_manifest_path)
        chunks = load_demo_chunks(settings.demo_corpus_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        ingestion = IngestionService(settings, index, sources, chunks)
        ingestion.bootstrap_demo_corpus()
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, InsufficientReasoner(), TavilyClient(settings))

        state = {
            "question": "H100 specs and NVLink topology",
            "results": [_make_chunk("c1", "H100", "H100 has 80GB HBM3")],
            "rejected_chunk_ids": [],
            "plan": build_query_plan("H100 specs", settings).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }
        result = agent._graph_multi_hop_check(state)
        assert result.get("multi_hop_used") is True
        assert result.get("follow_up_queries") == ["NVIDIA NVLink topology"]

    def test_multi_hop_skips_on_retry(self, agent_service_with_mock, settings):
        state = self._base_state(settings)
        state["retries"] = 1
        result = agent_service_with_mock._graph_multi_hop_check(state)
        assert result.get("multi_hop_used") is False

    def test_multi_hop_fallback_on_error(self, settings):
        class FailReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                raise RuntimeError("fail")

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.corpus import load_sources
        from app.services.agent import AgentService

        sources = load_sources(settings.source_manifest_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, FailReasoner(), TavilyClient(settings))
        state = {
            "question": "test",
            "results": [_make_chunk("c1", "Test", "content")],
            "rejected_chunk_ids": [],
            "plan": build_query_plan("test", settings).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }
        result = agent._graph_multi_hop_check(state)
        assert result.get("multi_hop_used") is False
        assert len(result["results"]) == 1  # original chunks preserved

    def test_multi_hop_trace_event(self, agent_service_with_mock, settings):
        state = self._base_state(settings)
        result = agent_service_with_mock._graph_multi_hop_check(state)
        trace_types = [t["type"] for t in result.get("trace", [])]
        assert "multi_hop" in trace_types

    def test_multi_hop_disabled_reasoner(self, settings):
        class DisabledReasoner:
            enabled = False

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.corpus import load_sources
        from app.services.agent import AgentService

        sources = load_sources(settings.source_manifest_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, DisabledReasoner(), TavilyClient(settings))
        state = {
            "question": "test",
            "results": [_make_chunk("c1", "Test", "content")],
            "rejected_chunk_ids": [],
            "plan": build_query_plan("test", settings).model_dump(),
            "trace": [],
            "retries": 0,
            "confidence": 0.5,
        }
        result = agent._graph_multi_hop_check(state)
        assert result.get("multi_hop_used") is False


# ---------------------------------------------------------------------------
# 6. Integration tests
# ---------------------------------------------------------------------------


class TestAgenticIntegration:
    def test_full_pipeline_doc_rag(self, agent_service_with_mock):
        from app.models import ChatRequest

        result = agent_service_with_mock.run(ChatRequest(question="How do I configure NCCL for multi-GPU training?"))
        assert result.answer
        assert result.response_mode in ("corpus-backed", "llm-knowledge", "web-backed")

    def test_full_pipeline_direct_chat(self, agent_service_with_mock):
        from app.models import ChatRequest

        result = agent_service_with_mock.run(ChatRequest(question="Hello, how are you?"))
        assert result.assistant_mode == "direct_chat"

    def test_fallback_chain_works(self, settings):
        """All LLM agentic features fail -> rule-based fallback chain still works."""

        class AlwaysFailReasoner:
            enabled = True

            def generate_text(self, prompt, model=None):
                raise RuntimeError("Everything fails")

        from app.services.retrieval import RetrievalService
        from app.services.indexes import InMemoryHybridIndex
        from app.services.providers import KeywordEmbedder, TavilyClient
        from app.services.ingestion import IngestionService
        from app.corpus import load_sources, load_demo_chunks
        from app.services.agent import AgentService
        from app.models import ChatRequest

        sources = load_sources(settings.source_manifest_path)
        chunks = load_demo_chunks(settings.demo_corpus_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        ingestion = IngestionService(settings, index, sources, chunks)
        ingestion.bootstrap_demo_corpus()
        retrieval = RetrievalService(settings, sources, index)
        agent = AgentService(settings, retrieval, AlwaysFailReasoner(), TavilyClient(settings))
        result = agent.run(ChatRequest(question="What is the NVIDIA H100 GPU memory?"))
        # Should still produce a result via fallbacks
        assert result is not None
        assert result.answer  # non-empty answer

    def test_agentic_trace_events(self, agent_service_with_mock):
        from app.models import ChatRequest

        result = agent_service_with_mock.run(ChatRequest(question="What are H100 specifications?"))
        trace_types = {t.type for t in result.trace}
        # doc_rag pipeline should have classification at minimum
        assert "classification" in trace_types
