from __future__ import annotations

from dataclasses import replace

from app.config import get_settings
from app.corpus import load_demo_chunks, load_sources
from app.models import ChatRequest
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import GeminiReasoner, KeywordEmbedder
from app.services.retrieval import RetrievalService


class StubTavilyClient:
    def search(self, query: str):
        return [
            {
                "title": "Fallback result",
                "url": "https://example.com/fallback",
                "content": "Web evidence about a recent NVIDIA runtime change.",
            }
        ]


def build_agent_with_demo() -> AgentService:
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_corpus()
    retrieval = RetrievalService(settings, sources, index)
    return AgentService(settings, retrieval, GeminiReasoner(settings), StubTavilyClient())


def test_agent_returns_citations_for_distributed_question():
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    assert state.citations
    assert state.query_plan is not None
    assert state.query_plan.query_class.value == "distributed_multi_gpu"
    assert "communication" in state.answer.lower() or "scaling" in state.answer.lower()
    assert any(event.type == "document_grading" for event in state.trace)
    assert state.response_mode == "corpus-backed"
    assert state.grounding_passed is True
    assert state.answer_quality_passed is True


def test_agent_graph_compiles_and_returns_required_fields():
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="When should I use mixed precision training?"))

    assert agent.graph is not None
    assert state.query_plan is not None
    assert state.retrieval_results
    assert state.citations
    assert state.trace
    assert state.answer
    assert state.response_mode
    assert isinstance(state.retry_count, int)


def test_generator_produces_non_empty_answer_and_populates_citations():
    state = build_agent_with_demo().run(ChatRequest(question="When should I use mixed precision training?"))

    assert state.answer.strip()
    assert all(citation.title and citation.url and citation.section_path and citation.snippet for citation in state.citations)
    assert all(citation.source_kind for citation in state.citations)


def test_fallback_triggers_when_local_retrieval_is_insufficient():
    settings = replace(get_settings(), use_tavily_fallback=True)
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, GeminiReasoner(settings), StubTavilyClient())

    state = agent.run(ChatRequest(question="What changed in the latest NVIDIA runtime yesterday?"))

    assert state.used_fallback is True
    assert state.response_mode == "web-backed"
    assert any(citation.source_kind == "web" for citation in state.citations)
