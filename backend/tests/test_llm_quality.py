"""LLM output quality verification tests.

Skipped by default — only runs when OPENAI_API_KEY is set.
These tests verify generation quality, citation accuracy, and grounding.
Cost: ~$0.50-2.00 per full run.

DO NOT RUN without explicit permission — costs real API credits.
"""
from __future__ import annotations

import os

import pytest

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest, AgentRunState
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService

SKIP_REASON = "requires OPENAI_API_KEY — costs real money"
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason=SKIP_REASON
)


class StubTavilyClient:
    """Tavily stub that returns no results (tests focus on knowledge base retrieval)."""

    def search(self, query: str):
        return []


def _build_real_agent() -> AgentService:
    """Build an agent with real OpenAI reasoner and demo knowledge base."""
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    reasoner = OpenAIReasoner(settings)
    return AgentService(settings, retrieval, reasoner, StubTavilyClient())


@pytest.fixture(scope="module")
def agent():
    """Module-scoped real agent to reduce repeated bootstrap cost."""
    return _build_real_agent()


class TestLLMQuality:
    """Tests that exercise real OpenAI generation against the demo knowledge base."""

    def test_knowledge_base_backed_response_is_grounded(self, agent: AgentService):
        """Ask a hardware question with known knowledge base data.

        Expects knowledge-base-backed mode with specific numbers from the knowledge base.
        """
        state = agent.run(ChatRequest(question="What is the memory bandwidth of the H100?"))

        assert state.response_mode == "knowledge-base-backed", (
            f"Expected knowledge-base-backed, got {state.response_mode}"
        )
        # The knowledge base contains specific bandwidth numbers for H100
        answer_lower = state.answer.lower()
        assert any(
            term in answer_lower
            for term in ["bandwidth", "tb/s", "gb/s", "hbm", "memory"]
        ), f"Answer should mention bandwidth/memory details, got: {state.answer[:200]}"

    def test_citation_markers_present(self, agent: AgentService):
        """A knowledge-base-backed answer must contain [N] citation markers and a non-empty citations list."""
        state = agent.run(
            ChatRequest(question="What are the key NCCL tuning parameters for bandwidth?")
        )

        assert state.response_mode == "knowledge-base-backed"
        assert state.citations, "Expected non-empty citations list"

        # Verify at least one [N] marker in the answer
        import re

        markers = re.findall(r"\[\d+\]", state.answer)
        assert markers, f"Answer should contain [N] citation markers, got: {state.answer[:200]}"

    def test_off_topic_not_knowledge_base_backed(self, agent: AgentService):
        """An off-topic question should NOT be knowledge-base-backed."""
        state = agent.run(ChatRequest(question="What is the best pizza in NYC?"))

        assert state.response_mode != "knowledge-base-backed", (
            f"Off-topic question should not be knowledge-base-backed, got {state.response_mode}"
        )
        assert state.response_mode in {
            "direct-chat",
            "llm-knowledge",
            "insufficient-evidence",
        }

    def test_response_mode_consistency(self, agent: AgentService):
        """Multiple knowledge base questions should all produce knowledge-base-backed responses."""
        questions = [
            "Why is 4-GPU training scaling poorly?",
            "What are the NCCL configuration parameters?",
            "How does GPUDirect Storage improve I/O performance?",
        ]
        for question in questions:
            state = agent.run(ChatRequest(question=question))
            assert state.response_mode == "knowledge-base-backed", (
                f"Question '{question}' should be knowledge-base-backed, got {state.response_mode}"
            )
            assert state.answer.strip(), f"Question '{question}' produced empty answer"

    def test_direct_chat_for_greeting(self, agent: AgentService):
        """A casual greeting should route to direct-chat, not trigger RAG."""
        state = agent.run(ChatRequest(question="Hello, how are you?"))

        assert state.assistant_mode == "direct_chat"
        assert state.response_mode == "direct-chat"
        assert not state.citations, "Greeting should not produce citations"
        assert not state.trace, "Greeting should not produce a RAG trace"
        assert state.answer.strip(), "Greeting should produce a non-empty answer"
