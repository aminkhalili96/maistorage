"""Edge case and robustness tests for the agentic RAG pipeline.

All tests use MockOpenAIReasoner — safe to run without API keys.
These verify the pipeline handles malformed, unusual, and boundary-condition
inputs gracefully without crashing.
"""
from __future__ import annotations

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest, ChatTurn
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder
from app.services.retrieval import RetrievalService
from conftest import MockOpenAIReasoner


class EmptyTavilyClient:
    """Tavily stub that always returns no results."""

    def search(self, query: str):
        return []


class DisabledReasoner:
    """Reasoner that is disabled (simulates no OpenAI API key)."""

    enabled = False

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        raise RuntimeError("Reasoner is disabled")


def build_agent_with_demo(reasoner=None) -> AgentService:
    """Build an agent with demo knowledge base loaded."""
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    if reasoner is None:
        reasoner = MockOpenAIReasoner()
    return AgentService(settings, retrieval, reasoner, EmptyTavilyClient())


def build_agent_empty_index(reasoner=None) -> AgentService:
    """Build an agent with an empty index (no knowledge base chunks)."""
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    if reasoner is None:
        reasoner = MockOpenAIReasoner()
    return AgentService(settings, retrieval, reasoner, EmptyTavilyClient())


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_empty_knowledge_base_graceful_fallback():
    """An empty index (no chunks) should not crash.

    With no retrieval results the pipeline should fall through to
    llm-knowledge (reasoner enabled) or insufficient-evidence.
    """
    agent = build_agent_empty_index(reasoner=MockOpenAIReasoner())
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters?"))

    assert state.response_mode in {"llm-knowledge", "insufficient-evidence"}, (
        f"Expected llm-knowledge or insufficient-evidence, got {state.response_mode}"
    )
    assert state.answer.strip(), "Answer should not be empty even with empty knowledge base"


def test_single_char_query():
    """A single-character query '?' should not crash the pipeline."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="?"))

    assert isinstance(state.answer, str)
    assert state.response_mode in {
        "knowledge-base-backed",
        "llm-knowledge",
        "insufficient-evidence",
        "direct-chat",
    }


def test_whitespace_only_query():
    """A whitespace-only query should not crash the pipeline."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="   "))

    assert isinstance(state.answer, str)
    assert state.response_mode in {
        "knowledge-base-backed",
        "llm-knowledge",
        "insufficient-evidence",
        "direct-chat",
    }


def test_very_long_query():
    """A 2000-character query (at the API limit) should not crash."""
    long_query = "What is the NVIDIA H100 GPU performance? " * 50  # ~2050 chars
    long_query = long_query[:2000]
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question=long_query))

    assert isinstance(state.answer, str)
    assert state.answer.strip(), "Long query should still produce an answer"


def test_unicode_query():
    """A query with emoji characters should not crash the pipeline."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="What is the H100 GPU \U0001f680?"))

    assert isinstance(state.answer, str)
    assert state.answer.strip()
    # The pipeline should still attempt retrieval for an NVIDIA-related query
    assert state.assistant_mode in {"doc_rag", "direct_chat"}


def test_query_with_newlines():
    """A query containing newline characters should be handled gracefully."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="What is\nthe H100\nGPU?"))

    assert isinstance(state.answer, str)
    assert state.answer.strip()


def test_empty_history_turns():
    """ChatRequest with empty-content history turns should not crash."""
    agent = build_agent_with_demo()
    state = agent.run(
        ChatRequest(
            question="What is the H100 GPU?",
            history=[
                ChatTurn(role="user", content=""),
                ChatTurn(role="assistant", content=""),
            ],
        )
    )

    assert isinstance(state.answer, str)
    assert state.answer.strip()


def test_reasoner_disabled_graceful_degradation():
    """When the reasoner is disabled, the pipeline should degrade gracefully.

    With a populated knowledge base, retrieval still works but LLM synthesis falls
    back to keyword-based answer assembly with generation_degraded=True.
    """
    agent = build_agent_with_demo(reasoner=DisabledReasoner())
    state = agent.run(
        ChatRequest(question="Why is 4-GPU training scaling poorly?")
    )

    # The pipeline should complete without crashing
    assert isinstance(state.answer, str)
    assert state.answer.strip(), "Disabled reasoner should still produce an answer"
    # generation_degraded should be True since the reasoner cannot synthesize
    assert state.generation_degraded is True, (
        f"Expected generation_degraded=True, got {state.generation_degraded}"
    )


def test_special_characters_in_query():
    """Special characters including HTML/script tags should not crash or execute."""
    agent = build_agent_with_demo()
    state = agent.run(
        ChatRequest(question="H100 <script>alert('xss')</script>")
    )

    assert isinstance(state.answer, str)
    # The script tag should not appear in the output as executable code
    assert "<script>" not in state.answer.lower() or "alert" not in state.answer.lower()


def test_repeated_identical_queries():
    """Running the same query 5 times should produce consistent response_mode."""
    agent = build_agent_with_demo()
    question = "Why is 4-GPU training scaling poorly?"
    modes = []
    for _ in range(5):
        state = agent.run(ChatRequest(question=question))
        modes.append(state.response_mode)

    # All runs should produce the same response mode
    assert len(set(modes)) == 1, (
        f"Expected consistent response_mode across 5 runs, got: {modes}"
    )
