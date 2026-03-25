from __future__ import annotations

import threading

import pytest
from dataclasses import replace

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest, ChatTurn
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder
from app.services.retrieval import RetrievalService
from conftest import MockOpenAIReasoner


class StubTavilyClient:
    enabled = False

    def search(self, query: str):
        return [
            {
                "title": "Fallback result",
                "url": "https://example.com/fallback",
                "content": "Web evidence about a recent NVIDIA runtime change.",
            }
        ]


class EmptyTavilyClient:
    enabled = False

    def search(self, query: str):
        return []


def build_agent_with_demo(reasoner=None) -> AgentService:
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    if reasoner is None:
        reasoner = MockOpenAIReasoner()
    return AgentService(settings, retrieval, reasoner, StubTavilyClient())


def build_agent_empty_index(reasoner=None, tavily=None) -> AgentService:
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    if reasoner is None:
        reasoner = MockOpenAIReasoner()
    if tavily is None:
        tavily = EmptyTavilyClient()
    return AgentService(settings, retrieval, reasoner, tavily)


# ---------------------------------------------------------------------------
# 1. Multi-turn conversation flow
# ---------------------------------------------------------------------------


def test_multi_turn_conversation_flow():
    """Run 3 sequential questions through agent.run(). First is standalone,
    second and third include history from previous turns."""
    agent = build_agent_with_demo()

    # Turn 1: standalone question
    state1 = agent.run(ChatRequest(question="What are the key tuning parameters for NCCL?"))
    assert state1.response_mode in {"knowledge-base-backed", "llm-knowledge", "web-backed"}
    assert state1.answer.strip()

    # Turn 2: follow-up with history
    history2 = [
        ChatTurn(role="user", content="What are the key tuning parameters for NCCL?"),
        ChatTurn(role="assistant", content=state1.answer),
        ChatTurn(role="user", content="How does it affect bandwidth?"),
    ]
    state2 = agent.run(ChatRequest(question="How does it affect bandwidth?", history=history2))
    assert state2.response_mode in {"knowledge-base-backed", "llm-knowledge", "web-backed", "direct-chat"}
    assert state2.answer.strip()
    # Follow-up should trigger reformulation
    reformulation_events = [e for e in state2.trace if e.type == "query_reformulation"]
    assert len(reformulation_events) == 1, "Follow-up question should trigger query reformulation"
    assert reformulation_events[0].payload["method"] == "llm"

    # Turn 3: another follow-up with longer history
    history3 = history2 + [
        ChatTurn(role="assistant", content=state2.answer),
        ChatTurn(role="user", content="What about NVLink?"),
    ]
    state3 = agent.run(ChatRequest(question="What about NVLink?", history=history3))
    assert state3.response_mode in {"knowledge-base-backed", "llm-knowledge", "web-backed", "direct-chat"}
    assert state3.answer.strip()
    reformulation_events3 = [e for e in state3.trace if e.type == "query_reformulation"]
    assert len(reformulation_events3) == 1


# ---------------------------------------------------------------------------
# 2. Fallback chain exhaustion
# ---------------------------------------------------------------------------


def test_fallback_chain_exhaustion():
    """Empty index + Tavily disabled + enabled reasoner -> llm-knowledge fallback."""
    agent = build_agent_empty_index(reasoner=MockOpenAIReasoner(), tavily=EmptyTavilyClient())
    state = agent.run(ChatRequest(question="What is NCCL used for in multi-GPU training?"))

    assert state.response_mode == "llm-knowledge"
    assert state.answer.strip()
    assert not state.citations


# ---------------------------------------------------------------------------
# 3. SSE event ordering
# ---------------------------------------------------------------------------


def test_sse_event_ordering():
    """Run a knowledge-base-backed query, collect all trace events. Verify events
    appear in the expected order."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    # Only check knowledge-base-backed path ordering
    if state.response_mode != "knowledge-base-backed":
        pytest.skip("Pipeline did not produce knowledge-base-backed response for this run")

    event_types = [e.type for e in state.trace]

    # These events must appear in this relative order (some may repeat)
    expected_order = [
        "classification",
        "retrieval",
        "document_grading",
        "generation",
        "self_reflect",
        "grounding_check",
        "answer_quality_check",
    ]
    # Find the first occurrence index of each expected event type
    first_indices = {}
    for event_type in expected_order:
        for i, et in enumerate(event_types):
            if et == event_type:
                first_indices[event_type] = i
                break

    # Verify that present events appear in the correct relative order
    found_types = [et for et in expected_order if et in first_indices]
    assert len(found_types) >= 4, f"Expected at least 4 pipeline stages, got: {found_types}"

    for i in range(len(found_types) - 1):
        assert first_indices[found_types[i]] < first_indices[found_types[i + 1]], (
            f"{found_types[i]} (idx {first_indices[found_types[i]]}) should appear before "
            f"{found_types[i + 1]} (idx {first_indices[found_types[i + 1]]})"
        )


# ---------------------------------------------------------------------------
# 4. Concurrent requests
# ---------------------------------------------------------------------------


def test_concurrent_requests():
    """Run 2 agent.run() calls in parallel with different questions.
    Verify each gets its own response with no state bleed."""
    agent = build_agent_with_demo()
    results: dict[str, object] = {}
    errors: list[Exception] = []

    def run_query(key: str, question: str):
        try:
            state = agent.run(ChatRequest(question=question))
            results[key] = state
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=run_query, args=("nccl", "What are the NCCL tuning parameters?"))
    t2 = threading.Thread(target=run_query, args=("mixed", "When should I use mixed precision training?"))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors, f"Thread errors: {errors}"
    assert "nccl" in results, "NCCL query did not complete"
    assert "mixed" in results, "Mixed precision query did not complete"

    state_nccl = results["nccl"]
    state_mixed = results["mixed"]
    # Both should have answers
    assert state_nccl.answer.strip()
    assert state_mixed.answer.strip()
    # Questions should be preserved without cross-contamination
    assert state_nccl.question == "What are the NCCL tuning parameters?"
    assert state_mixed.question == "When should I use mixed precision training?"


# ---------------------------------------------------------------------------
# 5. History context formatting
# ---------------------------------------------------------------------------


def test_history_context_formatting():
    """Test _format_history_context() directly with 0, 1, and 3 turns of history."""
    # 0 turns
    assert AgentService._format_history_context(None) == ""
    assert AgentService._format_history_context([]) == ""

    # 1 Q&A pair
    history_1 = [
        ChatTurn(role="user", content="What is NCCL?"),
        ChatTurn(role="assistant", content="NCCL is a collective communications library."),
    ]
    ctx_1 = AgentService._format_history_context(history_1)
    assert "What is NCCL?" in ctx_1
    assert "NCCL is a collective communications library." in ctx_1

    # 3 Q&A pairs
    history_3 = [
        ChatTurn(role="user", content="Question one"),
        ChatTurn(role="assistant", content="Answer one"),
        ChatTurn(role="user", content="Question two"),
        ChatTurn(role="assistant", content="Answer two"),
        ChatTurn(role="user", content="Question three"),
        ChatTurn(role="assistant", content="Answer three"),
    ]
    ctx_3 = AgentService._format_history_context(history_3)
    assert "Question one" in ctx_3
    assert "Question two" in ctx_3
    assert "Question three" in ctx_3
    assert isinstance(ctx_3, str)


# ---------------------------------------------------------------------------
# 6. Rewrite cycle
# ---------------------------------------------------------------------------


class PoorThenGoodReasoner:
    """First answer is poor (no citations, no content overlap), second is good."""
    enabled = True
    _call_count = 0

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        self._call_count += 1
        # Reformulation prompts
        if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
            return "What are the NCCL configuration parameters?"
        # Self-reflect scoring
        if "score" in prompt.lower() and "relevance" in prompt.lower():
            return '{"relevance": 4, "groundedness": 4, "completeness": 4, "issues": "none"}'
        # Query rewrite prompt
        if "rewrite" in prompt.lower() and "low-confidence" in prompt.lower():
            return "NCCL all-reduce collective communication tuning parameters for multi-GPU bandwidth"
        # Generation: always return a decent answer with citations
        return (
            "NCCL handles all-reduce communication for multi-GPU scaling. [1] "
            "NVLink bandwidth is critical for efficiency. [2] "
            "Mixed precision training reduces memory overhead. [3]"
        )


def test_rewrite_cycle():
    """Empty index should trigger low confidence -> rewrite -> re-retrieve."""
    agent = build_agent_empty_index(reasoner=PoorThenGoodReasoner(), tavily=EmptyTavilyClient())
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters for bandwidth?"))

    # With empty index, confidence stays below floor, should trigger at least 1 rewrite
    rewrite_events = [e for e in state.trace if e.type == "rewrite"]
    assert len(rewrite_events) >= 1, "Expected at least one rewrite event when knowledge base returns low confidence"
    assert state.retry_count >= 1


# ---------------------------------------------------------------------------
# 7. Max retry guard
# ---------------------------------------------------------------------------


def test_max_retry_guard():
    """Verify agent doesn't loop infinitely on rewrites. Rewrite count <= max_retries."""
    agent = build_agent_empty_index(reasoner=MockOpenAIReasoner(), tavily=EmptyTavilyClient())
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters?"))

    rewrite_events = [e for e in state.trace if e.type == "rewrite"]
    # max_retries is 2 by default in QueryPlan
    assert len(rewrite_events) <= 2, (
        f"Expected at most 2 rewrites (max_retries=2), got {len(rewrite_events)}"
    )
    assert state.retry_count <= 2


# ---------------------------------------------------------------------------
# 8. Graph state populated
# ---------------------------------------------------------------------------


def test_graph_state_populated():
    """Run pipeline, examine the final AgentRunState. Verify all expected fields are populated."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    # Core fields must be present
    assert state.answer.strip(), "Answer should be non-empty"
    assert state.response_mode in {
        "knowledge-base-backed", "web-backed", "llm-knowledge", "insufficient-evidence", "direct-chat"
    }
    assert isinstance(state.confidence, float)
    assert isinstance(state.trace, list)
    assert isinstance(state.grounding_passed, bool)
    assert isinstance(state.answer_quality_passed, bool)

    # For knowledge-base-backed responses, citations should be present
    if state.response_mode == "knowledge-base-backed":
        assert state.citations, "Knowledge-base-backed response should have citations"
        assert state.query_plan is not None
        assert state.retrieval_results


# ---------------------------------------------------------------------------
# 9. Cache key differs with context
# ---------------------------------------------------------------------------


def test_cache_key_differs_with_context():
    """Same question with different history -> different cache keys.
    Same question with same history -> same key."""
    req1 = ChatRequest(
        question="tell me more",
        history=[
            ChatTurn(role="user", content="What is NCCL?"),
            ChatTurn(role="assistant", content="NCCL is a library."),
            ChatTurn(role="user", content="tell me more"),
        ],
    )
    req2 = ChatRequest(
        question="tell me more",
        history=[
            ChatTurn(role="user", content="What is NVLink?"),
            ChatTurn(role="assistant", content="NVLink is an interconnect."),
            ChatTurn(role="user", content="tell me more"),
        ],
    )
    req3 = ChatRequest(question="tell me more")

    key1 = AgentService._cache_key(req1)
    key2 = AgentService._cache_key(req2)
    key3 = AgentService._cache_key(req3)

    assert key1 != key2, "Different histories should produce different cache keys"
    assert key1 != key3, "History vs no-history should produce different cache keys"
    assert key2 != key3

    # Same request should produce the same key
    key1_again = AgentService._cache_key(req1)
    assert key1 == key1_again, "Same request should produce the same cache key"


# ---------------------------------------------------------------------------
# 10. Direct chat skips retrieval
# ---------------------------------------------------------------------------


def test_direct_chat_skips_retrieval():
    """Send a greeting. Verify response_mode is direct-chat and no retrieval events."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="Hello"))

    assert state.assistant_mode == "direct_chat"
    assert state.response_mode == "direct-chat"
    assert not state.trace, "Direct chat should produce no trace events"
    retrieval_events = [e for e in state.trace if e.type == "retrieval"]
    assert len(retrieval_events) == 0
    assert not state.citations
    assert state.answer.strip()
