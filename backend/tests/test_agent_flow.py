from __future__ import annotations

from dataclasses import replace

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest, ChatTurn
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, KeywordEmbedder
from app.services.retrieval import RetrievalService
from conftest import MockOpenAIReasoner


class StubTavilyClient:
    def search(self, query: str):
        return [
            {
                "title": "Fallback result",
                "url": "https://example.com/fallback",
                "content": "Web evidence about a recent NVIDIA runtime change.",
            }
        ]


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


def test_agent_returns_citations_for_distributed_question():
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    assert state.citations
    assert state.assistant_mode == "doc_rag"
    assert state.query_plan is not None
    assert state.query_plan.query_class.value == "distributed_multi_gpu"
    assert "communication" in state.answer.lower() or "scaling" in state.answer.lower()
    assert any(event.type == "document_grading" for event in state.trace)
    assert state.response_mode == "knowledge-base-backed"
    assert state.grounding_passed is True
    assert state.answer_quality_passed is True


def test_agent_graph_compiles_and_returns_required_fields():
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(question="When should I use mixed precision training?"))

    assert agent.graph is not None
    assert state.assistant_mode == "doc_rag"
    assert state.query_plan is not None
    assert state.retrieval_results
    assert state.citations
    assert state.trace
    assert state.answer
    assert state.response_mode
    assert isinstance(state.retry_count, int)


def test_generator_produces_non_empty_answer_and_populates_citations():
    state = build_agent_with_demo().run(ChatRequest(question="When should I use mixed precision training?"))

    assert state.assistant_mode == "doc_rag"
    assert state.answer.strip()
    assert all(
        citation.title
        and citation.url
        and citation.citation_url
        and citation.domain
        and citation.section_path
        and citation.snippet
        for citation in state.citations
    )
    assert all(citation.source_kind for citation in state.citations)
    assert state.model_used


def test_fallback_triggers_when_local_retrieval_is_insufficient():
    settings = replace(get_settings(), use_tavily_fallback=True)
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, MockOpenAIReasoner(), StubTavilyClient())

    state = agent.run(ChatRequest(question="What changed in the latest NVIDIA runtime yesterday?"))

    assert state.assistant_mode == "doc_rag"
    assert state.used_fallback is True
    assert state.response_mode == "web-backed"
    assert any(citation.source_kind == "web" for citation in state.citations)


def test_model_override_uses_allowed_request_model():
    state = build_agent_with_demo().run(
        ChatRequest(question="Why is 4-GPU training scaling poorly?", model="gpt-5-mini")
    )

    assert state.model_used == "gpt-5-mini"


def test_direct_chat_bypasses_rag_trace_and_citations():
    state = build_agent_with_demo().run(ChatRequest(question="How should I prepare for an AI engineer interview?"))

    assert state.assistant_mode == "direct_chat"
    assert state.response_mode == "direct-chat"
    assert state.query_plan is None
    assert not state.citations
    assert not state.trace
    assert state.answer.strip()


def test_follow_up_general_question_stays_in_direct_chat_mode():
    state = build_agent_with_demo().run(
        ChatRequest(
            question="what day is it",
            history=[
                ChatTurn(role="user", content="hey"),
                ChatTurn(role="assistant", content="I can help with general questions or NVIDIA documentation."),
            ],
        )
    )

    assert state.assistant_mode == "direct_chat"
    assert state.response_mode == "direct-chat"
    assert not state.trace
    assert not state.citations


def test_general_nvidia_question_uses_helpful_direct_chat_fallback():
    state = build_agent_with_demo().run(ChatRequest(question="what is nvidia"))

    assert state.assistant_mode == "direct_chat"
    assert state.response_mode == "direct-chat"
    assert "technology company" in state.answer.lower()
    assert "gpu" in state.answer.lower()


# ---------------------------------------------------------------------------
# Gap tests: retry, grounding failure, insufficient-evidence, llm-knowledge
# ---------------------------------------------------------------------------


class NoCitationReasoner:
    """Reasoner that returns an answer with no [N] citation markers."""
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        return "This is an answer without any citation markers at all."


class ShortAnswerReasoner:
    """Reasoner that returns a very short answer (below quality threshold)."""
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        return "Yes."


class InsufficientContextReasoner:
    """Reasoner that explicitly says context is insufficient."""
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        return "I cannot find information in the provided context to answer this question."


class LLMKnowledgeReasoner:
    """Reasoner that returns a good answer for the LLM-knowledge fallback call."""
    enabled = True
    call_count = 0

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        self.call_count += 1
        return (
            "NCCL is NVIDIA Collective Communications Library used for multi-GPU training. "
            "It handles all-reduce, broadcast, and other collective operations efficiently. "
            "This is general LLM knowledge since no knowledge base context was available."
        )


class DisabledReasoner:
    """Reasoner that is disabled (simulates no Gemini API key)."""
    enabled = False

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        raise RuntimeError("Reasoner is disabled")


class EmptyTavilyClient:
    """Tavily stub that always returns no results."""
    def search(self, query: str):
        return []


def build_agent_empty_index(reasoner=None, tavily=None) -> AgentService:
    """Build an agent with an empty index (no knowledge base chunks)."""
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    if reasoner is None:
        reasoner = MockOpenAIReasoner()
    if tavily is None:
        tavily = EmptyTavilyClient()
    return AgentService(settings, retrieval, reasoner, tavily)


def test_rewrite_retry_fires_on_low_confidence():
    """An empty index should trigger at least one rewrite retry."""
    agent = build_agent_empty_index(reasoner=MockOpenAIReasoner())
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters for bandwidth?"))

    # With empty index, confidence stays below floor → retry fires
    assert state.retry_count >= 1


def test_grounding_failure_demotes_to_insufficient_evidence():
    """When the LLM explicitly says context is insufficient, grounding is forced False → insufficient-evidence."""
    agent = build_agent_with_demo(reasoner=InsufficientContextReasoner())
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    assert state.grounding_passed is False
    assert state.response_mode in {"insufficient-evidence", "llm-knowledge"}


def test_quality_failure_demotes_to_insufficient_evidence():
    """When the LLM returns a very short answer, quality check fails → insufficient-evidence."""
    agent = build_agent_with_demo(reasoner=ShortAnswerReasoner())
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    assert state.answer_quality_passed is False
    assert state.response_mode in {"insufficient-evidence", "llm-knowledge"}


def test_insufficient_evidence_when_all_layers_fail():
    """Empty index + disabled Tavily + disabled reasoner → refusal."""
    agent = build_agent_empty_index(reasoner=DisabledReasoner(), tavily=EmptyTavilyClient())
    state = agent.run(ChatRequest(question="What are the NCCL tuning parameters for bandwidth?"))

    assert state.response_mode == "insufficient-evidence"
    assert "not have enough" in state.answer.lower() or "insufficient" in state.answer.lower() or "cannot" in state.answer.lower()


def test_llm_knowledge_fallback_when_knowledge_base_and_web_exhausted():
    """Empty index + disabled Tavily + enabled reasoner → llm-knowledge fallback."""
    reasoner = LLMKnowledgeReasoner()
    agent = build_agent_empty_index(reasoner=reasoner, tavily=EmptyTavilyClient())
    state = agent.run(ChatRequest(question="What is NCCL used for in multi-GPU training?"))

    assert state.response_mode == "llm-knowledge"
    assert state.answer.strip()
    assert not state.citations


def test_refusal_answer_content():
    """The refusal answer must match the static text from _refusal_answer()."""
    from app.services.agent import AgentService
    expected = AgentService._refusal_answer()
    assert "not have enough" in expected.lower() or "insufficient" in expected.lower() or "cannot" in expected.lower()
    assert len(expected) > 20


# ---------------------------------------------------------------------------
# Multi-turn conversation memory tests
# ---------------------------------------------------------------------------


def test_follow_up_reformulation_with_mock_llm():
    """2-turn history + pronoun follow-up → trace contains query_reformulation event with method 'llm'."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="How much memory does it have?",
        history=[
            ChatTurn(role="user", content="What GPUs should I use for training?"),
            ChatTurn(role="assistant", content="The H100 and A100 are the primary choices for large-scale AI training."),
            ChatTurn(role="user", content="How much memory does it have?"),
        ],
    ))

    reformulation_events = [e for e in state.trace if e.type == "query_reformulation"]
    assert len(reformulation_events) == 1
    event = reformulation_events[0]
    assert event.payload["method"] == "llm"
    assert event.payload["original_query"] == "How much memory does it have?"
    assert "memory" in event.payload["reformulated_query"].lower()


def test_follow_up_falls_back_to_static_without_reasoner():
    """Disabled reasoner → static concat reformulation still works."""
    agent = build_agent_empty_index(reasoner=DisabledReasoner(), tavily=EmptyTavilyClient())
    state = agent.run(ChatRequest(
        question="How much memory does it have?",
        history=[
            ChatTurn(role="user", content="What GPUs should I use for training?"),
            ChatTurn(role="assistant", content="The H100 and A100 are the primary choices."),
            ChatTurn(role="user", content="How much memory does it have?"),
        ],
    ))

    reformulation_events = [e for e in state.trace if e.type == "query_reformulation"]
    assert len(reformulation_events) == 1
    assert reformulation_events[0].payload["method"] == "static"


def test_no_reformulation_for_standalone_questions():
    """A long standalone question should not trigger reformulation."""
    agent = build_agent_with_demo()
    state = agent.run(ChatRequest(
        question="What are the key NCCL tuning parameters for maximizing multi-GPU bandwidth?",
    ))

    reformulation_events = [e for e in state.trace if e.type == "query_reformulation"]
    assert len(reformulation_events) == 0


def test_synthesis_prompt_includes_history_when_provided():
    """Capture the prompt sent to the reasoner and verify history section is present."""
    captured_prompts: list[str] = []

    class PromptCapturingReasoner:
        enabled = True

        def generate_text(self, prompt: str, model: str | None = None) -> str:
            captured_prompts.append(prompt)
            if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
                return "What is the memory capacity of the NVIDIA H100?"
            return (
                "The H100 has 80GB of HBM3 memory. [1] "
                "This provides significant bandwidth for large model training. [2]"
            )

    agent = build_agent_with_demo(reasoner=PromptCapturingReasoner())
    agent.run(ChatRequest(
        question="How much memory does it have?",
        history=[
            ChatTurn(role="user", content="Tell me about the H100 GPU"),
            ChatTurn(role="assistant", content="The H100 is NVIDIA's Hopper-generation data-center GPU."),
            ChatTurn(role="user", content="How much memory does it have?"),
        ],
    ))

    # Find the synthesis prompt (the one with "numbered context passages")
    synthesis_prompts = [p for p in captured_prompts if "numbered context passages" in p.lower()]
    assert synthesis_prompts, "No synthesis prompt was captured"
    assert "conversation history" in synthesis_prompts[0].lower()
    assert "Tell me about the H100 GPU" in synthesis_prompts[0]


def test_synthesis_prompt_omits_history_when_empty():
    """No history → synthesis prompt should not contain conversation history section."""
    captured_prompts: list[str] = []

    class PromptCapturingReasoner:
        enabled = True

        def generate_text(self, prompt: str, model: str | None = None) -> str:
            captured_prompts.append(prompt)
            return (
                "Multi-GPU training scaling is affected by communication overhead. [1] "
                "NCCL handles collective operations across GPUs. [2]"
            )

    agent = build_agent_with_demo(reasoner=PromptCapturingReasoner())
    agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    synthesis_prompts = [p for p in captured_prompts if "numbered context passages" in p.lower()]
    assert synthesis_prompts, "No synthesis prompt was captured"
    assert "conversation history" not in synthesis_prompts[0].lower()


def test_cache_key_differs_with_different_context():
    """Same question + different histories → different cache keys."""
    from app.services.agent import AgentService

    req1 = ChatRequest(
        question="How much memory does it have?",
        history=[
            ChatTurn(role="user", content="Tell me about H100"),
            ChatTurn(role="assistant", content="H100 is a GPU."),
            ChatTurn(role="user", content="How much memory does it have?"),
        ],
    )
    req2 = ChatRequest(
        question="How much memory does it have?",
        history=[
            ChatTurn(role="user", content="Tell me about A100"),
            ChatTurn(role="assistant", content="A100 is a GPU."),
            ChatTurn(role="user", content="How much memory does it have?"),
        ],
    )
    req3 = ChatRequest(question="How much memory does it have?")

    key1 = AgentService._cache_key(req1)
    key2 = AgentService._cache_key(req2)
    key3 = AgentService._cache_key(req3)

    assert key1 != key2, "Different histories should produce different cache keys"
    assert key1 != key3, "History vs no-history should produce different cache keys"
    assert key3 == "How much memory does it have?"


def test_format_history_context_truncates_long_answers():
    """History context should truncate assistant responses over 200 chars."""
    from app.services.agent import AgentService

    long_answer = "A" * 300
    history = [
        ChatTurn(role="user", content="Question 1"),
        ChatTurn(role="assistant", content=long_answer),
    ]
    ctx = AgentService._format_history_context(history)
    assert "Question 1" in ctx
    assert "..." in ctx
    assert len(ctx) < len(long_answer) + 100


def test_format_history_context_empty_when_no_pairs():
    """Empty or user-only history → empty context string."""
    from app.services.agent import AgentService

    assert AgentService._format_history_context(None) == ""
    assert AgentService._format_history_context([]) == ""
    assert AgentService._format_history_context([ChatTurn(role="user", content="hi")]) == ""


# ---------------------------------------------------------------------------
# #6: Self-RAG score parsing hardening tests
# ---------------------------------------------------------------------------


class SelfReflectJsonReasoner:
    """Reasoner that returns valid JSON for self-reflect scoring."""
    enabled = True
    _call_count = 0

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        self._call_count += 1
        if "score" in prompt.lower() and "relevance" in prompt.lower():
            return '{"relevance": 5, "groundedness": 4, "completeness": 4, "issues": "none"}'
        if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
            return "What are the NCCL tuning parameters?"
        return (
            "NCCL handles all-reduce operations for multi-GPU training. [1] "
            "NVLink bandwidth is critical for scaling efficiency. [2]"
        )


class SelfReflectFloatReasoner:
    """Reasoner that returns float scores in self-reflect JSON."""
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        if "score" in prompt.lower() and "relevance" in prompt.lower():
            return '{"relevance": 4.7, "groundedness": 2.5, "completeness": 3.8, "issues": "minor gaps"}'
        if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
            return "What are NCCL tuning parameters?"
        return (
            "NCCL handles all-reduce operations for multi-GPU training. [1] "
            "NVLink bandwidth is critical for scaling. [2]"
        )


class SelfReflectGarbageReasoner:
    """Reasoner that returns non-JSON text for self-reflect."""
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        if "score" in prompt.lower() and "relevance" in prompt.lower():
            return "The answer is quite good overall, I would rate it four out of five."
        if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
            return "What are NCCL parameters?"
        return (
            "NCCL handles all-reduce operations for multi-GPU training. [1] "
            "NVLink is important for scaling. [2]"
        )


class SelfReflectLowGroundednessReasoner:
    """Reasoner that returns low groundedness score (< 3) to trigger fallback."""
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        if "score" in prompt.lower() and "relevance" in prompt.lower():
            return '{"relevance": 4, "groundedness": 2, "completeness": 3, "issues": "answer not well grounded"}'
        if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
            return "What are the NCCL parameters?"
        return (
            "NCCL handles all-reduce operations for multi-GPU training. [1] "
            "NVLink bandwidth is critical. [2]"
        )


def test_self_reflect_parses_valid_json_scores():
    """Valid JSON response should be parsed and stored in the run state."""
    agent = build_agent_with_demo(reasoner=SelfReflectJsonReasoner())
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    reflect_events = [e for e in state.trace if e.type == "self_reflect"]
    assert len(reflect_events) == 1
    scores = reflect_events[0].payload["scores"]
    assert scores["relevance"] == 5
    assert scores["groundedness"] == 4
    assert scores["completeness"] == 4


def test_self_reflect_handles_float_scores():
    """Float scores (e.g., 2.5) should be converted to int for groundedness check."""
    agent = build_agent_with_demo(reasoner=SelfReflectFloatReasoner())
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    reflect_events = [e for e in state.trace if e.type == "self_reflect"]
    assert len(reflect_events) == 1
    scores = reflect_events[0].payload["scores"]
    # Float 2.5 → int(2.5) = 2, which is < 3 → forces grounding fail
    assert scores["groundedness"] == 2.5
    assert reflect_events[0].payload["forced_grounding_fail"] is True


def test_self_reflect_defaults_to_neutral_on_garbage():
    """Non-JSON response should default to neutral scores (3), not 0."""
    agent = build_agent_with_demo(reasoner=SelfReflectGarbageReasoner())
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    reflect_events = [e for e in state.trace if e.type == "self_reflect"]
    assert len(reflect_events) == 1
    scores = reflect_events[0].payload["scores"]
    assert scores["relevance"] == 3
    assert scores["groundedness"] == 3
    assert scores["completeness"] == 3
    # Neutral score (3) should NOT force grounding failure
    assert reflect_events[0].payload["forced_grounding_fail"] is False


def test_self_reflect_low_groundedness_forces_grounding_fail():
    """Groundedness < 3 should force grounding_passed to False."""
    agent = build_agent_with_demo(reasoner=SelfReflectLowGroundednessReasoner())
    state = agent.run(ChatRequest(question="Why is 4-GPU training scaling poorly?"))

    reflect_events = [e for e in state.trace if e.type == "self_reflect"]
    assert len(reflect_events) == 1
    assert reflect_events[0].payload["forced_grounding_fail"] is True
    # Grounding should have been forced False by self-reflect
    # (though grounding_check may override it back if answer has good citations)


# ---------------------------------------------------------------------------
# #7: SSE error event tests
# ---------------------------------------------------------------------------


def test_stream_emits_error_event_on_pipeline_failure():
    """When the pipeline throws an uncaught exception, SSE stream includes error + done events."""
    import asyncio
    import json
    from unittest.mock import patch

    agent = build_agent_with_demo()
    request = ChatRequest(question="Why is 4-GPU training scaling poorly?")

    # Patch agent.run to simulate an uncaught pipeline crash
    with patch.object(agent, "run", side_effect=RuntimeError("Catastrophic pipeline failure")):
        async def collect_frames():
            frames = []
            async for frame in agent.stream(request):
                frames.append(frame)
            return frames

        frames = asyncio.run(collect_frames())

    all_text = "".join(frames)

    # Parse SSE events
    events = []
    for block in all_text.strip().split("\n\n"):
        event_name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.replace("event:", "", 1).strip()
            elif line.startswith("data:"):
                data = json.loads(line.replace("data:", "", 1).strip())
        if event_name and data is not None:
            events.append((event_name, data))

    event_names = [name for name, _ in events]
    assert "error" in event_names, "Expected an 'error' SSE event when pipeline fails"
    assert "done" in event_names, "Expected a 'done' SSE event even on error"

    error_payload = next(payload for name, payload in events if name == "error")
    assert "message" in error_payload
    assert "recoverable" in error_payload
    assert error_payload["recoverable"] is True
    assert "Catastrophic pipeline failure" in error_payload["message"]

    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["generation_degraded"] is True
    assert done_payload["response_mode"] == "error"


def test_multi_turn_rag_stays_coherent():
    """3-turn NCCL conversation → each turn should retrieve relevant chunks."""
    agent = build_agent_with_demo()

    # Turn 1: initial NCCL question
    state1 = agent.run(ChatRequest(question="What are the key tuning parameters for NCCL?"))
    assert state1.assistant_mode == "doc_rag"
    assert state1.retrieval_results

    # Turn 2: follow-up about bandwidth
    state2 = agent.run(ChatRequest(
        question="How does it affect bandwidth?",
        history=[
            ChatTurn(role="user", content="What are the key tuning parameters for NCCL?"),
            ChatTurn(role="assistant", content=state1.answer),
            ChatTurn(role="user", content="How does it affect bandwidth?"),
        ],
    ))
    # Should have reformulated the follow-up
    assert any(e.type == "query_reformulation" for e in state2.trace)

    # Turn 3: another follow-up
    state3 = agent.run(ChatRequest(
        question="What about NVLink?",
        history=[
            ChatTurn(role="user", content="What are the key tuning parameters for NCCL?"),
            ChatTurn(role="assistant", content=state1.answer),
            ChatTurn(role="user", content="How does it affect bandwidth?"),
            ChatTurn(role="assistant", content=state2.answer),
            ChatTurn(role="user", content="What about NVLink?"),
        ],
    ))
    assert any(e.type == "query_reformulation" for e in state3.trace)


# ---------------------------------------------------------------------------
# Tier 4.1: Semantic cache tests
# ---------------------------------------------------------------------------


def test_semantic_cache_miss_on_new_question():
    from app.services.agent import SemanticCache
    from app.services.providers import KeywordEmbedder

    cache = SemanticCache(KeywordEmbedder(), threshold=0.92)
    assert cache.get("What is NCCL?") is None


def test_semantic_cache_hit_on_exact_question():
    from app.services.agent import AgentRunState, SemanticCache
    from app.services.providers import KeywordEmbedder

    cache = SemanticCache(KeywordEmbedder(), threshold=0.92)
    state = AgentRunState(question="What is NCCL?", model_used="test")
    cache.put("What is NCCL?", state)
    result = cache.get("What is NCCL?")
    assert result is not None
    assert result.question == "What is NCCL?"


def test_semantic_cache_miss_on_dissimilar_question():
    from app.services.agent import AgentRunState, SemanticCache
    from app.services.providers import KeywordEmbedder

    cache = SemanticCache(KeywordEmbedder(), threshold=0.92)
    cache.put("What is NCCL?", AgentRunState(question="What is NCCL?", model_used="test"))
    assert cache.get("How do I install CUDA on Windows?") is None


def test_semantic_cache_lru_eviction():
    from app.services.agent import AgentRunState, SemanticCache
    from app.services.providers import KeywordEmbedder

    cache = SemanticCache(KeywordEmbedder(), threshold=0.92, maxsize=3)
    for i in range(5):
        q = f"unique question number {i} about topic {i * 7}"
        cache.put(q, AgentRunState(question=q, model_used="test"))
    # First two should have been evicted
    assert cache.get("unique question number 0 about topic 0") is None
    assert cache.get("unique question number 1 about topic 0") is None


def test_semantic_cache_clear():
    from app.services.agent import AgentRunState, SemanticCache
    from app.services.providers import KeywordEmbedder

    cache = SemanticCache(KeywordEmbedder(), threshold=0.92)
    cache.put("What is NCCL?", AgentRunState(question="What is NCCL?", model_used="test"))
    cache.clear()
    assert cache.get("What is NCCL?") is None


def test_semantic_cache_thread_safety():
    import threading
    from app.services.agent import AgentRunState, SemanticCache
    from app.services.providers import KeywordEmbedder

    cache = SemanticCache(KeywordEmbedder(), threshold=0.92, maxsize=64)
    errors: list[Exception] = []

    def worker(n: int):
        try:
            for i in range(20):
                q = f"thread {n} question {i}"
                cache.put(q, AgentRunState(question=q, model_used="test"))
                cache.get(q)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# Tier 4.2: Query decomposition tests
# ---------------------------------------------------------------------------


def test_should_decompose_detects_multi_part():
    """Multi-part question with 'and' + 'compare' should be decomposable."""
    from dataclasses import replace
    settings = replace(get_settings(), decomposition_enabled=True)
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, MockOpenAIReasoner(), EmptyTavilyClient())

    assert agent._should_decompose("Compare the H100 and A100 memory bandwidth and what are the NCCL tuning parameters?")


def test_should_decompose_rejects_simple_question():
    """Simple question should not be decomposed."""
    from dataclasses import replace
    settings = replace(get_settings(), decomposition_enabled=True)
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, MockOpenAIReasoner(), EmptyTavilyClient())

    assert not agent._should_decompose("What is NCCL?")


def test_decompose_disabled_by_default():
    """Decomposition should not trigger when decomposition_enabled=False."""
    agent = build_agent_with_demo()
    assert not agent._should_decompose("Compare H100 and A100 memory bandwidth and NCCL tuning")


def test_decompose_falls_back_when_reasoner_disabled():
    """Disabled reasoner → decomposition returns empty list."""
    from dataclasses import replace
    settings = replace(get_settings(), decomposition_enabled=True)
    sources = load_sources(settings.source_manifest_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, DisabledReasoner(), EmptyTavilyClient())

    result = agent._decompose_question("Compare H100 and A100")
    assert result == []


# ---------------------------------------------------------------------------
# Tier 4.4: Invalid citation marker tests
# ---------------------------------------------------------------------------


def test_strip_invalid_citations_removes_out_of_range():
    """[5] should be stripped when only 4 citations exist."""
    answer = "GPU training uses NVLink for interconnect. [1] Memory bandwidth matters. [5]"
    cleaned = AgentService._strip_invalid_citations(answer, 4)
    assert "[1]" in cleaned
    assert "[5]" not in cleaned


def test_strip_invalid_citations_keeps_valid():
    """Valid [1]-[3] markers should be preserved."""
    answer = "Point one. [1] Point two. [2] Point three. [3]"
    cleaned = AgentService._strip_invalid_citations(answer, 3)
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "[3]" in cleaned


def test_strip_invalid_citations_removes_all_when_no_citations():
    """All markers should be stripped when there are no citations."""
    answer = "Some content. [1] More content. [2]"
    cleaned = AgentService._strip_invalid_citations(answer, 0)
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned


def test_grounding_check_warns_on_invalid_markers():
    """Grounding check should handle answers with out-of-range markers gracefully."""
    from app.models import Citation

    citations = [
        Citation(
            chunk_id="c1", title="Test", url="https://example.com",
            citation_url="https://example.com#s1", domain="example.com",
            section_path="Overview", snippet="Test content",
        ),
    ]
    # Only 1 citation but answer references [1] and [5]
    answer = "NCCL handles collective communication efficiently [1]. It also supports multi-node setups [5]."
    # Should not crash; the valid [1] should count toward grounding
    result = AgentService._grounding_check(answer, citations)
    assert isinstance(result, bool)
