"""
Agentic RAG pipeline — the core intelligence of the system.

This module implements a 12-node LangGraph state machine that orchestrates the
full retrieve-grade-generate-verify cycle:

    classify → retrieve → document_grading → multi_hop_check
      → [LLM Router: rewrite | fallback | generate]
      → generate → self_reflect → claim_verification → grounding_check
      → answer_quality_check
      → [LLM Router: end | post_gen_fallback | rewrite]

Key design decisions:
  - Every LLM-powered decision (classification, routing, grading, multi-hop)
    has a rule-based fallback so the pipeline works without OpenAI.
  - Progressive SSE: trace events are emitted in real-time as each graph node
    completes, not batched at the end. Uses thread-local emit callback.
  - 5-mode response model with fallback chain:
    knowledge-base-backed → web-backed → llm-knowledge → insufficient-evidence → direct-chat
  - Guard rails override LLM routing decisions when confidence >= 0.85
    to prevent unnecessary API calls (Tavily) or rewrites.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, TypedDict
from urllib.parse import urlparse

_log = logging.getLogger("maistorage.agent")

from app.config import Settings
from app.models import AgentRunState, ChatRequest, ChatTurn, Citation, ChunkRecord, QueryPlan, RetrieverResult, SearchDebugResponse, TraceEvent
from app.services.providers import Embedder, OpenAIReasoner, TavilyClient, tokenize
from app.services.retrieval import RetrievalService, classify_assistant_mode, llm_classify_assistant_mode, llm_build_query_plan, generate_hyde_query, needs_retry, rewrite_query, estimate_confidence

try:  # pragma: no cover - graph wiring is exercised via run()
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None

try:  # pragma: no cover - exercised only when LangSmith is configured
    from langsmith import traceable, tracing_context
except ImportError:  # pragma: no cover
    def traceable(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    @contextmanager
    def tracing_context(**_kwargs):
        yield


class GraphState(TypedDict, total=False):
    """Shared state threaded through every LangGraph node.

    Each node reads what it needs and returns an updated copy (immutable pattern).
    The state accumulates trace events, retrieval results, and quality signals
    as the pipeline progresses through classify → retrieve → grade → generate → verify.
    """
    question: str                               # Original user question
    assistant_mode: str                         # doc_rag | direct_chat | live_query
    plan: dict[str, Any]                        # Serialized QueryPlan (search queries, source families, top_k)
    current_query: str                          # Active retrieval query (may differ from question after rewrite)
    rewritten_query: str | None                 # LLM-rewritten query for retry pass
    results: list[dict[str, Any]]               # Serialized RetrieverResult list (graded, reranked chunks)
    citations: list[dict[str, Any]]             # Serialized Citation list for the final answer
    trace: list[dict[str, Any]]                 # Accumulated trace events (emitted progressively via SSE)
    rejected_chunk_ids: list[str]               # Chunk IDs that failed document grading
    confidence: float                           # Weighted average of top-3 rerank scores (0.0–1.0)
    answer: str                                 # Generated answer text
    retries: int                                # Number of rewrite+re-retrieve cycles so far
    used_fallback: bool                         # Whether Tavily web search was invoked
    fallback_reason: str | None                 # Why fallback was triggered
    response_mode: str                          # Trust label: knowledge-base-backed | web-backed | llm-knowledge | insufficient-evidence
    grounding_passed: bool                      # Did the answer pass citation grounding check?
    answer_quality_passed: bool                 # Did the answer pass quality check?
    model_used: str                             # OpenAI model used for synthesis
    generation_degraded: bool                   # True if LLM synthesis failed and keyword fallback was used
    self_reflect_scores: dict[str, Any]         # Self-RAG scores: relevance, groundedness, completeness (1-5)
    sub_questions: list[str]                    # Decomposed sub-questions (when query decomposition is enabled)
    history_context: str                        # Formatted conversation history for multi-turn context
    classification_method: str                  # How classification was done: "llm" or "rule_fallback"
    plan_method: str                            # How query plan was built: "llm" or "rule_fallback"
    llm_graded: bool                            # Whether document grading used LLM (vs threshold filtering)
    routing_decisions: list[dict[str, Any]]     # Log of LLM routing decisions at each decision point
    multi_hop_used: bool                        # Whether multi-hop follow-up retrieval was triggered
    follow_up_queries: list[str]                # Multi-hop follow-up queries issued


def _slugify_section(section_path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section_path.lower()).strip("-") or "overview"


def _tool_payload(tool: str, tool_label: str, source_kind: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "tool_label": tool_label,
        "source_kind": source_kind,
        "brand": "nvidia" if tool == "nvidia_docs" else "web",
    }


def _result_preview(result: RetrieverResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk.id,
        "title": result.chunk.title,
        "url": result.chunk.url,
        "section_path": result.chunk.section_path,
        "source_kind": result.chunk.source_kind,
        "snippet": result.chunk.content[:160].strip(),
    }


def _resolve_citation_url(chunk: ChunkRecord) -> str:
    if chunk.source_kind != "knowledge_base":
        return chunk.url
    if "#" in chunk.url:
        return chunk.url
    if chunk.doc_type == "html":
        return f"{chunk.url}#{_slugify_section(chunk.section_path)}"
    return chunk.url


def _citation_domain(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc or "").removeprefix("www.") or "local"


def _citation_from_result(result: RetrieverResult) -> Citation:
    snippet = result.chunk.content[:240].strip()
    return Citation(
        chunk_id=result.chunk.id,
        title=result.chunk.title,
        url=result.chunk.url,
        citation_url=_resolve_citation_url(result.chunk),
        domain=_citation_domain(result.chunk.url),
        section_path=result.chunk.section_path,
        snippet=snippet,
        source_kind=result.chunk.source_kind,
        source_id=result.chunk.source_id,
        score=round(result.score, 2),
        char_count=len(result.chunk.content),
        page=result.chunk.metadata.get("page"),
    )


# Progressive SSE mechanism: each graph node calls _append_trace() which both
# appends to the state's trace list AND immediately emits the event to the SSE
# stream via a thread-local callback. This is what makes the agent trace appear
# in real-time in the frontend, not after the entire pipeline completes.
_thread_local_emit: threading.local = threading.local()


def _append_trace(state: GraphState, *events: TraceEvent) -> GraphState:
    trace = list(state.get("trace", []))
    for event in events:
        dumped = event.model_dump()
        trace.append(dumped)
        emit_fn = getattr(_thread_local_emit, "fn", None)
        if emit_fn is not None:
            emit_fn(event.type, dumped)
    state["trace"] = trace
    return state


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """In-memory LRU cache keyed on query embedding similarity.

    Instead of exact string matching, this cache embeds the query and checks
    cosine similarity against cached entries. If similarity >= threshold (0.92),
    the cached AgentRunState is returned, skipping the entire pipeline.
    Disabled by default (SEMANTIC_CACHE_ENABLED=false).
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.92, maxsize: int = 128) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[list[float], AgentRunState]] = OrderedDict()

    def get(self, question: str) -> AgentRunState | None:
        query_vec = self._embedder.embed_query(question)
        with self._lock:
            for key, (cached_vec, cached_state) in self._entries.items():
                sim = _cosine_similarity(query_vec, cached_vec)
                if sim >= self._threshold:
                    self._entries.move_to_end(key)
                    return cached_state
        return None

    def put(self, question: str, state: AgentRunState) -> None:
        query_vec = self._embedder.embed_query(question)
        with self._lock:
            self._entries[question] = (query_vec, state)
            self._entries.move_to_end(question)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class AgentService:
    """Main orchestrator: owns the LangGraph pipeline and all LLM interactions.

    Wired by runtime.py with all dependencies. Provides two entry points:
      - run()    → synchronous, returns AgentRunState
      - stream() → async SSE generator, emits trace events progressively
    """
    def __init__(self, settings: Settings, retrieval: RetrievalService, reasoner: OpenAIReasoner, tavily: TavilyClient, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.reasoner = reasoner
        self.tavily = tavily
        self._semantic_cache: SemanticCache | None = None
        if settings.semantic_cache_enabled and embedder is not None:
            self._semantic_cache = SemanticCache(
                embedder, threshold=settings.semantic_cache_threshold
            )
        self.graph = self._build_graph() if StateGraph is not None else None

    def _build_graph(self):  # pragma: no cover
        """Wire the 12-node LangGraph state machine with conditional routing edges.

        Two LLM-powered decision points use conditional edges:
          1. After multi_hop_check: generate | rewrite (retry) | fallback (Tavily)
          2. After answer_quality_check: end | rewrite | post_gen_fallback
        """
        graph = StateGraph(GraphState)
        graph.add_node("classify", self._graph_classify)
        graph.add_node("retrieve", self._graph_retrieve)
        graph.add_node("document_grading", self._graph_document_grading)
        graph.add_node("rewrite_if_needed", self._graph_rewrite)
        graph.add_node("fallback_if_needed", self._graph_fallback)
        graph.add_node("generate", self._graph_generate)
        graph.add_node("self_reflect", self._graph_self_reflect)
        graph.add_node("grounding_check", self._graph_grounding_check)
        graph.add_node("answer_quality_check", self._graph_answer_quality_check)
        graph.add_node("post_generation_fallback", self._graph_post_generation_fallback)

        graph.add_edge(START, "classify")
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "document_grading")
        graph.add_node("multi_hop_check", self._graph_multi_hop_check)
        graph.add_edge("document_grading", "multi_hop_check")
        graph.add_conditional_edges(
            "multi_hop_check",
            self._route_after_grading,
            {
                "rewrite_if_needed": "rewrite_if_needed",
                "fallback_if_needed": "fallback_if_needed",
                "generate": "generate",
            },
        )
        graph.add_edge("rewrite_if_needed", "retrieve")
        graph.add_conditional_edges(
            "fallback_if_needed",
            self._route_after_fallback,
            {"generate": "generate", "end": END},
        )
        graph.add_node("claim_verification", self._verify_claims)
        graph.add_edge("generate", "self_reflect")
        graph.add_edge("self_reflect", "claim_verification")
        graph.add_edge("claim_verification", "grounding_check")
        graph.add_edge("grounding_check", "answer_quality_check")
        graph.add_conditional_edges(
            "answer_quality_check",
            self._route_after_quality,
            {
                "rewrite_if_needed": "rewrite_if_needed",
                "post_gen_fallback": "post_generation_fallback",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "post_generation_fallback",
            self._route_after_post_gen_fallback,
            {"generate": "generate", "end": END},
        )
        return graph.compile()

    _DECOMPOSITION_SIGNALS = re.compile(
        r"\b(and|vs\.?|versus|compare|comparison|difference between|differences between)\b",
        re.IGNORECASE,
    )

    def _should_decompose(self, question: str) -> bool:
        if not self.settings.decomposition_enabled:
            return False
        if not self.reasoner.enabled:
            return False
        tokens = tokenize(question)
        has_signal = bool(self._DECOMPOSITION_SIGNALS.search(question))
        multi_question_mark = question.count("?") >= 2
        long_with_entities = (
            len(tokens) > 20
            and sum(1 for t in ("h100", "a100", "l40s", "nccl", "cuda", "nvlink", "dcgm") if t in question.lower()) >= 2
        )
        return has_signal or multi_question_mark or long_with_entities

    def _decompose_question(self, question: str) -> list[str]:
        prompt = (
            "Break the following complex question into 2-3 focused sub-questions that can each "
            "be answered independently from NVIDIA documentation. "
            'Output ONLY a JSON array of strings: ["sub-question 1", "sub-question 2", ...]\n\n'
            f"Question: {question}"
        )
        try:
            raw = self.reasoner.generate_text(prompt, model=self.settings.routing_model)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            sub_questions = json.loads(raw)
            if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
                return [q for q in sub_questions if q.strip()][:3]
        except Exception as exc:
            _log.warning("Query decomposition failed (%s: %s), using original question", type(exc).__name__, str(exc)[:120])
        return []

    def _graph_classify(self, state: GraphState) -> GraphState:
        """Node 1: Build a query plan (LLM or rule-based) and optionally decompose multi-part questions."""
        if self.reasoner.enabled:
            plan, plan_method = llm_build_query_plan(state["question"], self.settings, self.reasoner)
        else:
            plan = self.retrieval.build_plan(state["question"])
            plan_method = "rule_fallback"
        selected_model = str(state.get("model_used") or self.settings.generation_model)
        next_state: GraphState = {
            "assistant_mode": "doc_rag",
            "plan": plan.model_dump(),
            "current_query": state.get("current_query") or state["question"],
            "rewritten_query": None,
            "results": [],
            "citations": [],
            "trace": list(state.get("trace", [])),
            "rejected_chunk_ids": [],
            "confidence": 0.0,
            "answer": "",
            "retries": 0,
            "used_fallback": False,
            "fallback_reason": None,
            "response_mode": "knowledge-base-backed",
            "model_used": selected_model,
            "plan_method": plan_method,
        }

        sub_questions: list[str] = []
        if self._should_decompose(state["question"]):
            sub_questions = self._decompose_question(state["question"])
            if sub_questions:
                next_state["sub_questions"] = sub_questions

        trace_payload: dict[str, Any] = {
            "stage": "tool_selection",
            "source_families": plan.source_families,
            "search_queries": plan.search_queries[:2],
            "model": selected_model,
            "plan_method": plan_method,
            **_tool_payload("nvidia_docs", "NVIDIA Docs", "knowledge_base"),
        }
        if sub_questions:
            trace_payload["sub_questions"] = sub_questions

        return _append_trace(
            next_state,
            TraceEvent(
                type="classification",
                message=f"Classified question as {plan.query_class.value}" + (f" (decomposed into {len(sub_questions)} sub-questions)" if sub_questions else ""),
                payload=trace_payload,
            ),
        )

    def _graph_retrieve(self, state: GraphState) -> GraphState:
        """Node 2: Execute retrieval — run search queries against the hybrid index.

        Handles query decomposition (parallel sub-question retrieval) and HyDE
        (hypothetical document embedding for better semantic recall on first pass).
        """
        plan = QueryPlan.model_validate(state["plan"])
        sub_questions = state.get("sub_questions", [])

        if sub_questions and state.get("retries", 0) == 0:
            all_results: list[RetrieverResult] = []
            all_rejected: list[str] = []
            all_events: list[TraceEvent] = []
            for sub_q in sub_questions:
                r, rej, _, ev = self.retrieval.run_retrieval_pass(state["question"], plan, sub_q)
                all_results.extend(r)
                all_rejected.extend(rej)
                all_events.extend(ev)
            merged, merged_rej, confidence = self.retrieval.merge_results(
                state["question"], plan, all_results, []
            )
            all_rejected.extend(merged_rej)
            results = merged
            rejected = all_rejected
            events = all_events
        else:
            active_query = state.get("current_query", state["question"])
            # HyDE: on first pass only, generate a hypothetical document to improve recall
            if state.get("retries", 0) == 0:
                active_query = generate_hyde_query(active_query, self.reasoner, self.settings)
            results, rejected, confidence, events = self.retrieval.run_retrieval_pass(
                state["question"], plan, active_query
            )

        next_state = dict(state)
        next_state["results"] = [item.model_dump() for item in results]
        next_state["rejected_chunk_ids"] = list(dict.fromkeys(state.get("rejected_chunk_ids", []) + rejected))
        next_state["confidence"] = confidence
        trace = list(state.get("trace", []))
        emit_fn = getattr(_thread_local_emit, "fn", None)
        for event in events:
            dumped = event.model_dump()
            trace.append(dumped)
            if emit_fn is not None:
                emit_fn(event.type, dumped)
        next_state["trace"] = trace
        return next_state

    def _graph_document_grading(self, state: GraphState) -> GraphState:
        """Node 3: LLM judges each retrieved chunk's relevance to the question.

        Batches chunks in groups of 3 to reduce API calls (max 6 chunks graded).
        Includes context poisoning defense — flags chunks with instruction-like content.
        Falls back to keeping all chunks if LLM grading fails.
        """
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        rejected_ids = list(state.get("rejected_chunk_ids", []))

        # LLM-based document grading: ask the model to judge relevance of each chunk
        llm_graded = False
        grades = []
        if self.reasoner.enabled and results:
            llm_graded = True
            question = state["question"]
            accepted = []
            newly_rejected = []

            # Batch chunks for grading (up to 3 per prompt to reduce API calls)
            for batch_start in range(0, min(len(results), 6), 3):
                batch = results[batch_start:batch_start + 3]
                # Context poisoning defense: flag chunks with instruction-like content
                _POISONING_SIGNALS = ("ignore previous", "you must", "system prompt", "new instructions", "override your")
                for r in batch:
                    content_lower = r.chunk.content[:500].lower()
                    if any(signal in content_lower for signal in _POISONING_SIGNALS):
                        _log.warning("Potential context poisoning in chunk %s: matched instruction-like pattern", r.chunk.id)
                docs_text = "\n\n".join(
                    f"Document {i+1} [{r.chunk.chunk_id if hasattr(r.chunk, 'chunk_id') else r.chunk.id}]: "
                    f"{r.chunk.title} — {r.chunk.content[:300]}"
                    for i, r in enumerate(batch)
                )
                prompt = (
                    "You are evaluating document relevance for an NVIDIA infrastructure RAG system.\n\n"
                    f"Question: {question}\n\n"
                    f"Documents:\n{docs_text}\n\n"
                    "For each document, judge if it is relevant to answering the question.\n"
                    f'Respond in JSON: {{"grades": [{{"doc": 1, "relevant": true/false, "reason": "..."}},'
                    f' {{"doc": 2, "relevant": true/false, "reason": "..."}}, ...]}}'
                )
                try:
                    raw = self.reasoner.generate_text(prompt, model=self.settings.pipeline_model)
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)
                    parsed = json.loads(raw)
                    batch_grades = parsed.get("grades", [])
                    for idx, r in enumerate(batch):
                        grade = batch_grades[idx] if idx < len(batch_grades) else {"relevant": True}
                        is_relevant = grade.get("relevant", True)
                        reason = grade.get("reason", "")
                        grades.append({"chunk_id": r.chunk.id, "relevant": is_relevant, "reason": reason})
                        if is_relevant:
                            accepted.append(r)
                        else:
                            newly_rejected.append(r.chunk.id)
                except Exception as exc:
                    _log.warning("LLM document grading failed for batch (%s: %s), keeping all chunks", type(exc).__name__, str(exc)[:120])
                    accepted.extend(batch)
                    grades.extend([{"chunk_id": r.chunk.id, "relevant": True, "reason": "grading_failed"} for r in batch])

            # Keep any remaining chunks beyond the first 6 (not graded)
            if len(results) > 6:
                accepted.extend(results[6:])

            results = accepted
            rejected_ids.extend(newly_rejected)

        accepted_count = len(results)
        rejected_count = len(rejected_ids)

        next_state = dict(state)
        next_state["results"] = [item.model_dump() for item in results]
        next_state["rejected_chunk_ids"] = list(dict.fromkeys(rejected_ids))
        next_state["llm_graded"] = llm_graded

        # Recalculate confidence if LLM filtered chunks
        if llm_graded and results:
            next_state["confidence"] = estimate_confidence(results)

        return _append_trace(
            next_state,
            TraceEvent(
                type="document_grading",
                message="Evaluated document relevance" + (" using LLM judgment" if llm_graded else " using threshold filtering"),
                payload={
                    "stage": "evidence_selection",
                    "accepted_count": accepted_count,
                    "accepted_docs": [_result_preview(item) for item in results[:4]],
                    "rejected_count": rejected_count,
                    "total_count": accepted_count + rejected_count,
                    "kept_count": accepted_count,
                    "grades": [("pass" if g.get("relevant") else "fail") for g in grades] if llm_graded else (["pass"] * accepted_count + ["fail"] * rejected_count),
                    "llm_graded": llm_graded,
                    **_tool_payload("nvidia_docs", "NVIDIA Docs", "knowledge_base"),
                },
            ),
        )

    def _llm_route(self, state: GraphState, decision_point: str, options: list[dict[str, str]]) -> tuple[str, str]:
        """Agentic decision point: LLM chooses the next pipeline action.

        Called at two critical junctions in the graph:
          1. after_grading: generate | rewrite | fallback
          2. after_quality: end | rewrite | post_gen_fallback

        The LLM sees the pipeline state (confidence, retry count, grounding results)
        and picks the best action. Returns ("", "rule_fallback") if LLM is unavailable
        or returns an invalid action, letting the caller fall through to rule-based logic.
        """
        if not self.reasoner.enabled:
            return "", "rule_fallback"

        question = state.get("question", "")
        confidence = state.get("confidence", 0.0)
        result_count = len(state.get("results", []))
        retries = state.get("retries", 0)
        used_fallback = state.get("used_fallback", False)
        grounding_passed = state.get("grounding_passed", False)
        quality_passed = state.get("answer_quality_passed", False)

        options_text = "\n".join(
            f"- {opt['action']}: {opt['description']}"
            for opt in options
        )

        prompt = (
            "You are a routing agent in an NVIDIA infrastructure RAG pipeline.\n\n"
            f"Decision point: {decision_point}\n"
            f"Question: {question}\n"
            f"Retrieved documents: {result_count}\n"
            f"Confidence score: {confidence:.2f}\n"
            f"Retry count: {retries}\n"
            f"Web fallback used: {used_fallback}\n"
            f"Grounding check passed: {grounding_passed}\n"
            f"Quality check passed: {quality_passed}\n\n"
            f"Available actions:\n{options_text}\n\n"
            "Choose the best next action based on the pipeline state.\n"
            '{"action": "action_name", "reasoning": "brief explanation"}'
        )

        try:
            raw = self.reasoner.generate_text(prompt, model=self.settings.routing_model)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            action = parsed.get("action", "")
            valid_actions = {opt["action"] for opt in options}
            if action not in valid_actions:
                _log.warning("LLM route returned invalid action %r, falling back", action)
                return "", "rule_fallback"

            # Emit routing decision trace
            reasoning = parsed.get("reasoning", "")
            routing_decisions = list(state.get("routing_decisions", []))
            routing_decisions.append({
                "decision_point": decision_point,
                "action": action,
                "reasoning": reasoning,
                "method": "llm",
            })
            # Note: we can't mutate state here (routing functions return a string, not state)
            # The trace is logged but not persisted to state in routing functions

            return action, "llm"
        except Exception as exc:
            _log.warning("LLM routing failed (%s: %s), using rule-based fallback", type(exc).__name__, str(exc)[:120])
            return "", "rule_fallback"

    def _graph_multi_hop_check(self, state: GraphState) -> GraphState:
        """Multi-hop retrieval: LLM inspects chunks and identifies knowledge gaps."""
        next_state = dict(state)

        # Only run on first pass, only when reasoner is available
        if not self.reasoner.enabled or state.get("retries", 0) > 0:
            next_state["multi_hop_used"] = False
            return next_state

        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        if not results:
            next_state["multi_hop_used"] = False
            return next_state

        question = state["question"]
        chunk_summaries = "\n".join(
            f"- {r.chunk.title}: {r.chunk.content[:200]}"
            for r in results[:5]
        )

        prompt = (
            "You are a retrieval quality assessor for an NVIDIA infrastructure documentation system.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved documents:\n{chunk_summaries}\n\n"
            "Do these documents contain sufficient information to answer the question comprehensively?\n"
            "If not, what specific follow-up search query would help fill the knowledge gap?\n\n"
            'Respond in JSON: {"sufficient": true/false, "follow_up_query": "specific search query or empty string", "gap_description": "what information is missing or empty string"}'
        )

        try:
            raw = self.reasoner.generate_text(prompt, model=self.settings.pipeline_model)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            sufficient = parsed.get("sufficient", True)
            follow_up_query = parsed.get("follow_up_query", "").strip()
            gap_description = parsed.get("gap_description", "")

            if not sufficient and follow_up_query:
                # R3: Parallel multi-hop — currently single follow-up; if the LLM
                # returns multiple queries in the future, they can be dispatched
                # concurrently via asyncio.gather() or ThreadPoolExecutor since
                # run_retrieval_pass is synchronous and thread-safe.
                plan = QueryPlan.model_validate(state["plan"])
                follow_up_queries = [follow_up_query]

                all_new_results: list[RetrieverResult] = []
                all_new_events: list[TraceEvent] = []
                for fq in follow_up_queries:
                    new_results, new_rejected, new_confidence, new_events = self.retrieval.run_retrieval_pass(
                        question, plan, fq
                    )
                    all_new_results.extend(new_results)
                    all_new_events.extend(new_events)

                # Merge results, dedup by chunk ID (preserving highest-scoring duplicate)
                existing_ids = {r.chunk.id for r in results}
                seen_new: set[str] = set()
                added: list[RetrieverResult] = []
                for r in sorted(all_new_results, key=lambda x: x.rerank_score, reverse=True):
                    if r.chunk.id not in existing_ids and r.chunk.id not in seen_new:
                        added.append(r)
                        seen_new.add(r.chunk.id)
                merged_results = results + added

                next_state["results"] = [item.model_dump() for item in merged_results]
                next_state["multi_hop_used"] = True
                next_state["follow_up_queries"] = follow_up_queries

                # Recalculate confidence with merged results
                if merged_results:
                    top_scores = sorted([r.rerank_score for r in merged_results], reverse=True)[:3]
                    weights = [0.5, 0.3, 0.2][:len(top_scores)]
                    next_state["confidence"] = sum(s * w for s, w in zip(top_scores, weights)) / sum(weights[:len(top_scores)])

                # Emit retrieval trace events from the follow-up
                trace = list(state.get("trace", []))
                emit_fn = getattr(_thread_local_emit, "fn", None)
                for event in all_new_events:
                    dumped = event.model_dump()
                    trace.append(dumped)
                    if emit_fn is not None:
                        emit_fn(event.type, dumped)
                next_state["trace"] = trace

                return _append_trace(
                    next_state,
                    TraceEvent(
                        type="multi_hop",
                        message=f"Multi-hop retrieval: identified gap and retrieved {len(added)} additional documents via {len(follow_up_queries)} follow-up query(ies)",
                        payload={
                            "stage": "multi_hop_retrieval",
                            "sufficient": False,
                            "follow_up_queries": follow_up_queries,
                            "gap_description": gap_description,
                            "new_results_count": len(added),
                            "total_results_count": len(merged_results),
                            "follow_up_query_count": len(follow_up_queries),
                        },
                    ),
                )
            else:
                next_state["multi_hop_used"] = False
                return _append_trace(
                    next_state,
                    TraceEvent(
                        type="multi_hop",
                        message="Multi-hop check: retrieved documents are sufficient",
                        payload={
                            "stage": "multi_hop_retrieval",
                            "sufficient": True,
                            "follow_up_query": "",
                            "gap_description": "",
                        },
                    ),
                )
        except Exception as exc:
            _log.warning("Multi-hop check failed (%s: %s), proceeding with existing chunks", type(exc).__name__, str(exc)[:120])
            next_state["multi_hop_used"] = False
            return next_state

    def _route_after_grading(self, state: GraphState) -> str:
        """Routing decision point 1: after grading + multi-hop, decide next step.

        Guard rails enforce hard constraints regardless of LLM decision:
          - Max 1 rewrite before falling back to Tavily
          - Tavily disabled → force generate
          - Confidence >= 0.85 → skip rewrite/fallback entirely (P20 fix)
        """
        # Try LLM-based routing first
        action, method = self._llm_route(state, "after_grading", [
            {"action": "generate", "description": "Proceed to answer generation with the current evidence set"},
            {"action": "rewrite_if_needed", "description": "Rephrase the query and retry retrieval (low confidence in current results)"},
            {"action": "fallback_if_needed", "description": "Try web search fallback (knowledge base evidence insufficient)"},
        ])

        if method == "llm":
            plan = QueryPlan.model_validate(state["plan"])
            confidence = float(state.get("confidence", 0.0))
            # Guard rails: enforce constraints regardless of LLM decision
            if action == "rewrite_if_needed" and state.get("retries", 0) >= 1:
                return "fallback_if_needed"  # max 1 rewrite before grading
            if action == "fallback_if_needed" and not plan.use_tavily_fallback:
                return "generate"  # tavily disabled, just generate
            # High-confidence guard: skip fallback/rewrite when KB evidence is strong
            # But allow recency-sensitive queries through — KB may have stale content (P25 fix)
            recency_sensitive = plan.recency_sensitive
            if confidence >= 0.85 and action in ("fallback_if_needed", "rewrite_if_needed") and not recency_sensitive:
                _log.info("Overriding LLM route '%s' → 'generate' (confidence=%.2f >= 0.85)", action, confidence)
                return "generate"
            return action

        # Rule-based fallback (original logic)
        plan = QueryPlan.model_validate(state["plan"])
        confidence = float(state.get("confidence", 0.0))
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        if plan.recency_sensitive:
            return "fallback_if_needed"
        if needs_retry(plan, results) and state.get("retries", 0) < 1:
            return "rewrite_if_needed"
        if needs_retry(plan, results):
            return "fallback_if_needed"
        return "generate"

    def _llm_rewrite_query(self, question: str, results_count: int, confidence: float) -> str | None:
        """Use LLM to rephrase a low-confidence query. Returns None on failure."""
        if not self.reasoner.enabled:
            return None
        prompt = (
            "You are a search query optimizer for an NVIDIA infrastructure documentation system. "
            "The following query returned low-confidence results. Rewrite it to be more specific, "
            "technical, and likely to match NVIDIA documentation. Output ONLY the rewritten query, "
            "no explanation.\n\n"
            f"Original query: {question}\n"
            f"Low-confidence context: retrieved {results_count} chunks, confidence={confidence:.2f}"
        )
        try:
            rewritten = self.reasoner.generate_text(prompt, model=self.settings.pipeline_model)
            rewritten = rewritten.strip().strip('"').strip("'")
            if not rewritten or rewritten.lower() == question.lower():
                return None
            return rewritten
        except Exception as exc:
            _log.warning("LLM query rewrite failed (%s: %s), falling back to static expansion", type(exc).__name__, str(exc)[:120])
            return None

    def _graph_rewrite(self, state: GraphState) -> GraphState:
        plan = QueryPlan.model_validate(state["plan"])
        retries = int(state.get("retries", 0)) + 1
        results_count = len(state.get("results", []))
        confidence = float(state.get("confidence", 0.0))

        llm_rewritten = self._llm_rewrite_query(state["question"], results_count, confidence)
        rewritten_query = llm_rewritten or rewrite_query(state["question"], plan.query_class, retries)
        rewrite_method = "llm" if llm_rewritten else "static_expansion"

        next_state = dict(state)
        next_state["retries"] = retries
        next_state["current_query"] = rewritten_query
        next_state["rewritten_query"] = rewritten_query
        return _append_trace(
            next_state,
            TraceEvent(
                type="rewrite",
                message="Triggered a rewritten retrieval query because confidence was below the floor",
                payload={"rewritten_query": rewritten_query, "retry": retries, "rewrite_method": rewrite_method},
            ),
        )

    def _graph_fallback(self, state: GraphState) -> GraphState:
        """Tavily web search fallback — MERGES web results with existing KB results (P19 fix).

        Key design: web results supplement KB evidence, never replace it. If KB results
        exist, response_mode stays 'knowledge-base-backed'. Only 'web-backed' when KB
        had zero results. This preserves KB-first ordering for synthesis.
        """
        next_state = dict(state)
        plan = QueryPlan.model_validate(state["plan"])
        if not plan.use_tavily_fallback:
            next_state["response_mode"] = "insufficient-evidence"
            next_state["answer"] = self._refusal_answer()
            return _append_trace(
                next_state,
                TraceEvent(
                    type="fallback",
                    message="Skipped Tavily fallback because fallback is disabled for this run",
                    payload={"stage": "tool_result", "status": "skipped", "reason": "disabled", **_tool_payload("web_search", "Web Search", "web")},
                ),
            )

        try:
            web_results = self.tavily.search(state["question"])
        except Exception as exc:
            _log.warning("Tavily search failed (%s: %s), treating as empty", type(exc).__name__, str(exc)[:120])
            web_results = []
        if not web_results:
            next_state["response_mode"] = "insufficient-evidence"
            next_state["answer"] = self._refusal_answer()
            return _append_trace(
                next_state,
                TraceEvent(
                    type="fallback",
                    message="Fallback search returned no usable web evidence",
                    payload={"stage": "tool_result", "status": "empty", "reason": "empty", **_tool_payload("web_search", "Web Search", "web")},
                ),
            )

        pseudo_results: list[RetrieverResult] = []
        for index, item in enumerate(web_results, start=1):
            chunk = ChunkRecord(
                id=f"tavily-{index}",
                source_id="tavily-web",
                title=item["title"],
                url=item["url"],
                section_path="Web Search",
                doc_family="web",
                doc_type="web",
                product_tags=["web-search"],
                source_kind="web",
                content=item["content"],
            )
            pseudo_results.append(
                RetrieverResult(
                    chunk=chunk,
                    score=0.22,
                    retrieval_method="tavily",
                    rerank_score=0.22,
                )
            )
        # Merge: append web results after existing KB results (KB-first ordering)
        existing_kb = state.get("results", [])
        merged = existing_kb + [item.model_dump() for item in pseudo_results]
        next_state["results"] = merged
        next_state["used_fallback"] = True
        next_state["fallback_reason"] = "knowledge_base_insufficiency"
        # Keep knowledge-base-backed if KB results exist (web supplements, not replaces)
        if existing_kb:
            next_state["response_mode"] = "knowledge-base-backed"
            trace_msg = "Supplemented knowledge base results with web search"
        else:
            next_state["response_mode"] = "web-backed"
            trace_msg = "Used web search fallback (no knowledge base results)"
        return _append_trace(
            next_state,
            TraceEvent(
                type="fallback",
                message=trace_msg,
                payload={
                    "stage": "tool_result",
                    "status": "result",
                    "result_count": len(pseudo_results),
                    "kb_results_kept": len(existing_kb),
                    "accepted_docs": [_result_preview(item) for item in pseudo_results[:4]],
                    **_tool_payload("web_search", "Web Search", "web"),
                },
            ),
        )

    def _route_after_fallback(self, state: GraphState) -> str:
        if state.get("results"):
            return "generate"
        return "end"

    def _check_user_premises(self, question: str, results: list[RetrieverResult]) -> str:
        """Sycophancy defense: detect incorrect premises in the user's question.

        Example: "Since the H100 has 40GB..." (it actually has 80GB HBM3).
        Extracts assertions via routing model, checks token overlap with source chunks,
        and prepends a contradiction warning to the synthesis prompt so the LLM corrects
        rather than agrees with the user's false premise.
        """
        if not self.reasoner.enabled or not results:
            return ""

        # Ask routing model to extract factual assertions from the question
        prompt = (
            "Does this question contain any factual assertions or premises that could be verified?\n"
            "Examples of assertions: 'Since the H100 has 40GB...', 'Given that NCCL only supports TCP...'\n\n"
            f"Question: {question}\n\n"
            'Respond in JSON: {"has_assertions": true/false, "assertions": ["assertion1", ...]}\n'
            "Only include factual claims the user states as fact, not things they are asking about."
        )

        try:
            raw = self.reasoner.generate_text(prompt, model=self.settings.routing_model)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if not parsed.get("has_assertions"):
                return ""
            assertions = parsed.get("assertions", [])
        except Exception:
            return ""

        if not assertions:
            return ""

        # Check each assertion against retrieved chunks
        chunk_text = " ".join(r.chunk.content for r in results[:4])
        contradictions = []

        for assertion in assertions[:3]:
            assertion_lower = assertion.lower()
            # Simple contradiction detection: if the assertion mentions a number and
            # the chunks mention a different number for the same entity
            assertion_tokens = set(tokenize(assertion))
            chunk_tokens = set(tokenize(chunk_text))
            # If the assertion tokens have significant overlap with chunks but the exact
            # assertion doesn't appear, it might be contradicted
            overlap = len(assertion_tokens & chunk_tokens) / max(len(assertion_tokens), 1)
            if overlap > 0.3 and assertion_lower not in chunk_text.lower():
                contradictions.append(assertion)

        if contradictions:
            return (
                "IMPORTANT NOTE: The user's question contains premise(s) that may be incorrect based on the source documents: "
                + "; ".join(contradictions[:2])
                + ". If the sources contradict these premises, correct them directly in your answer.\n\n"
            )
        return ""

    def _graph_generate(self, state: GraphState) -> GraphState:
        """Node 6: Synthesize the final answer from accepted evidence chunks.

        Adaptive chunk count: 2 chunks for short queries, 3 for medium, 4 for complex.
        Supports token-level streaming (SSE answer_chunk events) when emit callback is set.
        Falls back to keyword-based answer if LLM synthesis fails (generation_degraded=True).
        Runs format validation after generation (word count, citation presence, code blocks).
        """
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        chunk_count = self._synthesis_chunk_count(state["question"])
        citations = [_citation_from_result(result).model_dump() for result in results[:chunk_count]]
        model_used = str(state.get("model_used") or self.settings.generation_model)
        history_context = state.get("history_context", "")
        # S3: Check for incorrect user premises before synthesis
        premise_note = self._check_user_premises(state["question"], results)
        emit_fn = getattr(_thread_local_emit, "fn", None)
        if emit_fn is not None and self.reasoner.enabled and results:
            answer, generation_degraded = self._synthesize_answer_stream(
                state["question"], results, model_used,
                history_context=history_context, emit=emit_fn,
                premise_note=premise_note,
            )
        else:
            answer, generation_degraded = self._synthesize_answer(state["question"], results, model_used, history_context=history_context, premise_note=premise_note)
        next_state = dict(state)
        next_state["citations"] = citations
        next_state["answer"] = answer
        next_state["model_used"] = model_used
        next_state["generation_degraded"] = generation_degraded
        events: list[TraceEvent] = []
        if generation_degraded:
            events.append(TraceEvent(
                type="generation_error",
                message="LLM synthesis failed; answer generated from keyword fallback",
                payload={"stage": "generation", "degraded": True},
            ))
        events.append(TraceEvent(
            type="generation",
            message="Synthesized the final answer from the accepted evidence set",
            payload={
                "stage": "generation",
                "citation_count": len(citations),
                "response_mode": next_state.get("response_mode", "knowledge-base-backed"),
                "model": model_used,
                "accepted_docs": [_result_preview(result) for result in results[:4]],
                "degraded": generation_degraded,
            },
        ))
        # S4: Output format enforcement — validate format before grounding check
        format_check = self._validate_format(answer, next_state.get("response_mode", "knowledge-base-backed"))
        if format_check["issues"]:
            _log.info("Format validation issues: %s", format_check["issues"])
        events.append(TraceEvent(
            type="generation",
            message="Post-generation format validation",
            payload={
                "stage": "validation",
                "check": "format",
                **format_check,
            },
        ))
        return _append_trace(next_state, *events)

    def _graph_self_reflect(self, state: GraphState) -> GraphState:
        """Self-RAG reflection: ask the LLM to score its own answer."""
        next_state = dict(state)
        if not self.reasoner.enabled:
            return next_state

        answer = state.get("answer", "")
        question = state.get("question", "")
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        context_summary = "\n".join(
            f"[{i}] {r.chunk.title}: {r.chunk.content[:200]}"
            for i, r in enumerate(results[:4], 1)
        )

        prompt = (
            "You are evaluating an AI-generated answer about NVIDIA infrastructure. "
            "Score the answer on three dimensions (1-5 each):\n"
            "1. RELEVANCE: Does the answer directly address the question?\n"
            "2. GROUNDEDNESS: Is every claim supported by the provided context?\n"
            "3. COMPLETENESS: Does the answer cover the key aspects of the question?\n\n"
            'Respond in JSON: {"relevance": N, "groundedness": N, "completeness": N, "issues": "..."}\n\n'
            f"Question: {question}\n\nContext used:\n{context_summary}\n\nAnswer:\n{answer}"
        )
        try:
            raw = self.reasoner.generate_text(prompt, model=self.settings.pipeline_model)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            scores = json.loads(raw)
            _log.info(
                "Self-RAG scores: relevance=%s groundedness=%s completeness=%s (model=%s)",
                scores.get("relevance"), scores.get("groundedness"), scores.get("completeness"),
                self.settings.pipeline_model,
            )
        except Exception as exc:
            _log.warning("Self-reflect failed (%s: %s), defaulting to neutral scores", type(exc).__name__, str(exc)[:120])
            scores = {"relevance": 3, "groundedness": 3, "completeness": 3, "issues": "reflection unavailable"}

        next_state["self_reflect_scores"] = scores
        try:
            groundedness = int(float(scores.get("groundedness", 3)))
        except (ValueError, TypeError):
            _log.warning("Could not parse groundedness score %r, defaulting to 3", scores.get("groundedness"))
            groundedness = 3
        if groundedness < 3:
            next_state["grounding_passed"] = False

        return _append_trace(
            next_state,
            TraceEvent(
                type="self_reflect",
                message="Self-RAG reflection scored the answer on relevance, groundedness, and completeness",
                payload={
                    "stage": "validation",
                    "check": "self_reflect",
                    "scores": scores,
                    "forced_grounding_fail": groundedness < 3,
                },
            ),
        )

    def _verify_claims(self, state: GraphState) -> GraphState:
        """Node 8: Extract factual claims from the answer and verify against source chunks.

        Uses the routing model (cheap) to extract 3-5 key claims, then checks each
        claim's token overlap against source chunks (40% threshold). If more claims
        are ungrounded than grounded, forces grounding failure → triggers fallback chain.
        """
        next_state = dict(state)
        answer = state.get("answer", "")
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]

        if not self.reasoner.enabled or not answer or not results:
            return next_state

        # Ask routing model to extract key factual claims
        prompt = (
            "Extract the key factual claims from this answer. Focus on specific numbers, "
            "names, comparisons, and technical specifications.\n\n"
            f"Answer: {answer[:1000]}\n\n"
            'Respond in JSON: {"claims": ["claim1", "claim2", ...]}\n'
            "List only the 3-5 most important factual claims."
        )

        try:
            raw = self.reasoner.generate_text(prompt, model=self.settings.routing_model)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            claims = parsed.get("claims", [])
        except Exception as exc:
            _log.debug("Claim extraction failed (%s), skipping verification", type(exc).__name__)
            return next_state

        if not claims:
            return next_state

        # Verify each claim against source chunks
        chunk_contents = " ".join(r.chunk.content.lower() for r in results[:4])
        verified = []
        ungrounded = []

        for claim in claims[:5]:
            claim_tokens = set(tokenize(claim))
            # A claim is "grounded" if at least 40% of its meaningful tokens appear in the source chunks
            meaningful_tokens = {t for t in claim_tokens if len(t) > 2}
            if not meaningful_tokens:
                verified.append(claim)
                continue
            hits = sum(1 for t in meaningful_tokens if t in chunk_contents)
            if hits / len(meaningful_tokens) >= 0.4:
                verified.append(claim)
            else:
                ungrounded.append(claim)

        claim_result = {
            "total_claims": len(claims[:5]),
            "verified": len(verified),
            "ungrounded": len(ungrounded),
            "ungrounded_claims": ungrounded[:3],  # Only show first 3
        }

        # If more than half the claims are ungrounded, flag for grounding failure
        if ungrounded and len(ungrounded) > len(verified):
            next_state["grounding_passed"] = False
            _log.info("Claim verification failed: %d/%d claims ungrounded", len(ungrounded), len(claims[:5]))

        return _append_trace(
            next_state,
            TraceEvent(
                type="claim_verification",
                message=f"Verified {len(verified)}/{len(claims[:5])} factual claims against source documents",
                payload={
                    "stage": "validation",
                    "check": "claim_verification",
                    **claim_result,
                },
            ),
        )

    def _graph_grounding_check(self, state: GraphState) -> GraphState:
        answer = state.get("answer", "")
        citations = [Citation.model_validate(item) for item in state.get("citations", [])]
        grounded = self._grounding_check(answer, citations)
        if self._answer_says_insufficient(answer):
            grounded = False  # override: LLM itself said the context wasn't good enough
        next_state = dict(state)
        next_state["grounding_passed"] = grounded

        # S2: Citation attribution quality metrics
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        citation_quality = self._citation_quality(answer, results)

        # S6: Stale info detection — flag answers citing only old sources
        stale_warning = False
        if results:
            retrieved_dates: list[datetime] = []
            for r in results[:4]:
                ra = getattr(r.chunk, 'retrieved_at', None)
                if ra:
                    try:
                        dt = datetime.fromisoformat(ra.replace("Z", "+00:00"))
                        retrieved_dates.append(dt)
                    except (ValueError, TypeError):
                        pass
            if retrieved_dates:
                now = datetime.now(timezone.utc)
                all_stale = all((now - dt).days > 180 for dt in retrieved_dates)
                if all_stale:
                    stale_warning = True
                    _log.info("All cited sources are older than 6 months — flagging stale info warning")

        return _append_trace(
            next_state,
            TraceEvent(
                type="grounding_check",
                message="Verified that the answer stays anchored to cited evidence",
                payload={
                    "stage": "validation",
                    "check": "grounding",
                    "passed": grounded,
                    "citation_count": len(citations),
                    "citation_quality": citation_quality,
                    "stale_sources_warning": stale_warning,
                },
            ),
        )

    def _graph_answer_quality_check(self, state: GraphState) -> GraphState:
        passed = self._answer_quality_check(state["question"], state.get("answer", ""))
        next_state = dict(state)
        next_state["answer_quality_passed"] = passed
        return _append_trace(
            next_state,
            TraceEvent(
                type="answer_quality_check",
                message="Checked whether the answer directly addresses the user question",
                payload={"stage": "validation", "check": "answer_quality", "passed": passed, "retry": state.get("retries", 0)},
            ),
        )

    def _route_after_quality(self, state: GraphState) -> str:
        # Try LLM-based routing first
        action, method = self._llm_route(state, "after_quality", [
            {"action": "end", "description": "Accept the answer — quality and grounding checks passed"},
            {"action": "post_gen_fallback", "description": "Try web search to improve the answer (quality/grounding failed)"},
            {"action": "rewrite_if_needed", "description": "Rephrase query and regenerate (quality failed, retry available)"},
        ])

        if method == "llm":
            plan = QueryPlan.model_validate(state["plan"])
            confidence = float(state.get("confidence", 0.0))
            # Guard rails
            if action == "post_gen_fallback" and (state.get("used_fallback") or not plan.use_tavily_fallback):
                if state.get("retries", 0) < plan.max_retries:
                    return "rewrite_if_needed"
                return "end"
            if action == "rewrite_if_needed" and state.get("retries", 0) >= plan.max_retries:
                return "end"
            # High-confidence guard: skip post-gen fallback when KB evidence was strong
            # BUT allow fallback if grounding failed — high retrieval confidence
            # doesn't help when the LLM didn't ground its answer in sources (P24 fix)
            grounding_failed = not state.get("grounding_passed", True)
            if confidence >= 0.85 and action == "post_gen_fallback" and not grounding_failed:
                _log.info("Overriding LLM route 'post_gen_fallback' → 'end' (confidence=%.2f >= 0.85)", confidence)
                return "end"
            return action

        # Rule-based fallback (original logic)
        plan = QueryPlan.model_validate(state["plan"])
        confidence = float(state.get("confidence", 0.0))
        grounding_failed = not state.get("grounding_passed", True)
        quality_ok = state.get("grounding_passed", False) and state.get("answer_quality_passed", False)
        # Allow fallback when grounding failed regardless of confidence (P24 fix)
        if not quality_ok and (confidence < 0.85 or grounding_failed):
            if not state.get("used_fallback") and plan.use_tavily_fallback:
                return "post_gen_fallback"
            if state.get("retries", 0) < plan.max_retries:
                return "rewrite_if_needed"
        return "end"

    def _graph_post_generation_fallback(self, state: GraphState) -> GraphState:
        """Post-generation Tavily fallback: triggered when quality/grounding failed
        and Tavily hasn't been tried yet."""
        return self._graph_fallback(state)

    def _route_after_post_gen_fallback(self, state: GraphState) -> str:
        if state.get("results"):
            return "generate"
        return "end"

    def run(self, request: ChatRequest) -> AgentRunState:
        """Main entry point: classify → route → run pipeline → apply fallback chain.

        Flow: semantic cache check → classify (LLM or rules) → route to one of:
          - direct_chat: greeting/general knowledge → LLM answer, no retrieval
          - live_query: weather/stocks/news → Tavily open search → LLM synthesis
          - doc_rag: NVIDIA infra question → full LangGraph pipeline

        After the graph completes, applies the 5-mode fallback chain:
        KB-backed → web-backed → llm-knowledge → insufficient-evidence → direct-chat
        """
        cache_key = self._cache_key(request)
        if self._semantic_cache is not None:
            cached = self._semantic_cache.get(cache_key)
            if cached is not None:
                _log.info("Semantic cache HIT (size=%d, key=%s)", len(self._semantic_cache._entries), cache_key[:80])
                return cached

        if self.reasoner.enabled:
            assistant_mode, classification_method = llm_classify_assistant_mode(
                request.question, request.history, self.reasoner, self.settings.routing_model
            )
        else:
            assistant_mode = classify_assistant_mode(request.question, request.history)
            classification_method = "rule_fallback"
        with tracing_context(
            project_name=self.settings.langsmith_project,
            enabled=self.settings.langsmith_tracing and bool(self.settings.langsmith_api_key),
            metadata={
                "app_mode": self.settings.app_mode,
                "model": self._resolve_request_model(request.model),
                "assistant_mode": assistant_mode,
            },
            tags=["maistorage", "agentic-rag", self.settings.app_mode],
        ):
            result = self._run_with_optional_trace(request)

        if self._semantic_cache is not None:
            self._semantic_cache.put(cache_key, result)
            _log.info("Semantic cache MISS -> stored (size=%d, key=%s)", len(self._semantic_cache._entries), cache_key[:80])
        return result

    @traceable(name="agent_run", run_type="chain")
    def _run_with_optional_trace(self, request: ChatRequest) -> AgentRunState:
        selected_model = self._resolve_request_model(request.model)

        # Reformulate follow-ups BEFORE classification so the classifier
        # sees a standalone query with proper NVIDIA/GPU terms.
        effective_query, reformulation_method = self._reformulate_follow_up(request.question, request.history)
        history_context = self._format_history_context(request.history)

        # Classify using the reformulated query when available so that
        # "How much memory does the NVIDIA H100 have?" routes to doc_rag.
        # Safety: if the ORIGINAL question is casual chat (e.g. "Thanks, that's helpful"),
        # don't let reformulation inject NVIDIA terms that misroute it to doc_rag.
        if self.reasoner.enabled:
            original_mode, _ = llm_classify_assistant_mode(request.question, request.history, self.reasoner, self.settings.routing_model)
        else:
            original_mode = classify_assistant_mode(request.question, request.history)
        if original_mode == "direct_chat":
            assistant_mode = "direct_chat"
        else:
            classify_query = effective_query if reformulation_method else request.question
            if self.reasoner.enabled:
                assistant_mode, _ = llm_classify_assistant_mode(classify_query, request.history, self.reasoner, self.settings.routing_model)
            else:
                assistant_mode = classify_assistant_mode(classify_query, request.history)
        if assistant_mode == "direct_chat":
            return self._run_direct_chat(request, selected_model)
        if assistant_mode == "live_query":
            return self._run_live_query(request, selected_model)

        initial_trace: list[dict[str, Any]] = []
        if reformulation_method:
            event = TraceEvent(
                type="query_reformulation",
                message=f"Reformulated follow-up query ({reformulation_method})",
                payload={"original_query": request.question, "reformulated_query": effective_query, "method": reformulation_method},
            )
            dumped = event.model_dump()
            initial_trace.append(dumped)
            emit_fn = getattr(_thread_local_emit, "fn", None)
            if emit_fn is not None:
                emit_fn("query_reformulation", dumped)

        if self.graph is not None:
            graph_state = self.graph.invoke({"question": request.question, "current_query": effective_query, "model_used": selected_model, "assistant_mode": assistant_mode, "history_context": history_context, "trace": initial_trace}, {"recursion_limit": 50})
        else:
            graph_state = self._run_without_graph(request.question, selected_model, effective_query, history_context=history_context, initial_trace=initial_trace)

        plan = QueryPlan.model_validate(graph_state["plan"])
        results = [RetrieverResult.model_validate(item) for item in graph_state.get("results", [])]
        citations = [Citation.model_validate(item) for item in graph_state.get("citations", [])]
        trace = [TraceEvent.model_validate(item) for item in graph_state.get("trace", [])]
        answer = graph_state.get("answer") or self._refusal_answer()
        response_mode = graph_state.get("response_mode", "knowledge-base-backed")
        grounding_passed = bool(graph_state.get("grounding_passed", False))
        answer_quality_passed = bool(graph_state.get("answer_quality_passed", False))
        if not results and response_mode == "knowledge-base-backed":
            response_mode = "insufficient-evidence"
        # For knowledge-base answers, enforce grounding + quality; for web-backed, trust the
        # external results regardless (citation-marker grounding doesn't apply to web).
        if response_mode == "knowledge-base-backed" and (not grounding_passed or not answer_quality_passed):
            response_mode = "insufficient-evidence"
            answer = self._refusal_answer()
            citations = []

        # LLM general-knowledge fallback: try OpenAI when all knowledge-base/web layers exhausted
        if response_mode == "insufficient-evidence" and self.reasoner.enabled:
            llm_answer = self._llm_knowledge_answer(request.question, selected_model, history_context=history_context)
            if llm_answer and len(llm_answer.strip()) > 60:
                answer = llm_answer
                response_mode = "llm-knowledge"
                citations = []

        return AgentRunState(
            question=request.question,
            model_used=str(graph_state.get("model_used") or selected_model),
            assistant_mode=str(graph_state.get("assistant_mode") or "doc_rag"),
            query_plan=plan,
            rewritten_query=graph_state.get("rewritten_query"),
            retrieval_results=results,
            citations=citations,
            trace=trace,
            answer=answer,
            used_fallback=bool(graph_state.get("used_fallback")),
            confidence=float(graph_state.get("confidence", 0.0)),
            response_mode=response_mode,
            retry_count=int(graph_state.get("retries", 0)),
            rejected_chunk_ids=list(graph_state.get("rejected_chunk_ids", [])),
            grounding_passed=grounding_passed,
            answer_quality_passed=answer_quality_passed,
            generation_degraded=bool(graph_state.get("generation_degraded", False)),
        )

    def _run_direct_chat(self, request: ChatRequest, model: str) -> AgentRunState:
        answer = self._direct_chat_answer(request, model)
        return AgentRunState(
            question=request.question,
            model_used=model,
            assistant_mode="direct_chat",
            query_plan=None,
            rewritten_query=None,
            retrieval_results=[],
            citations=[],
            trace=[],
            answer=answer,
            used_fallback=False,
            confidence=0.0,
            response_mode="direct-chat",
            retry_count=0,
            rejected_chunk_ids=[],
            grounding_passed=True,
            answer_quality_passed=bool(answer.strip()),
        )

    def _run_live_query(self, request: ChatRequest, model: str) -> AgentRunState:
        """Handle live data queries (weather, stocks, news) via Tavily → LLM synthesis."""
        question = request.question
        trace: list[TraceEvent] = []
        trace.append(TraceEvent(
            type="classification",
            message="Routed to live web search (weather/stock/news query detected)",
            payload={"stage": "tool_selection", "assistant_mode": "live_query"},
        ))

        # Try Tavily search (unrestricted domains for live queries like weather/stocks)
        web_results: list[dict[str, Any]] = []
        if self.tavily.enabled:
            try:
                web_results = self.tavily.search_open(question)
            except Exception as exc:
                _log.warning("Tavily live query failed (%s: %s)", type(exc).__name__, str(exc)[:120])

        if not web_results:
            # Fall back to LLM direct chat if Tavily unavailable or empty
            trace.append(TraceEvent(
                type="fallback",
                message="Tavily returned no results; falling back to LLM general knowledge",
                payload={"stage": "tool_result", "status": "empty"},
            ))
            answer = self._direct_chat_answer(request, model)
            return AgentRunState(
                question=question,
                model_used=model,
                assistant_mode="live_query",
                trace=trace,
                answer=answer,
                response_mode="llm-knowledge",
                grounding_passed=True,
                answer_quality_passed=bool(answer.strip()),
            )

        # Build pseudo-results and citations from web results
        results: list[RetrieverResult] = []
        for index, item in enumerate(web_results, start=1):
            chunk = ChunkRecord(
                id=f"live-tavily-{index}",
                source_id="tavily-web",
                title=item["title"],
                url=item["url"],
                section_path="Web Search",
                doc_family="web",
                doc_type="web",
                product_tags=["web-search"],
                source_kind="web",
                content=item["content"],
            )
            results.append(RetrieverResult(
                chunk=chunk,
                score=0.22,
                retrieval_method="tavily",
                rerank_score=0.22,
            ))

        citations = [_citation_from_result(r) for r in results[:4]]
        trace.append(TraceEvent(
            type="fallback",
            message="Tavily web search returned live results",
            payload={
                "stage": "tool_result",
                "status": "result",
                "result_count": len(results),
                "accepted_docs": [_result_preview(r) for r in results[:4]],
                **_tool_payload("web_search", "Web Search", "web"),
            },
        ))

        # Synthesize answer from web results
        answer, generation_degraded = self._synthesize_answer(question, results, model)
        trace.append(TraceEvent(
            type="generation",
            message="Synthesized answer from live web results",
            payload={"stage": "generation", "citation_count": len(citations), "response_mode": "web-backed", "model": model},
        ))

        return AgentRunState(
            question=question,
            model_used=model,
            assistant_mode="live_query",
            retrieval_results=results,
            citations=citations,
            trace=trace,
            answer=answer,
            used_fallback=True,
            confidence=0.5,
            response_mode="web-backed",
            grounding_passed=True,
            answer_quality_passed=bool(answer.strip()),
            generation_degraded=generation_degraded,
        )

    def _run_without_graph(self, question: str, model: str, effective_query: str | None = None, *, history_context: str = "", initial_trace: list[dict[str, Any]] | None = None) -> GraphState:
        """Imperative fallback that mirrors the LangGraph topology exactly."""
        state: GraphState = {"question": question, "current_query": effective_query or question, "model_used": model, "assistant_mode": "doc_rag", "history_context": history_context, "trace": list(initial_trace or [])}
        state = self._graph_classify(state)
        state = self._graph_retrieve(state)
        state = self._graph_document_grading(state)
        route = self._route_after_grading(state)
        if route == "rewrite_if_needed":
            state = self._graph_rewrite(state)
            state = self._graph_retrieve(state)
            state = self._graph_document_grading(state)
            route = self._route_after_grading(state)
        if route == "fallback_if_needed":
            state = self._graph_fallback(state)
            if self._route_after_fallback(state) == "end":
                return state
        state = self._graph_generate(state)
        state = self._graph_self_reflect(state)
        state = self._verify_claims(state)
        state = self._graph_grounding_check(state)
        state = self._graph_answer_quality_check(state)
        quality_route = self._route_after_quality(state)
        if quality_route == "post_gen_fallback":
            state = self._graph_post_generation_fallback(state)
            if self._route_after_post_gen_fallback(state) == "generate":
                state = self._graph_generate(state)
                state = self._graph_self_reflect(state)
                state = self._verify_claims(state)
                state = self._graph_grounding_check(state)
                state = self._graph_answer_quality_check(state)
            return state
        if quality_route == "rewrite_if_needed":
            state = self._graph_rewrite(state)
            state = self._graph_retrieve(state)
            state = self._graph_document_grading(state)
            state = self._graph_generate(state)
            state = self._graph_self_reflect(state)
            state = self._verify_claims(state)
            state = self._graph_grounding_check(state)
            state = self._graph_answer_quality_check(state)
        return state

    def _direct_chat_answer(self, request: ChatRequest, model: str) -> str:
        conversation = self._normalized_history(request)
        latest_question = conversation[-1].content if conversation else request.question
        known_answer = self._direct_chat_fallback(latest_question)
        if known_answer is not None:
            return known_answer

        if self.reasoner.enabled:
            transcript = "\n".join(f"{turn.role.title()}: {turn.content}" for turn in conversation[-8:])
            prompt = (
                "You are MaiStorage Assistant. Continue the conversation naturally, helpfully, and concisely. "
                "For ordinary general questions, answer directly in 2-4 sentences. "
                "Do not pretend you searched documents unless the system already routed into cited RAG. "
                "Do not invent citations. If you are unsure, say so plainly instead of bluffing.\n\n"
                f"Conversation:\n{transcript}\nAssistant:"
            )
            try:
                return self.reasoner.generate_text(prompt, model=model)
            except Exception as exc:
                _log.warning("LLM direct-chat generation failed (%s: %s), using static fallback", type(exc).__name__, str(exc)[:120])

        return (
            "I can answer general questions when the OpenAI API is available. "
            "For now, try asking me an NVIDIA infrastructure question — those are answered from the offline knowledge base and don't require a live API."
        )

    def _source_inventory_answer(self) -> str:
        highlighted = {
            source.id: source.title
            for source in self.retrieval.list_sources()
            if source.id
            in {
                "cuda-install",
                "container-toolkit",
                "gpu-operator",
                "nccl",
                "fabric-manager",
                "dl-performance",
                "gpudirect-storage",
                "h100",
                "a100",
                "l40s",
                "megatron-core",
                "infra-platforms",
                "infra-cluster-ops",
                "infra-storage",
                "infra-mlops",
            }
        }
        preview = ", ".join(list(highlighted.values())[:8])
        return (
            "I currently have a focused NVIDIA knowledge base covering Linux CUDA deployment, the NVIDIA Container Toolkit, "
            "GPU Operator for Kubernetes, NCCL and NVLink fabric behavior, mixed precision and GPU performance guidance, "
            "storage throughput considerations, hardware positioning for H100/A100/L40S, large-model parallelism, "
            "AI server platform design, cluster operations, data-path/storage choices, and container plus CI/CD delivery notes. "
            f"Representative sources include: {preview}."
        )

    def _direct_chat_fallback(self, latest_question: str) -> str | None:
        lowered = latest_question.strip().lower()
        if not lowered:
            return "How can I help?"

        if lowered in {"hi", "hello", "hey", "hellaur", "yo", "sup"}:
            return "Hello. Ask me anything, and if you want NVIDIA deployment or documentation help I can switch into cited RAG mode."

        if (
            lowered.startswith("how are you")
            or lowered.startswith("how's it going")
            or lowered.startswith("how do you do")
            or lowered in {"how are things", "how are things going", "hows it going"}
        ):
            return "Doing well, ready to help. Ask me a technical question about NVIDIA infrastructure or just chat."

        # Out-of-scope topics — give a polite redirect without touching the knowledge base
        # NOTE: weather/forecast/stock terms are handled by live_query route and won't reach here.
        _out_of_scope_hints = ("sports", "football", "basketball", "soccer", "cricket", "recipe",
                               "cooking", "restaurant", "flight", "airline", "hotel", "horoscope")
        if any(hint in lowered for hint in _out_of_scope_hints):
            return "That's outside what I'm set up for — I specialise in NVIDIA GPU infrastructure, CUDA, containers, distributed training, and storage. Try asking about one of those."

        if "what day is it" in lowered or "what date is it" in lowered:
            today = datetime.now().strftime("%A, %B %-d, %Y")
            return f"Today is {today}."

        if lowered.startswith("what is nvidia") or lowered == "nvidia":
            return (
                "NVIDIA is a technology company best known for GPUs and the software stack around accelerated computing. "
                "It is a major platform vendor for AI training, inference, graphics, CUDA, and data-center systems."
            )

        if lowered.startswith("tell me about nvidia") or lowered.startswith("what does nvidia do"):
            return (
                "NVIDIA builds GPUs, AI systems, and the CUDA software ecosystem used for training, inference, graphics, and high-performance computing. "
                "In practice, it is one of the core platform vendors behind modern AI infrastructure."
            )

        if lowered.startswith("who founded nvidia"):
            return "NVIDIA was founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993."

        if "who is the ceo of nvidia" in lowered:
            return "Jensen Huang is the CEO of NVIDIA."

        if "where is nvidia" in lowered and "headquartered" in lowered:
            return "NVIDIA is headquartered in Santa Clara, California."

        if "how big is nvidia" in lowered or "what is nvidia known for" in lowered:
            return (
                "NVIDIA is one of the most important companies in the AI hardware market because of its GPUs, CUDA ecosystem, and data-center platform. "
                "It is especially well known for AI training and inference infrastructure."
            )

        if "stock price of nvidia" in lowered or ("nvidia" in lowered and "stock price" in lowered):
            return "I’m not confident giving a live stock price from general knowledge alone. For that, I’d need a current external source."

        if lowered.startswith("what is cuda") or lowered == "cuda":
            return (
                "CUDA is NVIDIA's parallel computing platform and programming model for running workloads on NVIDIA GPUs. "
                "It includes the driver/runtime interface, libraries, and tooling used for AI, HPC, and accelerated applications."
            )

        if lowered.startswith("what does cuda stand for"):
            return "CUDA originally stood for Compute Unified Device Architecture."

        if lowered.startswith("tell me about h100") or lowered == "h100":
            return (
                "The NVIDIA H100 is a data-center GPU based on the Hopper architecture. "
                "It is designed for large-scale AI training and inference, with strong tensor performance, high memory bandwidth, and multi-GPU scaling support."
            )

        if lowered.startswith("what is an h100"):
            return (
                "An H100 is NVIDIA's Hopper-generation data-center GPU. "
                "It is built for demanding AI training and inference workloads, especially where tensor performance and multi-GPU scaling matter."
            )

        if lowered.startswith("what is rag") or "agentic rag" in lowered:
            return (
                "RAG stands for retrieval-augmented generation: the model first retrieves relevant source material, then uses it to answer. "
                "Agentic RAG adds routing, retrieval strategies, validation, and fallback decisions instead of relying on a single retrieval step."
            )

        if (
            "what docs do you have" in lowered
            or "what sources do you have" in lowered
            or "overview of all docs" in lowered
            or "overview of the docs" in lowered
            or "what docs are available" in lowered
        ):
            return self._source_inventory_answer()

        if lowered.startswith(("can you help me", "help me with", "what should i ask", "what can you do")) or (
            "what can you help" in lowered or "what do you help" in lowered or "what can you assist" in lowered
        ):
            return (
                "I can handle normal chat, and I can switch into cited RAG mode for NVIDIA infrastructure, deployment, performance, and troubleshooting questions. "
                "If you want grounded technical answers, ask about CUDA, drivers, containers, Kubernetes, profiling, scaling, storage, or hardware choices."
            )

        if len(tokenize(latest_question)) <= 3:
            return "I can help with that, but I need a bit more detail to answer well."
        return None

    _FOLLOW_UP_PRONOUNS = frozenset({
        "it", "this", "that", "those", "them", "above", "point",
        "elaborate", "more", "explain", "why", "how", "what",
    })

    @staticmethod
    def _contextualize_query(question: str, history: list | None) -> str:
        """Prepend prior user turn when the question looks like a follow-up.

        A question is treated as a follow-up if it is short (<=8 tokens) OR
        if it starts with a pronoun/reference word that implies prior context.
        """
        tokens = question.strip().split()
        lowered_first = tokens[0].lower().rstrip("?.,") if tokens else ""
        is_short = len(tokens) <= 8
        is_reference = lowered_first in AgentService._FOLLOW_UP_PRONOUNS
        if not (is_short or is_reference):
            return question
        recent = [t.content for t in (history or []) if t.role == "user" and t.content.strip()]
        if len(recent) < 2:
            return question
        prior = recent[-2]  # last user turn before current
        return f"{prior} {question}"

    def _reformulate_follow_up(self, question: str, history: list[ChatTurn] | None) -> tuple[str, str | None]:
        """Reformulate a follow-up question as a standalone query.

        Uses LLM when available, falls back to static concat (_contextualize_query).
        Returns (effective_query, method) where method is "llm", "static", or None.
        """
        tokens = question.strip().split()
        lowered_first = tokens[0].lower().rstrip("?.,") if tokens else ""
        is_short = len(tokens) <= 8
        is_reference = lowered_first in self._FOLLOW_UP_PRONOUNS
        if not (is_short or is_reference):
            return question, None

        recent_user = [t.content for t in (history or []) if t.role == "user" and t.content.strip()]
        if len(recent_user) < 2:
            return question, None

        # Try LLM reformulation
        if self.reasoner.enabled:
            pairs: list[tuple[str, str]] = []
            current_q: str | None = None
            for turn in (history or []):
                if turn.role == "user":
                    current_q = turn.content
                elif turn.role == "assistant" and current_q:
                    pairs.append((current_q, turn.content[:200]))
                    current_q = None
            recent_pairs = pairs[-2:]
            if recent_pairs:
                context_lines = [f"User: {q}\nAssistant: {a}" for q, a in recent_pairs]
                context = "\n\n".join(context_lines)
                prompt = (
                    "Rewrite the follow-up question as a standalone question using the conversation context.\n"
                    "Output ONLY the rewritten question.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Follow-up: {question}\nStandalone:"
                )
                try:
                    reformulated = self.reasoner.generate_text(prompt, model=self.settings.pipeline_model)
                    reformulated = reformulated.strip().strip('"').strip("'")
                    if reformulated and reformulated.lower() != question.lower():
                        return reformulated, "llm"
                except Exception as exc:
                    _log.warning("LLM query reformulation failed (%s: %s), falling back to static", type(exc).__name__, str(exc)[:120])

        # Static fallback: prepend prior user turn
        prior = recent_user[-2]
        return f"{prior} {question}", "static"

    @staticmethod
    def _format_history_context(history: list[ChatTurn] | None, max_pairs: int = 3) -> str:
        """Format recent Q&A pairs as a history context string for the synthesis prompt."""
        if not history:
            return ""
        pairs: list[tuple[str, str]] = []
        current_q: str | None = None
        for turn in history:
            if turn.role == "user":
                current_q = turn.content
            elif turn.role == "assistant" and current_q:
                pairs.append((current_q, turn.content))
                current_q = None
        if not pairs:
            return ""
        recent = pairs[-max_pairs:]
        lines: list[str] = []
        for q, a in recent:
            truncated_a = a[:200] + "..." if len(a) > 200 else a
            lines.append(f"User: {q}\nAssistant: {truncated_a}")
        return "\n\n".join(lines)

    @staticmethod
    def _cache_key(request: ChatRequest) -> str:
        """Build a cache key that incorporates recent conversation context."""
        recent_user = [t.content for t in request.history if t.role == "user" and t.content.strip()][-2:]
        if recent_user:
            return " | ".join(recent_user) + " | " + request.question
        return request.question

    @staticmethod
    def _normalized_history(request: ChatRequest) -> list[ChatTurn]:
        history = list(request.history)
        if not history or history[-1].role != "user" or history[-1].content.strip() != request.question.strip():
            history.append(ChatTurn(role="user", content=request.question))
        return history

    def _synthesize_answer(self, question: str, results: list[RetrieverResult], model: str, *, history_context: str = "", premise_note: str = "") -> tuple[str, bool]:
        """Return (answer_text, generation_degraded). generation_degraded is True when Gemini failed and keyword fallback was used."""
        if not results:
            return self._refusal_answer(), False

        chunk_count = self._synthesis_chunk_count(question)
        top_results = results[:chunk_count]
        if self.reasoner.enabled:
            context_blocks = []
            for index, result in enumerate(top_results, start=1):
                retrieved_note = f" (retrieved {result.chunk.retrieved_at[:10]})" if getattr(result.chunk, 'retrieved_at', None) else ""
                context_blocks.append(
                    f"[{index}] {result.chunk.content}\n— Source: {result.chunk.url}{retrieved_note}"
                )
            is_definitional = question.lower().strip().rstrip("?").strip().startswith(
                ("what is ", "what are ", "what's ", "define ", "explain ")
            )
            if is_definitional:
                prompt = (
                    "You are an NVIDIA AI infrastructure advisor.\n\n"
                    "Output EXACTLY ONE paragraph — no line breaks between sentences, no multiple paragraphs. "
                    "Do NOT use any headings, bold titles, bullet points, or lists. "
                    "Merge ALL passage information into that single paragraph. "
                    "Start with a direct definition that expands any acronym in the question to its full name. "
                    "Cite sources inline like [1] or [2]. "
                    "REPHRASE everything in your own words — never copy a passage sentence verbatim. "
                    "Ignore meta-text like 'This document describes...' or release note boilerplate.\n\n"
                )
            else:
                prompt = (
                    "You are an NVIDIA AI infrastructure advisor. "
                    "Directly and concisely answer the user question using only the numbered context passages below.\n\n"
                    "Rules:\n"
                    "1. Open your first sentence with a direct answer to the question.\n"
                    "2. Support each factual claim with an inline citation like [1] referencing the passage number.\n"
                    "3. Do not include information not in the context.\n"
                    "4. If context is insufficient, say so explicitly.\n"
                    "5. Keep the response under 300 words.\n"
                    "6. Prioritize technical accuracy and specific numbers (bandwidth, TFLOPS, memory sizes) over general descriptions.\n"
                    "7. When comparing hardware or configurations, use a structured format (bullet points or table).\n"
                    "8. End with a practical recommendation when the question implies a choice.\n"
                    "9. If the context contradicts the user's premise, correct it directly — do not agree with incorrect assumptions.\n"
                    "10. Do not pad with filler phrases or restate the question. Lead with the answer.\n"
                    "11. Synthesize all passages into one cohesive answer. Never use source titles or passage labels as headings. Write flowing paragraphs, not a section per source.\n\n"
                )
            if premise_note:
                prompt += premise_note
            prompt += f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
            if history_context:
                prompt += (
                    "\n\nConversation history (for continuity only — all factual claims must come from the numbered passages):\n"
                    + history_context
                )
            try:
                raw_answer = self.reasoner.generate_text(prompt, model=model)
                answer = self._ensure_citations(raw_answer, top_results)
                answer = self._strip_invalid_citations(answer, len(top_results))
                return answer, False
            except Exception as exc:
                _log.warning("LLM synthesis failed (%s: %s), using keyword fallback", type(exc).__name__, str(exc)[:120])

        lines = []
        is_def = question.lower().strip().rstrip("?").strip().startswith(
            ("what is ", "what are ", "what's ", "define ", "explain ")
        )
        for index, result in enumerate(top_results[:3], start=1):
            if is_def:
                lines.append(f"{result.chunk.content} [{index}]")
            else:
                section = result.chunk.section_path or result.chunk.title or "Source"
                lines.append(f"**{section}**\n\n{result.chunk.content} [{index}]")
        return "\n\n".join(lines), True

    @staticmethod
    def _synthesis_chunk_count(question: str) -> int:
        """Fewer chunks for simple questions to avoid content dumping."""
        lowered = question.lower().strip().rstrip("?").strip()
        token_count = len(question.split())
        # Definitional queries get 3 chunks for richer detail and source diversity
        if lowered.startswith(("what is ", "what are ", "what's ", "define ", "explain ")):
            return 3
        if token_count <= 6:
            return 2
        if token_count <= 10:
            return 3
        return 4

    def _synthesize_answer_stream(self, question: str, results: list[RetrieverResult], model: str, *, history_context: str = "", emit=None, premise_note: str = "") -> tuple[str, bool]:
        """Stream tokens via emit callback, return (answer_text, generation_degraded)."""
        if not results:
            return self._refusal_answer(), False

        chunk_count = self._synthesis_chunk_count(question)
        top_results = results[:chunk_count]
        context_blocks = []
        for index, result in enumerate(top_results, start=1):
            retrieved_note = f" (retrieved {result.chunk.retrieved_at[:10]})" if getattr(result.chunk, 'retrieved_at', None) else ""
            context_blocks.append(
                f"[{index}] {result.chunk.content}\n— Source: {result.chunk.url}{retrieved_note}"
            )
        is_definitional = question.lower().strip().rstrip("?").strip().startswith(
            ("what is ", "what are ", "what's ", "define ", "explain ")
        )
        if is_definitional:
            prompt = (
                "You are an NVIDIA AI infrastructure advisor.\n\n"
                "Output EXACTLY ONE paragraph — no line breaks between sentences, no multiple paragraphs. "
                "Do NOT use any headings, bold titles, bullet points, or lists. "
                "Merge ALL passage information into that single paragraph. "
                "Start with a direct definition that expands any acronym in the question to its full name. "
                "Cite sources inline like [1] or [2]. "
                "REPHRASE everything in your own words — never copy a passage sentence verbatim. "
                "Ignore meta-text like 'This document describes...' or release note boilerplate.\n\n"
            )
        else:
            prompt = (
                "You are an NVIDIA AI infrastructure advisor. "
                "Directly and concisely answer the user question using only the numbered context passages below.\n\n"
                "Rules:\n"
                "1. Open your first sentence with a direct answer to the question.\n"
                "2. Support each factual claim with an inline citation like [1] referencing the passage number.\n"
                "3. Do not include information not in the context.\n"
                "4. If context is insufficient, say so explicitly.\n"
                "5. Keep the response under 300 words.\n"
                "6. Prioritize technical accuracy and specific numbers (bandwidth, TFLOPS, memory sizes) over general descriptions.\n"
                "7. When comparing hardware or configurations, use a structured format (bullet points or table).\n"
                "8. End with a practical recommendation when the question implies a choice.\n"
                "9. If the context contradicts the user's premise, correct it directly — do not agree with incorrect assumptions.\n"
                "10. Do not pad with filler phrases or restate the question. Lead with the answer.\n"
                "11. Synthesize all passages into one cohesive answer. Never use source titles or passage labels as headings. Write flowing paragraphs, not a section per source.\n\n"
            )
        if premise_note:
            prompt += premise_note
        prompt += f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
        if history_context:
            prompt += (
                "\n\nConversation history (for continuity only — all factual claims must come from the numbered passages):\n"
                + history_context
            )
        try:
            accumulated: list[str] = []
            for token in self.reasoner.generate_text_stream(prompt, model=model):
                accumulated.append(token)
                if emit:
                    emit("answer_chunk", {"text": token})
            raw_answer = "".join(accumulated)
            answer = self._ensure_citations(raw_answer, top_results)
            answer = self._strip_invalid_citations(answer, len(top_results))
            return answer, False
        except Exception as exc:
            _log.warning("Streaming synthesis failed (%s: %s), falling back to non-streaming", type(exc).__name__, str(exc)[:120])
            return self._synthesize_answer(question, results, model, history_context=history_context, premise_note=premise_note)

    @staticmethod
    def _validate_format(answer: str, response_mode: str) -> dict[str, Any]:
        """Post-generation format validation. Returns a dict with validation results."""
        issues: list[str] = []
        word_count = len(answer.split())

        # Word count check
        if word_count > 500:
            issues.append(f"answer_too_long ({word_count} words)")
        elif word_count > 350:
            issues.append(f"answer_verbose ({word_count} words)")

        # Citation count check for knowledge-base-backed answers
        citation_count = len(re.findall(r'\[\d+\]', answer))
        if response_mode == "knowledge-base-backed" and citation_count == 0:
            issues.append("no_citations_in_knowledge_base_backed")

        # Markdown formatting issues
        if answer.count('```') % 2 != 0:
            issues.append("unclosed_code_block")
        if answer.count('|') > 3:
            # Check if table rows have consistent column counts
            table_rows = [line for line in answer.split('\n') if '|' in line and line.strip().startswith('|')]
            if table_rows:
                col_counts = [line.count('|') for line in table_rows]
                if len(set(col_counts)) > 1:
                    issues.append("inconsistent_table_columns")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "word_count": word_count,
            "citation_count": citation_count,
        }

    @staticmethod
    def _citation_quality(answer: str, results: list[RetrieverResult]) -> dict[str, Any]:
        """Compute citation attribution quality: strong vs weak citations."""
        if not results or not answer:
            return {"strong": 0, "weak": 0, "uncited_paragraphs": 0}

        chunk_term_sets = [
            set(r.chunk.sparse_terms or tokenize(r.chunk.content))
            for r in results
        ]

        strong = 0
        weak = 0
        uncited_paragraphs = 0

        # Split into units, handling structured content (tables/bullets)
        units: list[str] = []
        for raw_block in answer.split("\n\n"):
            stripped = raw_block.strip()
            if not stripped:
                continue
            lines = stripped.split("\n")
            has_structured = any(AgentService._STRUCTURED_LINE_RE.match(line) for line in lines)
            if has_structured:
                units.extend(line.strip() for line in lines if line.strip() and len(line.strip()) >= 40)
            elif len(stripped) >= 40:
                units.append(stripped)

        for unit in units:
            cited_indices = [int(m) - 1 for m in re.findall(r'\[(\d+)\]', unit)]
            cited_indices = [i for i in cited_indices if 0 <= i < len(results)]

            if not cited_indices:
                uncited_paragraphs += 1
                continue

            unit_tokens = set(tokenize(unit))
            for idx in cited_indices:
                overlap = len(unit_tokens & chunk_term_sets[idx])
                if overlap >= 3:
                    strong += 1
                else:
                    weak += 1

        return {"strong": strong, "weak": weak, "uncited_paragraphs": uncited_paragraphs}

    _STRUCTURED_LINE_RE = re.compile(r"^\s*(\|.+\||[-*+]\s|\d+[.)]\s)")

    @staticmethod
    def _ensure_citations(answer: str, results: list[RetrieverResult], max_citations_per_para: int = 3) -> str:
        if not results:
            return answer
        SKIP_PREFIXES = ("in summary", "to summarize", "overall,", "in conclusion")
        chunk_term_sets = [
            set(r.chunk.sparse_terms or tokenize(r.chunk.content))
            for r in results
        ]

        def _cite_unit(unit: str) -> str:
            already_cited = bool(re.search(r"\[\d+\]", unit))
            is_short = len(unit) < 40
            is_structural = unit.lower().startswith(SKIP_PREFIXES)
            if already_cited or is_short or is_structural:
                return unit
            unit_tokens = set(tokenize(unit))
            matches: list[tuple[int, int]] = []
            for idx, chunk_terms in enumerate(chunk_term_sets):
                overlap = len(unit_tokens & chunk_terms)
                if overlap >= 2:
                    matches.append((overlap, idx))
            if matches:
                matches.sort(reverse=True)
                citation_markers = "".join(f" [{idx + 1}]" for _, idx in matches[:max_citations_per_para])
                return f"{unit}{citation_markers}"
            return unit

        # Process each \n\n block; within structured blocks, cite each line
        output_blocks: list[str] = []
        for raw_block in answer.split("\n\n"):
            stripped = raw_block.strip()
            if not stripped:
                output_blocks.append("")
                continue
            lines = stripped.split("\n")
            has_structured = any(AgentService._STRUCTURED_LINE_RE.match(line) for line in lines)
            if has_structured:
                output_blocks.append("\n".join(_cite_unit(line.strip()) for line in lines if line.strip()))
            else:
                output_blocks.append(_cite_unit(stripped))
        return "\n\n".join(output_blocks)

    _LLM_HEDGING_PHRASES = (
        "based on my knowledge",
        "based on my training",
        "from my understanding",
        "i believe",
        "i think",
        "as an ai",
        "as a language model",
        "in my opinion",
        "from what i know",
        "to the best of my knowledge",
    )

    @staticmethod
    def _strip_invalid_citations(answer: str, num_citations: int) -> str:
        """Remove [N] markers where N > num_citations or N < 1."""
        if num_citations <= 0:
            return re.sub(r"\s*\[\d+\]", "", answer)

        def _replace(match: re.Match) -> str:
            n = int(match.group(1))
            if 1 <= n <= num_citations:
                return match.group(0)
            return ""

        return re.sub(r"\s*\[(\d+)\]", _replace, answer)

    @staticmethod
    def _grounding_check(answer: str, citations: list[Citation]) -> bool:
        if not answer.strip() or not citations:
            return False
        lowered = answer.lower()
        if any(phrase in lowered for phrase in AgentService._LLM_HEDGING_PHRASES):
            return False
        # Validate that referenced [N] markers map to actual citations
        referenced = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        valid_range = set(range(1, len(citations) + 1))
        invalid_refs = referenced - valid_range
        if invalid_refs:
            _log.warning("Answer contains invalid citation markers: %s (valid: 1-%d)", invalid_refs, len(citations))
        # Split into units: structured content (tables/bullets) split on \n too
        units: list[str] = []
        for raw_block in answer.split("\n\n"):
            stripped = raw_block.strip()
            if not stripped:
                continue
            lines = stripped.split("\n")
            has_structured = any(AgentService._STRUCTURED_LINE_RE.match(line) for line in lines)
            if has_structured:
                units.extend(line.strip() for line in lines if line.strip() and len(line.strip()) >= 40)
            elif len(stripped) >= 40:
                units.append(stripped)
        if not units:
            valid_cited = bool(referenced & valid_range)
            return valid_cited
        cited = sum(1 for u in units if re.search(r"\[\d+\]", u))
        return cited / len(units) >= 0.6

    @staticmethod
    def _answer_quality_check(question: str, answer: str) -> bool:
        if not answer.strip():
            return False
        if answer.strip().startswith("I do not have enough grounded evidence"):
            return False
        ignored = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "how",
            "should",
            "could",
            "would",
            "tell",
            "explain",
            "compare",
            "including",
            "about",
            "matter",
            "useful",
            "know",
        }
        question_tokens = [token for token in tokenize(question) if len(token) > 3 and token not in ignored]
        if not question_tokens:
            return len(answer) > 60
        hits = sum(1 for token in question_tokens[:4] if token in answer.lower())
        return hits >= 1 and len(answer) > 80

    @staticmethod
    def _answer_says_insufficient(answer: str) -> bool:
        lowered = answer.lower()
        PHRASES = (
            "does not contain information",
            "context does not contain",
            "provided context does not",
            "no information in the context",
            "cannot find information",
            "not found in the provided",
            "the passages do not address",
            "the documents do not",
            "i cannot answer",
            "unable to find",
        )
        return any(p in lowered for p in PHRASES)

    def _llm_knowledge_answer(self, question: str, model: str, *, history_context: str = "") -> str | None:
        if not self.reasoner.enabled:
            return None
        prompt = (
            "You are an AI infrastructure expert. Answer the following question from your general knowledge. "
            "Be concise (under 200 words). If you truly do not know, say so plainly. "
            "If the question contains an incorrect premise, correct it directly.\n\n"
            f"Question: {question}"
        )
        if history_context:
            prompt += "\n\nConversation history (for continuity):\n" + history_context
        try:
            return self.reasoner.generate_text(prompt, model=model)
        except Exception as exc:
            _log.warning("LLM knowledge fallback failed (%s: %s)", type(exc).__name__, str(exc)[:120])
            return None

    @staticmethod
    def _refusal_answer() -> str:
        return "I do not have enough grounded evidence in the knowledge base to answer this confidently."

    def _resolve_request_model(self, requested_model: str | None) -> str:
        if requested_model and requested_model in self.settings.openai_allowed_models:
            return requested_model
        return self.settings.generation_model

    async def stream(self, request: ChatRequest):
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        answer_streamed = [False]

        def _emit_to_queue(event_type: str, payload: dict[str, Any]) -> None:
            sse = self._format_sse(event_type, payload)
            loop.call_soon_threadsafe(queue.put_nowait, sse)
            if event_type == "answer_chunk":
                answer_streamed[0] = True

        def _sync_run() -> AgentRunState:
            _thread_local_emit.fn = _emit_to_queue
            try:
                return self.run(request)
            finally:
                _thread_local_emit.fn = None

        async def _run_and_enqueue() -> None:
            try:
                state = await asyncio.to_thread(_sync_run)
                for citation in state.citations:
                    await queue.put(self._format_sse("citation", citation.model_dump()))
                if not answer_streamed[0]:
                    for paragraph in state.answer.split("\n\n"):
                        if paragraph.strip():
                            await queue.put(self._format_sse("answer_chunk", {"text": paragraph + "\n\n"}))
                await queue.put(self._format_sse(
                    "done",
                    {
                        "assistant_mode": state.assistant_mode,
                        "confidence": state.confidence,
                        "used_fallback": state.used_fallback,
                        "answer": state.answer,
                        "response_mode": state.response_mode,
                        "retry_count": state.retry_count,
                        "grounding_passed": state.grounding_passed,
                        "answer_quality_passed": state.answer_quality_passed,
                        "rejected_chunk_count": len(state.rejected_chunk_ids),
                        "citation_count": len(state.citations),
                        "query_class": state.query_plan.query_class.value if state.query_plan is not None else "general",
                        "source_families": state.query_plan.source_families if state.query_plan is not None else [],
                        "model_used": state.model_used,
                        "generation_degraded": state.generation_degraded,
                    },
                ))
            except Exception as exc:
                _log.exception("Progressive stream pipeline failed")
                await queue.put(self._format_sse("error", {
                    "message": f"Pipeline error: {type(exc).__name__}: {str(exc)[:200]}",
                    "recoverable": True,
                }))
                await queue.put(self._format_sse("done", {
                    "assistant_mode": "error",
                    "confidence": 0.0,
                    "used_fallback": False,
                    "answer": "An internal error occurred while processing your question. Please try again.",
                    "response_mode": "error",
                    "retry_count": 0,
                    "grounding_passed": False,
                    "answer_quality_passed": False,
                    "rejected_chunk_count": 0,
                    "citation_count": 0,
                    "query_class": "general",
                    "source_families": [],
                    "model_used": "",
                    "generation_degraded": True,
                }))
            finally:
                await queue.put(None)

        asyncio.ensure_future(_run_and_enqueue())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    @staticmethod
    def _format_sse(event_type: str, payload: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
