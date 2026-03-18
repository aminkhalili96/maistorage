from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, TypedDict
from urllib.parse import urlparse

_log = logging.getLogger("maistorage.agent")

from app.config import Settings
from app.models import AgentRunState, ChatRequest, ChatTurn, Citation, ChunkRecord, QueryPlan, RetrieverResult, SearchDebugResponse, TraceEvent
from app.services.providers import Embedder, OpenAIReasoner, TavilyClient, tokenize
from app.services.retrieval import RetrievalService, classify_assistant_mode, needs_retry, rewrite_query

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
    question: str
    assistant_mode: str
    plan: dict[str, Any]
    current_query: str
    rewritten_query: str | None
    results: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    rejected_chunk_ids: list[str]
    confidence: float
    answer: str
    retries: int
    used_fallback: bool
    fallback_reason: str | None
    response_mode: str
    grounding_passed: bool
    answer_quality_passed: bool
    model_used: str
    generation_degraded: bool
    self_reflect_scores: dict[str, Any]
    sub_questions: list[str]
    history_context: str


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
    if chunk.source_kind != "corpus":
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
    """In-memory LRU cache keyed on query embedding similarity."""

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
        graph.add_conditional_edges(
            "document_grading",
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
        graph.add_edge("generate", "self_reflect")
        graph.add_edge("self_reflect", "grounding_check")
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
        plan = self.retrieval.build_plan(state["question"])
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
            "response_mode": "corpus-backed",
            "model_used": selected_model,
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
            **_tool_payload("nvidia_docs", "NVIDIA Docs", "corpus"),
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
            results, rejected, confidence, events = self.retrieval.run_retrieval_pass(
                state["question"], plan, state.get("current_query", state["question"])
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
        accepted = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        rejected_ids = state.get("rejected_chunk_ids", [])
        accepted_count = len(accepted)
        rejected_count = len(rejected_ids)
        return _append_trace(
            dict(state),
            TraceEvent(
                type="document_grading",
                message="Prepared the accepted evidence set for answer synthesis",
                payload={
                    "stage": "evidence_selection",
                    "accepted_count": accepted_count,
                    "accepted_docs": [_result_preview(item) for item in accepted[:4]],
                    "rejected_count": rejected_count,
                    "total_count": accepted_count + rejected_count,
                    "kept_count": accepted_count,
                    "grades": ["pass"] * accepted_count + ["fail"] * rejected_count,
                    **_tool_payload("nvidia_docs", "NVIDIA Docs", "corpus"),
                },
            ),
        )

    def _route_after_grading(self, state: GraphState) -> str:
        plan = QueryPlan.model_validate(state["plan"])
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
        next_state["results"] = [item.model_dump() for item in pseudo_results]
        next_state["used_fallback"] = True
        next_state["fallback_reason"] = "corpus_insufficiency"
        next_state["response_mode"] = "web-backed"
        return _append_trace(
            next_state,
            TraceEvent(
                type="fallback",
                message="Used Tavily fallback because the bundled corpus evidence stayed below confidence threshold",
                payload={
                    "stage": "tool_result",
                    "status": "result",
                    "result_count": len(pseudo_results),
                    "accepted_docs": [_result_preview(item) for item in pseudo_results[:4]],
                    **_tool_payload("web_search", "Web Search", "web"),
                },
            ),
        )

    def _route_after_fallback(self, state: GraphState) -> str:
        if state.get("results"):
            return "generate"
        return "end"

    def _graph_generate(self, state: GraphState) -> GraphState:
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        citations = [_citation_from_result(result).model_dump() for result in results[:4]]
        model_used = str(state.get("model_used") or self.settings.generation_model)
        history_context = state.get("history_context", "")
        answer, generation_degraded = self._synthesize_answer(state["question"], results, model_used, history_context=history_context)
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
                "response_mode": next_state.get("response_mode", "corpus-backed"),
                "model": model_used,
                "accepted_docs": [_result_preview(result) for result in results[:4]],
                "degraded": generation_degraded,
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

    def _graph_grounding_check(self, state: GraphState) -> GraphState:
        answer = state.get("answer", "")
        citations = [Citation.model_validate(item) for item in state.get("citations", [])]
        grounded = self._grounding_check(answer, citations)
        if self._answer_says_insufficient(answer):
            grounded = False  # override: LLM itself said the context wasn't good enough
        next_state = dict(state)
        next_state["grounding_passed"] = grounded
        return _append_trace(
            next_state,
            TraceEvent(
                type="grounding_check",
                message="Verified that the answer stays anchored to cited evidence",
                payload={"stage": "validation", "check": "grounding", "passed": grounded, "citation_count": len(citations)},
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
        plan = QueryPlan.model_validate(state["plan"])
        quality_ok = state.get("grounding_passed", False) and state.get("answer_quality_passed", False)
        if not quality_ok:
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
        cache_key = self._cache_key(request)
        if self._semantic_cache is not None:
            cached = self._semantic_cache.get(cache_key)
            if cached is not None:
                return cached

        assistant_mode = classify_assistant_mode(request.question, request.history)
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
        original_mode = classify_assistant_mode(request.question, request.history)
        if original_mode == "direct_chat":
            assistant_mode = "direct_chat"
        else:
            classify_query = effective_query if reformulation_method else request.question
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
        response_mode = graph_state.get("response_mode", "corpus-backed")
        grounding_passed = bool(graph_state.get("grounding_passed", False))
        answer_quality_passed = bool(graph_state.get("answer_quality_passed", False))
        if not results and response_mode == "corpus-backed":
            response_mode = "insufficient-evidence"
        # For corpus answers, enforce grounding + quality; for web-backed, trust the
        # external results regardless (citation-marker grounding doesn't apply to web).
        if response_mode == "corpus-backed" and (not grounding_passed or not answer_quality_passed):
            response_mode = "insufficient-evidence"
            answer = self._refusal_answer()
            citations = []

        # LLM general-knowledge fallback: try OpenAI when all corpus/web layers exhausted
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

        # Try Tavily search
        web_results: list[dict[str, Any]] = []
        if self.tavily.enabled:
            try:
                web_results = self.tavily.search(question)
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
        state = self._graph_grounding_check(state)
        state = self._graph_answer_quality_check(state)
        quality_route = self._route_after_quality(state)
        if quality_route == "post_gen_fallback":
            state = self._graph_post_generation_fallback(state)
            if self._route_after_post_gen_fallback(state) == "generate":
                state = self._graph_generate(state)
                state = self._graph_self_reflect(state)
                state = self._graph_grounding_check(state)
                state = self._graph_answer_quality_check(state)
            return state
        if quality_route == "rewrite_if_needed":
            state = self._graph_rewrite(state)
            state = self._graph_retrieve(state)
            state = self._graph_document_grading(state)
            state = self._graph_generate(state)
            state = self._graph_self_reflect(state)
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
            "For now, try asking me an NVIDIA infrastructure question — those are answered from the offline corpus and don't require a live API."
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
            "I currently have a focused NVIDIA corpus covering Linux CUDA deployment, the NVIDIA Container Toolkit, "
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

        # Out-of-scope topics — give a polite redirect without touching the corpus
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

        if lowered.startswith(("can you help me", "help me with", "what should i ask", "what can you do")):
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

    def _synthesize_answer(self, question: str, results: list[RetrieverResult], model: str, *, history_context: str = "") -> tuple[str, bool]:
        """Return (answer_text, generation_degraded). generation_degraded is True when Gemini failed and keyword fallback was used."""
        if not results:
            return self._refusal_answer(), False

        top_results = results[:4]
        if self.reasoner.enabled:
            context_blocks = []
            for index, result in enumerate(top_results, start=1):
                context_blocks.append(
                    f"[{index}] {result.chunk.title} | {result.chunk.section_path} | {result.chunk.url}\n{result.chunk.content}"
                )
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
                "8. End with a practical recommendation when the question implies a choice.\n\n"
                f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
            )
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
        for index, result in enumerate(top_results[:3], start=1):
            section = result.chunk.section_path or result.chunk.title or "Source"
            lines.append(f"**{section}**\n\n{result.chunk.content} [{index}]")
        return "\n\n".join(lines), True

    @staticmethod
    def _ensure_citations(answer: str, results: list[RetrieverResult], max_citations_per_para: int = 3) -> str:
        if not results:
            return answer
        paragraphs = [paragraph.strip() for paragraph in answer.split("\n\n") if paragraph.strip()]
        fixed: list[str] = []
        SKIP_PREFIXES = ("in summary", "to summarize", "overall,", "in conclusion")
        chunk_term_sets = [
            set(r.chunk.sparse_terms or tokenize(r.chunk.content))
            for r in results
        ]
        for paragraph in paragraphs:
            already_cited = bool(re.search(r"\[\d+\]", paragraph))
            is_short = len(paragraph) < 40
            is_structural = paragraph.lower().startswith(SKIP_PREFIXES)
            if already_cited or is_short or is_structural:
                fixed.append(paragraph)
                continue
            para_tokens = set(tokenize(paragraph))
            # Collect all matching citations with sufficient overlap
            matches: list[tuple[int, int]] = []  # (overlap, idx)
            for idx, chunk_terms in enumerate(chunk_term_sets):
                overlap = len(para_tokens & chunk_terms)
                if overlap >= 2:
                    matches.append((overlap, idx))
            if matches:
                # Sort by overlap descending and take top N
                matches.sort(reverse=True)
                citation_markers = "".join(f" [{idx + 1}]" for _, idx in matches[:max_citations_per_para])
                fixed.append(f"{paragraph}{citation_markers}")
            else:
                fixed.append(paragraph)
        return "\n\n".join(fixed)

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
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip() and len(p.strip()) >= 40]
        if not paragraphs:
            valid_cited = bool(referenced & valid_range)
            return valid_cited
        cited = sum(1 for p in paragraphs if re.search(r"\[\d+\]", p))
        return cited / len(paragraphs) >= 0.6

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
            "Be concise (under 200 words). If you truly do not know, say so plainly.\n\n"
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
        return "I do not have enough grounded evidence in the bundled corpus to answer this confidently."

    def _resolve_request_model(self, requested_model: str | None) -> str:
        if requested_model and requested_model in self.settings.openai_allowed_models:
            return requested_model
        return self.settings.generation_model

    async def stream(self, request: ChatRequest):
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _emit_to_queue(event_type: str, payload: dict[str, Any]) -> None:
            sse = self._format_sse(event_type, payload)
            loop.call_soon_threadsafe(queue.put_nowait, sse)

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
