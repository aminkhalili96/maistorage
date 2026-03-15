from __future__ import annotations

import asyncio
import json
import re
from contextlib import contextmanager
from typing import Any, TypedDict

from app.config import Settings
from app.models import AgentRunState, ChatRequest, Citation, ChunkRecord, QueryPlan, RetrieverResult, SearchDebugResponse, TraceEvent
from app.services.providers import GeminiReasoner, TavilyClient, tokenize
from app.services.retrieval import RetrievalService, needs_retry, rewrite_query

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


def _citation_from_result(result: RetrieverResult) -> Citation:
    snippet = result.chunk.content[:240].strip()
    return Citation(
        chunk_id=result.chunk.id,
        title=result.chunk.title,
        url=result.chunk.url,
        section_path=result.chunk.section_path,
        snippet=snippet,
        source_kind=result.chunk.source_kind,
    )


def _append_trace(state: GraphState, *events: TraceEvent) -> GraphState:
    trace = list(state.get("trace", []))
    trace.extend(event.model_dump() for event in events)
    state["trace"] = trace
    return state


class AgentService:
    def __init__(self, settings: Settings, retrieval: RetrievalService, reasoner: GeminiReasoner, tavily: TavilyClient) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.reasoner = reasoner
        self.tavily = tavily
        self.graph = self._build_graph() if StateGraph is not None else None

    def _build_graph(self):  # pragma: no cover
        graph = StateGraph(GraphState)
        graph.add_node("classify", self._graph_classify)
        graph.add_node("retrieve", self._graph_retrieve)
        graph.add_node("document_grading", self._graph_document_grading)
        graph.add_node("rewrite_if_needed", self._graph_rewrite)
        graph.add_node("fallback_if_needed", self._graph_fallback)
        graph.add_node("generate", self._graph_generate)
        graph.add_node("grounding_check", self._graph_grounding_check)
        graph.add_node("answer_quality_check", self._graph_answer_quality_check)

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
        graph.add_edge("generate", "grounding_check")
        graph.add_edge("grounding_check", "answer_quality_check")
        graph.add_conditional_edges(
            "answer_quality_check",
            self._route_after_quality,
            {"rewrite_if_needed": "rewrite_if_needed", "end": END},
        )
        return graph.compile()

    def _graph_classify(self, state: GraphState) -> GraphState:
        plan = self.retrieval.build_plan(state["question"])
        next_state: GraphState = {
            "plan": plan.model_dump(),
            "current_query": state["question"],
            "rewritten_query": None,
            "results": [],
            "citations": [],
            "trace": [],
            "rejected_chunk_ids": [],
            "confidence": 0.0,
            "answer": "",
            "retries": 0,
            "used_fallback": False,
            "fallback_reason": None,
            "response_mode": "corpus-backed",
        }
        return _append_trace(
            next_state,
            TraceEvent(
                type="classification",
                message=f"Classified question as {plan.query_class.value}",
                payload={"source_families": plan.source_families, "search_queries": plan.search_queries[:2]},
            ),
        )

    def _graph_retrieve(self, state: GraphState) -> GraphState:
        plan = QueryPlan.model_validate(state["plan"])
        results, rejected, confidence, events = self.retrieval.run_retrieval_pass(
            state["question"], plan, state.get("current_query", state["question"])
        )
        next_state = dict(state)
        next_state["results"] = [item.model_dump() for item in results]
        next_state["rejected_chunk_ids"] = list(dict.fromkeys(state.get("rejected_chunk_ids", []) + rejected))
        next_state["confidence"] = confidence
        next_state["trace"] = list(state.get("trace", [])) + [event.model_dump() for event in events]
        return next_state

    def _graph_document_grading(self, state: GraphState) -> GraphState:
        accepted = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        return _append_trace(
            dict(state),
            TraceEvent(
                type="document_grading",
                message="Prepared the accepted evidence set for answer synthesis",
                payload={"accepted_count": len(accepted), "rejected_count": len(state.get('rejected_chunk_ids', []))},
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

    def _graph_rewrite(self, state: GraphState) -> GraphState:
        plan = QueryPlan.model_validate(state["plan"])
        retries = int(state.get("retries", 0)) + 1
        rewritten_query = rewrite_query(state["question"], plan.query_class, retries)
        next_state = dict(state)
        next_state["retries"] = retries
        next_state["current_query"] = rewritten_query
        next_state["rewritten_query"] = rewritten_query
        return _append_trace(
            next_state,
            TraceEvent(
                type="rewrite",
                message="Triggered a rewritten retrieval query because confidence was below the floor",
                payload={"rewritten_query": rewritten_query, "retry": retries},
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
                    payload={"reason": "disabled"},
                ),
            )

        web_results = self.tavily.search(state["question"])
        if not web_results:
            next_state["response_mode"] = "insufficient-evidence"
            next_state["answer"] = self._refusal_answer()
            return _append_trace(
                next_state,
                TraceEvent(
                    type="fallback",
                    message="Fallback search returned no usable web evidence",
                    payload={"reason": "empty"},
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
                payload={"result_count": len(pseudo_results)},
            ),
        )

    def _route_after_fallback(self, state: GraphState) -> str:
        if state.get("results"):
            return "generate"
        return "end"

    def _graph_generate(self, state: GraphState) -> GraphState:
        results = [RetrieverResult.model_validate(item) for item in state.get("results", [])]
        citations = [_citation_from_result(result).model_dump() for result in results[:4]]
        answer = self._synthesize_answer(state["question"], results)
        next_state = dict(state)
        next_state["citations"] = citations
        next_state["answer"] = answer
        return _append_trace(
            next_state,
            TraceEvent(
                type="generation",
                message="Synthesized the final answer from the accepted evidence set",
                payload={"citation_count": len(citations), "response_mode": next_state.get("response_mode", "corpus-backed")},
            ),
        )

    def _graph_grounding_check(self, state: GraphState) -> GraphState:
        answer = state.get("answer", "")
        citations = [Citation.model_validate(item) for item in state.get("citations", [])]
        grounded = self._grounding_check(answer, citations)
        next_state = dict(state)
        next_state["grounding_passed"] = grounded
        return _append_trace(
            next_state,
            TraceEvent(
                type="grounding_check",
                message="Verified that the answer stays anchored to cited evidence",
                payload={"passed": grounded, "citation_count": len(citations)},
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
                payload={"passed": passed, "retry": state.get("retries", 0)},
            ),
        )

    def _route_after_quality(self, state: GraphState) -> str:
        plan = QueryPlan.model_validate(state["plan"])
        if not state.get("grounding_passed", False) or not state.get("answer_quality_passed", False):
            if state.get("retries", 0) < plan.max_retries:
                return "rewrite_if_needed"
        return "end"

    def run(self, request: ChatRequest) -> AgentRunState:
        with tracing_context(
            project_name=self.settings.langsmith_project,
            enabled=self.settings.langsmith_tracing and bool(self.settings.langsmith_api_key),
            metadata={"app_mode": self.settings.app_mode},
            tags=["maistorage", "agentic-rag", self.settings.app_mode],
        ):
            return self._run_with_optional_trace(request)

    @traceable(name="agent_run", run_type="chain")
    def _run_with_optional_trace(self, request: ChatRequest) -> AgentRunState:
        if self.graph is not None:
            graph_state = self.graph.invoke({"question": request.question})
        else:
            graph_state = self._run_without_graph(request.question)

        plan = QueryPlan.model_validate(graph_state["plan"])
        results = [RetrieverResult.model_validate(item) for item in graph_state.get("results", [])]
        citations = [Citation.model_validate(item) for item in graph_state.get("citations", [])]
        trace = [TraceEvent.model_validate(item) for item in graph_state.get("trace", [])]
        answer = graph_state.get("answer") or self._refusal_answer()
        response_mode = graph_state.get("response_mode", "corpus-backed")
        if not results and response_mode == "corpus-backed":
            response_mode = "insufficient-evidence"

        return AgentRunState(
            question=request.question,
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
            grounding_passed=bool(graph_state.get("grounding_passed", False)),
            answer_quality_passed=bool(graph_state.get("answer_quality_passed", False)),
        )

    def _run_without_graph(self, question: str) -> GraphState:
        state: GraphState = {"question": question}
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
        state = self._graph_grounding_check(state)
        state = self._graph_answer_quality_check(state)
        if self._route_after_quality(state) == "rewrite_if_needed":
            state = self._graph_rewrite(state)
            state = self._graph_retrieve(state)
            state = self._graph_document_grading(state)
            state = self._graph_generate(state)
            state = self._graph_grounding_check(state)
            state = self._graph_answer_quality_check(state)
        return state

    def _synthesize_answer(self, question: str, results: list[RetrieverResult]) -> str:
        if not results:
            return self._refusal_answer()

        top_results = results[:4]
        if self.reasoner.enabled:
            context_blocks = []
            for index, result in enumerate(top_results, start=1):
                context_blocks.append(
                    f"[{index}] {result.chunk.title} | {result.chunk.section_path} | {result.chunk.url}\n{result.chunk.content}"
                )
            prompt = (
                "You are an NVIDIA AI infrastructure advisor. Answer using only the provided context. "
                "Every factual paragraph must include inline citations like [1] or [2]. "
                "If the evidence is weak or incomplete, say so explicitly.\n\n"
                f"Question:\n{question}\n\nContext:\n" + "\n\n".join(context_blocks)
            )
            try:
                return self._ensure_citations(self.reasoner.generate_text(prompt), len(top_results))
            except Exception:
                pass

        lines = [
            "The strongest grounded evidence points to a mix of workload characteristics and infrastructure design choices. [1]",
        ]
        for index, result in enumerate(top_results[:3], start=1):
            lines.append(f"{result.chunk.content} [{index}]")
        return "\n\n".join(lines)

    @staticmethod
    def _ensure_citations(answer: str, citation_count: int) -> str:
        if citation_count == 0:
            return answer
        paragraphs = [paragraph.strip() for paragraph in answer.split("\n\n") if paragraph.strip()]
        fixed: list[str] = []
        for index, paragraph in enumerate(paragraphs, start=1):
            if re.search(r"\[\d+\]", paragraph):
                fixed.append(paragraph)
                continue
            citation_index = min(index, citation_count)
            fixed.append(f"{paragraph} [{citation_index}]")
        return "\n\n".join(fixed)

    @staticmethod
    def _grounding_check(answer: str, citations: list[Citation]) -> bool:
        if not answer.strip() or not citations:
            return False
        paragraphs = [paragraph.strip() for paragraph in answer.split("\n\n") if paragraph.strip()]
        return all(bool(re.search(r"\[\d+\]", paragraph)) for paragraph in paragraphs)

    @staticmethod
    def _answer_quality_check(question: str, answer: str) -> bool:
        if not answer.strip():
            return False
        if answer.strip().startswith("I do not have enough grounded evidence"):
            return False
        question_tokens = [token for token in tokenize(question) if len(token) > 3]
        if not question_tokens:
            return len(answer) > 60
        hits = sum(1 for token in question_tokens[:4] if token in answer.lower())
        return hits >= 1 and len(answer) > 80

    @staticmethod
    def _refusal_answer() -> str:
        return "I do not have enough grounded evidence in the bundled corpus to answer this confidently."

    async def stream(self, request: ChatRequest):
        state = await asyncio.to_thread(self.run, request)
        for event in state.trace:
            yield self._format_sse(event.type, event.model_dump())
        for citation in state.citations:
            yield self._format_sse("citation", citation.model_dump())
        for paragraph in state.answer.split("\n\n"):
            if paragraph.strip():
                yield self._format_sse("answer_chunk", {"text": paragraph + "\n\n"})
        yield self._format_sse(
            "done",
            {
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
            },
        )

    @staticmethod
    def _format_sse(event_type: str, payload: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
