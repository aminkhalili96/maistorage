from __future__ import annotations

from statistics import mean

from app.config import Settings
from app.models import DocumentSource, QueryClass, QueryPlan, RetrieverResult, SearchDebugResponse, TraceEvent
from app.services.indexes import SearchIndex
from app.services.providers import tokenize


QUERY_CLASS_RULES: list[tuple[QueryClass, tuple[str, ...]]] = [
    (QueryClass.distributed_multi_gpu, ("multi-gpu", "distributed", "nccl", "all-reduce", "nvlink", "nvswitch", "4-gpu", "scaling")),
    (QueryClass.deployment_runtime, ("deploy", "deployment", "runtime", "linux", "driver", "cuda", "container", "kubernetes", "operator")),
    (QueryClass.hardware_topology, ("h100", "h200", "a100", "l40s", "hardware", "topology", "sizing", "on-prem", "server")),
    (QueryClass.training_optimization, ("mixed precision", "throughput", "utilization", "training", "tensor core", "memory-bound", "profiling")),
]


CLASS_FAMILY_MAP: dict[QueryClass, list[str]] = {
    QueryClass.training_optimization: ["core", "advanced"],
    QueryClass.distributed_multi_gpu: ["distributed", "core", "advanced"],
    QueryClass.deployment_runtime: ["infrastructure", "core"],
    QueryClass.hardware_topology: ["hardware", "infrastructure", "advanced"],
    QueryClass.general: ["core", "distributed", "infrastructure", "advanced", "hardware"],
}


EXPANSION_TERMS: dict[QueryClass, list[str]] = {
    QueryClass.training_optimization: ["tensor cores", "throughput", "memory-bound"],
    QueryClass.distributed_multi_gpu: ["nccl", "all-reduce", "nvlink"],
    QueryClass.deployment_runtime: ["drivers", "cuda toolkit", "container runtime"],
    QueryClass.hardware_topology: ["h100", "a100", "l40s", "server design"],
    QueryClass.general: ["nvidia training infrastructure", "retrieval optimization", "ai infrastructure"],
}


QUERY_CLASS_SOURCE_HINTS: dict[QueryClass, dict[str, float]] = {
    QueryClass.training_optimization: {
        "dl-performance": 0.42,
        "cuda-best-practices": 0.24,
        "nsight-compute": 0.18,
        "gpudirect-storage": 0.12,
    },
    QueryClass.distributed_multi_gpu: {
        "nccl": 0.48,
        "fabric-manager": 0.32,
        "megatron-core": 0.14,
        "nemo-performance": 0.12,
    },
    QueryClass.deployment_runtime: {
        "cuda-install": 0.42,
        "container-toolkit": 0.34,
        "gpu-operator": 0.28,
        "dgx-basepod": 0.12,
    },
    QueryClass.hardware_topology: {
        "h100": 0.56,
        "h200": 0.48,
        "a100": 0.5,
        "l40s": 0.44,
        "dgx-basepod": 0.12,
    },
    QueryClass.general: {},
}

RECENCY_TERMS = ("latest", "current", "recent", "today", "yesterday", "newest", "release notes", "changed")


def classify_question(question: str) -> QueryClass:
    lowered = question.lower()
    for query_class, terms in QUERY_CLASS_RULES:
        if any(term in lowered for term in terms):
            return query_class
    return QueryClass.general


def is_recency_sensitive(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in RECENCY_TERMS)


def build_query_plan(question: str, settings: Settings) -> QueryPlan:
    query_class = classify_question(question)
    expansions = EXPANSION_TERMS[query_class][:2]
    search_queries = [question] + [f"{question} {term}".strip() for term in expansions]
    recency_sensitive = is_recency_sensitive(question)
    return QueryPlan(
        query_class=query_class,
        search_queries=search_queries,
        source_families=CLASS_FAMILY_MAP[query_class],
        top_k=7 if query_class in {QueryClass.distributed_multi_gpu, QueryClass.hardware_topology} else 5,
        use_tavily_fallback=settings.use_tavily_fallback,
        confidence_floor=0.26 if query_class == QueryClass.hardware_topology else 0.30,
        max_retries=2,
        recency_sensitive=recency_sensitive,
    )


def rewrite_query(question: str, query_class: QueryClass, attempt: int = 1) -> str:
    hint = " ".join(EXPANSION_TERMS[query_class][: 2 + min(attempt, 1)])
    return f"{question} {hint}".strip()


def rerank_results(question: str, plan: QueryPlan, results: list[RetrieverResult]) -> list[RetrieverResult]:
    query_tokens = set(tokenize(question))
    reranked: list[RetrieverResult] = []
    source_hints = QUERY_CLASS_SOURCE_HINTS.get(plan.query_class, {})
    for result in results:
        chunk_tokens = set(result.chunk.sparse_terms or tokenize(result.chunk.content))
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        family_bonus = 0.14 if result.chunk.doc_family in plan.source_families else 0.0
        if plan.query_class == QueryClass.hardware_topology and result.chunk.doc_family == "hardware":
            family_bonus += 0.22
        metadata_bonus = 0.06 if any(token in result.chunk.section_path.lower() for token in query_tokens) else 0.0
        source_bonus = source_hints.get(result.chunk.source_id, 0.0)
        tag_bonus = 0.0
        if any(token in " ".join(result.chunk.product_tags).lower() for token in query_tokens):
            tag_bonus = 0.08
        rerank_score = result.score + (0.28 * overlap) + family_bonus + metadata_bonus + source_bonus + tag_bonus
        reranked.append(result.model_copy(update={"rerank_score": rerank_score, "score": rerank_score}))
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


def grade_results(question: str, plan: QueryPlan, results: list[RetrieverResult]) -> tuple[list[RetrieverResult], list[str]]:
    query_tokens = set(tokenize(question))
    accepted: list[RetrieverResult] = []
    rejected: list[str] = []
    for result in results:
        chunk_tokens = set(result.chunk.sparse_terms or tokenize(result.chunk.content))
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        family_match = result.chunk.doc_family in plan.source_families
        threshold = 0.22 if plan.query_class == QueryClass.hardware_topology else 0.18
        if result.score >= threshold or (family_match and overlap >= 0.08):
            accepted.append(result)
        else:
            rejected.append(result.chunk.id)
    return accepted, rejected


def estimate_confidence(results: list[RetrieverResult]) -> float:
    if not results:
        return 0.0
    return round(mean(result.score for result in results[:3]), 4)


def needs_retry(plan: QueryPlan, results: list[RetrieverResult]) -> bool:
    return estimate_confidence(results) < plan.confidence_floor


class RetrievalService:
    def __init__(self, settings: Settings, sources: list[DocumentSource], index: SearchIndex) -> None:
        self.settings = settings
        self.sources = sources
        self.index = index

    def list_sources(self) -> list[DocumentSource]:
        return self.sources

    def build_plan(self, question: str) -> QueryPlan:
        return build_query_plan(question, self.settings)

    def run_retrieval_pass(self, question: str, plan: QueryPlan, query: str) -> tuple[list[RetrieverResult], list[str], float, list[TraceEvent]]:
        trace: list[TraceEvent] = []
        candidates: dict[str, RetrieverResult] = {}
        queries = [query]
        if query == question:
            queries = plan.search_queries[:3]
        retrieval_top_k = max(plan.top_k * 4, 12)

        total_retrieved = 0
        for active_query in queries:
            retrieved = self.index.search(active_query, top_k=retrieval_top_k, families=plan.source_families)
            total_retrieved += len(retrieved)
            trace.append(
                TraceEvent(
                    type="retrieval",
                    message=f"Retrieved {len(retrieved)} candidates from the hybrid index",
                    payload={"query": active_query, "families": plan.source_families, "top_k": retrieval_top_k},
                )
            )
            for item in retrieved:
                existing = candidates.get(item.chunk.id)
                if existing is None or item.score > existing.score:
                    candidates[item.chunk.id] = item

        reranked = rerank_results(question, plan, list(candidates.values()))
        confidence = estimate_confidence(reranked)
        trace.append(
            TraceEvent(
                type="rerank",
                message="Reranked candidates using lexical overlap, source-family routing, and metadata features",
                payload={"top_chunk_ids": [item.chunk.id for item in reranked[:3]], "confidence": confidence, "retrieved_total": total_retrieved},
            )
        )
        accepted, rejected = grade_results(question, plan, reranked)
        trace.append(
            TraceEvent(
                type="document_grading",
                message="Filtered weak chunks before synthesis",
                payload={"accepted": [item.chunk.id for item in accepted[:5]], "rejected": rejected[:8]},
            )
        )
        return accepted[: plan.top_k], rejected, estimate_confidence(accepted), trace

    def merge_results(self, question: str, plan: QueryPlan, left: list[RetrieverResult], right: list[RetrieverResult]) -> tuple[list[RetrieverResult], list[str], float]:
        merged: dict[str, RetrieverResult] = {item.chunk.id: item for item in left}
        for item in right:
            existing = merged.get(item.chunk.id)
            if existing is None or item.score > existing.score:
                merged[item.chunk.id] = item
        reranked = rerank_results(question, plan, list(merged.values()))
        accepted, rejected = grade_results(question, plan, reranked)
        return accepted[: plan.top_k], rejected, estimate_confidence(accepted)

    def search(self, question: str) -> SearchDebugResponse:
        plan = self.build_plan(question)
        trace = [
            TraceEvent(
                type="classification",
                message=f"Classified question as {plan.query_class.value}",
                payload={"source_families": plan.source_families, "search_queries": plan.search_queries[:2]},
            )
        ]

        results, rejected, confidence, pass_trace = self.run_retrieval_pass(question, plan, question)
        trace.extend(pass_trace)
        rewritten_query = None
        retry_count = 0

        if needs_retry(plan, results):
            retry_count = 1
            rewritten_query = rewrite_query(question, plan.query_class, retry_count)
            retry_results, retry_rejected, retry_confidence, retry_trace = self.run_retrieval_pass(question, plan, rewritten_query)
            trace.extend(
                [
                    TraceEvent(
                        type="rewrite",
                        message="Ran a second retrieval pass with a rewritten query",
                        payload={"rewritten_query": rewritten_query},
                    )
                ]
            )
            trace.extend(retry_trace)
            results, rejected, confidence = self.merge_results(question, plan, results, retry_results)
            rejected = list(dict.fromkeys(rejected + retry_rejected))
            confidence = max(confidence, retry_confidence)

        return SearchDebugResponse(
            plan=plan,
            results=results,
            rewritten_query=rewritten_query,
            confidence=confidence,
            retry_count=retry_count,
            rejected_chunk_ids=rejected,
            trace=trace,
        )
