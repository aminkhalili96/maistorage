from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from pinecone import Pinecone
from pypdf import PdfReader

from app.config import Settings, get_settings
from app.corpus import load_corpus_manifest, load_normalized_chunks, load_sources
from app.models import ChatRequest, ChunkRecord, Citation, DocumentSource, RetrieverResult
from app.services.agent import AgentService
from app.services.chunking import chunk_html_document, chunk_pdf_document
from app.services.evaluation import EvaluationService
from app.services.indexes import SearchIndex
from app.services.providers import Embedder, OpenAIReasoner, TavilyClient, build_embedder, tokenize
from app.services.retrieval import RetrievalService, grade_results, rerank_results, rewrite_query


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def benchmark_source_ids(question_items: list[dict[str, Any]]) -> list[str]:
    source_ids = {source_id for item in question_items for source_id in item.get("expected_sources", [])}
    return sorted(source_ids)


def sample_chunks_per_source(chunks: list[ChunkRecord], max_per_source: int | None) -> list[ChunkRecord]:
    if not max_per_source or max_per_source <= 0:
        return chunks
    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_id, []).append(chunk)

    sampled: list[ChunkRecord] = []
    for source_id in sorted(grouped):
        items = grouped[source_id]
        if len(items) <= max_per_source:
            sampled.extend(items)
            continue
        step = (len(items) - 1) / max(max_per_source - 1, 1)
        chosen = {round(index * step) for index in range(max_per_source)}
        sampled.extend(items[index] for index in sorted(chosen))
    return sampled


@dataclass(slots=True)
class BenchmarkBundle:
    settings: Settings
    sources: list[DocumentSource]
    source_ids: list[str]
    chunks: list[ChunkRecord]
    snapshot_id: str | None
    question_path: Path
    questions: list[dict[str, Any]]


def load_benchmark_bundle(
    settings: Settings,
    question_path: Path | None = None,
    source_ids: list[str] | None = None,
    max_chunks_per_source: int | None = None,
) -> BenchmarkBundle:
    question_path = question_path or settings.golden_questions_path
    questions = load_json(question_path)
    selected_source_ids = source_ids or benchmark_source_ids(questions)
    source_map = {source.id: source for source in load_sources(settings.source_manifest_path)}
    selected_sources = [source_map[source_id] for source_id in selected_source_ids if source_id in source_map]
    chunks = [chunk for chunk in load_normalized_chunks(settings.normalized_doc_root) if chunk.source_id in set(selected_source_ids)]
    chunks = sample_chunks_per_source(chunks, max_chunks_per_source)
    manifest = load_corpus_manifest(settings.corpus_manifest_path)
    return BenchmarkBundle(
        settings=settings,
        sources=selected_sources,
        source_ids=selected_source_ids,
        chunks=chunks,
        snapshot_id=manifest.get("snapshot_id"),
        question_path=question_path,
        questions=questions,
    )


def local_settings(settings: Settings, **overrides: Any) -> Settings:
    base = {
        "app_mode": "dev",
        "use_pinecone": False,
        "langsmith_tracing": False,
    }
    base.update(overrides)
    return replace(settings, **base)


def aggregate_retrieval_rows(rows: list[Any]) -> dict[str, float]:
    if not rows:
        return {"hit@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0, "routing@3": 0.0, "retry_count": 0.0}
    return {
        "hit@5": round(mean(float(row.metrics["hit@5"]) for row in rows), 4),
        "mrr": round(mean(float(row.metrics["mrr"]) for row in rows), 4),
        "ndcg@5": round(mean(float(row.metrics["ndcg@5"]) for row in rows), 4),
        "routing@3": round(mean(float(row.metrics["routing@3"]) for row in rows), 4),
        "retry_count": round(mean(float(row.metrics["retry_count"]) for row in rows), 4),
    }


def summarize_ragas(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "ok":
        return result
    rows = result.get("scores", [])
    if not rows:
        return {"status": "failed", "reason": "RAGAS returned no score rows."}
    numeric_keys = sorted(
        key for key in rows[0] if all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in rows)
    )
    summary = {key: round(mean(float(row[key]) for row in rows), 4) for key in numeric_keys}
    return {
        "status": "ok",
        "summary": summary,
        "question_count": len(rows),
        "scores": rows,
    }


class CachedEmbedder:
    def __init__(
        self,
        delegate: Embedder,
        settings: Settings,
        cache_root: Path,
        *,
        provider: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        document_task: str | None = None,
        query_task: str | None = None,
    ) -> None:
        self.delegate = delegate
        self.cache_root = cache_root
        self.provider = provider or getattr(delegate, "provider", settings.embedder_provider)
        self.model = model or getattr(delegate, "model", settings.openai_embedding_model if settings.embedder_provider == "openai" else "keyword")
        self.dimensions = dimensions or getattr(delegate, "dimensions", settings.openai_embedding_dimensions if settings.embedder_provider == "openai" else 128)
        self.document_task = document_task or getattr(delegate, "document_task_type", "documents")
        self.query_task = query_task or getattr(delegate, "query_task_type", "query")

    def _path_for(self, text: str, task_type: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.cache_root / self.provider / self.model / str(self.dimensions) / task_type / f"{digest}.json"

    def _read(self, path: Path) -> list[float] | None:
        if not path.exists():
            return None
        return [float(value) for value in json.loads(path.read_text())]

    def _write(self, path: Path, vector: list[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vector))

    def _embed_cached(self, text: str, task_type: str, embed_fn) -> list[float]:
        path = self._path_for(text, task_type)
        cached = self._read(path)
        if cached is not None:
            return cached
        vector = [float(value) for value in embed_fn(text)]
        self._write(path, vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_cached(text, self.document_task, lambda value: self.delegate.embed_documents([value])[0]) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_cached(text, self.query_task, self.delegate.embed_query)


class TrackingEmbedder:
    def __init__(self, delegate: Embedder) -> None:
        self.delegate = delegate
        self.reset()

    def reset(self) -> None:
        self.document_calls = 0
        self.document_texts = 0
        self.document_ms = 0.0
        self.query_calls = 0
        self.query_ms = 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "document_calls": self.document_calls,
            "document_texts": self.document_texts,
            "document_ms": round(self.document_ms, 2),
            "query_calls": self.query_calls,
            "query_ms": round(self.query_ms, 2),
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        vectors = self.delegate.embed_documents(texts)
        self.document_calls += 1
        self.document_texts += len(texts)
        self.document_ms += (time.perf_counter() - started) * 1000
        return vectors

    def embed_query(self, text: str) -> list[float]:
        started = time.perf_counter()
        vector = self.delegate.embed_query(text)
        self.query_calls += 1
        self.query_ms += (time.perf_counter() - started) * 1000
        return vector


class PineconeHostedEmbedder:
    def __init__(
        self,
        settings: Settings,
        *,
        model: str,
        dimensions: int,
        document_task_type: str = "passage",
        query_task_type: str = "query",
    ) -> None:
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone-hosted embedding experiments.")
        self.provider = "pinecone"
        self.model = model
        self.dimensions = dimensions
        self.document_task_type = document_task_type
        self.query_task_type = query_task_type
        self.client = Pinecone(api_key=settings.pinecone_api_key)

    def _embed(self, text: str, input_type: str) -> list[float]:
        response = self.client.inference.embed(
            model=self.model,
            inputs=[text],
            parameters={"input_type": input_type, "truncate": "END"},
        )
        item = response.data[0]
        values = item["values"] if isinstance(item, dict) else item.values
        return [float(value) for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, self.document_task_type) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, self.query_task_type)


class TrackingReasoner:
    def __init__(self, delegate: OpenAIReasoner) -> None:
        self.delegate = delegate
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.delegate.enabled

    def reset(self) -> None:
        self.calls = 0
        self.ms = 0.0

    def stats(self) -> dict[str, float | int]:
        return {"generation_calls": self.calls, "generation_ms": round(self.ms, 2)}

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        started = time.perf_counter()
        text = self.delegate.generate_text(prompt, model=model)
        self.calls += 1
        self.ms += (time.perf_counter() - started) * 1000
        return text


class TrackingTavilyClient:
    def __init__(self, delegate: TavilyClient) -> None:
        self.delegate = delegate
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.ms = 0.0

    def stats(self) -> dict[str, float | int]:
        return {"tavily_calls": self.calls, "tavily_ms": round(self.ms, 2)}

    def search(self, query: str) -> list[dict[str, str]]:
        started = time.perf_counter()
        payload = self.delegate.search(query)
        self.calls += 1
        self.ms += (time.perf_counter() - started) * 1000
        return payload


class DisabledTavilyClient:
    def search(self, _query: str) -> list[dict[str, str]]:
        return []


class WeightedInMemoryIndex:
    def __init__(self, embedder: Embedder, lexical_weight: float = 0.55, dense_weight: float = 0.45) -> None:
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.chunks: dict[str, ChunkRecord] = {}
        self.vectors: dict[str, list[float]] = {}
        self.term_counts: dict[str, Counter[str]] = {}

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        vectors = self.embedder.embed_documents([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=False):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = vector
            self.term_counts[chunk.id] = Counter(chunk.sparse_terms or tokenize(chunk.content))

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)
            self.vectors.pop(chunk_id, None)
            self.term_counts.pop(chunk_id, None)

    def search(self, query: str, top_k: int = 5, families: list[str] | None = None) -> list[RetrieverResult]:
        query_tokens = tokenize(query)
        query_counter = Counter(query_tokens)
        query_vector = self.embedder.embed_query(query) if self.dense_weight else []
        results: list[RetrieverResult] = []
        for chunk_id, chunk in self.chunks.items():
            if families and chunk.doc_family not in families:
                continue
            chunk_terms = self.term_counts[chunk_id]
            overlap = sum(min(query_counter[token], chunk_terms[token]) for token in query_counter)
            lexical_score = overlap / max(len(query_tokens), 1)
            dense_score = _cosine_similarity(query_vector, self.vectors[chunk_id]) if self.dense_weight else 0.0
            score = (self.lexical_weight * lexical_score) + (self.dense_weight * dense_score)
            results.append(
                RetrieverResult(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical_score,
                    dense_score=dense_score,
                    retrieval_method="dense" if self.lexical_weight == 0 else "hybrid",
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self.chunks)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    numerator = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return numerator / (left_norm * right_norm)


@dataclass(slots=True)
class ExperimentStack:
    settings: Settings
    sources: list[DocumentSource]
    chunks: list[ChunkRecord]
    embedder: TrackingEmbedder | CachedEmbedder | Embedder
    index: SearchIndex
    retrieval: RetrievalService
    agent: AgentService
    evaluation: EvaluationService
    reasoner: TrackingReasoner | OpenAIReasoner
    tavily: TrackingTavilyClient | TavilyClient | DisabledTavilyClient


def build_local_stack(
    settings: Settings,
    sources: list[DocumentSource],
    chunks: list[ChunkRecord],
    *,
    cache_root: Path,
    embedder_provider: str = "openai",
    embedder_model: str | None = None,
    openai_embedding_dimensions: int = 3072,
    pinecone_embedding_dimensions: int = 1024,
    pinecone_document_task_type: str = "passage",
    pinecone_query_task_type: str = "query",
    use_tavily_fallback: bool = False,
    lexical_weight: float = 0.55,
    dense_weight: float = 0.45,
    track_calls: bool = False,
) -> ExperimentStack:
    experiment_settings = local_settings(
        settings,
        embedder_provider=embedder_provider,
        openai_embedding_dimensions=openai_embedding_dimensions,
        use_tavily_fallback=use_tavily_fallback,
    )
    if embedder_provider == "pinecone":
        raw_embedder = PineconeHostedEmbedder(
            experiment_settings,
            model=embedder_model or "multilingual-e5-large",
            dimensions=pinecone_embedding_dimensions,
            document_task_type=pinecone_document_task_type,
            query_task_type=pinecone_query_task_type,
        )
        cached_embedder = CachedEmbedder(
            raw_embedder,
            experiment_settings,
            cache_root,
            provider="pinecone",
            model=raw_embedder.model,
            dimensions=raw_embedder.dimensions,
            document_task=raw_embedder.document_task_type,
            query_task=raw_embedder.query_task_type,
        )
    else:
        raw_embedder = build_embedder(experiment_settings)
        cached_embedder = CachedEmbedder(raw_embedder, experiment_settings, cache_root)
    embedder: TrackingEmbedder | CachedEmbedder = TrackingEmbedder(cached_embedder) if track_calls else cached_embedder
    index = WeightedInMemoryIndex(embedder, lexical_weight=lexical_weight, dense_weight=dense_weight)
    index.upsert(chunks)
    if track_calls and isinstance(embedder, TrackingEmbedder):
        embedder.reset()

    retrieval = RetrievalService(experiment_settings, sources, index)
    base_reasoner = OpenAIReasoner(experiment_settings)
    reasoner: TrackingReasoner | OpenAIReasoner = TrackingReasoner(base_reasoner) if track_calls else base_reasoner
    if track_calls and isinstance(reasoner, TrackingReasoner):
        reasoner.reset()

    if use_tavily_fallback:
        base_tavily = TavilyClient(experiment_settings)
        tavily: TrackingTavilyClient | TavilyClient = TrackingTavilyClient(base_tavily) if track_calls else base_tavily
        if track_calls and isinstance(tavily, TrackingTavilyClient):
            tavily.reset()
    else:
        tavily = DisabledTavilyClient()

    agent = AgentService(experiment_settings, retrieval, reasoner, tavily)
    evaluation = EvaluationService(experiment_settings, experiment_settings.golden_questions_path, retrieval, agent)
    return ExperimentStack(
        settings=experiment_settings,
        sources=sources,
        chunks=chunks,
        embedder=embedder,
        index=index,
        retrieval=retrieval,
        agent=agent,
        evaluation=evaluation,
        reasoner=reasoner,
        tavily=tavily,
    )


def citation_from_result(result: RetrieverResult) -> Citation:
    return Citation(
        chunk_id=result.chunk.id,
        title=result.chunk.title,
        url=result.chunk.url,
        section_path=result.chunk.section_path,
        snippet=result.chunk.content[:240].strip(),
        source_kind=result.chunk.source_kind,
    )


def keyword_hit(answer: str, expected_terms: list[str]) -> bool:
    lowered = answer.lower()
    return any(term.lower() in lowered for term in expected_terms)


def run_manual_pipeline_mode(stack: ExperimentStack, question: str, mode: str) -> dict[str, Any]:
    plan = stack.retrieval.build_plan(question)
    started = time.perf_counter()
    rewritten_query = None
    used_fallback = False
    response_mode = "corpus-backed"
    rejected_chunk_ids: list[str] = []
    confidence = 0.0

    if mode == "retrieval_only":
        results = stack.index.search(question, top_k=plan.top_k, families=plan.source_families)
        confidence = round(mean(item.score for item in results[:3]), 4) if results else 0.0
    else:
        results, rejected_chunk_ids, confidence, _ = stack.retrieval.run_retrieval_pass(question, plan, question)
        if mode in {"retrieval_grading_rewrite", "full_agentic"} and confidence < plan.confidence_floor:
            rewritten_query = rewrite_query(question, plan.query_class, 1)
            retry_results, retry_rejected, retry_confidence, _ = stack.retrieval.run_retrieval_pass(question, plan, rewritten_query)
            results, merged_rejected, confidence = stack.retrieval.merge_results(question, plan, results, retry_results)
            rejected_chunk_ids = list(dict.fromkeys(rejected_chunk_ids + retry_rejected + merged_rejected))
            confidence = max(confidence, retry_confidence)
        if mode == "full_agentic" and (plan.recency_sensitive or not results):
            state = stack.agent.run(ChatRequest(question=question))
            return {
                "answer": state.answer,
                "citations": [citation.model_dump() for citation in state.citations],
                "confidence": state.confidence,
                "grounding_passed": state.grounding_passed,
                "answer_quality_passed": state.answer_quality_passed,
                "response_mode": state.response_mode,
                "used_fallback": state.used_fallback,
                "retry_count": state.retry_count,
                "rejected_chunk_ids": state.rejected_chunk_ids,
                "rewritten_query": state.rewritten_query,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }

    citations = [citation_from_result(result) for result in results[:4]]
    if not results:
        response_mode = "insufficient-evidence"
        answer = stack.agent._refusal_answer()
    else:
        answer = stack.agent._synthesize_answer(question, results)
    grounding_passed = stack.agent._grounding_check(answer, citations)
    answer_quality_passed = stack.agent._answer_quality_check(question, answer)
    return {
        "answer": answer,
        "citations": [citation.model_dump() for citation in citations],
        "confidence": confidence,
        "grounding_passed": grounding_passed,
        "answer_quality_passed": answer_quality_passed,
        "response_mode": response_mode,
        "used_fallback": used_fallback,
        "retry_count": 1 if rewritten_query else 0,
        "rejected_chunk_ids": rejected_chunk_ids,
        "rewritten_query": rewritten_query,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def load_rechunked_chunks(settings: Settings, source_ids: list[str], max_chars: int, overlap: int) -> tuple[list[DocumentSource], list[ChunkRecord]]:
    source_map = {source.id: source for source in load_sources(settings.source_manifest_path)}
    selected_sources = [source_map[source_id] for source_id in source_ids if source_id in source_map]
    manifest = load_corpus_manifest(settings.corpus_manifest_path)
    all_chunks: list[ChunkRecord] = []

    for source in selected_sources:
        source_payload = manifest.get("sources", {}).get(source.id, {})
        bound_source = source.model_copy(
            update={
                "retrieved_at": source_payload.get("retrieved_at") or manifest.get("retrieved_at"),
                "snapshot_id": source_payload.get("snapshot_id") or manifest.get("snapshot_id"),
                "doc_version": source_payload.get("doc_version"),
            }
        )
        html_root = settings.raw_html_root / source.id
        pdf_root = settings.raw_pdf_root / source.id

        if html_root.exists():
            for html_path in sorted(html_root.glob("*.html")):
                page_url = source_payload.get("local_url_map", {}).get(html_path.name, source.url)
                all_chunks.extend(
                    chunk_html_document(
                        bound_source,
                        page_url,
                        html_path.read_text(),
                        updated_at=bound_source.retrieved_at,
                        max_chars=max_chars,
                        overlap=overlap,
                    )
                )

        if pdf_root.exists():
            for pdf_path in sorted(pdf_root.glob("*.pdf")):
                reader = PdfReader(str(pdf_path))
                pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
                all_chunks.extend(
                    chunk_pdf_document(
                        bound_source,
                        source_payload.get("pdf_url") or source.pdf_url or source.url,
                        pdf_text,
                        updated_at=bound_source.retrieved_at,
                        title=f"{source.title} PDF",
                        max_chars=max_chars,
                        overlap=overlap,
                    )
                )
    return selected_sources, all_chunks


def build_result_metadata(
    *,
    config_name: str,
    bundle: BenchmarkBundle,
    question_path: Path,
    notes: str,
    chunk_count: int | None = None,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    effective_source_ids = source_ids or bundle.source_ids
    return {
        "config_name": config_name,
        "snapshot_id": bundle.snapshot_id,
        "question_set": str(question_path),
        "source_ids": effective_source_ids,
        "source_count": len(effective_source_ids),
        "chunk_count": chunk_count if chunk_count is not None else len(bundle.chunks),
        "run_at": utc_now(),
        "notes": notes,
    }


def default_output_dir(settings: Settings) -> Path:
    return settings.project_root / "data/evals/results" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def get_project_settings() -> Settings:
    return get_settings()
