from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from typing import Protocol

from app.config import Settings
from app.models import ChunkRecord, RetrieverResult
from app.services.providers import Embedder, tokenize

_log = logging.getLogger("maistorage.indexes")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    numerator = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return numerator / (left_norm * right_norm)


def _sparse_vector(text: str, limit: int = 64) -> dict[str, list[float] | list[int]]:
    counts = Counter(tokenize(text))
    ranked = counts.most_common(limit)
    merged: dict[int, float] = {}
    for token, count in ranked:
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 20000
        merged[index] = merged.get(index, 0.0) + float(count)
    indices = list(merged.keys())
    values = list(merged.values())
    return {"indices": indices, "values": values}


def _sanitize_pinecone_metadata(payload: dict) -> dict:
    sanitized: dict = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            sanitized[key] = value
            continue
        if key == "metadata" and isinstance(value, dict) and not value:
            continue
    return sanitized


class SearchIndex(Protocol):
    def upsert(self, chunks: list[ChunkRecord]) -> None:
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        ...

    def search(self, query: str, top_k: int = 5, families: list[str] | None = None) -> list[RetrieverResult]:
        ...

    def count(self) -> int:
        ...


class InMemoryHybridIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: dict[str, ChunkRecord] = {}
        self.vectors: dict[str, list[float]] = {}
        self.term_counts: dict[str, Counter[str]] = {}

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        vectors = self.embedder.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks.")
        for chunk, vector in zip(chunks, vectors):
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
        query_vector = self.embedder.embed_query(query)
        results: list[RetrieverResult] = []

        for chunk_id, chunk in self.chunks.items():
            if families and chunk.doc_family not in families:
                continue
            chunk_terms = self.term_counts[chunk_id]
            overlap = sum(min(query_counter[token], chunk_terms[token]) for token in query_counter)
            lexical_score = overlap / max(len(query_tokens), 1)
            dense_score = _cosine_similarity(query_vector, self.vectors[chunk_id])
            score = (0.55 * lexical_score) + (0.45 * dense_score)
            results.append(
                RetrieverResult(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical_score,
                    dense_score=dense_score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self.chunks)


class PineconeHybridIndex:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        if not settings.pinecone_api_key or not settings.pinecone_index_name:
            raise ValueError("Pinecone configuration is incomplete.")
        from pinecone import Pinecone

        self.embedder = embedder
        self.namespace = settings.pinecone_namespace
        self.expected_dimensions = settings.openai_embedding_dimensions
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.client.Index(settings.pinecone_index_name)

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        vectors = self.embedder.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks.")
        payload = []
        for chunk, vector in zip(chunks, vectors):
            metadata = _sanitize_pinecone_metadata(chunk.model_dump())
            metadata["content"] = chunk.content
            payload.append(
                {
                    "id": chunk.id,
                    "values": vector,
                    "sparse_values": _sparse_vector(chunk.content),
                    "metadata": metadata,
                }
            )
        if payload:
            self.index.upsert(vectors=payload, namespace=self.namespace)

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.index.delete(ids=chunk_ids, namespace=self.namespace)

    def search(self, query: str, top_k: int = 5, families: list[str] | None = None) -> list[RetrieverResult]:
        filter_payload = None
        if families:
            filter_payload = {"doc_family": {"$in": families}}
        try:
            response = self.index.query(
                namespace=self.namespace,
                vector=self.embedder.embed_query(query),
                sparse_vector=_sparse_vector(query),
                top_k=top_k,
                include_metadata=True,
                filter=filter_payload,
            )
        except Exception as exc:
            _log.warning("Pinecone query failed (%s: %s), returning empty results", type(exc).__name__, str(exc)[:120])
            return []
        matches = getattr(response, "matches", [])
        results: list[RetrieverResult] = []
        for match in matches:
            metadata = dict(getattr(match, "metadata", {}) or {})
            chunk = ChunkRecord.model_validate(metadata)
            results.append(
                RetrieverResult(
                    chunk=chunk,
                    score=float(getattr(match, "score", 0.0)),
                    retrieval_method="pinecone-hybrid",
                )
            )
        return results

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", {}) or {}
        namespace_stats = namespaces.get(self.namespace, {})
        return int(namespace_stats.get("vector_count", 0))
