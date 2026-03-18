from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryClass(str, Enum):
    training_optimization = "training_optimization"
    distributed_multi_gpu = "distributed_multi_gpu"
    deployment_runtime = "deployment_runtime"
    hardware_topology = "hardware_topology"
    general = "general"


class DocumentSource(BaseModel):
    id: str
    title: str
    url: str
    doc_family: str
    doc_type: str
    crawl_prefix: str
    product_tags: list[str] = Field(default_factory=list)
    source_kind: str = "corpus"
    local_html_paths: list[str] = Field(default_factory=list)
    local_pdf_paths: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    doc_version: str | None = None
    retrieved_at: str | None = None
    snapshot_id: str | None = None
    enabled: bool = True


class ChunkRecord(BaseModel):
    id: str
    source_id: str
    title: str
    url: str
    section_path: str
    doc_family: str
    doc_type: str
    product_tags: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    retrieved_at: str | None = None
    content_hash: str | None = None
    doc_version: str | None = None
    snapshot_id: str | None = None
    source_kind: str = "corpus"
    content: str
    sparse_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieverResult(BaseModel):
    chunk: ChunkRecord
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_method: str = "hybrid"


class Citation(BaseModel):
    chunk_id: str
    title: str
    url: str
    citation_url: str
    domain: str
    section_path: str
    snippet: str
    source_kind: str = "corpus"
    source_id: str = ""
    score: float | None = None
    char_count: int | None = None
    page: int | None = None


class TraceEventType(str, Enum):
    classification = "classification"
    retrieval = "retrieval"
    retrieve = "retrieve"
    rerank = "rerank"
    document_grading = "document_grading"
    rewrite = "rewrite"
    fallback = "fallback"
    generation = "generation"
    generation_error = "generation_error"
    self_reflect = "self_reflect"
    grounding_check = "grounding_check"
    answer_quality_check = "answer_quality_check"
    citation = "citation"
    query_reformulation = "query_reformulation"
    error = "error"
    done = "done"


class TraceEvent(BaseModel):
    type: str  # kept as str for backwards compat; validated values in TraceEventType
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class QueryPlan(BaseModel):
    query_class: QueryClass
    search_queries: list[str]
    source_families: list[str]
    top_k: int = 5
    use_tavily_fallback: bool = False
    confidence_floor: float = 0.28
    max_retries: int = 2
    recency_sensitive: bool = False


class SearchDebugResponse(BaseModel):
    plan: QueryPlan
    results: list[RetrieverResult]
    rewritten_query: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    confidence: float = 0.0
    retry_count: int = 0
    rejected_chunk_ids: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)


class AgentRunState(BaseModel):
    question: str
    model_used: str
    assistant_mode: str = "doc_rag"
    query_plan: QueryPlan | None = None
    rewritten_query: str | None = None
    retrieval_results: list[RetrieverResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    answer: str = ""
    used_fallback: bool = False
    confidence: float = 0.0
    response_mode: str = "corpus-backed"
    retry_count: int = 0
    rejected_chunk_ids: list[str] = Field(default_factory=list)
    grounding_passed: bool = False
    answer_quality_passed: bool = False
    generation_degraded: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = Field(default_factory=list)
    model: str | None = None


class IngestRequest(BaseModel):
    families: list[str] = Field(default_factory=list)
    force_refresh: bool = False


class IngestionStatus(BaseModel):
    active: bool = False
    last_job_id: str | None = None
    snapshot_id: str | None = None
    source_counts: dict[str, int] = Field(default_factory=dict)
    chunk_counts: dict[str, int] = Field(default_factory=dict)
    changed_sources: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    last_refresh_at: str | None = None
    loaded_demo_corpus: bool = False


class SearchDebugRequest(BaseModel):
    question: str


class EvaluationRow(BaseModel):
    question: str
    query_class: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    metrics: dict[str, float | int | str | bool]
