"""
Pydantic models shared across the entire backend.

These models define the data contracts between all services (retrieval, agent,
ingestion, evaluation) and between backend and frontend (via SSE JSON payloads).
The frontend TypeScript types in types.ts mirror these definitions.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryClass(str, Enum):
    """Five query classes for retrieval routing — determines source families and expansion terms."""
    training_optimization = "training_optimization"
    distributed_multi_gpu = "distributed_multi_gpu"
    deployment_runtime = "deployment_runtime"
    hardware_topology = "hardware_topology"
    general = "general"


class DocumentSource(BaseModel):
    """A source in the knowledge base registry (data/sources/nvidia_sources.json).
    Each source maps to one JSONL file of chunks in data/knowledge_base/normalized/."""
    id: str
    title: str
    url: str
    doc_family: str                                     # core | distributed | infrastructure | advanced | hardware
    doc_type: str                                       # html | pdf | markdown
    crawl_prefix: str
    product_tags: list[str] = Field(default_factory=list)
    source_kind: str = "knowledge_base"
    local_html_paths: list[str] = Field(default_factory=list)
    local_pdf_paths: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    doc_version: str | None = None
    retrieved_at: str | None = None
    snapshot_id: str | None = None
    enabled: bool = True


class ChunkRecord(BaseModel):
    """A single chunk of content from the knowledge base — the atomic unit of retrieval.
    Stored as one line per chunk in JSONL files. Indexed in the hybrid search index."""
    id: str                                             # Deterministic: {source_id}-{section_slug}-{content_hash}
    source_id: str                                      # Links back to DocumentSource.id
    title: str                                          # Effective title (may differ from source title — P7 fix)
    url: str
    section_path: str                                   # Heading hierarchy, e.g., "Installation > Prerequisites"
    doc_family: str
    doc_type: str
    product_tags: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    retrieved_at: str | None = None
    content_hash: str | None = None                     # SHA-256 prefix for dedup
    doc_version: str | None = None
    snapshot_id: str | None = None
    source_kind: str = "knowledge_base"                 # "knowledge_base" or "web" (Tavily results)
    content: str                                        # The actual text content of the chunk
    sparse_terms: list[str] = Field(default_factory=list)  # Pre-tokenized terms for TF-IDF matching
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieverResult(BaseModel):
    """A chunk with its retrieval and reranking scores — passed through the pipeline."""
    chunk: ChunkRecord
    score: float                                        # Raw index score (TF-IDF or dense similarity)
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0                           # Composite score after multi-signal reranking
    retrieval_method: str = "hybrid"                    # hybrid | tavily


class Citation(BaseModel):
    """A citation chip shown in the frontend — links an answer passage to its source chunk."""
    chunk_id: str
    title: str
    url: str
    citation_url: str                                   # URL with section anchor for deep linking
    domain: str
    section_path: str
    snippet: str                                        # First 240 chars of chunk content
    source_kind: str = "knowledge_base"
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
    multi_hop = "multi_hop"
    routing_decision = "routing_decision"
    claim_verification = "claim_verification"


class TraceEvent(BaseModel):
    """An event in the agent trace — emitted progressively via SSE as each graph node completes.
    The frontend renders these as the "thinking" steps in the agent trace panel."""
    type: str  # kept as str for backwards compat; validated values in TraceEventType
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class QueryPlan(BaseModel):
    """Retrieval plan built by classify → plan step. Determines how retrieval is executed:
    which queries to run, which source families to prioritize, and when to retry or fall back."""
    query_class: QueryClass
    search_queries: list[str]                           # 2-3 queries: original + expansion variants
    source_families: list[str]                          # Families to boost during reranking
    top_k: int = 5                                      # Max chunks to return (adaptive: 3-15)
    use_tavily_fallback: bool = False
    confidence_floor: float = 0.28                      # Below this → trigger rewrite/fallback
    max_retries: int = 2
    recency_sensitive: bool = False                     # True for "latest/current/recent" queries


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
    """Complete result of an agent run — returned by AgentService.run() and cached by SemanticCache.
    Contains everything the frontend needs: answer, citations, trace, and quality signals."""
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
    response_mode: str = "knowledge-base-backed"
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
    loaded_demo_knowledge_base: bool = False


class SearchDebugRequest(BaseModel):
    question: str


class EvaluationRow(BaseModel):
    question: str
    query_class: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    metrics: dict[str, float | int | str | bool]
