"""
Service wiring and dependency injection — the composition root.

get_services() is the single entry point for the entire backend. It builds all
services in the correct order and returns a thread-safe singleton AppServices.
Uses double-checked locking (not @lru_cache) for safe concurrent first-request handling.

Boot sequence:
  1. Load settings from .env (get_settings)
  2. Validate runtime config (assessment mode requires OpenAI + Pinecone)
  3. Load source registry + demo chunks from JSON files
  4. Build embedder (keyword for dev, OpenAI for assessment)
  5. Build search index (InMemory for dev, Pinecone for assessment)
  6. Build RetrievalService → IngestionService → bootstrap knowledge base into index
  7. Build OpenAIReasoner, TavilyClient, AgentService, EvaluationService
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import DocumentSource
from app.services.agent import AgentService
from app.services.evaluation import EvaluationService
from app.services.indexes import InMemoryHybridIndex, PineconeHybridIndex, SearchIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, TavilyClient, build_embedder
from app.services.retrieval import RetrievalService

_log = logging.getLogger("maistorage.runtime")

_services_lock = threading.Lock()
_services_instance: AppServices | None = None


def _validate_langsmith(settings: Settings) -> Settings:
    """Ping the LangSmith API and disable tracing if the key is invalid.

    Returns a (possibly updated) Settings object so the caller can rely on
    ``settings.langsmith_tracing`` being accurate from startup onward.
    """
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return settings
    try:
        import httpx  # soft dependency — only needed at startup
        resp = httpx.get(
            "https://api.smith.langchain.com/api/v1/workspaces",
            headers={"X-API-Key": settings.langsmith_api_key},
            timeout=5,
        )
        if resp.status_code == 200:
            _log.info(
                "LangSmith tracing active — project: %s",
                settings.langsmith_project or "(default)",
            )
            return settings
        _log.warning(
            "LangSmith API key rejected (HTTP %s). Tracing disabled.",
            resp.status_code,
        )
    except Exception as exc:
        _log.warning("LangSmith reachability check failed (%s). Tracing disabled.", exc)
    return settings.model_copy(update={"langsmith_tracing": False})


@dataclass(slots=True)
class AppServices:
    settings: Settings
    sources: list[DocumentSource]
    index: SearchIndex
    retrieval: RetrievalService
    ingestion: IngestionService
    agent: AgentService
    evaluation: EvaluationService


def _build_services() -> AppServices:
    settings = get_settings()
    validation_errors = settings.validate_runtime()
    if validation_errors:
        raise RuntimeError(" | ".join(validation_errors))
    settings = _validate_langsmith(settings)  # warn loudly if the key is bad/expired
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    embedder = build_embedder(settings)
    index: SearchIndex

    if settings.use_pinecone:
        try:
            index = PineconeHybridIndex(settings, embedder)
        except Exception:
            if settings.is_assessment_mode:
                raise
            index = InMemoryHybridIndex(embedder)
    else:
        index = InMemoryHybridIndex(embedder)

    retrieval = RetrievalService(settings, sources, index)
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_local_knowledge_base()
    reasoner = OpenAIReasoner(settings)
    tavily = TavilyClient(settings)
    agent = AgentService(settings, retrieval, reasoner, tavily, embedder=embedder)
    evaluation = EvaluationService(settings, settings.golden_questions_path, retrieval, agent)

    return AppServices(
        settings=settings,
        sources=sources,
        index=index,
        retrieval=retrieval,
        ingestion=ingestion,
        agent=agent,
        evaluation=evaluation,
    )


def get_services() -> AppServices:
    """Thread-safe singleton accessor for AppServices.

    Uses double-checked locking to avoid redundant initialization under
    concurrent first requests while keeping the fast path lock-free.
    """
    global _services_instance
    if _services_instance is not None:
        return _services_instance
    with _services_lock:
        if _services_instance is not None:
            return _services_instance
        _services_instance = _build_services()
        return _services_instance


def reset_services() -> None:
    """Reset the singleton — used by tests that modify env vars between runs."""
    global _services_instance
    with _services_lock:
        _services_instance = None


# Backwards-compatible shim so existing code calling get_services.cache_clear() still works.
get_services.cache_clear = reset_services  # type: ignore[attr-defined]
