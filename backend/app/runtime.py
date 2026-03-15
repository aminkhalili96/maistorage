from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import Settings, get_settings
from app.corpus import load_demo_chunks, load_sources
from app.models import DocumentSource
from app.services.agent import AgentService
from app.services.evaluation import EvaluationService
from app.services.indexes import InMemoryHybridIndex, PineconeHybridIndex, SearchIndex
from app.services.ingestion import IngestionService
from app.services.providers import GeminiReasoner, TavilyClient, build_embedder
from app.services.retrieval import RetrievalService


@dataclass(slots=True)
class AppServices:
    settings: Settings
    sources: list[DocumentSource]
    index: SearchIndex
    retrieval: RetrievalService
    ingestion: IngestionService
    agent: AgentService
    evaluation: EvaluationService


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    settings = get_settings()
    validation_errors = settings.validate_runtime()
    if validation_errors:
        raise RuntimeError(" | ".join(validation_errors))
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)
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
    ingestion.bootstrap_local_corpus()
    reasoner = GeminiReasoner(settings)
    tavily = TavilyClient(settings)
    agent = AgentService(settings, retrieval, reasoner, tavily)
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
