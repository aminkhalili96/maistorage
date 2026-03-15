from __future__ import annotations

import os
from dataclasses import replace

import pytest

os.environ["APP_MODE"] = "dev"
os.environ["USE_PINECONE"] = "false"
os.environ["USE_TAVILY_FALLBACK"] = "false"
os.environ["EMBEDDER_PROVIDER"] = "keyword"
os.environ["LANGSMITH_TRACING"] = "false"

from app.config import get_settings
from app.corpus import load_demo_chunks, load_sources
from app.runtime import get_services
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import GeminiReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService

get_settings.cache_clear()
get_services.cache_clear()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def sources(settings):
    return load_sources(settings.source_manifest_path)


@pytest.fixture
def demo_chunks(settings):
    return load_demo_chunks(settings.demo_corpus_path)


@pytest.fixture
def demo_index(demo_chunks):
    index = InMemoryHybridIndex(KeywordEmbedder())
    index.upsert(demo_chunks)
    return index


@pytest.fixture
def retrieval_service(settings, sources, demo_index):
    return RetrievalService(settings, sources, demo_index)


@pytest.fixture
def agent_service(settings, sources, demo_chunks):
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_corpus()
    retrieval = RetrievalService(settings, sources, index)
    return AgentService(settings, retrieval, GeminiReasoner(settings), TavilyClient(settings))


@pytest.fixture
def dev_settings(settings, tmp_path):
    corpus_root = tmp_path / "data" / "corpus"
    return replace(
        settings,
        app_mode="dev",
        corpus_root=corpus_root,
        raw_html_root=corpus_root / "raw" / "html",
        raw_pdf_root=corpus_root / "raw" / "pdfs",
        raw_doc_root=corpus_root / "raw",
        normalized_doc_root=corpus_root / "normalized",
        corpus_manifest_path=corpus_root / "manifest.json",
    )
