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
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService

get_settings.cache_clear()
get_services.cache_clear()


class MockOpenAIReasoner:
    enabled = True

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        # Detect reformulation prompts and return a sensible standalone question
        if "standalone" in prompt.lower() and "follow-up" in prompt.lower():
            if "memory" in prompt.lower():
                return "How much memory does the NVIDIA H100 GPU have?"
            if "inference" in prompt.lower():
                return "What is the inference performance of NVIDIA H100 and A100 GPUs?"
            if "bandwidth" in prompt.lower():
                return "How does NCCL tuning affect bandwidth in multi-GPU scaling?"
            if "nvlink" in prompt.lower():
                return "What is the role of NVLink in NCCL multi-GPU communication?"
            return "What are the NCCL configuration parameters for NVIDIA multi-GPU training?"
        return (
            "Based on the NVIDIA documentation, multi-GPU training scaling issues arise from "
            "communication overhead in all-reduce operations across GPUs. [1] "
            "The NCCL library handles collective communication and NVLink bandwidth is critical "
            "for scaling efficiency beyond 4 GPUs. [2] "
            "Mixed precision training uses FP16 for compute and FP32 for accumulation, "
            "reducing memory footprint while maintaining accuracy. [3]"
        )


@pytest.fixture
def mock_reasoner():
    return MockOpenAIReasoner()


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
    return AgentService(settings, retrieval, OpenAIReasoner(settings), TavilyClient(settings))


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
