from __future__ import annotations

import json
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
        lowered = prompt.lower()

        # Agentic classification — extract the actual user question after "Question:"
        if "classify" in lowered and ("doc_rag" in lowered or "direct_chat" in lowered or "live_query" in lowered):
            # Extract the actual question from the end of the prompt
            q_marker = lowered.rfind("question:")
            user_question = lowered[q_marker + 9:].strip() if q_marker != -1 else lowered
            if any(t in user_question for t in ("weather", "stock price", "current news")):
                return '{"mode": "live_query", "reasoning": "Question asks about real-time data"}'
            if user_question in ("hello", "hi", "hey", "hello, how are you?", "how are you?") or user_question.startswith("hello"):
                return '{"mode": "direct_chat", "reasoning": "Casual greeting"}'
            # Non-NVIDIA topics: general knowledge, greetings, off-scope
            non_nvidia_signals = (
                "interview", "what day", "what is nvidia", "what time",
                "how are you", "thank", "bye", "sorry",
            )
            if any(s in user_question for s in non_nvidia_signals):
                return '{"mode": "direct_chat", "reasoning": "General question not about NVIDIA infrastructure docs"}'
            return '{"mode": "doc_rag", "reasoning": "Technical question about NVIDIA infrastructure"}'

        # Agentic query planning — choose class based on question keywords
        if "query_class" in lowered and "search_queries" in lowered and "source_families" in lowered:
            # Extract question from the prompt
            q_marker = lowered.rfind("question:")
            q_text = lowered[q_marker + 9:].strip() if q_marker != -1 else lowered
            if any(kw in q_text for kw in ("multi-gpu", "scaling", "nccl", "nvlink", "4-gpu", "multi gpu", "parallelism")):
                return json.dumps({
                    "query_class": "distributed_multi_gpu",
                    "search_queries": ["multi-GPU training scaling NCCL", "NVLink collective communication"],
                    "source_families": ["distributed", "core"],
                    "top_k": 7,
                    "confidence_floor": 0.26,
                    "reasoning": "Multi-GPU scaling question targeting distributed training docs",
                })
            return json.dumps({
                "query_class": "hardware_topology",
                "search_queries": ["NVIDIA H100 GPU specifications", "H100 memory bandwidth"],
                "source_families": ["hardware", "infrastructure"],
                "top_k": 5,
                "confidence_floor": 0.28,
                "reasoning": "Hardware specification question targeting GPU datasheets",
            })

        # Agentic document grading
        if "relevant" in lowered and "document" in lowered and "grades" in lowered:
            return '{"grades": [{"doc": 1, "relevant": true, "reason": "Directly relevant"}, {"doc": 2, "relevant": true, "reason": "Contains related info"}, {"doc": 3, "relevant": false, "reason": "Off-topic"}]}'

        # Agentic routing — context-aware based on pipeline state
        if "routing agent" in lowered and "action" in lowered:
            if "after_quality" in lowered:
                # Check if quality/grounding actually failed
                if "quality check passed: false" in lowered and "web fallback used: false" in lowered:
                    return '{"action": "post_gen_fallback", "reasoning": "Quality failed, try web search"}'
                return '{"action": "end", "reasoning": "Quality checks passed"}'
            # after_grading: check confidence and chunk count
            if "confidence score: 0.00" in lowered or "retrieved documents: 0" in lowered:
                return '{"action": "rewrite", "reasoning": "Zero confidence, need to rephrase query"}'
            return '{"action": "generate", "reasoning": "Sufficient evidence to generate answer"}'

        # Multi-hop check
        if "sufficient" in lowered and "follow_up_query" in lowered and "gap_description" in lowered:
            return '{"sufficient": true, "follow_up_query": "", "gap_description": ""}'

        # S1: Claim-level verification — extract factual claims from the answer
        if "factual claims" in lowered and "claims" in lowered:
            return '{"claims": ["NCCL handles collective communication", "NVLink bandwidth is critical for scaling"]}'

        # S3: Sycophancy detection — check for user assertions/premises
        if "assertions" in lowered and "premises" in lowered:
            return '{"has_assertions": false, "assertions": []}'

        # Self-reflection
        if "relevance" in lowered and "groundedness" in lowered and "completeness" in lowered:
            return '{"relevance": 4, "groundedness": 4, "completeness": 4, "issues": "none"}'

        # Reformulation
        if "standalone" in lowered and "follow-up" in lowered:
            if "memory" in lowered:
                return "How much memory does the NVIDIA H100 GPU have?"
            if "inference" in lowered:
                return "What is the inference performance of NVIDIA H100 and A100 GPUs?"
            if "bandwidth" in lowered:
                return "How does NCCL tuning affect bandwidth in multi-GPU scaling?"
            if "nvlink" in lowered:
                return "What is the role of NVLink in NCCL multi-GPU communication?"
            return "What are the NCCL configuration parameters for NVIDIA multi-GPU training?"

        # HyDE hypothetical document generation
        if "hypothetical" in lowered or ("factual paragraph" in lowered and "nvidia" in lowered):
            return (
                "NCCL provides high-bandwidth collective communication operations for multi-GPU "
                "training, leveraging NVLink interconnect for efficient all-reduce operations. "
                "Key tuning parameters include NCCL_ALGO, NCCL_PROTO, and buffer sizes."
            )

        # Default: standard synthesis response
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
def agent_service_with_mock(settings, sources):
    chunks = load_demo_chunks(settings.demo_corpus_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, chunks)
    ingestion.bootstrap_demo_corpus()
    retrieval = RetrievalService(settings, sources, index)
    return AgentService(settings, retrieval, MockOpenAIReasoner(), TavilyClient(settings))


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
