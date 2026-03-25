from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService


def _build_stack() -> AgentService:
    settings = get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    return AgentService(settings, retrieval, OpenAIReasoner(settings), TavilyClient(settings))


def test_harder_benchmark_slice() -> None:
    agent = _build_stack()
    path = Path(__file__).resolve().parents[2] / "data" / "evals" / "harder_questions.json"
    cases = json.loads(path.read_text())

    assert len(cases) >= 10

    for case in cases:
        state = agent.run(ChatRequest(question=case["question"]))
        assert state.assistant_mode == case["assistant_mode_expected"], case["question"]
        assert state.response_mode == case["response_mode_expected"], case["question"]

        if case["response_mode_expected"] == "knowledge-base-backed":
            retrieved = {result.chunk.source_id for result in state.retrieval_results[:5]}
            assert any(source_id in retrieved for source_id in case["expected_sources"]), case["question"]
            lowered_answer = state.answer.lower()
            assert any(term in lowered_answer for term in case["expected_terms"][:2]), case["question"]
            assert state.grounding_passed, case["question"]
            assert state.answer_quality_passed, case["question"]
            assert state.citations, case["question"]
        else:
            assert state.response_mode in {"insufficient-evidence", "web-backed"}, case["question"]
            assert not state.grounding_passed or state.response_mode != "knowledge-base-backed", case["question"]
