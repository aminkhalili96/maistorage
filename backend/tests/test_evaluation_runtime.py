from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.corpus import load_demo_chunks, load_sources
from app.services.agent import AgentService
from app.services.evaluation import EvaluationService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import GeminiReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService


def _build_evaluation_service(settings_override=None) -> EvaluationService:
    settings = settings_override or get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_corpus_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_corpus()
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, GeminiReasoner(settings), TavilyClient(settings))
    return EvaluationService(settings, settings.golden_questions_path, retrieval, agent)


def test_ragas_returns_skipped_without_gemini_key():
    settings = replace(get_settings(), gemini_api_key=None)
    evaluation = _build_evaluation_service(settings)

    result = evaluation.run_ragas()

    assert result["status"] == "skipped"
    assert "GEMINI_API_KEY" in result["reason"]


def test_build_ragas_rows_prefers_reference_answer(monkeypatch):
    evaluation = _build_evaluation_service()

    monkeypatch.setattr(
        evaluation,
        "load_golden_questions",
        lambda: [
            {
                "question": "Why is 4-GPU training scaling poorly?",
                "expected_terms": ["all-reduce", "communication"],
                "reference_answer": "Scaling drops when all-reduce communication, topology limits, or imbalance dominate the step time.",
            }
        ],
    )
    monkeypatch.setattr(
        evaluation.agent,
        "run",
        lambda _request: SimpleNamespace(answer="Synthetic answer", retrieval_results=[]),
    )

    rows = evaluation.build_ragas_rows()

    assert rows[0]["ground_truth"].startswith("Scaling drops when all-reduce communication")


@pytest.mark.skipif(
    (not get_settings().gemini_api_key) or os.getenv("RUN_LIVE_RAGAS_TESTS") != "true",
    reason="Requires GEMINI_API_KEY and RUN_LIVE_RAGAS_TESTS=true",
)
def test_ragas_runs_when_gemini_key_is_available():
    evaluation = _build_evaluation_service()

    result = evaluation.run_ragas()

    assert result["status"] in {"ok", "failed"}
