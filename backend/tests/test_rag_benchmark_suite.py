from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest
from app.services.agent import AgentService
from app.services.evaluation import EvaluationService
from app.services.indexes import InMemoryHybridIndex
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService


@pytest.fixture(scope="module")
def rag_suite() -> tuple[list[dict], EvaluationService, AgentService]:
    settings = replace(
        get_settings(),
        openai_model="gpt-5.4",
        openai_api_key=None,
        use_pinecone=False,
        use_tavily_fallback=False,
        embedder_provider="keyword",
    )
    sources = load_sources(settings.source_manifest_path)
    chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    index.upsert(chunks)
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, OpenAIReasoner(settings), TavilyClient(settings))
    evaluation = EvaluationService(settings, settings.golden_questions_path, retrieval, agent)
    questions = evaluation.load_retrieval_questions()
    return questions, evaluation, agent


def _unique_sources(state) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for result in state.retrieval_results:
        source_id = result.chunk.source_id
        if source_id not in seen:
            seen.add(source_id)
            ordered.append(source_id)
    return ordered


def _term_hit_count(answer: str, expected_terms: list[str]) -> int:
    lowered = answer.lower()
    return sum(1 for term in expected_terms if term.lower() in lowered)


def test_benchmark_question_set_has_50_knowledge_base_grounded_cases(rag_suite: tuple[list[dict], EvaluationService, AgentService]) -> None:
    questions, _, _ = rag_suite
    assert len(questions) == 50
    assert all(question["expected_sources"] for question in questions)
    assert all(question["reference_answer"] for question in questions)


def test_retrieval_benchmark_hits_expected_sources_for_every_question(
    rag_suite: tuple[list[dict], EvaluationService, AgentService],
) -> None:
    questions, evaluation, _ = rag_suite
    rows = evaluation.evaluate_retrieval()

    assert len(rows) == len(questions)
    for row in rows:
        assert row.metrics["hit@5"] == 1.0, row.question
        assert row.metrics["mrr"] > 0.0, row.question
        assert row.metrics["ndcg@5"] > 0.0, row.question


@pytest.mark.parametrize("index", list(range(50)))
def test_agent_rag_suite_returns_grounded_cited_answers(
    rag_suite: tuple[list[dict], EvaluationService, AgentService],
    index: int,
) -> None:
    questions, _, agent = rag_suite
    item = questions[index]

    state = agent.run(ChatRequest(question=item["question"], model="gemini-2.5-flash"))
    retrieved_sources = _unique_sources(state)

    assert state.assistant_mode == "doc_rag", item["question"]
    assert state.response_mode == "knowledge-base-backed", item["question"]
    assert state.citations, item["question"]
    assert state.grounding_passed, item["question"]
    assert state.answer_quality_passed, item["question"]
    assert any(source in item["expected_sources"] for source in retrieved_sources[:5]), item["question"]
    assert _term_hit_count(state.answer, item["expected_terms"]) >= 1, item["question"]
    assert "I do not have enough grounded evidence" not in state.answer, item["question"]
