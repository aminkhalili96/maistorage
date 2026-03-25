from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService


def _load_questions() -> list[dict]:
    path = Path(__file__).resolve().parents[2] / "data" / "evals" / "job_requirement_questions.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def job_requirements_suite() -> tuple[list[dict], AgentService]:
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
    return _load_questions(), agent


def _retrieved_source_ids(state) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for result in state.retrieval_results:
        source_id = result.chunk.source_id
        if source_id not in seen:
            seen.add(source_id)
            ordered.append(source_id)
    return ordered


def _answer_contains_terms(answer: str, expected_terms: list[str]) -> bool:
    lowered = answer.lower()
    return all(term.lower() in lowered for term in expected_terms[:2])


def test_job_requirement_question_set_has_expected_coverage(job_requirements_suite: tuple[list[dict], AgentService]) -> None:
    questions, _ = job_requirements_suite
    assert len(questions) == 12
    assert all(item["expected_sources"] for item in questions)
    assert all(item["expected_terms"] for item in questions)


@pytest.mark.parametrize("index", list(range(12)))
def test_job_requirement_topics_route_through_rag_with_citations(
    job_requirements_suite: tuple[list[dict], AgentService],
    index: int,
) -> None:
    questions, agent = job_requirements_suite
    item = questions[index]

    state = agent.run(ChatRequest(question=item["question"], model="gemini-2.5-flash"))
    retrieved = _retrieved_source_ids(state)

    assert state.assistant_mode == "doc_rag", item["question"]
    assert state.response_mode == "knowledge-base-backed", item["question"]
    assert state.citations, item["question"]
    assert state.grounding_passed, item["question"]
    assert state.answer_quality_passed, item["question"]
    assert any(source_id in item["expected_sources"] for source_id in retrieved[:5]), item["question"]
    assert _answer_contains_terms(state.answer, item["expected_terms"]), item["question"]
