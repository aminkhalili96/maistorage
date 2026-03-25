from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.services.agent import AgentService
from app.services.evaluation import EvaluationService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService


def _build_evaluation_service(settings_override=None) -> EvaluationService:
    settings = settings_override or get_settings()
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, OpenAIReasoner(settings), TavilyClient(settings))
    return EvaluationService(settings, settings.golden_questions_path, retrieval, agent)


def test_ragas_returns_skipped_without_openai_key():
    settings = replace(get_settings(), openai_api_key=None)
    evaluation = _build_evaluation_service(settings)

    result = evaluation.run_ragas()

    assert result["status"] == "skipped"
    assert "OPENAI_API_KEY" in result["reason"]


def test_build_ragas_rows_prefers_reference_answer(monkeypatch):
    evaluation = _build_evaluation_service()

    monkeypatch.setattr(
        evaluation,
        "load_ragas_questions",
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
    assert "metadata" in rows[0]


def test_build_ragas_rows_falls_back_to_local_benchmark_agent(monkeypatch):
    evaluation = _build_evaluation_service()

    monkeypatch.setattr(
        evaluation,
        "load_ragas_questions",
        lambda: [
            {
                "question": "What RAID levels should I compare for AI server redundancy and performance?",
                "reference_answer": "RAID 0 maximizes throughput without redundancy, RAID 1 mirrors data, RAID 5 and 6 add parity, and RAID 10 balances redundancy with better write performance.",
            }
        ],
    )

    def fail_run(_request):
        raise RuntimeError("rate limited")

    fallback_state = SimpleNamespace(
        answer="RAID 0 maximizes throughput without redundancy, RAID 1 mirrors data, RAID 5 and 6 add parity, and RAID 10 balances redundancy with better write performance.",
        retrieval_results=[],
        assistant_mode="doc_rag",
        response_mode="knowledge-base-backed",
        citations=[object()],
    )
    fallback_agent = SimpleNamespace(run=lambda _request: fallback_state)

    monkeypatch.setattr(evaluation.agent, "run", fail_run)
    monkeypatch.setattr(evaluation, "_build_local_benchmark_agent", lambda: fallback_agent)

    rows = evaluation.build_ragas_rows()

    assert rows[0]["answer"].startswith("RAID 0 maximizes throughput")
    assert rows[0]["metadata"]["generation_stack"] == "local_benchmark_fallback:RuntimeError"


def test_load_ragas_questions_includes_job_requirement_set():
    evaluation = _build_evaluation_service()

    rows = evaluation.load_ragas_questions()
    questions = {item["question"] for item in rows}

    assert "Why is 4-GPU training scaling poorly?" in questions
    assert "How does NVIDIA Transformer Engine enable FP8 training and what are the performance benefits compared to BF16?" in questions
    assert len(rows) == 50


def test_load_golden_questions_has_unified_50_question_benchmark():
    evaluation = _build_evaluation_service()

    rows = evaluation.load_golden_questions()
    ids = {item["id"] for item in rows}

    assert len(rows) == 54
    assert len(ids) == 54
    assert any(item["assistant_mode_expected"] == "direct_chat" for item in rows)
    assert any(item["response_mode_expected"] == "web-backed" for item in rows)
    assert any(item["response_mode_expected"] == "insufficient-evidence" for item in rows)


def test_load_retrieval_questions_filters_to_knowledge_base_backed_cases():
    evaluation = _build_evaluation_service()

    rows = evaluation.load_retrieval_questions()

    assert len(rows) == 50
    assert all(item["response_mode_expected"] == "knowledge-base-backed" for item in rows)
    assert all(item["assistant_mode_expected"] == "doc_rag" for item in rows)
    assert all(item["expected_sources"] for item in rows)


@pytest.mark.skipif(
    (not get_settings().openai_api_key) or os.getenv("RUN_LIVE_RAGAS_TESTS") != "true",
    reason="Requires OPENAI_API_KEY and RUN_LIVE_RAGAS_TESTS=true",
)
def test_ragas_runs_when_gemini_key_is_available():
    evaluation = _build_evaluation_service()

    result = evaluation.run_ragas()

    assert result["status"] in {"ok", "failed"}


def test_ragas_defaults_to_gpt41_as_judge(monkeypatch):
    settings = replace(get_settings(), openai_api_key="test-key")
    evaluation = _build_evaluation_service(settings)

    class DummyDataset:
        @staticmethod
        def from_list(rows):
            return rows

    created_models: list[str] = []

    class DummyChatModel:
        def __init__(self, *, model, api_key, temperature):
            created_models.append(model)

    class DummyEmbeddings:
        def __init__(self, **_kwargs):
            pass

    class DummyRunConfig:
        def __init__(self, **_kwargs):
            pass

    class DummyResult:
        def to_pandas(self):
            class DummyFrame:
                @staticmethod
                def to_dict(orient="records"):
                    return [{"faithfulness": 1.0}]

            return DummyFrame()

    monkeypatch.setattr(evaluation, "build_ragas_rows", lambda: [{"question": "q", "answer": "a", "contexts": [], "ground_truth": "g"}])

    import sys
    import types

    monkeypatch.delenv("RAGAS_EVALUATOR_MODEL", raising=False)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=DummyDataset))
    monkeypatch.setitem(sys.modules, "ragas", types.SimpleNamespace(evaluate=lambda *args, **kwargs: DummyResult()))
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        types.SimpleNamespace(answer_relevancy=object(), context_precision=object(), context_recall=object(), faithfulness=object()),
    )
    monkeypatch.setitem(sys.modules, "ragas.run_config", types.SimpleNamespace(RunConfig=DummyRunConfig))
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(ChatOpenAI=DummyChatModel, OpenAIEmbeddings=DummyEmbeddings),
    )

    result = evaluation.run_ragas()

    assert result["status"] == "ok"
    assert created_models == ["gpt-4.1"]


def test_ragas_respects_question_limit(monkeypatch):
    settings = replace(get_settings(), openai_api_key="test-key")
    evaluation = _build_evaluation_service(settings)

    class DummyDataset:
        @staticmethod
        def from_list(rows):
            return rows

    class DummyChatModel:
        def __init__(self, **_kwargs):
            pass

    class DummyEmbeddings:
        def __init__(self, **_kwargs):
            pass

    class DummyRunConfig:
        def __init__(self, **_kwargs):
            pass

    class DummyResult:
        def to_pandas(self):
            class DummyFrame:
                @staticmethod
                def to_dict(orient="records"):
                    return [{"faithfulness": 1.0}]

            return DummyFrame()

    monkeypatch.setattr(
        evaluation,
        "build_ragas_rows",
        lambda: [{"question": f"q{i}", "answer": "a", "contexts": [], "ground_truth": "g"} for i in range(5)],
    )

    import sys
    import types

    monkeypatch.setenv("RAGAS_QUESTION_LIMIT", "2")
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=DummyDataset))
    monkeypatch.setitem(sys.modules, "ragas", types.SimpleNamespace(evaluate=lambda dataset, **kwargs: DummyResult()))
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        types.SimpleNamespace(answer_relevancy=object(), context_precision=object(), context_recall=object(), faithfulness=object()),
    )
    monkeypatch.setitem(sys.modules, "ragas.run_config", types.SimpleNamespace(RunConfig=DummyRunConfig))
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(ChatOpenAI=DummyChatModel, OpenAIEmbeddings=DummyEmbeddings),
    )

    result = evaluation.run_ragas()

    assert result["status"] == "ok"
    assert result["question_count"] == 2
    assert result["question_limit"] == 2


def test_evaluate_trajectory_returns_expected_shape(monkeypatch):
    evaluation = _build_evaluation_service()
    question_row = {
        "id": "agentic-test",
        "question": "What is NVIDIA?",
        "category": "direct_chat",
        "assistant_mode_expected": "direct_chat",
        "response_mode_expected": "direct-chat",
        "expected_sources": [],
        "expected_terms": ["technology company"],
        "reference_answer": "NVIDIA is a technology company.",
        "expected_tool_path": ["direct_chat"],
        "max_retries": 0,
        "should_require_citations": False,
        "should_use_fallback": False,
    }
    monkeypatch.setattr(evaluation, "load_golden_questions", lambda: [question_row])
    async def fake_timings(question, model=None):
        return (12.5, 18.0)

    monkeypatch.setattr(evaluation, "_measure_stream_timings", fake_timings)
    monkeypatch.setattr(
        evaluation.agent,
        "run",
        lambda _request: SimpleNamespace(
            assistant_mode="direct_chat",
            response_mode="direct-chat",
            retrieval_results=[],
            citations=[],
            retry_count=0,
            used_fallback=False,
            rejected_chunk_ids=[],
            grounding_passed=True,
            answer_quality_passed=True,
            trace=[],
        ),
    )

    result = evaluation.evaluate_trajectory()

    assert result["status"] == "ok"
    assert result["aggregate"]["question_count"] == 1
    assert result["rows"][0]["assistant_mode_match"] is True
    assert result["rows"][0]["tool_path_match"] is True
    assert result["rows"][0]["ttft_ms"] == 12.5


def test_agentic_judge_skips_without_flag(monkeypatch):
    settings = replace(get_settings(), openai_api_key="test-key")
    evaluation = _build_evaluation_service(settings)

    monkeypatch.delenv("RUN_LIVE_JUDGE_TESTS", raising=False)

    result = evaluation.run_agentic_judge()

    assert result["status"] == "skipped"
    assert "RUN_LIVE_JUDGE_TESTS" in result["reason"]


def test_agentic_judge_uses_gpt41_and_returns_summary(monkeypatch):
    settings = replace(get_settings(), openai_api_key="test-key")
    evaluation = _build_evaluation_service(settings)

    monkeypatch.setenv("RUN_LIVE_JUDGE_TESTS", "true")
    monkeypatch.setattr(
        "app.services.evaluation.EvaluationService.evaluate_trajectory",
        lambda self: {
            "status": "ok",
            "rows": [
                {
                    "id": "agentic-judge-1",
                    "question": "What is NVIDIA?",
                    "assistant_mode_expected": "direct_chat",
                    "assistant_mode_actual": "direct_chat",
                    "response_mode_expected": "direct-chat",
                    "response_mode_actual": "direct-chat",
                    "expected_tool_path": ["direct_chat"],
                    "actual_tool_path": ["direct_chat"],
                    "expected_sources": [],
                    "grounding_passed": True,
                    "answer_quality_passed": True,
                    "citation_count": 0,
                    "retry_count": 0,
                }
            ],
        },
    )

    created_models: list[str] = []

    class DummyJudge:
        def __init__(self, settings):
            created_models.append(settings.openai_model)

        @property
        def enabled(self):
            return True

        def generate_text(self, prompt, model=None):
            assert "What is NVIDIA?" in prompt
            assert model == "gpt-4.1"
            return (
                '{"faithfulness": 1.0, "answer_relevance": 0.9, "trajectory_correctness": 1.0, '
                '"tool_path_correctness": 1.0, "citation_support": 1.0, "overall_pass": true, '
                '"rationale": "Looks correct."}'
            )

    monkeypatch.setattr("app.services.evaluation.OpenAIReasoner", DummyJudge)

    result = evaluation.run_agentic_judge()

    assert result["status"] == "ok"
    assert created_models[-1] == "gpt-4.1"
    assert "gpt-4.1" in created_models
    assert result["summary"]["overall_pass_rate"] == 1.0


def test_agentic_judge_returns_failed_on_provider_error(monkeypatch):
    settings = replace(get_settings(), openai_api_key="test-key")
    evaluation = _build_evaluation_service(settings)

    monkeypatch.setenv("RUN_LIVE_JUDGE_TESTS", "true")
    monkeypatch.setattr(
        "app.services.evaluation.EvaluationService.evaluate_trajectory",
        lambda self: {
            "status": "ok",
            "rows": [
                {
                    "id": "agentic-judge-1",
                    "question": "What is NVIDIA?",
                    "assistant_mode_expected": "direct_chat",
                    "assistant_mode_actual": "direct_chat",
                    "response_mode_expected": "direct-chat",
                    "response_mode_actual": "direct-chat",
                    "expected_tool_path": ["direct_chat"],
                    "actual_tool_path": ["direct_chat"],
                    "expected_sources": [],
                    "grounding_passed": True,
                    "answer_quality_passed": True,
                    "citation_count": 0,
                    "retry_count": 0,
                }
            ],
        },
    )

    class DummyJudge:
        def __init__(self, settings):
            self.settings = settings

        @property
        def enabled(self):
            return True

        def generate_text(self, prompt, model=None):
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr("app.services.evaluation.OpenAIReasoner", DummyJudge)

    result = evaluation.run_agentic_judge()

    assert result["status"] == "failed"
    assert "429" in result["reason"]
