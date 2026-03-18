from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from app.corpus import load_demo_chunks, load_sources
from app.config import Settings
from app.models import ChatRequest, EvaluationRow
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService


def _unique_sources(sources: list[str]) -> list[str]:
    return list(dict.fromkeys(sources))


def hit_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 5) -> float:
    top_sources = _unique_sources(retrieved_sources)[:k]
    return 1.0 if any(source in top_sources for source in expected_sources) else 0.0


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    for index, source in enumerate(_unique_sources(retrieved_sources), start=1):
        if source in expected_sources:
            return 1.0 / index
    return 0.0


def ndcg(retrieved_sources: list[str], expected_sources: list[str], k: int = 5) -> float:
    dcg = 0.0
    for index, source in enumerate(_unique_sources(retrieved_sources)[:k], start=1):
        relevance = 1.0 if source in expected_sources else 0.0
        if relevance:
            dcg += relevance / math.log2(index + 1)
    ideal_hits = min(len(expected_sources), k)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


class EvaluationService:
    def __init__(self, settings: Settings, golden_questions_path: Path, retrieval: RetrievalService, agent: AgentService) -> None:
        self.settings = settings
        self.golden_questions_path = golden_questions_path
        self.retrieval = retrieval
        self.agent = agent

    def _build_local_benchmark_agent(self) -> AgentService:
        local_settings = replace(self.settings, openai_api_key=None, use_tavily_fallback=True)
        sources = load_sources(local_settings.source_manifest_path)
        demo_chunks = load_demo_chunks(local_settings.demo_corpus_path)
        index = InMemoryHybridIndex(KeywordEmbedder())
        ingestion = IngestionService(local_settings, index, sources, demo_chunks)
        ingestion.bootstrap_demo_corpus()
        retrieval = RetrievalService(local_settings, sources, index)

        class LocalFallbackClient:
            def search(self, query: str) -> list[dict[str, str]]:
                return [
                    {
                        "title": "Current external result",
                        "url": "https://example.com/current",
                        "content": f"Live web evidence placeholder for: {query}",
                    }
                ]

        return AgentService(
            local_settings,
            retrieval,
            OpenAIReasoner(local_settings),
            LocalFallbackClient(),
        )

    def _normalize_question_row(self, item: dict[str, Any], index: int) -> dict[str, Any]:
        expected_sources = list(item.get("expected_sources", []))
        response_mode = item.get("response_mode_expected")
        assistant_mode = item.get("assistant_mode_expected")

        if assistant_mode is None:
            assistant_mode = "doc_rag" if expected_sources else "direct_chat"
        if response_mode is None:
            if assistant_mode == "direct_chat":
                response_mode = "direct-chat"
            elif expected_sources:
                response_mode = "corpus-backed"
            else:
                response_mode = "insufficient-evidence"

        return {
            "id": item.get("id", f"agentic-{index:03d}"),
            "question": item["question"],
            "category": item.get("category") or item.get("query_class", "general"),
            "assistant_mode_expected": assistant_mode,
            "response_mode_expected": response_mode,
            "expected_sources": expected_sources,
            "expected_terms": list(item.get("expected_terms", [])),
            "reference_answer": item.get("reference_answer") or item.get("ground_truth", ""),
            "expected_tool_path": list(item.get("expected_tool_path", ["direct_chat"] if assistant_mode == "direct_chat" else ["nvidia_docs"])),
            "max_retries": int(item.get("max_retries", 0 if assistant_mode == "direct_chat" else 2)),
            "should_require_citations": bool(
                item.get(
                    "should_require_citations",
                    response_mode in {"corpus-backed", "web-backed"},
                )
            ),
            "should_use_fallback": bool(item.get("should_use_fallback", response_mode == "web-backed")),
        }

    def load_golden_questions(self) -> list[dict[str, Any]]:
        raw_rows = json.loads(self.golden_questions_path.read_text())
        return [self._normalize_question_row(item, index) for index, item in enumerate(raw_rows, start=1)]

    def load_retrieval_questions(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.load_golden_questions()
            if item["assistant_mode_expected"] == "doc_rag"
            and item["response_mode_expected"] == "corpus-backed"
            and item["expected_sources"]
        ]

    def load_ragas_questions(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.load_golden_questions()
            if item["assistant_mode_expected"] == "doc_rag"
            and item["response_mode_expected"] == "corpus-backed"
            and bool(item.get("reference_answer"))
        ]

    def evaluate_retrieval(self) -> list[EvaluationRow]:
        rows: list[EvaluationRow] = []
        for item in self.load_retrieval_questions():
            response = self.retrieval.search(item["question"])
            retrieved_sources = _unique_sources([result.chunk.source_id for result in response.results])
            routing_hit = 1.0 if any(source in item["expected_sources"] for source in retrieved_sources[:3]) else 0.0
            rows.append(
                EvaluationRow(
                    question=item["question"],
                    query_class=item["category"],
                    expected_sources=item["expected_sources"],
                    retrieved_sources=retrieved_sources,
                    metrics={
                        "hit@5": hit_at_k(retrieved_sources, item["expected_sources"], 5),
                        "mrr": reciprocal_rank(retrieved_sources, item["expected_sources"]),
                        "ndcg@5": ndcg(retrieved_sources, item["expected_sources"], 5),
                        "routing@3": routing_hit,
                        "retry_count": response.retry_count,
                    },
                )
            )
        return rows

    @staticmethod
    def _actual_tool_path(run_state: Any) -> list[str]:
        if getattr(run_state, "assistant_mode", "doc_rag") == "direct_chat":
            return ["direct_chat"]
        ordered: list[str] = []
        seen: set[str] = set()
        for event in getattr(run_state, "trace", []):
            tool = event.payload.get("tool")
            if tool and tool not in seen:
                seen.add(tool)
                ordered.append(str(tool))
        return ordered

    @staticmethod
    def _expected_source_hit(run_state: Any, expected_sources: list[str]) -> bool:
        if not expected_sources:
            return True
        retrieved = []
        seen: set[str] = set()
        for result in getattr(run_state, "retrieval_results", []):
            source_id = result.chunk.source_id
            if source_id not in seen:
                seen.add(source_id)
                retrieved.append(source_id)
        return any(source in expected_sources for source in retrieved[:5])

    async def _measure_stream_timings(self, question: str, model: str | None = None) -> tuple[float, float]:
        started = time.perf_counter()
        first_signal_ms: float | None = None
        async for _frame in self.agent.stream(ChatRequest(question=question, model=model)):
            if first_signal_ms is None:
                first_signal_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        return round(first_signal_ms or total_ms, 2), round(total_ms, 2)

    def evaluate_trajectory(self, *, model: str | None = None) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for item in self.load_golden_questions():
            ttft_ms, total_ms = asyncio.run(self._measure_stream_timings(item["question"], model))
            run_state = self.agent.run(ChatRequest(question=item["question"], model=model))
            actual_tool_path = self._actual_tool_path(run_state)
            row = {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "assistant_mode_expected": item["assistant_mode_expected"],
                "assistant_mode_actual": run_state.assistant_mode,
                "assistant_mode_match": run_state.assistant_mode == item["assistant_mode_expected"],
                "response_mode_expected": item["response_mode_expected"],
                "response_mode_actual": run_state.response_mode,
                "response_mode_match": run_state.response_mode == item["response_mode_expected"],
                "expected_tool_path": item["expected_tool_path"],
                "actual_tool_path": actual_tool_path,
                "tool_path_match": actual_tool_path == item["expected_tool_path"],
                "expected_sources": item["expected_sources"],
                "expected_source_hit": self._expected_source_hit(run_state, item["expected_sources"]),
                "retry_count": run_state.retry_count,
                "max_retries": item["max_retries"],
                "within_retry_budget": run_state.retry_count <= item["max_retries"],
                "should_require_citations": item["should_require_citations"],
                "citation_count": len(run_state.citations),
                "citation_requirement_match": (len(run_state.citations) > 0) == item["should_require_citations"],
                "should_use_fallback": item["should_use_fallback"],
                "used_fallback": run_state.used_fallback,
                "fallback_match": run_state.used_fallback == item["should_use_fallback"],
                "accepted_evidence_ids": [result.chunk.id for result in run_state.retrieval_results],
                "rejected_evidence_ids": list(run_state.rejected_chunk_ids),
                "grounding_passed": run_state.grounding_passed,
                "answer_quality_passed": run_state.answer_quality_passed,
                "trace_types": [event.type for event in run_state.trace],
                "latency_ms": total_ms,
                "ttft_ms": ttft_ms,
            }
            rows.append(row)

        aggregate = {
            "question_count": len(rows),
            "assistant_mode_accuracy": round(mean(1.0 if row["assistant_mode_match"] else 0.0 for row in rows), 4),
            "response_mode_accuracy": round(mean(1.0 if row["response_mode_match"] else 0.0 for row in rows), 4),
            "tool_path_accuracy": round(mean(1.0 if row["tool_path_match"] else 0.0 for row in rows), 4),
            "retry_budget_pass_rate": round(mean(1.0 if row["within_retry_budget"] else 0.0 for row in rows), 4),
            "citation_requirement_pass_rate": round(mean(1.0 if row["citation_requirement_match"] else 0.0 for row in rows), 4),
            "fallback_pass_rate": round(mean(1.0 if row["fallback_match"] else 0.0 for row in rows), 4),
            "expected_source_hit_rate": round(
                mean(1.0 if row["expected_source_hit"] else 0.0 for row in rows if row["expected_sources"]),
                4,
            )
            if any(row["expected_sources"] for row in rows)
            else 0.0,
            "grounding_pass_rate": round(mean(1.0 if row["grounding_passed"] else 0.0 for row in rows), 4),
            "answer_quality_pass_rate": round(mean(1.0 if row["answer_quality_passed"] else 0.0 for row in rows), 4),
            "avg_ttft_ms": round(mean(float(row["ttft_ms"]) for row in rows), 2),
            "avg_total_latency_ms": round(mean(float(row["latency_ms"]) for row in rows), 2),
        }
        return {"status": "ok", "rows": rows, "aggregate": aggregate}

    def build_ragas_rows(self) -> list[dict[str, Any]]:
        answer_model = os.getenv("RAGAS_GENERATION_MODEL")
        agent = self.agent
        if answer_model:
            override_settings = replace(self.settings, openai_model=answer_model)
            agent = AgentService(
                override_settings,
                self.retrieval,
                OpenAIReasoner(override_settings),
                TavilyClient(override_settings),
            )
        fallback_agent: AgentService | None = None

        rows: list[dict[str, Any]] = []
        for item in self.load_ragas_questions():
            generation_stack = "configured"
            try:
                run_state = agent.run(ChatRequest(question=item["question"]))
            except Exception as exc:
                if fallback_agent is None:
                    fallback_agent = self._build_local_benchmark_agent()
                run_state = fallback_agent.run(ChatRequest(question=item["question"]))
                generation_stack = f"local_benchmark_fallback:{type(exc).__name__}"
            ground_truth = item.get("reference_answer") or item.get("ground_truth") or " | ".join(item.get("expected_terms", []))
            rows.append(
                {
                    "question": item["question"],
                    "answer": run_state.answer,
                    "contexts": [result.chunk.content for result in run_state.retrieval_results[:4]],
                    "ground_truth": ground_truth,
                    "metadata": {
                        "id": item.get("id", item["question"]),
                        "category": item.get("category", item.get("query_class", "general")),
                        "expected_sources": item.get("expected_sources", []),
                        "expected_tool_path": item.get("expected_tool_path", []),
                        "assistant_mode": getattr(run_state, "assistant_mode", None),
                        "response_mode": getattr(run_state, "response_mode", None),
                        "citation_count": len(getattr(run_state, "citations", [])),
                        "generation_stack": generation_stack,
                    },
                }
            )
        return rows

    def run_ragas(self) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            return {"status": "skipped", "reason": "Set OPENAI_API_KEY before running RAGAS evaluation."}

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
            from ragas.run_config import RunConfig
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        except Exception as exc:  # pragma: no cover - depends on optional environment
            return {"status": "skipped", "reason": str(exc)}

        rows = self.build_ragas_rows()
        question_limit = int(os.getenv("RAGAS_QUESTION_LIMIT", "0") or "0")
        if question_limit > 0:
            rows = rows[:question_limit]
        dataset = Dataset.from_list(rows)
        try:  # pragma: no cover - external evaluation call
            evaluator_model = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4.1")
            llm = ChatOpenAI(
                model=evaluator_model,
                api_key=self.settings.openai_api_key,
                temperature=0.0,
            )
            embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
            )
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=1, max_retries=6, max_wait=60),
                batch_size=1,
            )
            return {
                "status": "ok",
                "scores": result.to_pandas().to_dict(orient="records"),
                "question_count": len(rows),
                "question_limit": question_limit or None,
                "question_sets": [
                    str(self.golden_questions_path),
                    str(self.settings.project_root / "data" / "evals" / "job_requirement_questions.json"),
                ],
                "thresholds": {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.55,
                    "context_precision": 0.75,
                    "context_recall": 0.7,
                },
            }
        except Exception as exc:  # pragma: no cover
            return {"status": "failed", "reason": str(exc)}

    def run_agentic_judge(self, *, model: str = "gpt-4.1") -> dict[str, Any]:
        if not self.settings.openai_api_key:
            return {"status": "skipped", "reason": "Set OPENAI_API_KEY before running the live judge."}
        if os.getenv("RUN_LIVE_JUDGE_TESTS", "").strip().lower() != "true":
            return {"status": "skipped", "reason": "Set RUN_LIVE_JUDGE_TESTS=true to enable the live GPT-4.1 judge."}

        local_agent = self._build_local_benchmark_agent()
        trajectory = EvaluationService(
            self.settings,
            self.golden_questions_path,
            local_agent.retrieval,
            local_agent,
        ).evaluate_trajectory()
        if trajectory.get("status") != "ok":
            return {"status": "failed", "reason": "Trajectory benchmark did not complete."}

        judge = OpenAIReasoner(replace(self.settings, openai_model=model))
        if not judge.enabled:
            return {"status": "skipped", "reason": "OpenAI judge is unavailable in the current runtime."}

        question_limit = int(os.getenv("AGENTIC_JUDGE_QUESTION_LIMIT", "0") or "0")
        rows = trajectory["rows"][:question_limit] if question_limit > 0 else trajectory["rows"]
        judgments: list[dict[str, Any]] = []
        try:
            for row in rows:
                prompt = (
                    "You are grading an Agentic RAG run. Return strict JSON only with keys: "
                    "faithfulness, answer_relevance, trajectory_correctness, tool_path_correctness, "
                    "citation_support, overall_pass, rationale.\n\n"
                    f"Question: {row['question']}\n"
                    f"Expected assistant mode: {row['assistant_mode_expected']}\n"
                    f"Actual assistant mode: {row['assistant_mode_actual']}\n"
                    f"Expected response mode: {row['response_mode_expected']}\n"
                    f"Actual response mode: {row['response_mode_actual']}\n"
                    f"Expected tool path: {row['expected_tool_path']}\n"
                    f"Actual tool path: {row['actual_tool_path']}\n"
                    f"Expected sources: {row['expected_sources']}\n"
                    f"Grounding passed: {row['grounding_passed']}\n"
                    f"Answer quality passed: {row['answer_quality_passed']}\n"
                    f"Citation count: {row['citation_count']}\n"
                    f"Retry count: {row['retry_count']}\n"
                )
                response = judge.generate_text(prompt, model=model)
                payload = json.loads(response)
                payload["id"] = row["id"]
                payload["question"] = row["question"]
                judgments.append(payload)
        except Exception as exc:
            return {
                "status": "failed",
                "judge_model": model,
                "question_count": len(judgments),
                "question_limit": question_limit or None,
                "reason": str(exc),
            }

        summary = {
            "faithfulness": round(mean(float(item["faithfulness"]) for item in judgments), 4),
            "answer_relevance": round(mean(float(item["answer_relevance"]) for item in judgments), 4),
            "trajectory_correctness": round(mean(float(item["trajectory_correctness"]) for item in judgments), 4),
            "tool_path_correctness": round(mean(float(item["tool_path_correctness"]) for item in judgments), 4),
            "citation_support": round(mean(float(item["citation_support"]) for item in judgments), 4),
            "overall_pass_rate": round(mean(1.0 if item["overall_pass"] else 0.0 for item in judgments), 4),
        }
        return {
            "status": "ok",
            "judge_model": model,
            "question_count": len(judgments),
            "question_limit": question_limit or None,
            "summary": summary,
            "judgments": judgments,
        }
