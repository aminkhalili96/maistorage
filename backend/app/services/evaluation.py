from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import ChatRequest, EvaluationRow
from app.services.agent import AgentService
from app.services.providers import GeminiReasoner, TavilyClient
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

    def load_golden_questions(self) -> list[dict[str, Any]]:
        return json.loads(self.golden_questions_path.read_text())

    def evaluate_retrieval(self) -> list[EvaluationRow]:
        rows: list[EvaluationRow] = []
        for item in self.load_golden_questions():
            response = self.retrieval.search(item["question"])
            retrieved_sources = _unique_sources([result.chunk.source_id for result in response.results])
            routing_hit = 1.0 if any(source in item["expected_sources"] for source in retrieved_sources[:3]) else 0.0
            rows.append(
                EvaluationRow(
                    question=item["question"],
                    query_class=item["query_class"],
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

    def build_ragas_rows(self) -> list[dict[str, Any]]:
        answer_model = os.getenv("RAGAS_GENERATION_MODEL")
        agent = self.agent
        if answer_model:
            override_settings = replace(self.settings, gemini_model=answer_model)
            agent = AgentService(
                override_settings,
                self.retrieval,
                GeminiReasoner(override_settings),
                TavilyClient(override_settings),
            )

        rows: list[dict[str, Any]] = []
        for item in self.load_golden_questions():
            run_state = agent.run(ChatRequest(question=item["question"]))
            ground_truth = item.get("reference_answer") or item.get("ground_truth") or " | ".join(item.get("expected_terms", []))
            rows.append(
                {
                    "question": item["question"],
                    "answer": run_state.answer,
                    "contexts": [result.chunk.content for result in run_state.retrieval_results[:4]],
                    "ground_truth": ground_truth,
                }
            )
        return rows

    def run_ragas(self) -> dict[str, Any]:
        if not self.settings.gemini_api_key:
            return {"status": "skipped", "reason": "Set GEMINI_API_KEY before running RAGAS evaluation."}

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
            from ragas.run_config import RunConfig
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        except Exception as exc:  # pragma: no cover - depends on optional environment
            return {"status": "skipped", "reason": str(exc)}

        dataset = Dataset.from_list(self.build_ragas_rows())
        try:  # pragma: no cover - external evaluation call
            evaluator_model = os.getenv("RAGAS_EVALUATOR_MODEL", self.settings.generation_model)
            llm = ChatGoogleGenerativeAI(
                model=evaluator_model,
                google_api_key=self.settings.gemini_api_key,
                temperature=0.0,
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.settings.gemini_embedding_model,
                google_api_key=self.settings.gemini_api_key,
            )
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=1, max_retries=6, max_wait=60),
                batch_size=1,
            )
            return {"status": "ok", "scores": result.to_pandas().to_dict(orient="records")}
        except Exception as exc:  # pragma: no cover
            return {"status": "failed", "reason": str(exc)}
