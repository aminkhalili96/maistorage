"""Run RAGAS evaluation on the slim 10-question subset only.

Usage:
    LANGSMITH_TRACING=false PYTHONPATH=backend backend/.venv/bin/python scripts/run_ragas_slim.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest
from app.services.agent import AgentService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import KeywordEmbedder, OpenAIReasoner, TavilyClient
from app.services.retrieval import RetrievalService

SLIM_QUESTIONS_PATH = PROJECT_ROOT / "data/evals/ragas_slim_10.json"
OUTPUT_DIR = PROJECT_ROOT / "data/evals/results"


def main() -> None:
    settings = get_settings()

    # Load slim questions
    questions = json.loads(SLIM_QUESTIONS_PATH.read_text())
    print(f"Loaded {len(questions)} slim RAGAS questions")

    # Build local stack (keyword embedder, in-memory index)
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    embedder = KeywordEmbedder()
    index = InMemoryHybridIndex(embedder)
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    reasoner = OpenAIReasoner(settings)
    tavily = TavilyClient(settings)
    agent = AgentService(settings, retrieval, reasoner, tavily)

    print(f"Index loaded: {index.count()} chunks")
    print(f"Generation model: {settings.generation_model}")
    print(f"Pipeline model: {settings.pipeline_model}")

    # Step 1: Generate answers for each question
    print("\n--- Generating answers ---")
    rows: list[dict] = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"  [{i}/{len(questions)}] {q[:80]}...")
        started = time.perf_counter()
        try:
            run_state = agent.run(ChatRequest(question=q))
            elapsed = time.perf_counter() - started
            ground_truth = item.get("reference_answer") or " | ".join(item.get("expected_terms", []))
            rows.append({
                "question": q,
                "answer": run_state.answer,
                "contexts": [r.chunk.content for r in run_state.retrieval_results[:4]],
                "ground_truth": ground_truth,
                "metadata": {
                    "id": item["id"],
                    "category": item["category"],
                    "response_mode_expected": item["response_mode_expected"],
                    "response_mode_actual": run_state.response_mode,
                    "confidence": run_state.confidence,
                    "citation_count": len(run_state.citations),
                    "latency_ms": round(elapsed * 1000, 2),
                },
            })
            print(f"    -> {run_state.response_mode} | confidence={run_state.confidence:.2f} | {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"    -> FAILED: {exc} ({elapsed:.1f}s)")
            rows.append({
                "question": q,
                "answer": f"Error: {exc}",
                "contexts": [],
                "ground_truth": item.get("reference_answer", ""),
                "metadata": {"id": item["id"], "category": item["category"], "error": str(exc)},
            })

    # Step 2: Run RAGAS evaluation
    print("\n--- Running RAGAS scoring ---")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        evaluator_model = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4.1")
        print(f"RAGAS judge model: {evaluator_model}")

        dataset = Dataset.from_list(rows)
        llm = ChatOpenAI(model=evaluator_model, api_key=settings.openai_api_key, temperature=0.0)
        embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1, max_retries=6, max_wait=60),
            batch_size=1,
        )
        scores_df = result.to_pandas()
        scores = scores_df.to_dict(orient="records")
        summary = {
            "faithfulness": round(float(scores_df["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(scores_df["answer_relevancy"].mean()), 4),
            "context_precision": round(float(scores_df["context_precision"].mean()), 4),
            "context_recall": round(float(scores_df["context_recall"].mean()), 4),
        }
        ragas_result = {"status": "ok", "summary": summary, "scores": scores, "question_count": len(rows)}
        print("\n--- RAGAS Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        ragas_result = {"status": "failed", "reason": str(exc)}
        print(f"RAGAS evaluation failed: {exc}")

    # Step 3: Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "ragas_slim_10",
        "timestamp": timestamp,
        "question_file": str(SLIM_QUESTIONS_PATH),
        "question_count": len(questions),
        "categories": sorted(set(q["category"] for q in questions)),
        "response_modes": sorted(set(q["response_mode_expected"] for q in questions)),
        "generation_rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows],
        "ragas": ragas_result,
    }
    out_path = out_dir / "ragas_slim_10.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResults saved to: {out_path}")

    # Also save the full log-friendly summary
    log_path = out_dir / "ragas_slim_10_log.txt"
    lines = [
        f"RAGAS Slim 10 Evaluation — {timestamp}",
        f"Questions: {len(questions)}",
        f"Categories: {', '.join(sorted(set(q['category'] for q in questions)))}",
        "",
    ]
    for r in rows:
        meta = r["metadata"]
        lines.append(f"[{meta.get('id', '?')}] {r['question'][:80]}")
        lines.append(f"  mode={meta.get('response_mode_actual','?')} confidence={meta.get('confidence','?')} citations={meta.get('citation_count','?')} latency={meta.get('latency_ms','?')}ms")
    lines.append("")
    if ragas_result["status"] == "ok":
        lines.append("RAGAS Summary:")
        for k, v in ragas_result["summary"].items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append(f"RAGAS: {ragas_result.get('reason', 'failed')}")
    log_path.write_text("\n".join(lines) + "\n")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
