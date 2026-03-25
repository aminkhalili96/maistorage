from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.knowledge_base import load_demo_chunks, load_sources
from app.models import ChatRequest
from app.services.agent import AgentService
from app.services.evaluation import EvaluationService
from app.services.indexes import InMemoryHybridIndex
from app.services.ingestion import IngestionService
from app.services.providers import OpenAIReasoner, KeywordEmbedder, TavilyClient
from app.services.retrieval import RetrievalService
from app.services.retrieval import grade_results, rerank_results

from compare_embeddings import run_comparison
from eval_common import (
    DisabledTavilyClient,
    ExperimentStack,
    TrackingEmbedder,
    TrackingReasoner,
    TrackingTavilyClient,
    WeightedInMemoryIndex,
    aggregate_retrieval_rows,
    benchmark_source_ids,
    build_local_stack,
    build_result_metadata,
    default_output_dir,
    keyword_hit,
    load_benchmark_bundle,
    load_json,
    load_rechunked_chunks,
    summarize_ragas,
    utc_now,
    write_json,
)


DEMO_QUESTION_PATH = Path("data/evals/demo_queries.json")
MAX_BENCHMARK_CHUNKS_PER_SOURCE = 12
CHUNKING_CONFIGS = [
    {"name": "chunk-600-overlap-80", "max_chars": 600, "overlap": 80},
    {"name": "chunk-900-overlap-120", "max_chars": 900, "overlap": 120},
    {"name": "chunk-1200-overlap-200", "max_chars": 1200, "overlap": 200},
]
PIPELINE_MODES = [
    ("retrieval_only", "Retrieval only"),
    ("retrieval_grading", "Retrieval + grading"),
    ("retrieval_grading_rewrite", "Retrieval + grading + rewrite"),
    ("full_agentic", "Retrieval + grading + rewrite + fallback gating"),
]


def _build_stack_with_keyword_fallback(
    settings: Any,
    sources: list[Any],
    chunks: list[Any],
    *,
    cache_root: Path,
    notes_on_fallback: str,
    **kwargs: Any,
) -> tuple[ExperimentStack, str]:
    try:
        return (
            build_local_stack(
                settings,
                sources,
                chunks,
                cache_root=cache_root,
                **kwargs,
            ),
            "configured",
        )
    except Exception:
        offline_settings = replace(
            settings,
            app_mode="dev",
            use_pinecone=False,
            langsmith_tracing=False,
            openai_api_key=None,
            use_tavily_fallback=False,
            embedder_provider="keyword",
        )
        lexical_weight = float(kwargs.get("lexical_weight", 0.55))
        dense_weight = float(kwargs.get("dense_weight", 0.45))
        track_calls = bool(kwargs.get("track_calls", False))
        raw_embedder = KeywordEmbedder()
        embedder = TrackingEmbedder(raw_embedder) if track_calls else raw_embedder
        index = WeightedInMemoryIndex(embedder, lexical_weight=lexical_weight, dense_weight=dense_weight)
        index.upsert(chunks)
        retrieval = RetrievalService(offline_settings, sources, index)
        base_reasoner = OpenAIReasoner(offline_settings)
        reasoner = TrackingReasoner(base_reasoner) if track_calls else base_reasoner
        tavily = DisabledTavilyClient()
        agent = AgentService(offline_settings, retrieval, reasoner, tavily)
        evaluation = EvaluationService(offline_settings, offline_settings.golden_questions_path, retrieval, agent)
        return (
            ExperimentStack(
                settings=offline_settings,
                sources=sources,
                chunks=chunks,
                embedder=embedder,
                index=index,
                retrieval=retrieval,
                agent=agent,
                evaluation=evaluation,
                reasoner=reasoner,
                tavily=tavily,
            ),
            notes_on_fallback,
        )


def _safe_comparison_result(experiment: str, reason: str) -> dict[str, Any]:
    return {"experiment": experiment, "status": "failed", "reason": reason, "results": []}


def _run_command(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "cmd": args,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "latency_ms": round(elapsed, 2),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_command_result(output_dir: Path, name: str, result: dict[str, Any]) -> dict[str, Any]:
    log_path = output_dir / f"{name}.log"
    log_path.write_text((result.get("stdout", "") + "\n" + result.get("stderr", "")).strip() + "\n")
    return {
        "exit_code": result["exit_code"],
        "latency_ms": result["latency_ms"],
        "log_path": str(log_path),
    }


def _demo_questions(project_root: Path) -> list[dict[str, Any]]:
    return load_json(project_root / DEMO_QUESTION_PATH)


def _aggregate_mode_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    response_modes: dict[str, int] = {}
    for run in runs:
        response_modes[run["response_mode"]] = response_modes.get(run["response_mode"], 0) + 1
    return {
        "avg_latency_ms": round(mean(float(run["latency_ms"]) for run in runs), 2),
        "avg_confidence": round(mean(float(run["confidence"]) for run in runs), 4),
        "grounding_pass_rate": round(mean(1.0 if run["grounding_passed"] else 0.0 for run in runs), 4),
        "answer_quality_pass_rate": round(mean(1.0 if run["answer_quality_passed"] else 0.0 for run in runs), 4),
        "keyword_hit_rate": round(mean(1.0 if run["keyword_hit"] else 0.0 for run in runs), 4),
        "citation_rate": round(mean(1.0 if run["citation_count"] > 0 else 0.0 for run in runs), 4),
        "avg_retry_count": round(mean(float(run["retry_count"]) for run in runs), 2),
        "response_modes": response_modes,
    }


def _ranking_key(entry: dict[str, Any]) -> tuple[float, float, float]:
    metrics = entry["retrieval_metrics"]
    return (float(metrics["hit@5"]), float(metrics["ndcg@5"]), float(metrics["mrr"]))


def _run_retrieval_benchmark(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    stack, stack_mode = _build_stack_with_keyword_fallback(
        settings,
        bundle.sources,
        bundle.chunks,
        cache_root=cache_root,
        embedder_provider="google",
        openai_embedding_dimensions=settings.openai_embedding_dimensions,
        notes_on_fallback="keyword_fallback_after_live_embedding_failure",
    )
    rows = stack.evaluation.evaluate_retrieval()
    payload = {
        "experiment": "retrieval_benchmark",
        "metadata": build_result_metadata(
            config_name="text-embedding-3-large-3072",
            bundle=bundle,
            question_path=bundle.question_path,
            notes=(
                f"Selected assessment configuration on a capped benchmark subset covering all golden questions "
                f"(max {MAX_BENCHMARK_CHUNKS_PER_SOURCE} chunks per source id). "
                + (
                    "Live OpenAI embeddings were unavailable, so the benchmark fell back to the keyword baseline."
                    if stack_mode != "configured"
                    else ""
                )
            ).strip(),
        ),
        "aggregate_metrics": aggregate_retrieval_rows(rows),
        "rows": [row.model_dump() for row in rows],
        "stack_mode": stack_mode,
    }
    write_json(output_dir / "retrieval_benchmark.json", payload)
    return payload


def _run_ragas(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    stack, stack_mode = _build_stack_with_keyword_fallback(
        settings,
        bundle.sources,
        bundle.chunks,
        cache_root=cache_root,
        embedder_provider="google",
        openai_embedding_dimensions=settings.openai_embedding_dimensions,
        notes_on_fallback="keyword_fallback_after_live_embedding_failure",
    )
    payload = {
        "experiment": "ragas",
        "metadata": build_result_metadata(
            config_name="text-embedding-3-large-3072",
            bundle=bundle,
            question_path=bundle.question_path,
            notes=(
                f"RAGAS run on the selected assessment configuration using authored reference answers on the capped benchmark subset "
                f"(max {MAX_BENCHMARK_CHUNKS_PER_SOURCE} chunks per source id). "
                + (
                    "Live OpenAI embeddings were unavailable, so context generation fell back to the keyword baseline."
                    if stack_mode != "configured"
                    else ""
                )
            ).strip(),
        ),
        "result": summarize_ragas(stack.evaluation.run_ragas()),
        "stack_mode": stack_mode,
    }
    write_json(output_dir / "ragas.json", payload)
    return payload


def _run_trajectory_benchmark(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, question_path=settings.golden_questions_path, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    sources = load_sources(settings.source_manifest_path)
    demo_chunks = load_demo_chunks(settings.demo_knowledge_base_path)
    index = InMemoryHybridIndex(KeywordEmbedder())
    ingestion = IngestionService(settings, index, sources, demo_chunks)
    ingestion.bootstrap_demo_knowledge_base()
    retrieval = RetrievalService(settings, sources, index)
    agent = AgentService(settings, retrieval, OpenAIReasoner(settings), TavilyClient(settings))
    eval_service = EvaluationService(settings, settings.golden_questions_path, retrieval, agent)
    local_agent = eval_service._build_local_benchmark_agent()
    local_evaluation = EvaluationService(settings, settings.golden_questions_path, local_agent.retrieval, local_agent)
    trajectory = local_evaluation.evaluate_trajectory()
    payload = {
        "experiment": "trajectory_benchmark",
        "metadata": build_result_metadata(
            config_name="agentic-trajectory",
            bundle=bundle,
            question_path=settings.golden_questions_path,
            notes="50-question deterministic local benchmark across direct chat, knowledge-base-backed RAG, refusal, and controlled fallback. TTFT currently equals total latency because the SSE path emits after the full agent run completes.",
        ),
        "rows": trajectory["rows"],
        "aggregate": trajectory["aggregate"],
    }
    write_json(output_dir / "trajectory_benchmark.json", payload)
    return payload


def _run_agentic_judge(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, question_path=settings.golden_questions_path, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    stack = build_local_stack(
        settings,
        bundle.sources,
        bundle.chunks,
        cache_root=cache_root,
        embedder_provider="google",
        openai_embedding_dimensions=settings.openai_embedding_dimensions,
        use_tavily_fallback=True,
    )
    result = stack.evaluation.run_agentic_judge()
    payload = {
        "experiment": "agentic_judge",
        "metadata": build_result_metadata(
            config_name="gpt-4.1-judge",
            bundle=bundle,
            question_path=settings.golden_questions_path,
            notes="Optional Gemini 3.1 Pro live judge for trajectory and final-answer quality. Skips cleanly when credentials, enable flag, or provider quota are unavailable.",
        ),
        "result": result,
    }
    write_json(output_dir / "agentic_judge.json", payload)
    return payload


def _run_dimension_ablation(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    payload = run_comparison(
        config_path=settings.project_root / "data/evals/embedding_dimensions.example.json",
        output_path=output_dir / "embedding_dimension_comparison.json",
        source_ids=benchmark_source_ids(load_json(settings.golden_questions_path)),
        max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE,
    )
    payload["experiment"] = "embedding_dimension_comparison"
    payload["comparison_label"] = "Gemini embedding dimension ablation"
    write_json(output_dir / "embedding_dimension_comparison.json", payload)
    return payload


def _run_chunking_ablation(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    results: list[dict[str, Any]] = []
    for config in CHUNKING_CONFIGS:
        sources, chunks = load_rechunked_chunks(settings, bundle.source_ids, config["max_chars"], config["overlap"])
        stack = build_local_stack(
            settings,
            sources,
            chunks,
            cache_root=cache_root,
            embedder_provider="keyword",
            openai_embedding_dimensions=settings.openai_embedding_dimensions,
        )
        rows = stack.evaluation.evaluate_retrieval()
        results.append(
            {
                "metadata": build_result_metadata(
                    config_name=config["name"],
                    bundle=bundle,
                    question_path=bundle.question_path,
                    notes="Chunking ablation on the benchmark subset using the keyword baseline as a fast local proxy.",
                    chunk_count=len(chunks),
                ),
                "chunking": config,
                "retrieval_metrics": aggregate_retrieval_rows(rows),
                "row_count": len(rows),
            }
        )
    best = max(results, key=_ranking_key)
    payload = {
        "experiment": "chunking_ablation",
        "evaluation_mode": "keyword-proxy",
        "results": results,
        "winner": {
            "config_name": best["metadata"]["config_name"],
            "chunking": best["chunking"],
            "retrieval_metrics": best["retrieval_metrics"],
        },
    }
    write_json(output_dir / "chunking_ablation.json", payload)
    return payload


def _run_pipeline_ablation(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    demo_questions = _demo_questions(settings.project_root)
    stack, stack_mode = _build_stack_with_keyword_fallback(
        settings,
        bundle.sources,
        bundle.chunks,
        cache_root=cache_root,
        embedder_provider="google",
        openai_embedding_dimensions=settings.openai_embedding_dimensions,
        use_tavily_fallback=True,
        notes_on_fallback="keyword_fallback_after_live_embedding_failure",
    )
    results: list[dict[str, Any]] = []
    for mode, label in PIPELINE_MODES:
        mode_runs: list[dict[str, Any]] = []
        for item in demo_questions:
            if mode != "full_agentic":
                from eval_common import run_manual_pipeline_mode

                payload = run_manual_pipeline_mode(stack, item["question"], mode)
            else:
                started = time.perf_counter()
                run = stack.agent.run(ChatRequest(question=item["question"]))
                payload = {
                    "answer": run.answer,
                    "citations": [citation.model_dump() for citation in run.citations],
                    "confidence": run.confidence,
                    "grounding_passed": run.grounding_passed,
                    "answer_quality_passed": run.answer_quality_passed,
                    "response_mode": run.response_mode,
                    "used_fallback": run.used_fallback,
                    "retry_count": run.retry_count,
                    "rejected_chunk_ids": run.rejected_chunk_ids,
                    "rewritten_query": run.rewritten_query,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            payload["question_id"] = item["id"]
            payload["question"] = item["question"]
            payload["expected_mode"] = item["expected_mode"]
            payload["expected_terms"] = item["expected_terms"]
            payload["keyword_hit"] = keyword_hit(payload["answer"], item["expected_terms"])
            payload["citation_count"] = len(payload["citations"])
            mode_runs.append(payload)
        results.append(
            {
                "mode": mode,
                "label": label,
                "aggregate": _aggregate_mode_runs(mode_runs),
                "runs": mode_runs,
            }
        )
    payload = {
        "experiment": "pipeline_ablation",
        "metadata": build_result_metadata(
            config_name="text-embedding-3-large-3072",
            bundle=bundle,
            question_path=settings.project_root / DEMO_QUESTION_PATH,
            notes=(
                f"End-to-end pipeline comparison across the rehearsed demo questions using the capped benchmark subset "
                f"(max {MAX_BENCHMARK_CHUNKS_PER_SOURCE} chunks per source id). "
                + (
                    "Live OpenAI embeddings were unavailable, so retrieval used the keyword baseline."
                    if stack_mode != "configured"
                    else ""
                )
            ).strip(),
        ),
        "results": results,
        "stack_mode": stack_mode,
    }
    write_json(output_dir / "pipeline_ablation.json", payload)
    return payload


def _run_hybrid_vs_dense(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    variants = [
        ("hybrid", 0.55, 0.45, "Dense + sparse hybrid retrieval"),
        ("dense_only", 0.0, 1.0, "Dense semantic retrieval only"),
    ]
    results: list[dict[str, Any]] = []
    for name, lexical_weight, dense_weight, notes in variants:
        stack, stack_mode = _build_stack_with_keyword_fallback(
            settings,
            bundle.sources,
            bundle.chunks,
            cache_root=cache_root,
            embedder_provider="google",
            openai_embedding_dimensions=settings.openai_embedding_dimensions,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            notes_on_fallback="keyword_fallback_after_live_embedding_failure",
        )
        rows = stack.evaluation.evaluate_retrieval()
        results.append(
            {
                "metadata": build_result_metadata(
                    config_name=name,
                    bundle=bundle,
                    question_path=bundle.question_path,
                    notes=notes
                    + (
                        " Keyword fallback was used because live OpenAI embeddings were unavailable."
                        if stack_mode != "configured"
                        else ""
                    ),
                ),
                "retrieval_metrics": aggregate_retrieval_rows(rows),
                "stack_mode": stack_mode,
            }
        )
    payload = {
        "experiment": "hybrid_vs_dense_only",
        "results": results,
    }
    write_json(output_dir / "hybrid_vs_dense_only.json", payload)
    return payload


def _run_latency_summary(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    demo_questions = _demo_questions(settings.project_root)
    stack, stack_mode = _build_stack_with_keyword_fallback(
        settings,
        bundle.sources,
        bundle.chunks,
        cache_root=cache_root,
        embedder_provider="google",
        openai_embedding_dimensions=settings.openai_embedding_dimensions,
        use_tavily_fallback=True,
        track_calls=True,
        notes_on_fallback="keyword_fallback_after_live_embedding_failure",
    )
    per_query: list[dict[str, Any]] = []
    for item in demo_questions:
        question = item["question"]
        plan = stack.retrieval.build_plan(question)

        if isinstance(stack.embedder, TrackingEmbedder):
            stack.embedder.reset()
        if isinstance(stack.reasoner, TrackingReasoner):
            stack.reasoner.reset()
        if isinstance(stack.tavily, TrackingTavilyClient):
            stack.tavily.reset()

        started = time.perf_counter()
        stack.embedder.embed_query(question)
        embedding_ms = (time.perf_counter() - started) * 1000

        retrieval_top_k = max(plan.top_k * 4, 12)
        started = time.perf_counter()
        candidates = stack.index.search(question, top_k=retrieval_top_k, families=plan.source_families)
        retrieval_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        reranked = rerank_results(question, plan, candidates)
        accepted, _ = grade_results(question, plan, reranked)
        grading_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        answer = stack.agent._synthesize_answer(question, accepted[: plan.top_k]) if accepted else stack.agent._refusal_answer()
        generation_ms = (time.perf_counter() - started) * 1000

        fallback_ms = 0.0
        if plan.recency_sensitive and isinstance(stack.tavily, TrackingTavilyClient):
            stack.tavily.reset()
            started = time.perf_counter()
            stack.tavily.search(question)
            fallback_ms = (time.perf_counter() - started) * 1000

        if isinstance(stack.embedder, TrackingEmbedder):
            stack.embedder.reset()
        if isinstance(stack.reasoner, TrackingReasoner):
            stack.reasoner.reset()
        if isinstance(stack.tavily, TrackingTavilyClient):
            stack.tavily.reset()

        started = time.perf_counter()
        run_state = stack.agent.run(ChatRequest(question=question))
        total_ms = (time.perf_counter() - started) * 1000
        embed_stats = stack.embedder.stats() if isinstance(stack.embedder, TrackingEmbedder) else {}
        reasoner_stats = stack.reasoner.stats() if isinstance(stack.reasoner, TrackingReasoner) else {}
        tavily_stats = stack.tavily.stats() if isinstance(stack.tavily, TrackingTavilyClient) else {}

        per_query.append(
            {
                "question_id": item["id"],
                "question": question,
                "response_mode": run_state.response_mode,
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "grading_ms": round(grading_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "fallback_ms": round(fallback_ms, 2),
                "total_agent_ms": round(total_ms, 2),
                "answer_length": len(answer),
                "api_call_counts": {**embed_stats, **reasoner_stats, **tavily_stats},
            }
        )

    payload = {
        "experiment": "latency_summary",
        "metadata": build_result_metadata(
            config_name="text-embedding-3-large-3072",
            bundle=bundle,
            question_path=settings.project_root / DEMO_QUESTION_PATH,
            notes=(
                "Per-query latency and API call counts on the rehearsed demo set. API call counts are used as a cost proxy."
                + (
                    " Retrieval embeddings fell back to the keyword baseline because live OpenAI embeddings were unavailable."
                    if stack_mode != "configured"
                    else ""
                )
            ),
        ),
        "queries": per_query,
        "aggregate": {
            "avg_embedding_ms": round(mean(float(item["embedding_ms"]) for item in per_query), 2),
            "avg_retrieval_ms": round(mean(float(item["retrieval_ms"]) for item in per_query), 2),
            "avg_grading_ms": round(mean(float(item["grading_ms"]) for item in per_query), 2),
            "avg_generation_ms": round(mean(float(item["generation_ms"]) for item in per_query), 2),
            "avg_total_agent_ms": round(mean(float(item["total_agent_ms"]) for item in per_query), 2),
        },
        "stack_mode": stack_mode,
    }
    write_json(output_dir / "latency_summary.json", payload)
    return payload


def _run_demo_query_validation(output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE)
    cache_root = settings.project_root / "data/evals/cache/embeddings"
    demo_questions = _demo_questions(settings.project_root)
    stack, stack_mode = _build_stack_with_keyword_fallback(
        settings,
        bundle.sources,
        bundle.chunks,
        cache_root=cache_root,
        embedder_provider="google",
        openai_embedding_dimensions=settings.openai_embedding_dimensions,
        use_tavily_fallback=True,
        notes_on_fallback="keyword_fallback_after_live_embedding_failure",
    )
    runs: list[dict[str, Any]] = []
    for item in demo_questions:
        state = stack.agent.run(ChatRequest(question=item["question"]))
        runs.append(
            {
                "question_id": item["id"],
                "question": item["question"],
                "expected_mode": item["expected_mode"],
                "response_mode": state.response_mode,
                "grounding_passed": state.grounding_passed,
                "answer_quality_passed": state.answer_quality_passed,
                "citation_count": len(state.citations),
                "retry_count": state.retry_count,
                "confidence": state.confidence,
                "keyword_hit": keyword_hit(state.answer, item["expected_terms"]),
                "answer": state.answer,
            }
        )
    payload = {
        "experiment": "demo_query_validation",
        "metadata": build_result_metadata(
            config_name="text-embedding-3-large-3072",
            bundle=bundle,
            question_path=settings.project_root / DEMO_QUESTION_PATH,
            notes=(
                f"Full-agent validation across the 3 primary demo queries and the controlled fallback query using the capped benchmark subset "
                f"(max {MAX_BENCHMARK_CHUNKS_PER_SOURCE} chunks per source id). "
                + (
                    "Live OpenAI embeddings were unavailable, so retrieval used the keyword baseline."
                    if stack_mode != "configured"
                    else ""
                )
            ).strip(),
        ),
        "runs": runs,
        "stack_mode": stack_mode,
    }
    write_json(output_dir / "demo_query_validation.json", payload)
    return payload


def _write_summary(
    output_dir: Path,
    suite_status: dict[str, Any],
    retrieval: dict[str, Any],
    ragas: dict[str, Any],
    trajectory: dict[str, Any],
    judge: dict[str, Any],
    embeddings: dict[str, Any],
    dimensions: dict[str, Any],
    chunking: dict[str, Any],
    pipeline: dict[str, Any],
    hybrid: dict[str, Any],
    latency: dict[str, Any],
    demo_queries: dict[str, Any],
) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"Generated at: `{utc_now()}`",
        "",
        "## Verification",
        "",
        f"- Backend tests: `exit {suite_status['backend_tests']['exit_code']}`",
        f"- Frontend build: `exit {suite_status['frontend_build']['exit_code']}`",
        "",
        "## Retrieval Benchmark",
        "",
        f"- `hit@5`: `{retrieval['aggregate_metrics']['hit@5']}`",
        f"- `MRR`: `{retrieval['aggregate_metrics']['mrr']}`",
        f"- `nDCG@5`: `{retrieval['aggregate_metrics']['ndcg@5']}`",
        f"- `routing@3`: `{retrieval['aggregate_metrics']['routing@3']}`",
        "",
        "## RAGAS",
        "",
    ]
    ragas_result = ragas["result"]
    if ragas_result.get("status") == "ok":
        for key, value in ragas_result["summary"].items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append(f"- Status: `{ragas_result.get('status')}`")
        lines.append(f"- Reason: `{ragas_result.get('reason', 'unknown')}`")
    lines.extend(
        [
            "",
            "## Trajectory Benchmark",
            "",
            f"- Questions: `{trajectory['aggregate']['question_count']}`",
            f"- Assistant-mode accuracy: `{trajectory['aggregate']['assistant_mode_accuracy']}`",
            f"- Response-mode accuracy: `{trajectory['aggregate']['response_mode_accuracy']}`",
            f"- Tool-path accuracy: `{trajectory['aggregate']['tool_path_accuracy']}`",
            "",
            "## GPT-4.1 Judge",
            "",
        ]
    )
    judge_result = judge["result"]
    if judge_result.get("status") == "ok":
        lines.append(f"- Questions judged: `{judge_result['question_count']}`")
        for key, value in judge_result["summary"].items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append(f"- Status: `{judge_result.get('status')}`")
        lines.append(f"- Reason: `{judge_result.get('reason', 'unknown')}`")
    lines.extend(
        [
            "",
            "## Slide-Worthy Comparisons",
            "",
            f"- Embedding models compared: `{len(embeddings['results'])}`",
            f"- Dimension variants compared: `{len(dimensions['results'])}`",
            f"- Chunking winner: `{chunking['winner']['config_name']}`",
            f"- Pipeline modes compared: `{len(pipeline['results'])}`",
            f"- Hybrid vs dense-only compared: `{len(hybrid['results'])}`",
            "",
            "## Latency",
            "",
            f"- Average end-to-end latency: `{latency['aggregate']['avg_total_agent_ms']} ms`",
            f"- Average retrieval latency: `{latency['aggregate']['avg_retrieval_ms']} ms`",
            f"- Average generation latency: `{latency['aggregate']['avg_generation_ms']} ms`",
            "",
            "## Demo Query Validation",
            "",
        ]
    )
    for run in demo_queries["runs"]:
        lines.append(
            f"- `{run['question_id']}` -> mode `{run['response_mode']}`, citations `{run['citation_count']}`, "
            f"grounding `{run['grounding_passed']}`, answer-quality `{run['answer_quality_passed']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The embedding comparison now uses 3 real models: Gemini plus two Pinecone-hosted models.",
            "- The chunking ablation uses the keyword baseline as a fast local proxy; the selected assessment benchmark still uses `gemini-embedding-001` at 3072 dimensions.",
            "- API call counts in the latency and trajectory artifacts are cost proxies, not billing reports.",
            "- The benchmark question file now unifies 50 questions across direct chat, knowledge-base-backed RAG, refusal, and recency-sensitive fallback.",
            "- The current RAGAS run measures only the knowledge-base-backed subset of the benchmark; the Gemini 3.1 Pro judge covers the broader agent trajectory when enabled.",
        ]
    )
    (output_dir / "slide_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation evidence suite and write slide-ready artifacts.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to data/evals/results/<timestamp>.")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip the live RAGAS run.")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip the hybrid vs dense-only comparison.")
    parser.add_argument("--skip-verification", action="store_true", help="Skip pytest and frontend build inside the suite runner.")
    args = parser.parse_args()

    settings = get_settings()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "backend"
    env["LANGSMITH_TRACING"] = "false"

    if args.skip_verification:
        suite_status = {
            "run_at": utc_now(),
            "backend_tests": {"exit_code": "skipped", "latency_ms": 0.0, "log_path": ""},
            "frontend_build": {"exit_code": "skipped", "latency_ms": 0.0, "log_path": ""},
        }
    else:
        backend_result = _run_command(
            ["backend/.venv/bin/python", "-m", "pytest", "backend/tests"],
            cwd=settings.project_root,
            env=env,
        )
        frontend_result = _run_command(
            ["npm", "run", "build"],
            cwd=settings.project_root / "frontend",
            env=env,
        )
        suite_status = {
            "run_at": utc_now(),
            "backend_tests": _write_command_result(output_dir, "backend-tests", backend_result),
            "frontend_build": _write_command_result(output_dir, "frontend-build", frontend_result),
        }
    write_json(output_dir / "suite_status.json", suite_status)

    print("running retrieval benchmark...")
    retrieval = _run_retrieval_benchmark(output_dir)
    print("running ragas..." if not args.skip_ragas else "skipping ragas...")
    ragas = {"result": {"status": "skipped", "reason": "Skipped by flag."}} if args.skip_ragas else _run_ragas(output_dir)
    print("running trajectory benchmark...")
    trajectory = _run_trajectory_benchmark(output_dir)
    print("running GPT-4.1 judge...")
    judge = _run_agentic_judge(output_dir)
    print("running embedding model comparison...")
    try:
        embeddings = run_comparison(
            config_path=settings.project_root / "data/evals/embedding_experiments.example.json",
            output_path=output_dir / "embedding_model_comparison.json",
            source_ids=benchmark_source_ids(load_json(settings.golden_questions_path)),
            max_chunks_per_source=MAX_BENCHMARK_CHUNKS_PER_SOURCE,
        )
    except Exception as exc:
        embeddings = _safe_comparison_result("embedding_model_comparison", str(exc))
        write_json(output_dir / "embedding_model_comparison.json", embeddings)
    print("running embedding dimension ablation...")
    try:
        dimensions = _run_dimension_ablation(output_dir)
    except Exception as exc:
        dimensions = _safe_comparison_result("embedding_dimension_comparison", str(exc))
        write_json(output_dir / "embedding_dimension_comparison.json", dimensions)
    print("running chunking ablation...")
    chunking = _run_chunking_ablation(output_dir)
    print("running pipeline ablation...")
    pipeline = _run_pipeline_ablation(output_dir)
    print("running hybrid vs dense-only comparison..." if not args.skip_hybrid else "skipping hybrid vs dense-only comparison...")
    hybrid = {"experiment": "hybrid_vs_dense_only", "results": []} if args.skip_hybrid else _run_hybrid_vs_dense(output_dir)
    print("running latency summary...")
    latency = _run_latency_summary(output_dir)
    print("running demo query validation...")
    demo_queries = _run_demo_query_validation(output_dir)
    _write_summary(output_dir, suite_status, retrieval, ragas, trajectory, judge, embeddings, dimensions, chunking, pipeline, hybrid, latency, demo_queries)

    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
