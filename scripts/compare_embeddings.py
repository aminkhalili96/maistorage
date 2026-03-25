from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.knowledge_base import load_normalized_chunks, load_sources

from eval_common import (
    aggregate_retrieval_rows,
    build_local_stack,
    build_result_metadata,
    load_benchmark_bundle,
    load_json,
    summarize_ragas,
    utc_now,
    write_json,
)


def load_experiment_configs(path: Path) -> list[dict]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("Embedding config file must contain a JSON array.")
    return payload


def run_comparison(
    *,
    config_path: Path,
    output_path: Path | None = None,
    include_ragas: bool = False,
    source_ids: list[str] | None = None,
    use_all_sources: bool = False,
    max_chunks_per_source: int | None = 12,
) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_benchmark_bundle(settings, source_ids=source_ids, max_chunks_per_source=None if use_all_sources else max_chunks_per_source)

    cache_root = settings.project_root / "data/evals/cache/embeddings"
    configs = load_experiment_configs(config_path)
    report: list[dict[str, Any]] = []

    selected_sources = bundle.sources
    selected_chunks = bundle.chunks
    if use_all_sources:
        selected_sources = load_sources(settings.source_manifest_path)
        selected_chunks = load_normalized_chunks(settings.normalized_doc_root)

    for config in configs:
        name = config["name"]
        provider = config.get("embedder_provider", settings.embedder_provider)
        experiment_settings = replace(
            settings,
            embedder_provider=provider,
            gemini_embedding_model=config.get("gemini_embedding_model", settings.gemini_embedding_model),
            gemini_embedding_dimensions=int(config.get("gemini_embedding_dimensions", settings.gemini_embedding_dimensions)),
        )
        stack = build_local_stack(
            experiment_settings,
            selected_sources,
            selected_chunks,
            cache_root=cache_root,
            embedder_provider=provider,
            embedder_model=config.get("embedder_model"),
            gemini_embedding_dimensions=int(config.get("gemini_embedding_dimensions", experiment_settings.gemini_embedding_dimensions)),
            pinecone_embedding_dimensions=int(config.get("pinecone_embedding_dimensions", config.get("embedding_dimensions", 1024))),
            pinecone_document_task_type=config.get("pinecone_document_task_type", "passage"),
            pinecone_query_task_type=config.get("pinecone_query_task_type", "query"),
        )
        retrieval_rows = stack.evaluation.evaluate_retrieval()
        result: dict[str, Any] = {
            "metadata": build_result_metadata(
                config_name=name,
                bundle=bundle,
                question_path=bundle.question_path,
                notes=f"Local benchmark subset comparison across retrieval configurations (max {max_chunks_per_source or 'all'} chunks per source id).",
                chunk_count=len(selected_chunks),
                source_ids=[source.id for source in selected_sources],
            ),
            "config": config,
            "retrieval_metrics": aggregate_retrieval_rows(retrieval_rows),
            "retrieval_rows": [row.model_dump() for row in retrieval_rows],
        }

        if include_ragas and experiment_settings.openai_api_key and provider == "openai":
            result["ragas"] = summarize_ragas(stack.evaluation.run_ragas())

        report.append(result)

    payload = {
        "experiment": "embedding_configuration_comparison",
        "comparison_label": "3 embedding models" if len(report) == 3 else "embedding model comparison",
        "run_at": utc_now(),
        "results": report,
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple embedding configurations on the local NVIDIA knowledge base.")
    parser.add_argument("--configs", required=True, help="Path to a JSON array of embedding experiment configs.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--include-ragas", action="store_true", help="Run RAGAS after retrieval metrics when credentials are available.")
    parser.add_argument("--source-id", action="append", dest="source_ids", help="Restrict the comparison to specific source ids.")
    parser.add_argument("--all-sources", action="store_true", help="Compare against the full normalized knowledge base instead of the benchmark subset.")
    parser.add_argument("--max-chunks-per-source", type=int, default=12, help="Cap benchmark chunks per source id to keep live comparisons tractable.")
    args = parser.parse_args()

    payload = run_comparison(
        config_path=Path(args.configs),
        output_path=Path(args.output) if args.output else None,
        include_ragas=args.include_ragas,
        source_ids=args.source_ids,
        use_all_sources=args.all_sources,
        max_chunks_per_source=args.max_chunks_per_source,
    )
    if not args.output:
        import json

        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
