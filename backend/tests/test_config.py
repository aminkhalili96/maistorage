from app.config import RerankConfig, Settings


def test_assessment_mode_validation_requires_openai_stack(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        app_mode="assessment",
        api_title="test",
        openai_api_key=None,
        openai_model="gpt-5.4",
        openai_pipeline_model="gpt-5-mini",
        openai_routing_model="gpt-5.4-nano",
        openai_allowed_models=("gpt-5.4-nano", "gpt-5-mini", "gpt-5.4"),
        openai_embedding_model="text-embedding-3-large",
        openai_embedding_dimensions=3072,
        pinecone_api_key=None,
        pinecone_index_name=None,
        pinecone_namespace="demo",
        langsmith_api_key=None,
        langsmith_project="demo",
        langsmith_tracing=False,
        tavily_api_key=None,
        use_pinecone=False,
        use_tavily_fallback=False,
        embedder_provider="keyword",
        corpus_root=tmp_path / "data/corpus",
        raw_html_root=tmp_path / "data/corpus/raw/html",
        raw_pdf_root=tmp_path / "data/corpus/raw/pdfs",
        raw_md_root=tmp_path / "data/corpus/raw/markdown",
        raw_doc_root=tmp_path / "data/corpus/raw",
        normalized_doc_root=tmp_path / "data/corpus/normalized",
        corpus_manifest_path=tmp_path / "data/corpus/manifest.json",
        demo_corpus_path=tmp_path / "data/demo_chunks.json",
        source_manifest_path=tmp_path / "data/sources/nvidia_sources.json",
        golden_questions_path=tmp_path / "data/evals/golden_questions.json",
        cors_origin="http://127.0.0.1:5173",
        rerank_config=RerankConfig(),
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.92,
        decomposition_enabled=False,
        openai_temperature=0.2,
        openai_timeout=60.0,
        tavily_timeout=30.0,
        embedder_timeout=30.0,
    )

    errors = settings.validate_runtime()
    assert any("OPENAI_API_KEY" in error for error in errors)
    assert not any("OPENAI_MODEL" in error for error in errors)
    assert any("EMBEDDER_PROVIDER" in error for error in errors)
    assert any("USE_PINECONE" in error for error in errors)
