from app.config import Settings


def test_assessment_mode_validation_requires_google_stack(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        app_mode="assessment",
        api_title="test",
        gemini_api_key=None,
        gemini_model="gemini-2.5-pro",
        gemini_embedding_model="text-embedding-004",
        gemini_embedding_dimensions=768,
        gemini_embedding_document_task_type="RETRIEVAL_DOCUMENT",
        gemini_embedding_query_task_type="RETRIEVAL_QUERY",
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
        raw_doc_root=tmp_path / "data/corpus/raw",
        normalized_doc_root=tmp_path / "data/corpus/normalized",
        corpus_manifest_path=tmp_path / "data/corpus/manifest.json",
        demo_corpus_path=tmp_path / "data/demo_chunks.json",
        source_manifest_path=tmp_path / "data/sources/nvidia_sources.json",
        golden_questions_path=tmp_path / "data/evals/golden_questions.json",
        cors_origin="http://127.0.0.1:5173",
    )

    errors = settings.validate_runtime()
    assert any("GEMINI_API_KEY" in error for error in errors)
    assert any("GEMINI_MODEL" in error for error in errors)
    assert any("GEMINI_EMBEDDING_MODEL" in error for error in errors)
    assert any("EMBEDDER_PROVIDER" in error for error in errors)
    assert any("USE_PINECONE" in error for error in errors)
