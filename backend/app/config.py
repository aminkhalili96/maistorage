from __future__ import annotations

import os
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    project_root: Path
    app_mode: str
    api_title: str
    gemini_api_key: str | None
    gemini_model: str
    gemini_embedding_model: str
    gemini_embedding_dimensions: int
    gemini_embedding_document_task_type: str
    gemini_embedding_query_task_type: str
    pinecone_api_key: str | None
    pinecone_index_name: str | None
    pinecone_namespace: str
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_tracing: bool
    tavily_api_key: str | None
    use_pinecone: bool
    use_tavily_fallback: bool
    embedder_provider: str
    corpus_root: Path
    raw_html_root: Path
    raw_pdf_root: Path
    raw_doc_root: Path
    normalized_doc_root: Path
    corpus_manifest_path: Path
    demo_corpus_path: Path
    source_manifest_path: Path
    golden_questions_path: Path
    cors_origin: str

    @property
    def generation_model(self) -> str:
        return self.gemini_model

    @property
    def is_assessment_mode(self) -> bool:
        return self.app_mode == "assessment"

    def validate_runtime(self) -> list[str]:
        errors: list[str] = []
        if not self.is_assessment_mode:
            return errors

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is required in assessment mode.")
        if self.gemini_model != "gemini-3.1-pro-preview":
            errors.append("Assessment mode requires GEMINI_MODEL=gemini-3.1-pro-preview.")
        if self.gemini_embedding_model != "gemini-embedding-001":
            errors.append("Assessment mode requires GEMINI_EMBEDDING_MODEL=gemini-embedding-001.")
        if self.embedder_provider != "google":
            errors.append("Assessment mode requires EMBEDDER_PROVIDER=google.")
        if not self.use_pinecone:
            errors.append("Assessment mode requires USE_PINECONE=true.")
        if not self.pinecone_api_key or not self.pinecone_index_name:
            errors.append("Assessment mode requires PINECONE_API_KEY and PINECONE_INDEX_NAME.")
        if not self.corpus_manifest_path.exists():
            errors.append("Assessment mode requires data/corpus/manifest.json.")
        elif not json.loads(self.corpus_manifest_path.read_text()).get("sources"):
            errors.append("Assessment mode requires a populated corpus manifest with bundled sources.")
        if not any(self.raw_html_root.rglob("*.html")) and not any(self.raw_pdf_root.rglob("*.pdf")):
            errors.append("Assessment mode requires a bundled local corpus under data/corpus/raw.")
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    app_mode = os.getenv("APP_MODE", "dev").strip().lower()
    corpus_root = project_root / os.getenv("CORPUS_ROOT", "data/corpus")
    raw_html_root = corpus_root / "raw/html"
    raw_pdf_root = corpus_root / "raw/pdfs"
    raw_doc_root = project_root / os.getenv("RAW_DOC_ROOT", "data/corpus/raw")
    normalized_doc_root = project_root / os.getenv("NORMALIZED_DOC_ROOT", "data/corpus/normalized")
    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "google" if app_mode == "assessment" else "keyword")
    use_pinecone = _as_bool(os.getenv("USE_PINECONE"), default=app_mode == "assessment")

    return Settings(
        project_root=project_root,
        app_mode=app_mode,
        api_title="NVIDIA AI Infrastructure Agentic RAG",
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        gemini_embedding_dimensions=int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "3072")),
        gemini_embedding_document_task_type=os.getenv("GEMINI_DOCUMENT_TASK_TYPE", "RETRIEVAL_DOCUMENT"),
        gemini_embedding_query_task_type=os.getenv("GEMINI_QUERY_TASK_TYPE", "RETRIEVAL_QUERY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "nvidia-rag"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "nvidia-agentic-rag"),
        langsmith_tracing=_as_bool(os.getenv("LANGSMITH_TRACING"), default=False),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        use_pinecone=use_pinecone,
        use_tavily_fallback=_as_bool(os.getenv("USE_TAVILY_FALLBACK"), default=False),
        embedder_provider=embedder_provider,
        corpus_root=corpus_root,
        raw_html_root=raw_html_root,
        raw_pdf_root=raw_pdf_root,
        raw_doc_root=raw_doc_root,
        normalized_doc_root=normalized_doc_root,
        corpus_manifest_path=project_root / os.getenv("CORPUS_MANIFEST_PATH", "data/corpus/manifest.json"),
        demo_corpus_path=project_root / os.getenv("DEMO_CORPUS_PATH", "data/demo_chunks.json"),
        source_manifest_path=project_root / "data/sources/nvidia_sources.json",
        golden_questions_path=project_root / "data/evals/golden_questions.json",
        cors_origin=os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
    )
