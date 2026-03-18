from __future__ import annotations

import os
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ALLOWED_OPENAI_MODELS = (
    "gpt-5.4-nano",
    "gpt-5-mini",
    "gpt-5.4",
)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class RerankConfig:
    """Weights and caps used by rerank_results(). Extracted here so they can be
    tuned explicitly and tested independently of the retrieval logic."""
    lexical_overlap_weight: float = 0.28
    family_bonus: float = 0.14
    hardware_family_bonus: float = 0.22
    metadata_bonus: float = 0.06
    tag_bonus: float = 0.08
    max_per_source: int = 3


@dataclass(slots=True)
class Settings:
    project_root: Path
    app_mode: str
    api_title: str
    openai_api_key: str | None
    openai_model: str
    openai_pipeline_model: str
    openai_routing_model: str
    openai_allowed_models: tuple[str, ...]
    openai_embedding_model: str
    openai_embedding_dimensions: int
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
    raw_md_root: Path
    raw_doc_root: Path
    normalized_doc_root: Path
    corpus_manifest_path: Path
    demo_corpus_path: Path
    source_manifest_path: Path
    golden_questions_path: Path
    cors_origin: str
    rerank_config: RerankConfig
    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    decomposition_enabled: bool
    openai_temperature: float
    openai_timeout: float
    tavily_timeout: float
    embedder_timeout: float

    @property
    def generation_model(self) -> str:
        return self.openai_model

    @property
    def pipeline_model(self) -> str:
        return self.openai_pipeline_model

    @property
    def routing_model(self) -> str:
        return self.openai_routing_model

    @property
    def is_assessment_mode(self) -> bool:
        return self.app_mode == "assessment"

    def validate_runtime(self) -> list[str]:
        errors: list[str] = []
        if not self.is_assessment_mode:
            return errors

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required in assessment mode.")
        if self.openai_model not in self.openai_allowed_models:
            errors.append("Assessment mode requires OPENAI_MODEL to be an allowed OpenAI model.")
        if self.embedder_provider != "openai":
            errors.append("Assessment mode requires EMBEDDER_PROVIDER=openai.")
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
    raw_md_root = corpus_root / "raw/markdown"
    raw_doc_root = project_root / os.getenv("RAW_DOC_ROOT", "data/corpus/raw")
    normalized_doc_root = project_root / os.getenv("NORMALIZED_DOC_ROOT", "data/corpus/normalized")
    embedder_provider = os.getenv("EMBEDDER_PROVIDER", "openai" if app_mode == "assessment" else "keyword")
    use_pinecone = _as_bool(os.getenv("USE_PINECONE"), default=app_mode == "assessment")

    return Settings(
        project_root=project_root,
        app_mode=app_mode,
        api_title="NVIDIA AI Infrastructure Agentic RAG",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        openai_pipeline_model=os.getenv("OPENAI_PIPELINE_MODEL", "gpt-5-mini"),
        openai_routing_model=os.getenv("OPENAI_ROUTING_MODEL", "gpt-5.4-nano"),
        openai_allowed_models=ALLOWED_OPENAI_MODELS,
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        openai_embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "3072")),
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
        raw_md_root=raw_md_root,
        raw_doc_root=raw_doc_root,
        normalized_doc_root=normalized_doc_root,
        corpus_manifest_path=project_root / os.getenv("CORPUS_MANIFEST_PATH", "data/corpus/manifest.json"),
        demo_corpus_path=project_root / os.getenv("DEMO_CORPUS_PATH", "data/demo_chunks.json"),
        source_manifest_path=project_root / "data/sources/nvidia_sources.json",
        golden_questions_path=project_root / "data/evals/ragas_slim_10.json",
        cors_origin=os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
        rerank_config=RerankConfig(),
        semantic_cache_enabled=_as_bool(os.getenv("SEMANTIC_CACHE_ENABLED"), default=False),
        semantic_cache_threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92")),
        decomposition_enabled=_as_bool(os.getenv("QUERY_DECOMPOSITION_ENABLED"), default=False),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        openai_timeout=float(os.getenv("OPENAI_TIMEOUT", "60.0")),
        tavily_timeout=float(os.getenv("TAVILY_TIMEOUT", "30.0")),
        embedder_timeout=float(os.getenv("EMBEDDER_TIMEOUT", "30.0")),
    )
