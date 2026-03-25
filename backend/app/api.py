"""
FastAPI router — all API endpoints for the agentic RAG system.

Endpoints:
  GET  /health           → system health + indexed chunk count
  GET  /api/sources      → knowledge base source registry grouped by family
  GET  /api/ingest/status → ingestion job status
  POST /api/ingest/start  → trigger background ingestion job
  POST /api/search/debug  → debug retrieval without generation (returns raw results)
  GET  /api/evals/retrieval → run retrieval benchmark against golden questions
  GET  /api/evals/ragas    → run RAGAS evaluation (costs API credits!)
  POST /api/chat/stream    → main chat endpoint — returns SSE stream of trace + answer
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.knowledge_base import group_sources_by_family
from app.models import ChatRequest, IngestRequest, SearchDebugRequest
from app.runtime import get_services


router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    services = get_services()
    s = services.settings
    chunk_count = services.index.count()
    return {
        "status": "ok" if chunk_count > 0 else "degraded",
        "mode": s.app_mode,
        "knowledge_base_loaded": chunk_count > 0,
        "indexed_chunks": chunk_count,
    }


@router.get("/api/sources")
def sources():
    services = get_services()
    grouped = group_sources_by_family(services.sources)
    status = services.ingestion.get_status()
    return {
        "sources": [source.model_dump() for source in services.sources],
        "families": {family: len(items) for family, items in grouped.items()},
        "indexed_chunks": services.index.count(),
        "app_mode": services.settings.app_mode,
        "snapshot_id": status.snapshot_id,
        "last_refresh_at": status.last_refresh_at,
    }


@router.get("/api/ingest/status")
def ingest_status():
    return get_services().ingestion.get_status()


@router.post("/api/ingest/start")
def ingest_start(request: IngestRequest, background_tasks: BackgroundTasks):
    services = get_services()
    job_id = services.ingestion.prepare_job()
    background_tasks.add_task(services.ingestion.run_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@router.post("/api/search/debug")
def search_debug(request: SearchDebugRequest):
    return get_services().retrieval.search(request.question)


@router.get("/api/evals/retrieval")
def eval_retrieval():
    rows = get_services().evaluation.evaluate_retrieval()
    return {"rows": [row.model_dump() for row in rows]}


@router.get("/api/evals/ragas")
def eval_ragas():
    return get_services().evaluation.run_ragas()


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="Question exceeds maximum length of 2000 characters.")
    if len(request.history) > 20:
        raise HTTPException(status_code=400, detail="Conversation history exceeds maximum of 20 turns.")
    if any(len(turn.content) > 5000 for turn in request.history):
        raise HTTPException(status_code=400, detail="A history message exceeds maximum length of 5000 characters.")
    services = get_services()
    if request.model and request.model not in services.settings.openai_allowed_models:
        raise HTTPException(status_code=400, detail="Unsupported model.")
    return StreamingResponse(services.agent.stream(request), media_type="text/event-stream")
