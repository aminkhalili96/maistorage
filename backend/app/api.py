from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.corpus import group_sources_by_family
from app.models import ChatRequest, IngestRequest, SearchDebugRequest
from app.runtime import get_services


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_services().settings
    return {"status": "ok", "mode": settings.app_mode}


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
    services = get_services()
    return StreamingResponse(services.agent.stream(request), media_type="text/event-stream")
