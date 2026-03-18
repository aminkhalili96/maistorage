from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    from app.runtime import reset_services
    reset_services()


settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    description=(
        "Agentic RAG assistant for NVIDIA AI infrastructure. "
        "LangGraph pipeline with Self-RAG, adaptive retrieval, and 4-layer fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
    settings.cors_origin,
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app.add_middleware(RequestIDMiddleware)
app.include_router(router)
