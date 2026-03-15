from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import httpx

from app.config import Settings


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class KeywordEmbedder:
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dimensions
        for token, count in counts.items():
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[index] += float(count)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GoogleGeminiEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Google embeddings.")
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_embedding_model
        self.dimensions = settings.gemini_embedding_dimensions
        self.document_task_type = settings.gemini_embedding_document_task_type
        self.query_task_type = settings.gemini_embedding_query_task_type
        self.client = httpx.Client(timeout=30.0)

    def _embed(self, text: str, task_type: str) -> list[float]:
        response = self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
            params={"key": self.api_key},
            json={
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": self.dimensions,
            },
        )
        response.raise_for_status()
        payload = response.json()
        values = payload.get("embedding", {}).get("values")
        if not values and payload.get("embeddings"):
            values = payload["embeddings"][0].get("values")
        if not values:
            raise RuntimeError("Gemini embedding response did not include embedding values.")
        return [float(value) for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, self.document_task_type) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, self.query_task_type)


class GeminiReasoner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(timeout=60.0)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("Gemini is not configured.")
        target_model = model or self.settings.generation_model
        response = self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent",
            params={"key": self.settings.gemini_api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini did not return any candidate content.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise RuntimeError("Gemini response did not include text content.")
        return text


class TavilyClient:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.tavily_api_key
        self.enabled = bool(settings.use_tavily_fallback and self.api_key)
        self.client = httpx.Client(timeout=30.0)

    def search(self, query: str) -> list[dict[str, str]]:
        if not self.enabled:
            return []
        response = self.client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
            },
        )
        response.raise_for_status()
        payload = response.json()
        results: list[dict[str, str]] = []
        for item in payload.get("results", []):
            results.append(
                {
                    "title": item.get("title", "Web result"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
            )
        return results


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder_provider == "google":
        try:
            return GoogleGeminiEmbedder(settings)
        except Exception:
            if settings.is_assessment_mode:
                raise
            return KeywordEmbedder()
    if settings.is_assessment_mode:
        raise ValueError("Assessment mode requires the Google embedder.")
    return KeywordEmbedder()
