from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol

import httpx

from app.config import Settings


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+./-]*")

STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "not", "no", "nor",
    "so", "if", "then", "than", "too", "very", "just", "about", "above",
    "after", "before", "between", "into", "through", "during", "each",
    "all", "both", "such", "only", "also", "how", "what", "which",
    "who", "when", "where", "why", "up", "out", "over", "under",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class KeywordEmbedder:
    def __init__(self, dimensions: int = 2048) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dimensions
        for token, count in counts.items():
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[index] += math.log(1.0 + count)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.embedder_timeout)
        self.model = settings.openai_embedding_model
        self.dimensions = settings.openai_embedding_dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIReasoner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        if settings.openai_api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # Models that only accept default temperature (no custom values)
    _DEFAULT_TEMP_ONLY_MODELS = {"gpt-5.4-nano", "gpt-5-mini"}

    def generate_text(self, prompt: str, model: str | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("OpenAI is not configured.")
        target_model = model or self.settings.generation_model
        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if target_model not in self._DEFAULT_TEMP_ONLY_MODELS:
            kwargs["temperature"] = self.settings.openai_temperature
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("OpenAI response did not include text content.")
        return text.strip()


class TavilyClient:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.tavily_api_key
        self.enabled = bool(settings.use_tavily_fallback and self.api_key)
        self.client = httpx.Client(timeout=settings.tavily_timeout)

    # Domains that are acceptable Tavily sources for NVIDIA infra questions
    _ALLOWED_DOMAINS = (
        "nvidia.com",
        "developer.nvidia.com",
        "docs.nvidia.com",
        "ngc.nvidia.com",
        "github.com",
        "arxiv.org",
        "huggingface.co",
        "pytorch.org",
        "kubernetes.io",
        "lwn.net",
        "linuxfoundation.org",
    )

    def _is_relevant_result(self, url: str, content: str, query: str) -> bool:
        """Return True only if the result is from a plausible technical source."""
        domain_ok = any(d in url for d in self._ALLOWED_DOMAINS)
        if domain_ok:
            return True
        # For non-whitelisted domains, require at least 2 query tokens in the content
        query_tokens = set(query.lower().split())
        content_lower = content.lower()
        matches = sum(1 for t in query_tokens if len(t) > 3 and t in content_lower)
        return matches >= 3

    def search(self, query: str) -> list[dict[str, str]]:
        if not self.enabled:
            return []
        response = self.client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 8,
                "include_domains": [
                    "nvidia.com",
                    "developer.nvidia.com",
                    "docs.nvidia.com",
                    "github.com",
                    "kubernetes.io",
                    "pytorch.org",
                ],
            },
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            return []
        raw_results = payload.get("results") or []
        results: list[dict[str, str]] = []
        for item in raw_results:
            url = item.get("url", "")
            content = item.get("content", "")
            if self._is_relevant_result(url, content, query):
                results.append(
                    {
                        "title": item.get("title", "Web result"),
                        "url": url,
                        "content": content,
                    }
                )
        return results


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder_provider == "openai":
        try:
            return OpenAIEmbedder(settings)
        except Exception:
            if settings.is_assessment_mode:
                raise
            return KeywordEmbedder()
    if settings.is_assessment_mode:
        raise ValueError("Assessment mode requires the OpenAI embedder.")
    return KeywordEmbedder()
