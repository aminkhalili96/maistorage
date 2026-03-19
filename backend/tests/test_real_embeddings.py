"""Real OpenAI embedding quality verification.

Skipped by default — only runs when OPENAI_API_KEY is set.
Cost: ~$0.01 per run (a handful of embed calls × ~100 tokens each).
"""
from __future__ import annotations

import math
import os

import pytest

SKIP_REASON = "requires OPENAI_API_KEY"
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason=SKIP_REASON
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def embedder():
    """Build an OpenAIEmbedder using real credentials."""
    from app.config import get_settings

    get_settings.cache_clear()
    os.environ["EMBEDDER_PROVIDER"] = "openai"
    settings = get_settings()
    from app.services.providers import OpenAIEmbedder

    emb = OpenAIEmbedder(settings)
    yield emb
    # Restore default
    os.environ["EMBEDDER_PROVIDER"] = "keyword"
    get_settings.cache_clear()


class TestOpenAIEmbeddings:
    def test_returns_expected_dimensions(self, embedder):
        """OpenAI embedder returns vectors of the configured dimension (3072)."""
        vec = embedder.embed_query("NVIDIA H100 GPU memory bandwidth")
        assert isinstance(vec, list)
        assert len(vec) == embedder.dimensions

    def test_similar_queries_high_cosine(self, embedder):
        """Semantically similar queries produce high cosine similarity (>0.80)."""
        q1 = "What is the memory bandwidth of the NVIDIA H100 GPU?"
        q2 = "How much memory bandwidth does the H100 have?"
        v1 = embedder.embed_query(q1)
        v2 = embedder.embed_query(q2)
        sim = _cosine_similarity(v1, v2)
        assert sim > 0.80, f"Similar queries should have cosine > 0.80, got {sim:.3f}"

    def test_unrelated_queries_low_cosine(self, embedder):
        """Unrelated queries produce lower cosine similarity (<0.60)."""
        q1 = "NVIDIA NCCL all-reduce bandwidth optimization"
        q2 = "chocolate cake recipe with frosting"
        v1 = embedder.embed_query(q1)
        v2 = embedder.embed_query(q2)
        sim = _cosine_similarity(v1, v2)
        assert sim < 0.60, f"Unrelated queries should have cosine < 0.60, got {sim:.3f}"

    def test_empty_string_does_not_crash(self, embedder):
        """Embedding an empty string returns a valid vector without crashing."""
        vec = embedder.embed_query("")
        assert isinstance(vec, list)
        assert len(vec) == embedder.dimensions
