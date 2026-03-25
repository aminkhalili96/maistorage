"""Tests verifying that the original rule-based functions still work correctly.

These are the fallback paths used when LLM agentic features fail or are disabled.
"""

from __future__ import annotations

import os

import pytest

os.environ["APP_MODE"] = "dev"
os.environ["USE_PINECONE"] = "false"
os.environ["USE_TAVILY_FALLBACK"] = "false"
os.environ["EMBEDDER_PROVIDER"] = "keyword"
os.environ["LANGSMITH_TRACING"] = "false"

from app.models import QueryClass, QueryPlan
from app.services.retrieval import (
    classify_assistant_mode,
    classify_question,
    build_query_plan,
    _adaptive_retrieval_params,
)


class TestRuleBasedFallbacks:
    def test_classify_assistant_mode_doc_rag(self):
        assert classify_assistant_mode("How do I configure NCCL?") == "doc_rag"

    def test_classify_assistant_mode_direct_chat(self):
        assert classify_assistant_mode("Hello") == "direct_chat"

    def test_classify_question_distributed(self):
        assert classify_question("NCCL multi-GPU scaling") == QueryClass.distributed_multi_gpu

    def test_build_query_plan_valid(self, settings):
        plan = build_query_plan("H100 specs", settings)
        assert isinstance(plan, QueryPlan)
        assert plan.query_class == QueryClass.hardware_topology

    def test_adaptive_params_simple(self):
        top_k, floor = _adaptive_retrieval_params("H100 memory", QueryClass.hardware_topology)
        assert 3 <= top_k <= 10
        assert 0.2 <= floor <= 0.4

    def test_adaptive_params_complex(self):
        top_k, floor = _adaptive_retrieval_params(
            "Compare NCCL ring algorithm performance across H100 and A100 in multi-node training with pipeline parallelism",
            QueryClass.distributed_multi_gpu,
        )
        assert top_k == 10
        assert floor == 0.22
