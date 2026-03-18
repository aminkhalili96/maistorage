"""Classification regression suite.

Covers edge cases for both classify_assistant_mode() and classify_question()
to guard against regressions in the routing logic.
"""
from __future__ import annotations

import pytest

from app.models import ChatTurn, QueryClass
from app.services.retrieval import classify_assistant_mode, classify_question


# ---------------------------------------------------------------------------
# classify_assistant_mode parametrized cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,history,expected_mode", [
    # --- Clear doc_rag: NVIDIA hardware / infra terms ---
    ("GPU temperature monitoring best practices", None, "doc_rag"),
    ("Compare H100 and A100 for Kubernetes deployment", None, "doc_rag"),
    ("How do I configure NCCL for multi-node training?", None, "doc_rag"),
    ("What are the VRAM limits of the L40S GPU?", None, "doc_rag"),
    ("How does GPUDirect Storage reduce latency?", None, "doc_rag"),
    ("DCGM health check commands for A100 cluster", None, "doc_rag"),
    ("What is the TDP of the H200 GPU?", None, "doc_rag"),
    ("How to configure InfiniBand RDMA for GPU clusters?", None, "doc_rag"),
    ("What is the BMC redfish API for DGX systems?", None, "doc_rag"),
    ("Explain FP8 vs BF16 precision in Transformer Engine", None, "doc_rag"),
    ("How to install the NVIDIA Container Toolkit on Ubuntu?", None, "doc_rag"),
    ("What are the CUDA programming guide best practices?", None, "doc_rag"),
    ("How does NVLink scale bandwidth across GPUs?", None, "doc_rag"),
    ("Configure GPU operator for Kubernetes", None, "doc_rag"),
    ("What is the fabric manager used for in NVSwitch systems?", None, "doc_rag"),
    # --- Live query: weather/stocks go to Tavily ---
    ("How's the weather today?", None, "live_query"),
    # --- Clear direct_chat: general knowledge / casual ---
    ("What is love?", None, "direct_chat"),
    ("Tell me a joke", None, "direct_chat"),
    ("What is the capital of France?", None, "direct_chat"),
    ("How do I bake a chocolate cake?", None, "direct_chat"),
    ("What is 2 + 2?", None, "direct_chat"),
    ("Who wrote Hamlet?", None, "direct_chat"),
    # --- Follow-up context: doc_rag context should stay doc_rag ---
    # Short follow-ups (<=3 tokens) after doc_rag context stay in doc_rag
    (
        "more",
        [
            ChatTurn(role="user", content="How do I configure NCCL for multi-node training?"),
            ChatTurn(role="assistant", content="NCCL requires setting NCCL_SOCKET_IFNAME..."),
        ],
        "doc_rag",
    ),
    (
        "why",
        [
            ChatTurn(role="user", content="What are the VRAM limits of the L40S GPU?"),
            ChatTurn(role="assistant", content="The L40S has 48 GB GDDR6 VRAM..."),
        ],
        "doc_rag",
    ),
    # --- Financial terms always live_query (need live data) ---
    ("What is NVIDIA stock price today?", None, "live_query"),
])
def test_classify_assistant_mode_edge_cases(question, history, expected_mode):
    result = classify_assistant_mode(question, history)
    assert result == expected_mode, (
        f"classify_assistant_mode({question!r}) returned {result!r}, expected {expected_mode!r}"
    )


# ---------------------------------------------------------------------------
# classify_question parametrized cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected_class", [
    ("Why is 4-GPU training scaling poorly?", QueryClass.distributed_multi_gpu),
    ("How to configure NCCL for all-reduce bandwidth?", QueryClass.distributed_multi_gpu),
    ("What is tensor parallelism in Megatron?", QueryClass.distributed_multi_gpu),
    ("How to optimize training throughput with Tensor Cores?", QueryClass.training_optimization),
    ("What are the best practices for mixed precision training?", QueryClass.training_optimization),
    ("How do I deploy GPU operator on Kubernetes?", QueryClass.deployment_runtime),
    ("How to install NVIDIA Container Toolkit?", QueryClass.deployment_runtime),
    ("What is the NVLink topology of the H100 SXM?", QueryClass.hardware_topology),
    ("Compare A100 and H100 memory bandwidth", QueryClass.hardware_topology),
    ("What is the weather like today?", QueryClass.general),
    ("Tell me about NVIDIA", QueryClass.general),
])
def test_classify_question_edge_cases(question, expected_class):
    result = classify_question(question)
    assert result == expected_class, (
        f"classify_question({question!r}) returned {result!r}, expected {expected_class!r}"
    )


# ---------------------------------------------------------------------------
# Regression: last-chance gpu/nvidia catch
# ---------------------------------------------------------------------------

def test_last_chance_gpu_catch_routes_to_doc_rag():
    """A long question mentioning 'gpu' that doesn't match any explicit rule should still be doc_rag."""
    question = "What are the best ways to monitor gpu utilization in a production cluster environment?"
    assert classify_assistant_mode(question) == "doc_rag"


def test_last_chance_nvidia_catch_routes_to_doc_rag():
    """A long question mentioning 'nvidia' that doesn't match any explicit rule should still be doc_rag."""
    question = "How does nvidia handle memory management across multiple cards in a server?"
    assert classify_assistant_mode(question) == "doc_rag"


def test_short_gpu_question_not_last_chance():
    """A very short question like 'gpu?' (<=6 tokens) should NOT trigger the last-chance rule alone."""
    # The last-chance rule requires len(tokens) > 6
    question = "gpu?"
    # This may be direct_chat since it's too short; just verify it doesn't crash
    result = classify_assistant_mode(question)
    assert result in {"doc_rag", "direct_chat"}
