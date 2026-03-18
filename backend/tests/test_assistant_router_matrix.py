from __future__ import annotations

import pytest

from app.models import ChatTurn
from app.services.retrieval import classify_assistant_mode


DIRECT_CHAT_CASES = [
    ("hello", []),
    ("what is nvidia", []),
    ("who founded nvidia", []),
    ("where is nvidia headquartered", []),
    ("what does nvidia do", []),
    ("tell me about nvidia", []),
    ("how big is nvidia", []),
    ("tell me about H100", []),
    ("what is an H100", []),
    ("what does cuda stand for", []),
    ("what docs do you have", []),
    ("tell me the overview of all docs you have for rag", []),
    ("can you check the docs for me?", []),
    ("who is the ceo of nvidia?", []),
]

LIVE_QUERY_CASES = [
    ("what is the stock price of nvidia today?", []),
    ("How's the weather today?", []),
]

DOC_RAG_CASES = [
    ("What NVIDIA stack is needed on Linux for GPU containers?", []),
    ("How do I install CUDA on Ubuntu?", []),
    ("Why is 4-GPU training scaling poorly?", []),
    ("According to official NVIDIA docs, when should I use mixed precision?", []),
    ("What changed in the latest NVIDIA Container Toolkit release?", []),
    ("Show me the NVIDIA guide for GPU Operator installation", []),
    ("Show me the guide for TensorRT", []),
    ("Which official docs explain RAID choices for AI servers?", []),
    ("Find the official docs for Slurm or Kubernetes cluster management", []),
    ("Show me official docs for server motherboard and PCIe planning", []),
]


@pytest.mark.parametrize(("question", "history"), DIRECT_CHAT_CASES)
def test_router_matrix_direct_chat_cases(question: str, history: list[ChatTurn]) -> None:
    assert classify_assistant_mode(question, history) == "direct_chat"


@pytest.mark.parametrize(("question", "history"), LIVE_QUERY_CASES)
def test_router_matrix_live_query_cases(question: str, history: list[ChatTurn]) -> None:
    assert classify_assistant_mode(question, history) == "live_query"


@pytest.mark.parametrize(("question", "history"), DOC_RAG_CASES)
def test_router_matrix_doc_rag_cases(question: str, history: list[ChatTurn]) -> None:
    assert classify_assistant_mode(question, history) == "doc_rag"


def test_router_matrix_short_follow_up_inherits_doc_rag_context() -> None:
    history = [
        ChatTurn(role="user", content="Show me the official NVIDIA docs for CUDA installation"),
        ChatTurn(role="assistant", content="I found the main installation guides."),
    ]

    assert classify_assistant_mode("where?", history) == "doc_rag"
    assert classify_assistant_mode("which guide?", history) == "doc_rag"
    assert classify_assistant_mode("CUDA", history) == "doc_rag"


def test_router_matrix_assistant_doc_wording_does_not_flip_general_follow_up() -> None:
    history = [
        ChatTurn(role="user", content="hello"),
        ChatTurn(role="assistant", content="I can help with general questions or official NVIDIA docs."),
    ]

    assert classify_assistant_mode("what day is it", history) == "direct_chat"


def test_router_matrix_short_follow_up_after_general_chat_stays_direct_chat() -> None:
    history = [
        ChatTurn(role="user", content="What is NVIDIA?"),
        ChatTurn(role="assistant", content="NVIDIA is a technology company focused on GPUs and AI systems."),
    ]

    assert classify_assistant_mode("where?", history) == "direct_chat"
    assert classify_assistant_mode("who founded it?", history) == "direct_chat"


def test_router_matrix_generic_docs_wording_without_context_stays_direct_chat() -> None:
    assert classify_assistant_mode("can you check the documentation?", []) == "direct_chat"
    assert classify_assistant_mode("look in the manual for me", []) == "direct_chat"
