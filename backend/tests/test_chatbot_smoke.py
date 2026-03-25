from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app


def _parse_sse(response_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in response_text.strip().split("\n\n"):
        event_name = None
        payload = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.replace("event:", "", 1).strip()
            elif line.startswith("data:"):
                payload = json.loads(line.replace("data:", "", 1).strip())
        if event_name and payload is not None:
            events.append((event_name, payload))
    return events


def test_normal_chat_runs_as_direct_chat():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "hey", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert event_names == ["answer_chunk", "done"]
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "direct_chat"
    assert done_payload["response_mode"] == "direct-chat"
    assert done_payload["citation_count"] == 0


def test_doc_question_routes_to_rag_with_citations():
    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        json={"question": "What NVIDIA stack is needed on Linux for GPU containers?", "history": []},
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert "classification" in event_names
    assert "retrieval" in event_names
    assert "document_grading" in event_names
    assert "citation" in event_names
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "doc_rag"
    assert done_payload["citation_count"] > 0


def test_streaming_emits_trace_then_answer_then_done_for_doc_runs():
    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        json={"question": "Why is 4-GPU training scaling poorly?", "history": []},
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert event_names.index("classification") < event_names.index("generation")
    assert "answer_chunk" in event_names
    assert event_names[-1] == "done"


def test_api_router_follow_up_general_question_stays_out_of_rag():
    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        json={
            "question": "thanks, that makes sense",
            "history": [
                {"role": "user", "content": "hey"},
                {"role": "assistant", "content": "I can help with general questions or NVIDIA documentation."},
            ],
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert "classification" not in event_names
    assert "retrieval" not in event_names
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "direct_chat"
