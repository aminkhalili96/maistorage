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


def test_simple_casual_answer_stays_in_direct_chat():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "hello there", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["answer_chunk", "done"]
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "direct_chat"
    assert done_payload["response_mode"] == "direct-chat"


def test_simple_general_nvidia_question_stays_in_direct_chat():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "what is nvidia", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["answer_chunk", "done"]
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "direct_chat"
    assert done_payload["response_mode"] == "direct-chat"
    assert "technology company" in done_payload["answer"].lower()


def test_simple_docs_inventory_question_stays_in_direct_chat_and_lists_sources():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "what docs do you have", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["answer_chunk", "done"]
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "direct_chat"
    assert done_payload["response_mode"] == "direct-chat"
    assert "cuda" in done_payload["answer"].lower()
    assert "container toolkit" in done_payload["answer"].lower()


def test_simple_rag_answer_routes_to_nvidia_docs():
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
    assert "citation" in event_names
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "doc_rag"
    assert done_payload["citation_count"] > 0


def test_simple_doc_streaming_emits_trace_before_answer():
    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        json={"question": "Why is 4-GPU training scaling poorly?", "history": []},
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names.index("classification") < names.index("retrieval")
    assert names.index("retrieval") < names.index("answer_chunk")
    assert names[-1] == "done"
