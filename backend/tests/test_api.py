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


def test_ingestion_status_exposes_snapshot_and_refresh_fields():
    client = TestClient(app)

    response = client.get("/api/ingest/status")

    assert response.status_code == 200
    payload = response.json()
    assert "snapshot_id" in payload
    assert "last_refresh_at" in payload
    assert "changed_sources" in payload
    assert "chunk_counts" in payload


def test_chat_stream_emits_expected_event_order_and_citations():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "Why is 4-GPU training scaling poorly?", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert event_names.index("classification") < event_names.index("retrieval")
    assert event_names.index("retrieval") < event_names.index("document_grading")
    assert event_names.index("document_grading") < event_names.index("generation")
    assert event_names[-1] == "done"
    assert any(name == "citation" for name, _ in events)
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "doc_rag"
    assert "grounding_passed" in done_payload
    assert "answer_quality_passed" in done_payload
    assert "rejected_chunk_count" in done_payload
    assert "query_class" in done_payload
    assert "source_families" in done_payload
    assert "model_used" in done_payload


def test_chat_stream_rejects_unsupported_model():
    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        json={"question": "Why is 4-GPU training scaling poorly?", "history": [], "model": "unsupported-model"},
    )

    assert response.status_code == 400


def test_end_to_end_known_question_contains_expected_keywords():
    client = TestClient(app)
    response = client.post(
        "/api/chat/stream",
        json={"question": "What NVIDIA stack is needed to deploy training workloads on Linux?", "history": []},
    )

    events = _parse_sse(response.text)
    done_payload = next(payload for name, payload in events if name == "done")
    answer = str(done_payload["answer"]).lower()

    assert "driver" in answer or "container" in answer or "cuda" in answer
    assert done_payload["assistant_mode"] == "doc_rag"
    assert done_payload["response_mode"] in {"knowledge-base-backed", "web-backed", "insufficient-evidence"}
    assert isinstance(done_payload["citation_count"], int)


def test_direct_chat_stream_skips_doc_trace_and_sources():
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "How should I prepare for a technical interview?", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert "classification" not in event_names
    assert "retrieval" not in event_names
    assert "citation" not in event_names
    done_payload = next(payload for name, payload in events if name == "done")
    assert done_payload["assistant_mode"] == "direct_chat"
    assert done_payload["response_mode"] == "direct-chat"
    assert done_payload["citation_count"] == 0


def test_chat_stream_rejects_missing_question():
    """POST with no question field should return 422 Unprocessable Entity."""
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={})

    assert response.status_code == 422


def test_chat_stream_handles_empty_question():
    """POST with empty string question should return a response (not crash)."""
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "", "history": []})

    # FastAPI validates non-empty via Pydantic; either 422 or a valid SSE response is acceptable
    assert response.status_code in {200, 422}


def test_search_debug_returns_results():
    """POST to /api/search/debug should return 200 with plan and results."""
    client = TestClient(app)
    response = client.post("/api/search/debug", json={"question": "How to configure NCCL for multi-GPU training?"})

    assert response.status_code == 200
    payload = response.json()
    assert "plan" in payload
    assert "results" in payload
    assert "confidence" in payload
    assert isinstance(payload["results"], list)


def test_search_debug_rejects_missing_question():
    """POST to /api/search/debug with no question should return 422."""
    client = TestClient(app)
    response = client.post("/api/search/debug", json={})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# #8: Request validation tests
# ---------------------------------------------------------------------------


def test_chat_stream_rejects_question_exceeding_max_length():
    """Question > 2000 chars should return 400."""
    client = TestClient(app)
    long_question = "x" * 2001
    response = client.post("/api/chat/stream", json={"question": long_question, "history": []})
    assert response.status_code == 400
    assert "2000" in response.json()["detail"]


def test_chat_stream_rejects_too_many_history_turns():
    """History > 20 turns should return 400."""
    client = TestClient(app)
    history = [{"role": "user", "content": f"turn {i}"} for i in range(21)]
    response = client.post("/api/chat/stream", json={"question": "test", "history": history})
    assert response.status_code == 400
    assert "20" in response.json()["detail"]


def test_chat_stream_rejects_oversized_history_message():
    """A history message > 5000 chars should return 400."""
    client = TestClient(app)
    history = [{"role": "user", "content": "a" * 5001}]
    response = client.post("/api/chat/stream", json={"question": "test", "history": history})
    assert response.status_code == 400
    assert "5000" in response.json()["detail"]


def test_chat_stream_accepts_valid_request_at_limits():
    """Request at exactly the limits should succeed."""
    client = TestClient(app)
    # 2000-char question, 20 turns, 5000-char messages
    response = client.post(
        "/api/chat/stream",
        json={
            "question": "x" * 2000,
            "history": [{"role": "user", "content": "a" * 5000}] * 20,
        },
    )
    # Should not return 400 — either 200 (success) or possibly other status, but not validation error
    assert response.status_code != 400


def test_done_event_includes_generation_degraded_field():
    """The done SSE event should include the generation_degraded field."""
    client = TestClient(app)
    response = client.post("/api/chat/stream", json={"question": "Why is 4-GPU training scaling poorly?", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    done_payload = next(payload for name, payload in events if name == "done")
    assert "generation_degraded" in done_payload
    assert isinstance(done_payload["generation_degraded"], bool)
