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
    assert "grounding_passed" in done_payload
    assert "answer_quality_passed" in done_payload
    assert "rejected_chunk_count" in done_payload
    assert "query_class" in done_payload
    assert "source_families" in done_payload


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
    assert done_payload["response_mode"] in {"corpus-backed", "web-backed", "insufficient-evidence"}
    assert isinstance(done_payload["citation_count"], int)
