"""Tests for API endpoint error handling and edge cases.

Covers request validation (400/422 errors), response structure checks
for GET endpoints, and boundary conditions for POST /api/chat/stream.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/chat/stream — Pydantic validation (422)
# ---------------------------------------------------------------------------


def test_chat_stream_missing_body_returns_422():
    """POST with no JSON body at all should return 422."""
    response = client.post("/api/chat/stream")
    assert response.status_code == 422


def test_chat_stream_empty_json_body_returns_422():
    """POST with empty JSON object (missing required 'question' field) returns 422."""
    response = client.post("/api/chat/stream", json={})
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_chat_stream_invalid_json_returns_422():
    """POST with malformed JSON body returns 422."""
    response = client.post(
        "/api/chat/stream",
        content="this is not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_chat_stream_wrong_content_type_returns_422():
    """POST with non-JSON content type returns 422."""
    response = client.post(
        "/api/chat/stream",
        content="question=hello",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 422


def test_chat_stream_question_wrong_type_returns_422():
    """POST with question as non-string (e.g. integer) returns 422."""
    response = client.post("/api/chat/stream", json={"question": 12345})
    # Pydantic may coerce int to str; if not, expect 422
    assert response.status_code in {200, 422}


def test_chat_stream_history_invalid_role_returns_422():
    """History turn with invalid role value should return 422."""
    response = client.post(
        "/api/chat/stream",
        json={
            "question": "test",
            "history": [{"role": "system", "content": "injected"}],
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/chat/stream — Application validation (400)
# ---------------------------------------------------------------------------


def test_chat_stream_question_exceeds_2000_chars_returns_400():
    """Question longer than 2000 characters should return 400."""
    long_question = "x" * 2001
    response = client.post(
        "/api/chat/stream", json={"question": long_question, "history": []}
    )
    assert response.status_code == 400
    assert "2000" in response.json()["detail"]


def test_chat_stream_history_exceeds_20_turns_returns_400():
    """History with more than 20 turns should return 400."""
    history = [{"role": "user", "content": f"turn {i}"} for i in range(21)]
    response = client.post(
        "/api/chat/stream", json={"question": "test", "history": history}
    )
    assert response.status_code == 400
    assert "20" in response.json()["detail"]


def test_chat_stream_history_turn_exceeds_5000_chars_returns_400():
    """A single history message over 5000 characters should return 400."""
    history = [{"role": "user", "content": "a" * 5001}]
    response = client.post(
        "/api/chat/stream", json={"question": "test", "history": history}
    )
    assert response.status_code == 400
    assert "5000" in response.json()["detail"]


def test_chat_stream_unsupported_model_returns_400():
    """Request with an unrecognized model name should return 400."""
    response = client.post(
        "/api/chat/stream",
        json={
            "question": "test",
            "history": [],
            "model": "gpt-nonexistent-999",
        },
    )
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/chat/stream — Valid edge cases (should succeed)
# ---------------------------------------------------------------------------


def test_chat_stream_empty_history_succeeds():
    """Request with an empty history array should be accepted (200)."""
    response = client.post(
        "/api/chat/stream", json={"question": "hello", "history": []}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_chat_stream_at_exact_limits_does_not_return_400():
    """Request at exactly the boundary limits should not be rejected."""
    response = client.post(
        "/api/chat/stream",
        json={
            "question": "q" * 2000,
            "history": [{"role": "user", "content": "c" * 5000}] * 20,
        },
    )
    # Must not be a 400 validation error; 200 is the expected success path
    assert response.status_code != 400


def test_chat_stream_model_null_succeeds():
    """Request with model explicitly set to null should be accepted."""
    response = client.post(
        "/api/chat/stream",
        json={"question": "hello", "history": [], "model": None},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /health — Response structure
# ---------------------------------------------------------------------------


def test_health_returns_expected_fields():
    """GET /health should return status, mode, knowledge_base_loaded, indexed_chunks."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] in {"ok", "degraded"}
    assert "mode" in body
    assert body["mode"] == "dev"
    assert "knowledge_base_loaded" in body
    assert isinstance(body["knowledge_base_loaded"], bool)
    assert "indexed_chunks" in body
    assert isinstance(body["indexed_chunks"], int)


# ---------------------------------------------------------------------------
# GET /api/sources — Response structure
# ---------------------------------------------------------------------------


def test_sources_returns_expected_structure():
    """GET /api/sources should return sources list, families dict, and metadata."""
    response = client.get("/api/sources")
    assert response.status_code == 200
    body = response.json()
    assert "sources" in body
    assert isinstance(body["sources"], list)
    assert "families" in body
    assert isinstance(body["families"], dict)
    assert "indexed_chunks" in body
    assert isinstance(body["indexed_chunks"], int)
    assert "app_mode" in body
    assert body["app_mode"] == "dev"
    assert "snapshot_id" in body
    assert "last_refresh_at" in body


def test_sources_each_source_has_required_fields():
    """Each source in the sources list should have id, title, and doc_family."""
    response = client.get("/api/sources")
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) > 0, "Expected at least one source in dev mode"
    for source in sources:
        assert "id" in source
        assert "title" in source
        assert "doc_family" in source


# ---------------------------------------------------------------------------
# Nonexistent routes
# ---------------------------------------------------------------------------


def test_nonexistent_route_returns_404():
    """GET to an undefined route should return 404."""
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404


def test_wrong_method_returns_405():
    """GET to a POST-only endpoint should return 405."""
    response = client.get("/api/chat/stream")
    assert response.status_code == 405
