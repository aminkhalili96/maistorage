"""
Contract tests: validate that frontend TypeScript types stay aligned with
backend Pydantic models.

No LLM API calls, no external services — pure structural checks.

Run:
    PYTHONPATH=backend backend/.venv/bin/pytest backend/tests/test_contract.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.models import (
    AgentRunState,
    Citation,
    DocumentSource,
    IngestionStatus,
    TraceEvent,
    TraceEventType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRONTEND_TYPES_PATH = Path(__file__).resolve().parents[2] / "frontend" / "src" / "types.ts"
FRONTEND_API_PATH = Path(__file__).resolve().parents[2] / "frontend" / "src" / "api.ts"
BACKEND_AGENT_PATH = Path(__file__).resolve().parents[2] / "backend" / "app" / "services" / "agent.py"


def _read_file(path: Path) -> str:
    assert path.exists(), f"File not found: {path}"
    return path.read_text(encoding="utf-8")


def extract_ts_interface_fields(ts_content: str, interface_name: str) -> set[str]:
    """Extract field names from a TypeScript interface definition."""
    pattern = rf"export interface {interface_name}\s*\{{([^}}]+)\}}"
    match = re.search(pattern, ts_content, re.DOTALL)
    if not match:
        return set()
    body = match.group(1)
    fields: set[str] = set()
    for line in body.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("/*"):
            continue
        field_match = re.match(r"(\w+)\??:", line)
        if field_match:
            fields.add(field_match.group(1))
    return fields


def _pydantic_field_names(model_cls: type) -> set[str]:
    """Return the set of field names declared on a Pydantic model."""
    return set(model_cls.model_fields.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFrontendBackendContract:
    """Ensure frontend TS types are structurally compatible with backend models."""

    @pytest.fixture(autouse=True)
    def _load_ts(self) -> None:
        self.ts_content = _read_file(FRONTEND_TYPES_PATH)

    # -- 1. Citation ----------------------------------------------------------

    def test_citation_ts_fields_subset_of_backend(self) -> None:
        """Every field declared in the TS Citation interface must exist in the
        backend Citation Pydantic model."""
        ts_fields = extract_ts_interface_fields(self.ts_content, "Citation")
        backend_fields = _pydantic_field_names(Citation)

        assert ts_fields, "Failed to parse TS Citation interface"
        missing = ts_fields - backend_fields
        assert not missing, (
            f"Frontend Citation has fields not in backend model: {sorted(missing)}"
        )

    # -- 2. TraceEvent --------------------------------------------------------

    def test_trace_event_ts_fields_match_backend(self) -> None:
        """TS TraceEvent fields must match the backend TraceEvent model."""
        ts_fields = extract_ts_interface_fields(self.ts_content, "TraceEvent")
        backend_fields = _pydantic_field_names(TraceEvent)

        assert ts_fields, "Failed to parse TS TraceEvent interface"
        assert ts_fields == backend_fields, (
            f"TraceEvent mismatch — "
            f"only in TS: {sorted(ts_fields - backend_fields)}, "
            f"only in backend: {sorted(backend_fields - ts_fields)}"
        )

    # -- 3. SourceRecord vs DocumentSource -----------------------------------

    def test_source_record_ts_fields_subset_of_backend(self) -> None:
        """Frontend SourceRecord is a projection of backend DocumentSource.
        Every TS field must exist in the backend model (backend may have more)."""
        ts_fields = extract_ts_interface_fields(self.ts_content, "SourceRecord")
        backend_fields = _pydantic_field_names(DocumentSource)

        assert ts_fields, "Failed to parse TS SourceRecord interface"
        missing = ts_fields - backend_fields
        assert not missing, (
            f"Frontend SourceRecord has fields not in backend DocumentSource: "
            f"{sorted(missing)}"
        )

    # -- 4. IngestionStatus ---------------------------------------------------

    def test_ingestion_status_ts_fields_match_backend(self) -> None:
        """TS IngestionStatus fields must match the backend IngestionStatus model."""
        ts_fields = extract_ts_interface_fields(self.ts_content, "IngestionStatus")
        backend_fields = _pydantic_field_names(IngestionStatus)

        assert ts_fields, "Failed to parse TS IngestionStatus interface"
        assert ts_fields == backend_fields, (
            f"IngestionStatus mismatch — "
            f"only in TS: {sorted(ts_fields - backend_fields)}, "
            f"only in backend: {sorted(backend_fields - ts_fields)}"
        )

    # -- 5. SSE event types ---------------------------------------------------

    def test_all_backend_sse_events_handled_by_frontend(self) -> None:
        """Every SSE event type the backend can emit must be handled (or passed
        through to onTrace) by the frontend SSE parser in api.ts."""
        # Backend event types: everything in TraceEventType enum + "answer_chunk"
        backend_events = {e.value for e in TraceEventType}
        backend_events.add("answer_chunk")

        # Frontend explicitly handles: citation, answer_chunk, done, error
        # Everything else falls through to onTrace — which is fine, it accepts
        # any event name. We verify the frontend code at least references the
        # four special-cased event names so the routing logic is intact.
        api_content = _read_file(FRONTEND_API_PATH)

        explicitly_handled = {"citation", "answer_chunk", "done", "error"}
        for event_name in explicitly_handled:
            assert f'"{event_name}"' in api_content, (
                f"Frontend api.ts does not explicitly handle SSE event '{event_name}'"
            )

        # The else branch in api.ts sends unrecognized events to onTrace,
        # so all backend events are handled. Verify no backend event is
        # silently dropped by checking there's an else/default branch.
        assert "else" in api_content, (
            "Frontend api.ts has no else/default branch for unknown SSE events"
        )

        # Additionally, verify the backend doesn't emit event types that are
        # completely unknown (not in TraceEventType and not the known extras).
        known_extras = {"answer_chunk"}
        enum_values = {e.value for e in TraceEventType}
        all_known = enum_values | known_extras
        # Scan agent.py for _format_sse("...", calls to find all emitted events
        agent_content = _read_file(BACKEND_AGENT_PATH)
        emitted = set(re.findall(r'_format_sse\(\s*["\'](\w+)["\']', agent_content))
        unknown = emitted - all_known
        assert not unknown, (
            f"Backend agent.py emits SSE events not in TraceEventType or known extras: "
            f"{sorted(unknown)}"
        )

    # -- 6. ChatDonePayload ---------------------------------------------------

    def test_chat_done_payload_matches_backend_emission(self) -> None:
        """ChatDonePayload TS fields must match the keys emitted in the backend
        'done' SSE event (constructed in agent.py stream method)."""
        ts_fields = extract_ts_interface_fields(self.ts_content, "ChatDonePayload")
        assert ts_fields, "Failed to parse TS ChatDonePayload interface"

        # Extract the actual keys emitted in the done event from agent.py.
        # The done payload is built as a dict literal passed to _format_sse("done", {...}).
        agent_content = _read_file(BACKEND_AGENT_PATH)

        # Find the primary (non-error) done payload block.
        # Pattern: _format_sse(\n  "done",\n  {\n    "key": ..., ...
        # We look for the block starting after 'state.answer,' which is the
        # success path (not the error fallback).
        done_pattern = re.compile(
            r'_format_sse\(\s*\n\s*"done",\s*\n\s*\{(.*?)\}\s*,?\s*\)',
            re.DOTALL,
        )
        done_matches = done_pattern.findall(agent_content)
        assert done_matches, "Could not find _format_sse('done', {...}) in agent.py"

        # Parse keys from the first (success-path) done payload dict.
        # The success path references state.answer (the error path has a string literal).
        success_block = None
        for block in done_matches:
            if "state.answer" in block:
                success_block = block
                break
        assert success_block is not None, (
            "Could not identify the success-path done payload in agent.py"
        )

        backend_done_keys: set[str] = set()
        for line in success_block.splitlines():
            key_match = re.match(r'\s*"(\w+)"\s*:', line)
            if key_match:
                backend_done_keys.add(key_match.group(1))

        only_in_ts = ts_fields - backend_done_keys
        only_in_backend = backend_done_keys - ts_fields
        assert not only_in_ts, (
            f"ChatDonePayload has fields not emitted by backend done event: "
            f"{sorted(only_in_ts)}"
        )
        assert not only_in_backend, (
            f"Backend done event emits keys not in ChatDonePayload TS type: "
            f"{sorted(only_in_backend)}"
        )
