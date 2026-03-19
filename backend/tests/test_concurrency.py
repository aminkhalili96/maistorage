"""Thread-safety tests for key components.

Validates:
- Double-checked locking singleton in get_services()
- InMemoryHybridIndex.search() under concurrent queries
- _thread_local_emit isolation across threads
- Concurrent agent.run() calls
- reset_services() clearing the singleton
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.models import ChatRequest
from app.runtime import get_services, reset_services, _build_services
from app.services.agent import _thread_local_emit
from app.services.indexes import InMemoryHybridIndex
from app.services.providers import KeywordEmbedder


class TestGetServicesSingleton:
    """get_services() returns the same instance from concurrent threads."""

    def setup_method(self):
        get_settings.cache_clear()
        reset_services()

    def teardown_method(self):
        reset_services()

    def test_same_instance_from_concurrent_threads(self):
        """10 threads all calling get_services() must receive the identical object."""
        barrier = threading.Barrier(10)
        results: list = [None] * 10

        def _call(index: int) -> None:
            barrier.wait()
            results[index] = get_services()

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = results[0]
        assert first is not None, "get_services() returned None"
        for i, svc in enumerate(results):
            assert svc is first, f"Thread {i} got a different instance (id={id(svc)} vs {id(first)})"

    def test_build_services_called_once_under_contention(self):
        """Even with 10 concurrent first-callers, _build_services runs exactly once."""
        call_count = 0
        original_build = _build_services

        def _counting_build():
            nonlocal call_count
            call_count += 1
            return original_build()

        barrier = threading.Barrier(10)

        def _call():
            barrier.wait()
            return get_services()

        with patch("app.runtime._build_services", side_effect=_counting_build):
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(_call) for _ in range(10)]
                for f in as_completed(futures):
                    f.result()  # propagate any exception

        assert call_count == 1, f"_build_services called {call_count} times, expected 1"


class TestResetServices:
    """reset_services() properly clears the singleton for re-initialization."""

    def setup_method(self):
        get_settings.cache_clear()
        reset_services()

    def teardown_method(self):
        reset_services()

    def test_reset_allows_new_instance(self):
        first = get_services()
        reset_services()
        second = get_services()
        # Both should be valid AppServices but different objects since we reset
        assert first is not second, "After reset, get_services() should return a new instance"

    def test_reset_under_concurrent_access(self):
        """Calling reset while other threads call get_services should not raise."""
        errors: list[Exception] = []

        def _getter():
            try:
                for _ in range(20):
                    get_services()
            except Exception as exc:
                errors.append(exc)

        def _resetter():
            try:
                for _ in range(5):
                    reset_services()
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=_getter) for _ in range(5)]
            + [threading.Thread(target=_resetter) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent get/reset raised: {errors}"


class TestIndexConcurrentSearch:
    """InMemoryHybridIndex.search() is safe under concurrent queries."""

    def test_concurrent_searches_no_exceptions(self, demo_index):
        """10 threads searching the same index concurrently should not crash."""
        queries = [
            "NCCL multi-GPU training",
            "NVLink bandwidth scaling",
            "DGX BasePOD architecture",
            "GPU operator Kubernetes",
            "container toolkit installation",
            "mixed precision FP16 training",
            "GPUDirect storage RDMA",
            "H100 memory capacity",
            "fabric manager NVSwitch",
            "inference optimization TensorRT",
        ]
        barrier = threading.Barrier(10)
        results: list = [None] * 10
        errors: list = [None] * 10

        def _search(index: int) -> None:
            barrier.wait()
            try:
                results[index] = demo_index.search(queries[index], top_k=5)
            except Exception as exc:
                errors[index] = exc

        threads = [threading.Thread(target=_search, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, err in enumerate(errors):
            assert err is None, f"Thread {i} raised: {err}"
        for i, res in enumerate(results):
            assert res is not None, f"Thread {i} got no results"
            assert isinstance(res, list), f"Thread {i} result is not a list"

    def test_concurrent_search_results_are_correct(self, demo_index):
        """Each thread gets properly sorted results with valid scores."""
        barrier = threading.Barrier(5)
        results: list = [None] * 5

        def _search(index: int, query: str) -> None:
            barrier.wait()
            results[index] = demo_index.search(query, top_k=3)

        queries = [
            "NCCL tuning bandwidth",
            "GPU memory",
            "Kubernetes deployment",
            "NVLink topology",
            "container runtime",
        ]
        threads = [threading.Thread(target=_search, args=(i, q)) for i, q in enumerate(queries)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, res in enumerate(results):
            assert res is not None
            assert len(res) <= 3
            # Scores should be monotonically non-increasing
            scores = [r.score for r in res]
            for j in range(len(scores) - 1):
                assert scores[j] >= scores[j + 1], (
                    f"Thread {i}: scores not sorted: {scores}"
                )


class TestThreadLocalEmit:
    """_thread_local_emit is isolated per thread."""

    def test_emit_fn_isolated_across_threads(self):
        """Setting emit fn in one thread must not be visible in another."""
        seen_in_other = []
        barrier = threading.Barrier(2)

        def _setter():
            _thread_local_emit.fn = lambda t, p: None
            barrier.wait()
            # Hold the value while the reader checks
            barrier.wait()
            _thread_local_emit.fn = None

        def _reader():
            barrier.wait()
            seen_in_other.append(getattr(_thread_local_emit, "fn", None))
            barrier.wait()

        t1 = threading.Thread(target=_setter)
        t2 = threading.Thread(target=_reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert seen_in_other[0] is None, (
            "_thread_local_emit.fn leaked across threads"
        )

    def test_emit_fn_available_in_same_thread(self):
        """Setting emit fn should be readable within the same thread."""
        sentinel = lambda t, p: "test"  # noqa: E731
        _thread_local_emit.fn = sentinel
        try:
            assert getattr(_thread_local_emit, "fn", None) is sentinel
        finally:
            _thread_local_emit.fn = None

    def test_multiple_threads_have_independent_emit_fns(self):
        """Each thread's emit fn is independent — no cross-contamination."""
        results: dict[int, object] = {}
        barrier = threading.Barrier(5)

        def _set_and_read(index: int):
            sentinel = object()
            _thread_local_emit.fn = sentinel
            barrier.wait()
            # After all threads set their own sentinel, read back
            results[index] = getattr(_thread_local_emit, "fn", None)
            _thread_local_emit.fn = None

        threads = [threading.Thread(target=_set_and_read, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have gotten back its own unique sentinel object
        values = list(results.values())
        assert len(values) == 5
        assert all(v is not None for v in values)
        # All values should be distinct objects
        assert len(set(id(v) for v in values)) == 5, (
            "Thread-local values are not independent"
        )


class TestConcurrentAgentRun:
    """Concurrent agent.run() calls on different questions don't crash."""

    def test_concurrent_runs_no_crash(self, agent_service):
        """Multiple threads calling agent.run() with different questions should not raise."""
        questions = [
            "What is NCCL?",
            "How does NVLink work?",
            "What is DGX BasePOD?",
            "How to install GPU Operator?",
            "What is GPUDirect Storage?",
        ]
        barrier = threading.Barrier(len(questions))
        results: list = [None] * len(questions)
        errors: list = [None] * len(questions)

        def _run(index: int) -> None:
            barrier.wait()
            try:
                request = ChatRequest(question=questions[index])
                results[index] = agent_service.run(request)
            except Exception as exc:
                errors[index] = exc

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(len(questions))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, err in enumerate(errors):
            assert err is None, f"Thread {i} raised: {err}"
        for i, res in enumerate(results):
            assert res is not None, f"Thread {i} returned None"
            assert res.question == questions[i], (
                f"Thread {i}: question mismatch, got {res.question!r}"
            )
            assert res.answer, f"Thread {i} produced empty answer"
