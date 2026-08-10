import asyncio
from types import SimpleNamespace

import pytest

import uni_api.runtime as runtime
from uni_api.runtime import RuntimeGauges
from uni_api.streaming.bounded_queue import ByteBoundedQueue


def test_runtime_snapshot_uses_authoritative_admission_and_cached_network_state(monkeypatch):
    gauges = RuntimeGauges()
    monkeypatch.setattr(
        runtime,
        "cpu_phase_snapshot",
        lambda: {"capacity": 2, "active": 1, "waiters": 3},
    )
    gauges.inflight_requests = 99
    gauges.attach_request_admission(
        lambda: {
            "mode": "weighted_resources",
            "active": 7,
            "waiters": 11,
            "capacity": None,
            "waiter_limit": None,
            "control_memory_bytes_per_request": 65536,
            "control_memory_reserved_bytes": 458752,
            "reserved_body_bytes": 1234,
            "reserved_response_bytes": 4321,
            "reserved_retained_bytes": 5555,
            "deferred_memory_requests": 2,
            "deferred_memory_bytes": 3333,
            "body_budget": 5678,
            "large_body_threshold_weighted_bytes": 1024,
            "large_body_limit": 2,
            "large_body_active": 1,
            "large_body_oldest_holder_age_ms": 2500,
            "large_body_decision_events_recorded_total": 3,
            "large_body_decision_history_overwritten_total": 0,
            "large_body_decision_record_failures_total": 0,
            "large_body_decision_observer_errors_total": 0,
            "large_body_decision_observer_enqueue_failures_total": 0,
            "rejection_decision_total": 3,
            "rejected": {"queue_full": 3},
        }
    )
    gauges.open_sockets = 4
    gauges.record_admission_503_response_write("queue_full", completed=True)
    gauges.record_admission_503_response_write("queue_full", completed=False)
    gauges.tcp_states = {"ESTABLISHED": 2, "CLOSE_WAIT": 1}
    gauges.attach_stream_parser_budget(
        lambda: {
            "used_bytes": 777,
            "peak_bytes": 999,
            "capacity_bytes": 4096,
            "rejected": 3,
        }
    )
    gauges.attach_observability_exporter(
        lambda: {
            "large_body_decision_enqueued_total": 3,
            "large_body_decision_enqueue_dropped_total": 1,
            "large_body_decision_build_errors_total": 0,
            "large_body_decision_export_errors_total": 2,
            "admission_503_outcome_enqueued_total": 2,
            "admission_503_outcome_enqueue_dropped_total": 1,
            "admission_503_outcome_build_errors_total": 0,
            "admission_503_outcome_export_errors_total": 1,
        }
    )
    monkeypatch.setattr(
        runtime,
        "_open_socket_count",
        lambda: (_ for _ in ()).throw(AssertionError("snapshot must not scan /proc")),
    )

    snapshot = gauges.snapshot()

    assert snapshot["inflight_requests"] == 7
    assert snapshot["request_active"] == 7
    assert snapshot["request_waiters"] == 11
    assert snapshot["request_capacity"] is None
    assert snapshot["request_waiter_limit"] is None
    assert snapshot["request_admission_mode"] == "weighted_resources"
    assert snapshot["request_control_memory_bytes_per_request"] == 65536
    assert snapshot["request_control_memory_reserved_bytes"] == 458752
    assert snapshot["cpu_phase_capacity"] == 2
    assert snapshot["cpu_phase_active"] == 1
    assert snapshot["cpu_phase_waiters"] == 3
    assert snapshot["middleware_inflight_requests"] == 99
    assert snapshot["request_body_reserved_weighted_bytes"] == 1234
    assert snapshot["runtime_global_request_body_reserved_weighted_bytes"] == 1234
    assert snapshot["upstream_response_reserved_weighted_bytes"] == 4321
    assert snapshot["runtime_global_upstream_response_reserved_weighted_bytes"] == 4321
    assert snapshot["request_retained_reserved_weighted_bytes"] == 5555
    assert snapshot["runtime_global_retained_reserved_weighted_bytes"] == 5555
    assert snapshot["request_large_body_threshold_weighted_bytes"] == 1024
    assert snapshot["request_large_body_limit"] == 2
    assert snapshot["request_large_body_active"] == 1
    assert snapshot["runtime_global_large_body_active"] == 1
    assert snapshot["runtime_global_large_body_oldest_holder_age_ms"] == 2500
    assert snapshot["request_deferred_memory_requests"] == 2
    assert snapshot["request_deferred_memory_weighted_bytes"] == 3333
    assert snapshot["runtime_global_deferred_memory_requests"] == 2
    assert snapshot["runtime_global_deferred_memory_weighted_bytes"] == 3333
    assert snapshot["stream_parser_reserved_bytes"] == 777
    assert snapshot["stream_parser_budget_bytes"] == 4096
    assert snapshot["stream_parser_peak_bytes"] == 999
    assert snapshot["stream_parser_rejected_total"] == 3
    assert "request_body_reserved_bytes" not in snapshot
    assert snapshot["request_admission_rejected_total"] == 3
    assert snapshot["runtime_global_admission_rejection_decision_total"] == 3
    assert snapshot["runtime_global_admission_503_response_write_completed_total"] == 1
    assert snapshot["runtime_global_admission_503_response_write_failed_total"] == 1
    assert snapshot["runtime_global_admission_503_response_write_completed"] == {
        "queue_full": 1
    }
    assert snapshot["runtime_global_large_body_decision_export_enqueued_total"] == 3
    assert snapshot[
        "runtime_global_large_body_decision_export_enqueue_dropped_total"
    ] == 1
    assert snapshot[
        "runtime_global_admission_503_outcome_export_enqueue_dropped_total"
    ] == 1
    assert "large_body_holders" not in snapshot
    assert snapshot["open_sockets"] == 4
    assert snapshot["tcp_close_wait"] == 1


def test_bounded_env_int_rejects_unsafe_override(monkeypatch):
    monkeypatch.setenv("TEST_BOUNDED_LIMIT", "101")
    with pytest.raises(ValueError, match="startup safety limit 100"):
        runtime._bounded_env_int("TEST_BOUNDED_LIMIT", 50, 100)


def test_admission_503_write_callback_emits_joinable_post_write_outcome(monkeypatch):
    captured = []
    request_facts = []
    monkeypatch.setattr(runtime, "runtime_gauges", RuntimeGauges())
    monkeypatch.setattr(runtime, "_emit_request_observability", request_facts.append)
    monkeypatch.setattr(
        runtime,
        "emit_admission_503_response_write_outcome",
        captured.append,
    )
    scope = {
        "method": "POST",
        "path": "/v1/responses",
        "headers": [(b"x-request-id", b"req-123")],
        "state": {
            "uni_api_admission_request_id": "req-123",
            "uni_api_admission_trace_id": "0123456789abcdef0123456789abcdef",
            "uni_api_admission_lease": SimpleNamespace(lease_id="lease-safe-1"),
        },
    }

    rejection = SimpleNamespace(
        status_code=503,
        reason="large_body_capacity_exhausted",
    )
    runtime._observe_request_admission_rejection(scope, rejection, 3.0)
    runtime._observe_request_admission_response_write(
        scope,
        rejection,
        True,
    )
    runtime._observe_request_admission_response_write(
        scope,
        rejection,
        False,
    )

    assert len(request_facts) == 1
    assert request_facts[0]["request_id"] == "req-123"
    assert request_facts[0]["trace_id"] == "0123456789abcdef0123456789abcdef"
    assert len(captured) == 2
    outcome = captured[0]
    assert outcome.request_self_request_id == "req-123"
    assert outcome.request_self_trace_id == "0123456789abcdef0123456789abcdef"
    assert outcome.request_self_lease_id == "lease-safe-1"
    assert outcome.asgi_response_write_completed is True
    assert outcome.runtime_global_admission_503_response_write_completed_total_after == 1
    assert outcome.runtime_global_admission_503_response_write_failed_total_after == 0
    failed = captured[1]
    assert failed.request_self_request_id == outcome.request_self_request_id
    assert failed.request_self_trace_id == outcome.request_self_trace_id
    assert failed.request_self_lease_id == outcome.request_self_lease_id
    assert failed.asgi_response_write_completed is False
    assert failed.runtime_global_admission_503_response_write_completed_total_after == 1
    assert failed.runtime_global_admission_503_response_write_failed_total_after == 1


def test_bounded_env_int_rejects_invalid_override(monkeypatch):
    monkeypatch.setenv("TEST_BOUNDED_LIMIT", "not-an-int")
    with pytest.raises(ValueError, match="must be an integer"):
        runtime._bounded_env_int("TEST_BOUNDED_LIMIT", 50, 100)


def test_positive_env_int_rejects_nonpositive_override(monkeypatch):
    monkeypatch.setenv("TEST_POSITIVE_LIMIT", "0")
    with pytest.raises(ValueError, match="must be positive"):
        runtime._positive_env_int("TEST_POSITIVE_LIMIT", 50)


def test_runtime_observability_endpoint_bypasses_request_admission():
    assert runtime._bypass_request_admission(
        {"method": "GET", "path": "/v1/observability/runtime"}
    )
    assert not runtime._bypass_request_admission(
        {"method": "POST", "path": "/v1/observability/runtime"}
    )


def test_runtime_receive_failure_does_not_fabricate_disconnect():
    async def scenario():
        event = asyncio.Event()

        class Request:
            async def receive(self):
                raise RuntimeError("adapter failed")

        await runtime.monitor_disconnect(Request(), event)
        assert not event.is_set()

    asyncio.run(scenario())


def test_runtime_stream_queue_metrics_track_live_and_retired_totals():
    async def scenario():
        gauges = RuntimeGauges()
        queue = ByteBoundedQueue(max_items=1, max_bytes=16)
        gauges.register_stream_queue(queue)
        await queue.put(b"first")
        blocked = asyncio.create_task(queue.put(b"second"))
        await asyncio.sleep(0)

        live = gauges.snapshot()
        assert live["stream_queue_active"] == 1
        assert live["stream_queue_bytes"] == 5
        assert live["stream_queue_waiting_putters"] == 1

        lease = await queue.get_lease()
        assert lease.item == b"first"
        await lease.release()
        await blocked
        await queue.close(discard=True)
        gauges.unregister_stream_queue(queue)

        retired = gauges.snapshot()
        assert retired["stream_queue_active"] == 0
        assert retired["stream_queue_bytes"] == 0
        assert retired["stream_queue_blocked_puts"] == 1

    asyncio.run(scenario())


def test_runtime_network_sampler_runs_proc_scans_off_event_loop(monkeypatch):
    async def scenario():
        gauges = RuntimeGauges()
        monkeypatch.setattr(runtime, "_open_socket_count", lambda: 8)
        monkeypatch.setattr(
            runtime,
            "_tcp_state_counts",
            lambda: {"ESTABLISHED": 6},
        )

        await gauges.start_network_sampler(interval_seconds=60)
        try:
            assert gauges.snapshot()["open_sockets"] == 8
            assert gauges.snapshot()["tcp_states"] == {"ESTABLISHED": 6}
        finally:
            await gauges.stop_network_sampler()

    asyncio.run(scenario())
