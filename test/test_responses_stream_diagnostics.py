import asyncio
import errno
import gc
import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpcore
import httpx
import pytest
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from uni_api.observability.responses_stream import (
    OAIX_TERMINAL_FLUSH_MARKER_CONTRACT,
    ObservedResponseByteIterator,
    ResponsesStreamDiagnostics,
)
from uni_api.streaming.logging_response import LoggingStreamingResponse
from uni_api.streaming.bounded_queue import ObservedStreamChunk
from uni_api.streaming.cleanup import (
    force_close_response_httpcore_stream_chain_safely,
)
from uni_api.streaming.sse import IncrementalSSEParser, SSEIncompleteEventError
from uni_api.streaming.usage import stream_usage_snapshot
import uni_api.runtime as runtime
from uni_api.observability import responses_stream as responses_stream_observability


def _tracker() -> tuple[ResponsesStreamDiagnostics, dict]:
    current_info = {"upstream_attempts": [{}]}
    tracker = ResponsesStreamDiagnostics(
        current_info=current_info,
        attempt_index=0,
        logical_authority="oaix.example",
        proxy_configured=False,
    )
    return tracker, current_info


class _FakeNetworkStream:
    def get_extra_info(self, name):
        return {
            "client_addr": ("10.0.0.7", 43123),
            "server_addr": ("192.0.2.9", 443),
            "socket": None,
        }.get(name)


def test_response_metadata_is_exact_but_endpoints_are_payload_free_hmacs():
    tracker, _current_info = _tracker()
    response = httpx.Response(
        200,
        headers={"X-OAIX-Connection-ID": "oaixc-test-7"},
        extensions={
            "http_version": b"HTTP/2",
            "stream_id": 11,
            "network_stream": _FakeNetworkStream(),
        },
    )

    tracker.capture_response(response)

    facts = tracker.facts
    assert facts["http_version"] == "HTTP/2"
    assert facts["httpcore_stream_id"] == 11
    assert facts["oaix_connection_id"] == "oaixc-test-7"
    assert len(facts["transport_local_endpoint_hmac"]) == 64
    assert len(facts["transport_peer_endpoint_hmac"]) == 64
    assert len(facts["transport_four_tuple_hmac"]) == 64
    serialized = json.dumps(facts, sort_keys=True)
    assert "10.0.0.7" not in serialized
    assert "192.0.2.9" not in serialized
    assert "43123" not in serialized


def test_missing_http_version_and_network_metadata_remain_unknown():
    tracker, _current_info = _tracker()
    tracker.capture_response(httpx.Response(200))

    assert "http_version" not in tracker.facts
    assert tracker.facts["transport_metadata_available"] is False


def test_oaix_terminal_flush_marker_produces_hash_bound_exact_hop_observation():
    observations = []
    chunks = []
    events = []
    current_info = {
        "request_id": "request-hop-1",
        "trace_id": "trace-hop-1",
        "upstream_attempts": [{}],
    }
    tracker = ResponsesStreamDiagnostics(
        current_info=current_info,
        attempt_index=0,
        logical_authority="oaix.example",
        proxy_configured=False,
        sse_chunk_observer=chunks.append,
        sse_event_observer=events.append,
        terminal_hop_observer=observations.append,
    )
    tracker.capture_response(
        httpx.Response(
            200,
            headers={
                "X-OAIX-Connection-ID": "oaixc-hop-1",
                "X-OAIX-Terminal-Flush-Marker": (
                    OAIX_TERMINAL_FLUSH_MARKER_CONTRACT
                ),
            },
        )
    )
    tracker.observe_upstream_chunk(b"12345")
    terminal = (
        "event: response.completed\n"
        'data: {"type":"response.completed","response":{"status":"completed"}}'
    )
    tracker.observe_complete_event(terminal)
    terminal_received = datetime.fromisoformat(
        tracker.facts["declared_terminal_received_at"]
    )
    flush_completed = terminal_received - timedelta(milliseconds=7654.25)
    flush_attempted = flush_completed - timedelta(milliseconds=0.25)
    wire_sha256 = hashlib.sha256(
        terminal.encode("utf-8") + b"\n\n"
    ).hexdigest()
    marker_payload = {
        "schema_version": 1,
        "connection_id": "oaixc-hop-1",
        "terminal_wire_sha256": wire_sha256,
        "flush_attempted_unix_nano": int(flush_attempted.timestamp() * 1e9),
        "flush_completed_unix_nano": int(flush_completed.timestamp() * 1e9),
    }
    marker = ": oaix-terminal-flush-v1 " + json.dumps(
        marker_payload,
        separators=(",", ":"),
        sort_keys=True,
    )
    tracker.observe_complete_event(marker, has_data_field=False)

    assert tracker.expects_oaix_terminal_flush_marker is True
    assert tracker.terminal_hop_observed is True
    assert tracker.facts["oaix_terminal_flush_marker_hash_matched"] is True
    assert abs(
        tracker.facts["oaix_terminal_flush_to_ember_receive_ms"] - 7654.25
    ) < 0.01
    assert chunks == [5]
    assert events == [len(terminal.encode("utf-8")) + 2]
    assert len(observations) == 1
    assert observations[0]["request_id"] == "request-hop-1"
    assert observations[0]["terminal_wire_sha256"] == wire_sha256


def test_complete_event_preserves_transport_receive_timestamp():
    tracker, _current_info = _tracker()
    received_at = datetime(2026, 7, 24, 12, 34, 56, tzinfo=timezone.utc)
    terminal = (
        "event: response.completed\n"
        'data: {"type":"response.completed",'
        '"response":{"status":"completed"}}'
    )

    tracker.observe_complete_event(terminal, received_at=received_at)

    assert tracker.facts["last_event_received_at"] == received_at.isoformat()
    assert tracker.facts["declared_terminal_received_at"] == (
        received_at.isoformat()
    )


def test_nonterminal_event_records_metadata_without_hashing():
    tracker, _current_info = _tracker()
    received_at = datetime(2026, 7, 24, 12, 34, 56, tzinfo=timezone.utc)
    raw_event = (
        "event: response.output_text.delta\n"
        'data: {"type":"response.output_text.delta","delta":"hello"}'
    )
    wire = raw_event.encode("utf-8") + b"\n\n"

    tracker.observe_complete_event(
        raw_event,
        event_type="response.output_text.delta",
        wire_bytes=wire,
        received_at=received_at,
    )
    tracker.observe_parsed_event(
        raw_event,
        "response.output_text.delta",
        {"type": "response.output_text.delta", "delta": "hello"},
        semantic_outcome="nonterminal",
        wire_bytes=wire,
        received_at=received_at,
    )

    assert tracker.facts["schema_version"] == 4
    assert tracker.facts["event_hash_policy"] == "terminal_or_error_only_v1"
    assert tracker.facts["complete_event_count"] == 1
    assert tracker.facts["last_event_type"] == "response.output_text.delta"
    assert tracker.facts["last_event_bytes"] == len(wire)
    assert "last_event_sha256" not in tracker.facts
    assert "last_event_received_at" not in tracker.facts


def test_terminal_timeline_uses_exact_monotonic_stage_boundaries():
    wall_values = iter(
        [
            1_700_000_000_000_001_000,
            1_700_000_000_000_002_000,
            1_700_000_000_000_003_000,
            1_700_000_000_000_004_000,
            1_700_000_000_000_005_000,
            1_700_000_000_000_006_000,
        ]
    )
    monotonic_values = iter([11_000, 12_000, 13_000, 14_000, 15_000, 16_000])
    tracker = ResponsesStreamDiagnostics(
        current_info={"upstream_attempts": [{}]},
        attempt_index=0,
        logical_authority="oaix.example",
        proxy_configured=False,
        wall_time_ns=lambda: next(wall_values),
        monotonic_ns=lambda: next(monotonic_values),
    )
    received_unix_nano = 1_700_000_000_000_000_000
    received_monotonic_nano = 10_000
    raw_event = _completed_event().decode("utf-8").rstrip("\n")
    payload = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            },
        },
    }

    tracker.mark_terminal_parse_completed(
        "response.completed",
        received_unix_nano=received_unix_nano,
        received_monotonic_nano=received_monotonic_nano,
    )
    tracker.observe_complete_event(
        raw_event,
        received_unix_nano=received_unix_nano,
        received_monotonic_nano=received_monotonic_nano,
    )
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        payload,
        semantic_outcome="completed",
    )
    tracker.mark_terminal_queue_handoff_completed()
    tracker.mark_terminal_asgi_write_attempted()
    tracker.mark_terminal_asgi_write_completed()

    facts = tracker.facts
    assert facts["terminal_timeline_complete"] is True
    assert facts["terminal_received_unix_nano"] == received_unix_nano
    assert facts["terminal_received_monotonic_nano"] == (
        received_monotonic_nano
    )
    assert facts["terminal_parse_completed_monotonic_nano"] == 11_000
    assert facts["terminal_observer_completed_monotonic_nano"] == 12_000
    assert facts["terminal_semantic_classified_monotonic_nano"] == 13_000
    assert facts["terminal_queue_handoff_completed_monotonic_nano"] == 14_000
    assert facts["terminal_asgi_write_attempted_monotonic_nano"] == 15_000
    assert facts["terminal_asgi_write_completed_monotonic_nano"] == 16_000
    assert facts["terminal_received_to_parse_completed_us"] == 1.0
    assert facts["terminal_parse_completed_to_observer_completed_us"] == 1.0
    assert facts[
        "terminal_queue_handoff_completed_to_asgi_write_attempted_us"
    ] == 1.0


def test_phase_sample_records_only_timing_and_numeric_work_counters():
    observed = []
    monotonic_values = iter([100, 350])
    cpu_values = iter([20, 70])
    tracker = ResponsesStreamDiagnostics(
        current_info={"upstream_attempts": [{}]},
        attempt_index=0,
        logical_authority="oaix.example",
        proxy_configured=False,
        phase_sampled=True,
        phase_sample_rate=128,
        phase_observer=lambda phase, **metrics: observed.append(
            (phase, metrics)
        ),
        monotonic_ns=lambda: next(monotonic_values),
        thread_cpu_ns=lambda: next(cpu_values),
    )

    token = tracker.begin_phase("json_parse")
    tracker.finish_phase(
        "json_parse",
        token,
        bytes_count=512,
        events=2,
    )

    assert tracker.facts["phase_json_parse_samples"] == 1
    assert tracker.facts["phase_json_parse_wall_ns"] == 250
    assert tracker.facts["phase_json_parse_cpu_ns"] == 50
    assert tracker.facts["phase_json_parse_bytes"] == 512
    assert tracker.facts["phase_json_parse_events"] == 2
    assert observed == [
        (
            "json_parse",
            {
                "wall_ns": 250,
                "cpu_ns": 50,
                "bytes_count": 512,
                "events": 2,
            },
        )
    ]


def test_kernel_socket_unread_sample_revalidates_fd_inode(monkeypatch):
    ioctl_calls = []
    monkeypatch.setattr(responses_stream_observability.sys, "platform", "linux")
    monkeypatch.setattr(
        responses_stream_observability.os,
        "readlink",
        lambda _path: "socket:[123]",
    )
    monkeypatch.setattr(
        responses_stream_observability.fcntl,
        "ioctl",
        lambda fd, operation, value: (
            ioctl_calls.append((fd, operation, value)) or b"\x00\x10\x00\x00"
        ),
    )

    assert responses_stream_observability._kernel_socket_unread_bytes(
        9,
        "123",
    ) == 4096
    assert len(ioctl_calls) == 1
    assert responses_stream_observability._kernel_socket_unread_bytes(
        9,
        "999",
    ) is None
    assert len(ioctl_calls) == 1


def test_semantic_failure_hashes_existing_wire_bytes_once_classified():
    tracker, _current_info = _tracker()
    received_at = datetime(2026, 7, 24, 12, 34, 56, tzinfo=timezone.utc)
    raw_event = (
        "event: response.status\n"
        'data: {"type":"response.status","status":"failed"}'
    )
    wire = raw_event.encode("utf-8") + b"\n\n"

    tracker.observe_complete_event(
        raw_event,
        event_type="response.status",
        wire_bytes=wire,
        received_at=received_at,
    )
    tracker.observe_parsed_event(
        raw_event,
        "response.status",
        {"type": "response.status", "status": "failed"},
        semantic_outcome="failed",
        wire_bytes=wire,
        received_at=received_at,
    )

    expected = hashlib.sha256(wire).hexdigest()
    assert tracker.facts["last_event_sha256"] == expected
    assert tracker.facts["semantic_terminal_sha256"] == expected
    assert tracker.facts["last_event_received_at"] == received_at.isoformat()


def test_canonical_standard_event_reuses_the_observed_wire_object():
    raw_event = (
        "event: response.output_text.delta\n"
        'data: {"type":"response.output_text.delta","delta":"hello"}'
    )
    wire = raw_event.encode("utf-8") + b"\n\n"

    canonical, normalized = runtime._canonical_responses_sse_event_bytes(
        raw_event,
        event_type="response.output_text.delta",
        has_event_field=True,
        wire_bytes=wire,
    )

    assert normalized is False
    assert canonical is wire


def test_canonical_data_only_event_derives_output_without_reencoding_text():
    raw_event = 'data: {"type":"response.output_text.delta","delta":"hello"}'
    wire = raw_event.encode("utf-8") + b"\n\n"

    canonical, normalized = runtime._canonical_responses_sse_event_bytes(
        raw_event,
        event_type="response.output_text.delta",
        has_event_field=False,
        wire_bytes=wire,
    )

    assert normalized is True
    assert canonical == b"event: response.output_text.delta\n" + wire


def test_oaix_terminal_flush_marker_rejects_hash_mismatch_without_metric():
    observations = []
    tracker = ResponsesStreamDiagnostics(
        current_info={"upstream_attempts": [{}]},
        attempt_index=0,
        logical_authority="oaix.example",
        proxy_configured=False,
        terminal_hop_observer=observations.append,
    )
    tracker.capture_response(
        httpx.Response(
            200,
            headers={
                "X-OAIX-Connection-ID": "oaixc-hop-2",
                "X-OAIX-Terminal-Flush-Marker": (
                    OAIX_TERMINAL_FLUSH_MARKER_CONTRACT
                ),
            },
        )
    )
    terminal = (
        "event: response.completed\n"
        'data: {"type":"response.completed","response":{"status":"completed"}}'
    )
    tracker.observe_complete_event(terminal)
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    marker = ": oaix-terminal-flush-v1 " + json.dumps(
        {
            "schema_version": 1,
            "connection_id": "oaixc-hop-2",
            "terminal_wire_sha256": "f" * 64,
            "flush_attempted_unix_nano": now_ns - 1_000_000,
            "flush_completed_unix_nano": now_ns,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    tracker.observe_complete_event(marker, has_data_field=False)

    assert tracker.terminal_hop_observed is False
    assert tracker.facts["oaix_terminal_flush_marker_hash_matched"] is False
    assert tracker.facts["oaix_terminal_flush_marker_invalid_reason"] == (
        "terminal_hash_mismatch"
    )
    assert observations == []


@pytest.mark.parametrize("schema_version", [[], {}, "not-an-integer"])
def test_oaix_terminal_flush_marker_rejects_malformed_schema_without_raising(
    schema_version,
):
    tracker, _current_info = _tracker()
    marker = ": oaix-terminal-flush-v1 " + json.dumps(
        {
            "schema_version": schema_version,
            "terminal_wire_sha256": "a" * 64,
            "flush_attempted_unix_nano": 1,
            "flush_completed_unix_nano": 2,
        },
        separators=(",", ":"),
    )

    tracker.observe_complete_event(marker, has_data_field=False)

    assert tracker.facts["oaix_terminal_flush_marker_valid"] is False
    assert tracker.facts["oaix_terminal_flush_marker_invalid_reason"] == (
        "invalid_payload"
    )
    assert "event_observer_error" not in tracker.facts


def test_oaix_terminal_flush_marker_requires_advertised_connection_identity():
    tracker, _current_info = _tracker()
    tracker.capture_response(
        httpx.Response(
            200,
            headers={
                "X-OAIX-Connection-ID": "oaixc-required",
                "X-OAIX-Terminal-Flush-Marker": (
                    OAIX_TERMINAL_FLUSH_MARKER_CONTRACT
                ),
            },
        )
    )
    marker = ": oaix-terminal-flush-v1 " + json.dumps(
        {
            "schema_version": 1,
            "terminal_wire_sha256": "a" * 64,
            "flush_attempted_unix_nano": 1,
            "flush_completed_unix_nano": 2,
        },
        separators=(",", ":"),
    )

    tracker.observe_complete_event(marker, has_data_field=False)

    assert tracker.facts["oaix_terminal_flush_marker_valid"] is False
    assert tracker.facts["oaix_terminal_flush_marker_invalid_reason"] == (
        "connection_id_mismatch"
    )


def _chained_read_error() -> httpx.ReadError:
    try:
        raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")
    except ConnectionResetError as os_exc:
        try:
            raise httpcore.ReadError("core read failed") from os_exc
        except httpcore.ReadError as core_exc:
            try:
                raise httpx.ReadError(
                    "httpx read failed",
                    request=httpx.Request(
                        "POST",
                        "https://example.test/v1/responses",
                    ),
                ) from core_exc
            except httpx.ReadError as exc:
                return exc


def test_exception_chain_errno_and_httpcore_reactive_close_order_are_preserved():
    tracker, _current_info = _tracker()
    exc = _chained_read_error()
    core_exc = exc.__cause__
    assert isinstance(core_exc, httpcore.ReadError)

    async def observe():
        await tracker.httpcore_trace(
            "http11.receive_response_body.failed",
            {"exception": core_exc},
        )
        await tracker.httpcore_trace("http11.response_closed.started", {})
        await tracker.httpcore_trace("http11.response_closed.complete", {})

    asyncio.run(observe())
    tracker.observe_exception(exc, origin="upstream_body_iterator")

    facts = tracker.facts
    assert [row["type"] for row in facts["exception_chain"]] == [
        "ReadError",
        "ReadError",
        "ConnectionResetError",
    ]
    assert facts["exception_errno"] == errno.ECONNRESET
    assert facts["exception_errno_name"] == "ECONNRESET"
    assert facts["exception_chain_truncated"] is False
    assert facts["httpcore_response_close_trigger"] == (
        "httpcore_reactive_close_after_body_read_failure"
    )
    assert facts["local_cleanup_claimed_before_body_read_failure"] is False
    serialized = json.dumps(facts, sort_keys=True)
    assert "Connection reset by peer" not in serialized
    assert "core read failed" not in serialized
    assert "httpx read failed" not in serialized


def test_httpcore_trace_drops_successful_reads_without_hiding_a_late_failure():
    tracker, _current_info = _tracker()
    core_exc = httpcore.ReadError("late read failure")

    async def observe():
        for _ in range(100):
            await tracker.httpcore_trace(
                "http2.receive_response_body.started",
                {},
            )
            await tracker.httpcore_trace(
                "http2.receive_response_body.complete",
                {},
            )
        await tracker.httpcore_trace(
            "http2.receive_response_body.failed",
            {"exception": core_exc},
        )
        await tracker.httpcore_trace("http2.response_closed.started", {})

    asyncio.run(observe())

    assert [event["name"] for event in tracker.facts["httpcore_events"]] == [
        "http2.receive_response_body.failed",
        "http2.response_closed.started",
    ]
    assert "httpcore_events_truncated" not in tracker.facts
    assert tracker.facts["httpcore_response_close_trigger"] == (
        "httpcore_reactive_close_after_body_read_failure"
    )


def test_httpcore_cancellation_after_local_claim_is_not_mislabeled_read_error():
    tracker, _current_info = _tracker()
    tracker.begin_cleanup(
        owner="responses_queue_body",
        trigger="downstream_body_iterator_closed_or_shutdown",
    )

    async def observe():
        await tracker.httpcore_trace(
            "http11.receive_response_body.failed",
            {"exception": asyncio.CancelledError()},
        )
        await tracker.httpcore_trace("http11.response_closed.started", {})

    asyncio.run(observe())

    assert "httpcore_body_read_failed_at" not in tracker.facts
    assert tracker.facts["httpcore_body_read_cancelled_at"]
    assert tracker.facts["transport_end_trigger"] == "ember_local_cleanup"
    assert tracker.facts["httpcore_response_close_trigger"] == (
        "ember_explicit_cleanup"
    )


def test_first_cleanup_owner_cannot_be_overwritten():
    tracker, _current_info = _tracker()

    assert tracker.begin_cleanup(owner="pool_sweeper", trigger="kernel_close_wait")
    assert not tracker.begin_cleanup(
        owner="responses_proxy_finally",
        trigger="after_upstream_read_failure",
    )
    tracker.observe_cleanup_transport_outcome(
        {
            "method": "pool_eviction",
            "transport_evicted": True,
            "transport_isolated": True,
        }
    )
    tracker.finish_cleanup(
        transport_safe=True,
        context_exit_succeeded=True,
    )
    tracker.finish_cleanup(
        transport_safe=False,
        context_exit_succeeded=False,
    )

    assert tracker.facts["cleanup_owner"] == "pool_sweeper"
    assert tracker.facts["cleanup_trigger"] == "kernel_close_wait"
    assert tracker.facts["cleanup_method"] == "pool_eviction"
    assert tracker.facts["cleanup_transport_evicted"] is True
    assert tracker.facts["cleanup_result"] == "succeeded"


def test_cleanup_outcome_sink_reports_cooperative_close_without_guessing_eviction():
    outcomes = []

    class Response:
        stream = None

        async def aclose(self):
            return None

    result = asyncio.run(
        force_close_response_httpcore_stream_chain_safely(
            Response(),
            label="diagnostic-test",
            outcome_sink=outcomes.append,
        )
    )

    assert result is True
    assert outcomes == [
        {
            "method": "cooperative_response_aclose",
            "cooperative_close_started": True,
            "cooperative_close_completed": True,
            "transport_evicted": False,
            "transport_isolated": False,
            "detached_cleanup": False,
        }
    ]


def test_pool_sweeper_is_correlated_to_the_active_transport_before_close():
    tracker, _current_info = _tracker()
    local_socket, peer_socket = socket.socketpair()

    class NetworkStream:
        def get_extra_info(self, name):
            return {
                "client_addr": ("127.0.0.1", 41000),
                "server_addr": ("127.0.0.1", 42000),
                "socket": local_socket,
            }.get(name)

    network_stream = NetworkStream()
    tracker.capture_response(
        httpx.Response(200, extensions={"network_stream": network_stream})
    )

    class Connection:
        _network_stream = network_stream

        def is_closed(self):
            return False

        def has_expired(self):
            return True

        async def aclose(self):
            local_socket.close()

    connection = Connection()

    class Pool:
        _optional_thread_lock = None

        def __init__(self):
            self._connections = [connection]

        def _assign_requests_to_connections(self):
            return []

        async def _close_connections(self, connections):
            for item in connections:
                await item.aclose()

    client = SimpleNamespace(
        _transport=SimpleNamespace(_pool=Pool())
    )
    try:
        assert asyncio.run(runtime._sweep_httpx_client_idle_connections(client)) == 1
    finally:
        local_socket.close()
        peer_socket.close()

    assert tracker.facts["pool_sweeper_close_observed"] is True
    assert tracker.facts["pool_sweeper_trigger"] == "httpcore_has_expired"
    assert tracker.facts["pool_sweeper_close_succeeded"] is True
    assert tracker.facts["cleanup_owner"] == "pool_sweeper"
    assert tracker.facts["cleanup_method"] == "pool_sweeper_connection_close"


def test_pool_sweeper_correlates_every_active_http2_stream_on_shared_transport():
    first_tracker, _first_info = _tracker()
    second_tracker, _second_info = _tracker()

    class NetworkStream:
        def get_extra_info(self, name):
            return {
                "client_addr": ("127.0.0.1", 41000),
                "server_addr": ("127.0.0.1", 42000),
                "socket": None,
            }.get(name)

    network_stream = NetworkStream()
    first_tracker.capture_response(
        httpx.Response(
            200,
            extensions={
                "http_version": b"HTTP/2",
                "stream_id": 1,
                "network_stream": network_stream,
            },
        )
    )
    second_tracker.capture_response(
        httpx.Response(
            200,
            extensions={
                "http_version": b"HTTP/2",
                "stream_id": 3,
                "network_stream": network_stream,
            },
        )
    )

    connection = SimpleNamespace(_network_stream=network_stream)
    observed = runtime.observe_pool_sweeper_connection_close(
        connection,
        trigger="httpcore_is_closed",
    )

    assert set(observed) == {first_tracker, second_tracker}
    for tracker in observed:
        assert tracker.facts["pool_sweeper_close_observed"] is True
        assert tracker.facts["pool_sweeper_trigger"] == "httpcore_is_closed"
        assert tracker.facts["cleanup_owner"] == "pool_sweeper"


def test_transport_tracker_registry_keys_are_removed_when_requests_are_collected():
    gc.collect()
    baseline = len(responses_stream_observability._NETWORK_STREAM_TRACKERS)

    for index in range(100):
        current_info = {"upstream_attempts": [{}]}
        tracker = ResponsesStreamDiagnostics(
            current_info=current_info,
            attempt_index=0,
            logical_authority="oaix.example",
            proxy_configured=False,
        )
        tracker.capture_response(
            httpx.Response(
                200,
                extensions={
                    "network_stream": _FakeNetworkStream(),
                    "stream_id": index * 2 + 1,
                },
            )
        )

    del current_info
    del tracker
    gc.collect()

    assert len(responses_stream_observability._NETWORK_STREAM_TRACKERS) == baseline


def test_nonterminal_and_partial_diagnostics_do_not_store_payload():
    tracker, _current_info = _tracker()
    parser = IncrementalSSEParser()
    complete = parser.feed(
        b"\xef\xbb\xbfevent: response.created\r\ndata: {\"type\":\"response.created\"}\r\n\r\n"
        b"event: response.completed\r\ndata: {\"secret\":\"VERY_PRIVATE_PARTIAL_BODY"
    )
    assert len(complete) == 1
    tracker.observe_complete_event(complete[0])
    tracker.observe_partial_diagnostics(parser.pending_diagnostics())

    assert tracker.facts["last_event_type"] == "response.created"
    assert "last_event_sha256" not in tracker.facts
    assert tracker.facts["partial_event_bytes"] > 0
    assert tracker.facts["partial_event_sha256"]
    serialized = json.dumps(tracker.facts, sort_keys=True)
    assert "VERY_PRIVATE_PARTIAL_BODY" not in serialized

    try:
        parser.finish()
    except SSEIncompleteEventError:
        pass
    else:
        raise AssertionError("expected incomplete event")
    assert parser.failure_pending_diagnostics["bytes"] == tracker.facts[
        "partial_event_bytes"
    ]


def test_malformed_completed_label_is_not_reported_as_semantic_completion():
    tracker, _current_info = _tracker()
    raw_event = (
        "event: response.completed\n"
        'data: {"type":"response.completed","response":null}'
    )

    tracker.observe_complete_event(raw_event)
    assert tracker.facts["terminal_frame_seen"] is True
    assert tracker.facts["upstream_terminal_seen"] is False
    assert tracker.facts["upstream_terminal_validated"] is False
    assert tracker.facts["semantic_status"] == "unknown"
    assert tracker.facts["diagnosis"] == "responses_terminal_event_unvalidated"

    tracker.observe_exception(
        SSEIncompleteEventError(pending_bytes=1),
        origin="postcommit_stream",
    )
    assert tracker.facts["semantic_status"] == "error"
    assert tracker.facts["diagnosis"] == "responses_sse_protocol_error"


def test_no_data_terminal_frame_is_observed_without_becoming_a_terminal():
    tracker, _current_info = _tracker()
    raw_event = "event: response.completed"

    tracker.observe_complete_event(raw_event, has_data_field=False)
    tracker.observe_normalization(
        "ignored_no_data_event_block",
        "response.completed",
    )

    assert tracker.facts["complete_event_count"] == 1
    assert tracker.facts["last_event_type"] == "response.completed"
    assert tracker.facts["terminal_frame_seen"] is False
    assert tracker.facts["ignored_no_data_event_count"] == 1
    assert tracker.facts["normalization_applied"] is True


def test_provider_error_to_response_failed_normalization_is_explicit():
    tracker, _current_info = _tracker()

    tracker.observe_normalization(
        "provider_error_to_response_failed",
        "error",
    )

    assert tracker.facts["provider_error_to_response_failed_count"] == 1
    assert tracker.facts["last_normalization_rule"] == (
        "provider_error_to_response_failed"
    )
    assert tracker.facts["last_normalized_event_type"] == "error"
    assert tracker.facts["normalization_applied"] is True


def test_validated_terminal_is_not_completed_until_ember_queue_handoff():
    tracker, _current_info = _tracker()
    raw_event = _completed_event().decode("utf-8").rstrip("\n")
    payload = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            }
        },
    }

    tracker.observe_complete_event(raw_event)
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        payload,
        semantic_outcome="completed",
    )

    assert tracker.facts["diagnosis"] == (
        "responses_terminal_pending_queue_handoff"
    )
    assert tracker.facts["semantic_status"] == "unknown"

    tracker.mark_local_end(origin="local_backpressure_abort")
    assert tracker.facts["diagnosis"] == "responses_local_backpressure_abort"
    assert tracker.facts["semantic_status"] == "unknown"


def test_completed_diagnosis_requires_successful_ember_queue_handoff():
    tracker, _current_info = _tracker()
    raw_event = _completed_event().decode("utf-8").rstrip("\n")
    payload = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            }
        },
    }

    tracker.observe_complete_event(raw_event)
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        payload,
        semantic_outcome="completed",
    )
    tracker.mark_terminal_queue_handoff_completed()

    assert tracker.facts["diagnosis"] == "responses_completed_with_usage"
    assert tracker.facts["semantic_status"] == "completed"


def test_observed_iterator_counts_body_and_keeps_full_read_error_chain():
    tracker, _current_info = _tracker()
    exc = _chained_read_error()

    async def source():
        yield b"abc"
        yield b"defg"
        raise exc

    async def consume():
        iterator = ObservedResponseByteIterator(source(), tracker)
        assert await iterator.__anext__() == b"abc"
        assert await iterator.__anext__() == b"defg"
        try:
            await iterator.__anext__()
        except httpx.ReadError:
            return
        raise AssertionError("expected read error")

    asyncio.run(consume())
    assert tracker.facts["upstream_body_bytes"] == 7
    assert tracker.facts["upstream_chunk_count"] == 2
    assert tracker.facts["exception_type"] == "ReadError"
    assert tracker.facts["diagnosis"] == "responses_read_error"


def _completed_event() -> bytes:
    return (
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"usage":'
        b'{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}\n\n'
    )


def _observe_completed_terminal(tracker: ResponsesStreamDiagnostics) -> None:
    raw_event = _completed_event().decode("utf-8").rstrip("\n")
    tracker.mark_terminal_parse_completed("response.completed")
    tracker.observe_complete_event(raw_event)
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                }
            },
        },
        semantic_outcome="completed",
    )
    tracker.mark_terminal_queue_handoff_completed()


def test_downstream_terminal_is_recorded_only_after_asgi_send_returns():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)

    async def body():
        yield ObservedStreamChunk(
            _completed_event(),
            event_type="response.completed",
            semantic_outcome="completed",
            terminal_timeline_event=True,
        )

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    assert any(b"response.completed" in message.get("body", b"") for message in sent)
    assert tracker.facts["downstream_terminal_seen"] is True
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is True
    assert tracker.facts["terminal_timeline_complete"] is True
    monotonic_points = [
        tracker.facts[f"terminal_{point}_monotonic_nano"]
        for point in (
            "received",
            "parse_completed",
            "observer_completed",
            "semantic_classified",
            "queue_handoff_completed",
            "asgi_write_attempted",
            "asgi_write_completed",
        )
    ]
    assert monotonic_points == sorted(monotonic_points)
    assert tracker.facts["downstream_final_body_completed"] is True
    assert current_info["usage_seen"] is True


def test_logging_response_lazily_binds_tracker_created_by_body_iterator():
    current_info = {
        "request_kind": "/v1/responses",
        "upstream_attempts": [{}],
    }
    trackers = []

    async def body():
        tracker = ResponsesStreamDiagnostics(
            current_info=current_info,
            attempt_index=0,
            logical_authority="oaix.example",
            proxy_configured=False,
            phase_sampled=True,
        )
        trackers.append(tracker)
        _observe_completed_terminal(tracker)
        yield ObservedStreamChunk(
            _completed_event(),
            event_type="response.completed",
            semantic_outcome="completed",
            terminal_timeline_event=True,
        )

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        assert response._responses_diagnostics_tracker is None
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    assert len(trackers) == 1
    tracker = trackers[0]
    assert tracker.facts["terminal_timeline_complete"] is True
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is True
    assert tracker.facts["phase_asgi_write_samples"] >= 1
    assert any(b"response.completed" in item.get("body", b"") for item in sent)


class _MetadataBypassParser:
    pending_bytes = 0

    def feed(self, _chunk):
        raise AssertionError("metadata-complete SSE must not be framed twice")

    def finish(self):
        return []

    def discard(self):
        return None


def test_complete_stream_metadata_bypasses_downstream_sse_parser():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)
    usage = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    snapshot = stream_usage_snapshot(usage)
    assert snapshot is not None
    sent = []

    async def body():
        yield ObservedStreamChunk(
            _completed_event(),
            event_type="response.completed",
            semantic_outcome="completed",
            sse_metadata_complete=True,
            usage_snapshot=snapshot,
        )

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        response._usage_sse_parser = _MetadataBypassParser()
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message.get("type") == "http.response.body"
    )
    assert wire == _completed_event()
    assert current_info["prompt_tokens"] == 1
    assert current_info["completion_tokens"] == 2
    assert current_info["total_tokens"] == 3
    assert current_info["usage_seen"] is True
    assert tracker.facts["downstream_usage_observer_status"] == "completed"
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is True


def test_stream_metadata_respects_disabled_usage_observation():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)
    snapshot = stream_usage_snapshot(
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    )
    assert snapshot is not None
    sent = []

    async def body():
        yield ObservedStreamChunk(
            _completed_event(),
            event_type="response.completed",
            semantic_outcome="completed",
            sse_metadata_complete=True,
            usage_snapshot=snapshot,
        )

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
            observe_usage=False,
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message.get("type") == "http.response.body"
    )
    assert wire == _completed_event()
    assert "prompt_tokens" not in current_info
    assert "completion_tokens" not in current_info
    assert "total_tokens" not in current_info
    assert tracker.facts["downstream_usage_observer_status"] == (
        "not_applicable"
    )
    assert tracker.facts["downstream_usage_observer_reason"] == (
        "disabled_by_policy"
    )


def test_segmented_stream_metadata_preserves_boundary_without_reparse():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)
    event = _completed_event()
    split_at = len(event) // 2
    snapshot = stream_usage_snapshot(
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    )
    assert snapshot is not None
    sent = []

    async def body():
        yield ObservedStreamChunk(
            event[:split_at],
            event_type="response.completed",
            semantic_outcome="completed",
            final_event_segment=False,
            sse_metadata_complete=True,
        )
        yield ObservedStreamChunk(
            event[split_at:],
            event_type="response.completed",
            semantic_outcome="completed",
            final_event_segment=True,
            sse_metadata_complete=True,
            usage_snapshot=snapshot,
        )

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        response._usage_sse_parser = _MetadataBypassParser()
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message.get("type") == "http.response.body"
    )
    assert wire == event
    assert current_info["usage_seen"] is True
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is True


def test_incomplete_segmented_metadata_preserves_usage_observer_failure():
    _tracker_instance, current_info = _tracker()
    event_prefix = _completed_event()[:37]
    sent = []

    async def body():
        yield ObservedStreamChunk(
            event_prefix,
            event_type="response.completed",
            semantic_outcome="completed",
            final_event_segment=False,
            sse_metadata_complete=True,
        )

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        response._usage_sse_parser = _MetadataBypassParser()
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message.get("type") == "http.response.body"
    )
    assert wire == event_prefix
    assert current_info["usage_parse_error"] == "SSEIncompleteEventError"
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["downstream_usage_observer_status"] == "disabled"
    assert diagnostics["downstream_usage_observer_error_type"] == (
        "SSEIncompleteEventError"
    )


def test_failed_terminal_asgi_send_does_not_claim_downstream_terminal():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)

    async def body():
        yield ObservedStreamChunk(
            _completed_event(),
            event_type="response.completed",
            semantic_outcome="completed",
        )

    async def send(message):
        if b"response.completed" in message.get("body", b""):
            raise BrokenPipeError(errno.EPIPE, "broken downstream")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    assert tracker.facts["upstream_terminal_seen"] is True
    assert tracker.facts["downstream_terminal_seen"] is False
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is False
    assert tracker.facts["downstream_final_body_attempted"] is False
    assert tracker.facts["diagnosis"] == "responses_downstream_disconnect"
    assert tracker.facts["downstream_usage_observer_status"] == "aborted"
    assert tracker.facts["downstream_usage_observer_abort_reason"] == (
        "downstream_disconnected"
    )


@pytest.mark.parametrize(
    (
        "usage",
        "input_known",
        "output_known",
        "total_known",
        "values_valid",
        "usage_seen",
    ),
    [
        ({}, False, False, False, None, False),
        ({"total_tokens": 3}, False, False, True, True, False),
        ({"input_tokens": 3}, True, False, False, True, False),
        ({"output_tokens": 3}, False, True, False, True, False),
        (
            {"input_tokens": 1, "output_tokens": 2},
            True,
            True,
            True,
            True,
            True,
        ),
        (
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            True,
            True,
            True,
            True,
            True,
        ),
        (
            {"input_tokens": True, "output_tokens": 0, "total_tokens": 0},
            True,
            True,
            True,
            False,
            False,
        ),
        (
            {"input_tokens": -1, "output_tokens": 0, "total_tokens": 0},
            True,
            True,
            True,
            False,
            False,
        ),
    ],
)
def test_usage_facts_distinguish_missing_components_from_known_zero(
    usage,
    input_known,
    output_known,
    total_known,
    values_valid,
    usage_seen,
):
    tracker, _current_info = _tracker()
    payload = {
        "type": "response.completed",
        "response": {"status": "completed", "usage": usage},
    }
    raw_event = (
        "event: response.completed\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}"
    )

    tracker.observe_complete_event(raw_event)
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        payload,
        semantic_outcome="completed",
    )
    tracker.mark_terminal_queue_handoff_completed()

    facts = tracker.facts
    assert facts["usage_object_seen"] is True
    assert facts.get("usage_input_known", False) is input_known
    assert facts.get("usage_output_known", False) is output_known
    assert facts.get("usage_total_known", False) is total_known
    assert facts.get("usage_values_valid") is values_valid
    assert facts["usage_seen"] is usage_seen
    assert facts["diagnosis"] == (
        "responses_completed_with_usage"
        if usage_seen
        else "responses_completed_without_usage"
    )


def test_last_sse_event_field_wins_and_unknown_event_name_is_not_exported():
    tracker, _current_info = _tracker()
    tracker.observe_complete_event(
        "event: response.created\n"
        "event: response.completed\n"
        'data: {"type":"response.completed"}'
    )
    assert tracker.facts["last_event_type"] == "response.completed"

    tracker.observe_complete_event(
        "event: response.secret_customer_123456\n"
        'data: {"type":"message"}'
    )
    assert tracker.facts["last_event_type"] == "other"
    secret_event = (
        "event: response.secret_customer_123456\n"
        'data: {"response":{"status":"failed"}}'
    )
    tracker.observe_parsed_event(
        secret_event,
        "response.secret_customer_123456",
        {"response": {"status": "failed"}},
        semantic_outcome="failed",
    )
    assert tracker.facts["semantic_terminal_type"] == "other"
    assert "secret_customer" not in json.dumps(tracker.facts)


def test_remote_protocol_error_has_safe_deterministic_transport_code():
    tracker, _current_info = _tracker()
    tracker.observe_exception(
        httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)",
            request=httpx.Request("POST", "https://example.test/v1/responses"),
        ),
        origin="postcommit_stream",
    )

    assert tracker.facts["transport_error_code"] == (
        "peer_closed_incomplete_chunked_body"
    )
    assert tracker.facts["transport_error_code_source"] == (
        "known_message_pattern"
    )
    assert "incomplete chunked read" not in json.dumps(tracker.facts)


def test_transport_failure_remains_primary_after_terminal_inconsistency():
    tracker, _current_info = _tracker()
    raw_event = (
        "event: response.completed\n"
        'data: {"type":"response.completed","response":{"status":"failed"}}'
    )
    payload = {
        "type": "response.completed",
        "response": {"status": "failed"},
    }
    tracker.observe_complete_event(raw_event)
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        payload,
        semantic_outcome="failed",
    )
    tracker.mark_terminal_queue_handoff_completed()
    tracker.observe_exception(
        httpx.ReadError(
            "read failed",
            request=httpx.Request("POST", "https://example.test/v1/responses"),
        ),
        origin="postcommit_stream",
    )

    assert tracker.facts["diagnosis"] == "responses_read_error"
    assert tracker.facts["semantic_status"] == "error"
    assert tracker.facts["terminal_consistency_status"] == "inconsistent"


def test_final_empty_body_failure_is_distinct_from_terminal_event_delivery():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)

    async def body():
        yield ObservedStreamChunk(
            _completed_event(),
            event_type="response.completed",
            semantic_outcome="completed",
        )

    async def send(message):
        if message.get("type") == "http.response.body" and not message.get(
            "more_body", False
        ):
            raise BrokenPipeError(errno.EPIPE, "final body failed")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is True
    assert tracker.facts["downstream_final_body_attempted"] is True
    assert tracker.facts["downstream_final_body_completed"] is False
    assert tracker.facts["downstream_final_body_outcome"] == (
        "downstream_disconnected"
    )
    assert tracker.facts["diagnosis"] == "responses_downstream_disconnect"


def test_terminal_metadata_survives_starlette_base_http_middleware():
    tracker, current_info = _tracker()
    _observe_completed_terminal(tracker)

    async def endpoint(_request: Request):
        async def body():
            yield ObservedStreamChunk(
                _completed_event(),
                event_type="response.completed",
                semantic_outcome="completed",
            )

        return StreamingResponse(body(), media_type="text/event-stream")

    class WrapForLogging(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            return LoggingStreamingResponse(
                response.body_iterator,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type,
                current_info=current_info,
            )

    app = Starlette(routes=[Route("/v1/responses", endpoint, methods=["POST"])])
    app.add_middleware(WrapForLogging)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://ember.test",
        ) as client:
            response = await client.post("/v1/responses")
            assert response.status_code == 200
            assert response.content == _completed_event()

    asyncio.run(run())
    assert tracker.facts["downstream_terminal_seen"] is True
    assert tracker.facts["downstream_terminal_asgi_write_completed"] is True
    assert tracker.facts["downstream_final_body_completed"] is True


@pytest.mark.parametrize(
    ("usage", "usage_seen", "input_known", "output_known", "expected_tokens"),
    [
        ({"total_tokens": 0}, False, False, False, {}),
        (
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            True,
            True,
            True,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    ],
)
def test_downstream_usage_parser_keeps_missing_distinct_from_known_zero(
    usage,
    usage_seen,
    input_known,
    output_known,
    expected_tokens,
):
    tracker, current_info = _tracker()
    payload = {"type": "response.completed", "usage": usage}
    event = (
        "event: response.completed\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")

    async def body():
        yield event

    async def send(_message):
        return None

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    facts = tracker.facts
    assert facts["downstream_usage_seen"] is usage_seen
    assert facts["downstream_usage_input_known"] is input_known
    assert facts["downstream_usage_output_known"] is output_known
    for key, value in expected_tokens.items():
        assert current_info[key] == value
    if not input_known:
        assert "prompt_tokens" not in current_info
    if not output_known:
        assert "completion_tokens" not in current_info


def test_attacker_sized_string_usage_counter_is_fail_open_observability():
    tracker, current_info = _tracker()
    huge_counter = "9" * 5000
    usage = {
        "prompt_tokens": huge_counter,
        "input_tokens": huge_counter,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    payload = {
        "type": "response.completed",
        "response": {"status": "completed", "usage": usage},
    }
    raw_event = (
        "event: response.completed\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}"
    )

    tracker.observe_complete_event(raw_event)
    tracker.observe_parsed_event(
        raw_event,
        "response.completed",
        payload,
        semantic_outcome="completed",
    )
    assert tracker.facts["usage_values_valid"] is False
    assert tracker.facts["usage_alias_consistent"] is False
    assert tracker.facts["usage_seen"] is False

    downstream_event = (
        "event: response.completed\n"
        f"data: {json.dumps({'type': 'response.completed', 'usage': usage}, separators=(',', ':'))}\n\n"
    ).encode("utf-8")
    sent = []

    async def body():
        yield downstream_event

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

    asyncio.run(run())
    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message.get("type") == "http.response.body"
    )
    assert wire == downstream_event
    assert current_info["stream_outcome"] == "completed"
    assert current_info["usage_parse_error"] == "invalid_usage_counter"
    assert tracker.facts["downstream_usage_seen"] is False
