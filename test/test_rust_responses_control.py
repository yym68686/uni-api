import asyncio
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from uni_api.runtime import (
    ResponsesRequestExecution,
    _is_trusted_rust_spooled_request,
    _record_rust_request_spool_observability,
)
from uni_api.rust_responses_control import (
    RustResponsesControlError,
    RustResponsesControlPlane,
)


def test_rust_stream_report_records_usage_without_completion_error():
    execution = SimpleNamespace(current_info={})
    ResponsesRequestExecution._apply_rust_stream_report(
        execution,
        {
            "upstream_bytes": 120,
            "downstream_bytes": 90,
            "event_count": 3,
            "delta_events": 1,
            "normalized_events": 2,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
        },
    )
    assert execution.current_info == {
        "rust_responses_upstream_bytes": 120,
        "rust_responses_downstream_bytes": 90,
        "rust_responses_sse_events": 3,
        "rust_responses_delta_events": 1,
        "rust_responses_normalized_events": 2,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "usage_seen": True,
    }


def test_rust_spool_observability_only_accepts_trusted_internal_headers(monkeypatch):
    monkeypatch.setattr("uni_api.runtime.RUST_RESPONSES_CONTROL_TOKEN", "control-token")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [
            (b"x-uni-api-rust-control-token", b"control-token"),
            (b"x-uni-api-rust-request-spool-body-bytes", b"20000000"),
            (b"x-uni-api-rust-request-spool-memory-peak-bytes", b"65536"),
            (b"x-uni-api-rust-request-spool-local-disk-bytes", b"20000000"),
            (b"x-uni-api-rust-request-spool-resource-wait-ms", b"19"),
            (b"x-uni-api-rust-request-spool-final-tier", b"local_disk"),
        ],
    }
    current_info = {}
    _record_rust_request_spool_observability(Request(scope), current_info)
    assert current_info == {
        "rust_request_spool": True,
        "rust_request_spool_body_bytes": 20_000_000,
        "rust_request_spool_memory_peak_bytes": 65_536,
        "rust_request_spool_local_disk_bytes": 20_000_000,
        "rust_request_spool_resource_wait_ms": 19,
        "rust_request_spool_final_tier": "local_disk",
    }

    spoofed_scope = dict(scope)
    spoofed_scope["headers"] = [
        (b"x-uni-api-rust-control-token", b"wrong"),
        (b"x-uni-api-rust-request-spool-body-bytes", b"999"),
    ]
    spoofed = {}
    _record_rust_request_spool_observability(Request(spoofed_scope), spoofed)
    assert spoofed == {}


def test_resource_only_body_admission_requires_complete_trusted_local_spool(
    monkeypatch,
):
    monkeypatch.setattr("uni_api.runtime.RUST_RESPONSES_CONTROL_TOKEN", "control-token")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [
            (b"x-uni-api-rust-control-token", b"control-token"),
            (b"x-uni-api-rust-request-spool-body-bytes", b"20000000"),
            (b"x-uni-api-rust-request-spool-local-disk-bytes", b"20000000"),
            (b"x-uni-api-rust-request-spool-final-tier", b"local_disk"),
        ],
    }

    assert _is_trusted_rust_spooled_request(scope) is True

    for header_index, replacement in (
        (0, (b"x-uni-api-rust-control-token", b"wrong")),
        (2, (b"x-uni-api-rust-request-spool-local-disk-bytes", b"19999999")),
        (3, (b"x-uni-api-rust-request-spool-final-tier", b"invalid")),
    ):
        untrusted = dict(scope)
        untrusted["headers"] = list(scope["headers"])
        untrusted["headers"][header_index] = replacement
        assert _is_trusted_rust_spooled_request(untrusted) is False


class _FakeExecution:
    def __init__(self):
        self.current_info = {}
        self.commits = []
        self.attempts = [
            SimpleNamespace(routing_attempt_id="attempt-1"),
            SimpleNamespace(routing_attempt_id="attempt-2"),
        ]

    async def _run_attempts(self, *, execute_attempt):
        for attempt in self.attempts:
            try:
                return await execute_attempt(attempt)
            except RuntimeError:
                continue
        return Response("exhausted", status_code=502)

    async def _build_rust_stream_plan(self, attempt):
        return {
            "kind": "plan",
            "attempt_id": attempt.routing_attempt_id,
            "url": "https://provider.example/v1/responses",
        }

    async def _resolve_rust_stream_outcome(self, _attempt, outcome):
        if outcome["kind"] == "http_error":
            raise RuntimeError("retry")
        return Response("", status_code=200)

    async def _observe_rust_stream_commit(self, attempt, observation):
        self.commits.append((attempt.routing_attempt_id, dict(observation)))


def test_control_plane_coordinates_retry_commit_and_completion():
    async def scenario():
        plane = RustResponsesControlPlane()
        execution = _FakeExecution()
        session = await plane.create(execution)

        first = await session.next_message()
        assert first["attempt_id"] == "attempt-1"
        second = await session.advance(
            attempt_id="attempt-1",
            outcome={"kind": "http_error"},
        )
        assert second["attempt_id"] == "attempt-2"

        ack = await session.observe_commit(
            attempt_id="attempt-2",
            observation={"commit_reason": "real_output"},
        )
        assert ack == {"kind": "ack", "attempt_id": "attempt-2"}
        final = await session.complete(
            attempt_id="attempt-2",
            outcome={"kind": "completed"},
        )
        assert final["kind"] == "final"
        assert final["status_code"] == 200
        assert execution.commits == [
            ("attempt-2", {"commit_reason": "real_output"})
        ]

        body = session.control_body()
        with pytest.raises(StopAsyncIteration):
            await body.__anext__()
        await plane.release(session)
        assert plane.snapshot() == {"active_sessions": 0}

    asyncio.run(scenario())


def test_control_plane_rejects_stale_attempt_outcomes():
    async def scenario():
        plane = RustResponsesControlPlane()
        session = await plane.create(_FakeExecution())
        await session.next_message()
        with pytest.raises(RustResponsesControlError, match="stale attempt"):
            await session.advance(
                attempt_id="wrong-attempt",
                outcome={"kind": "http_error"},
            )
        await plane.release(session)

    asyncio.run(scenario())
