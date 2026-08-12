import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.responses import JSONResponse, StreamingResponse

import uni_api.runtime as runtime
from uni_api.admission import (
    RequestAdmissionController,
    UpstreamResponseBudgetExhausted,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.runtime import ResponsesRequestExecution, _prime_responses_upstream_stream
from uni_api.observability.responses_stream import ResponsesStreamDiagnostics
from uni_api.streaming.bounded_queue import (
    ObservedStreamChunk,
    ObservedStreamFrame,
    StreamQueuePutTimeout,
)
from uni_api.streaming.logging_response import LoggingStreamingResponse
from uni_api.streaming.sse import IncrementalSSEParser


class _ControlledResponsesExecution(ResponsesRequestExecution):
    def __init__(self, body_iterator, *, disconnect_event=None):
        super().__init__(
            handler=None,
            http_request=None,
            request_data=SimpleNamespace(stream=True),
            api_index=0,
            background_tasks=None,
            endpoint="/v1/responses",
            config={},
            current_info={
                "request_id": "req-bounded-stream",
                "api_key": "redacted",
            },
            disconnect_event=disconnect_event,
            request_id="req-bounded-stream",
            request_model_name="gpt-test",
            plan=None,
            runner=None,
        )
        self._controlled_body_iterator = body_iterator

    async def _run_attempts(self):
        return StreamingResponse(
            self._controlled_body_iterator,
            media_type="text/event-stream",
        )


class _ControlledFiniteResponseExecution(_ControlledResponsesExecution):
    async def _run_attempts(self):
        return JSONResponse(
            status_code=400,
            content={"error": "bounded-worker-response"},
        )


async def _wait_until(predicate, *, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def test_observed_responses_frame_borrows_wire_until_queue_handoff():
    wire = b'event: response.output_text.delta\ndata: {"delta":"x"}\n\n'

    frame = runtime._observed_responses_stream_chunk(
        wire,
        None,
        event_type="response.output_text.delta",
        semantic_outcome="nonterminal",
    )

    assert isinstance(frame, ObservedStreamFrame)
    assert frame.data is wire
    assert frame.sse_metadata_complete is True


def test_responses_retry_payload_releases_after_early_response_start():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        lease = await controller.acquire(initial_body_bytes=100)

        async def body():
            if False:
                yield b""

        execution = _ControlledResponsesExecution(body())
        execution.http_request = SimpleNamespace(
            scope={
                "state": {
                    "uni_api_admission_lease": lease,
                    "uni_api_body_response_started": True,
                }
            }
        )
        await execution._release_request_retry_payload()

        assert execution.request_data is None
        assert lease.reserved_body_bytes == 0
        assert execution.http_request.scope["state"][
            "uni_api_release_body_at_response_start"
        ] is True
        await lease.release()

    asyncio.run(scenario())


def test_logging_response_reuses_small_metadata_chunk_for_asgi_write():
    async def scenario():
        chunk = ObservedStreamChunk(
            b'event: response.output_text.delta\ndata: {"delta":"x"}\n\n',
            event_type="response.output_text.delta",
            semantic_outcome="nonterminal",
            sse_metadata_complete=True,
        )

        async def body():
            yield chunk

        sent_bodies = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            body_bytes = message.get("body", b"")
            if body_bytes:
                sent_bodies.append(body_bytes)

        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info={"request_id": "zero-copy-asgi"},
        )
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )
        return chunk, sent_bodies

    chunk, sent_bodies = asyncio.run(scenario())
    assert sent_bodies == [chunk]
    assert sent_bodies[0] is chunk


def test_responses_stream_queue_backpressures_producer_without_dropping(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 1)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 16)
        monkeypatch.setattr(
            runtime,
            "RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS",
            1.0,
        )
        producer_finished = asyncio.Event()
        disconnect_event = asyncio.Event()

        async def upstream_body():
            try:
                yield b"first"
                yield b"second"
                yield b"third"
            finally:
                producer_finished.set()

        execution = _ControlledResponsesExecution(
            upstream_body(),
            disconnect_event=disconnect_event,
        )
        response = await execution._run_stream()
        queue = execution.stream_output_queue
        assert queue is not None

        await _wait_until(
            lambda: queue.snapshot().items == 1
            and queue.snapshot().waiting_putters == 1
        )
        assert queue.snapshot().items == 1
        snapshot = queue.snapshot()
        assert snapshot.bytes == len(b"first")
        assert snapshot.inflight_bytes == len(b"first")
        assert snapshot.queued_bytes == 0
        assert not producer_finished.is_set()

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        assert chunks == [b"first", b"second", b"third"]
        assert producer_finished.is_set()
        assert not disconnect_event.is_set()
        assert execution.current_info["stream_queue_blocked_puts"] >= 1
        assert execution.current_info["stream_queue_peak_items"] == 1
        assert execution.current_info["stream_queue_peak_bytes"] <= 16
        assert runtime.runtime_gauges.snapshot()["stream_queue_active"] == 0

    asyncio.run(scenario())


def test_finite_worker_response_budget_rejection_retires_queue_ownership():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        lease = await controller.acquire(initial_body_bytes=1024)
        token = bind_request_admission_lease(lease)
        execution = _ControlledFiniteResponseExecution(iter(()))
        try:
            with pytest.raises(UpstreamResponseBudgetExhausted):
                await execution._run_stream()
            queue = execution.stream_output_queue
            assert queue is not None
            snapshot = queue.snapshot()
            assert snapshot.closed is True
            assert snapshot.items == 0
            assert snapshot.bytes == 0
            assert runtime.runtime_gauges.snapshot()["stream_queue_active"] == 0
        finally:
            reset_request_admission_lease(token)
            await lease.release()
        assert controller.snapshot()["reserved_retained_bytes"] == 0

    asyncio.run(scenario())


def test_responses_stream_splits_large_transport_chunk_without_changing_bytes(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 4)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 16)
        original = b"0123456789" * 5

        async def upstream_body():
            yield original

        execution = _ControlledResponsesExecution(upstream_body())
        response = await execution._run_stream()
        chunks = [chunk async for chunk in response.body_iterator]

        assert b"".join(chunks) == original
        assert all(len(chunk) <= 16 for chunk in chunks)
        assert execution.current_info["stream_queue_peak_bytes"] <= 16

    asyncio.run(scenario())


def test_responses_stream_local_iterator_close_does_not_forge_disconnect(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 1)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 16)
        producer_closed = asyncio.Event()
        disconnect_event = asyncio.Event()

        async def upstream_body():
            try:
                yield b"first"
                await asyncio.Event().wait()
            finally:
                producer_closed.set()

        execution = _ControlledResponsesExecution(
            upstream_body(),
            disconnect_event=disconnect_event,
        )
        response = await execution._run_stream()
        iterator = response.body_iterator
        assert await anext(iterator) == b"first"

        await iterator.aclose()

        assert not disconnect_event.is_set()
        assert producer_closed.is_set()
        assert execution.stream_output_queue.snapshot().bytes == 0
        assert runtime.runtime_gauges.snapshot()["stream_queue_active"] == 0

    asyncio.run(scenario())


def test_responses_stream_stalled_consumer_times_out_and_aborts(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 1)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 16)
        monkeypatch.setattr(
            runtime,
            "RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS",
            0.01,
        )
        producer_closed = asyncio.Event()
        disconnect_event = asyncio.Event()

        async def upstream_body():
            try:
                yield b"first"
                yield b"second"
                yield b"third"
            finally:
                producer_closed.set()

        execution = _ControlledResponsesExecution(
            upstream_body(),
            disconnect_event=disconnect_event,
        )
        response = await execution._run_stream()
        queue = execution.stream_output_queue
        assert queue is not None
        await _wait_until(lambda: queue.snapshot().put_timeouts == 1)

        iterator = response.body_iterator
        assert await anext(iterator) == b"first"
        with pytest.raises(StreamQueuePutTimeout):
            await anext(iterator)

        assert producer_closed.is_set()
        assert not disconnect_event.is_set()
        assert execution.current_info["stream_outcome"] == "local_backpressure_abort"
        assert execution.current_info["stream_error_status_code"] == 503
        assert "status_code" not in execution.current_info
        assert execution.current_info["stream_queue_put_timeouts"] == 1
        assert runtime.runtime_gauges.snapshot()["stream_queue_active"] == 0

    asyncio.run(scenario())


def test_responses_precommit_buffer_has_total_item_and_byte_limits(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_PRECOMMIT_MAX_ITEMS", 2)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_PRECOMMIT_MAX_BYTES", 1024)

        async def upstream_chunks():
            for sequence in range(3):
                yield (
                    "event: response.created\n"
                    f'data: {{"type":"response.created","sequence_number":{sequence}}}\n\n'
                ).encode()

        with pytest.raises(HTTPException) as exc_info:
            await _prime_responses_upstream_stream(upstream_chunks())
        assert exc_info.value.status_code == 502
        assert "precommit buffer limit" in str(exc_info.value.detail)

    asyncio.run(scenario())


def test_responses_committing_frame_can_exceed_precommit_byte_limit(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_PRECOMMIT_MAX_BYTES", 64)
        event = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta",'
            '"delta":"this frame commits the response"}\n\n'
        ).encode()

        async def upstream_chunks():
            yield event

        buffered, committed = await _prime_responses_upstream_stream(
            upstream_chunks(),
            precommit_semantic_guard=True,
        )
        try:
            assert committed is True
            assert b"".join(buffered) == event
        finally:
            await buffered.clear()

    asyncio.run(scenario())


def test_responses_precommit_rejects_incomplete_sse_eof():
    async def scenario():
        async def upstream_chunks():
            yield b'event: response.created\ndata: {"type":"response.created"}'

        with pytest.raises(HTTPException) as exc_info:
            await _prime_responses_upstream_stream(upstream_chunks())
        assert exc_info.value.status_code == 502
        assert "incomplete trailing event" in str(exc_info.value.detail)

    asyncio.run(scenario())


def test_responses_transparent_precommit_commits_response_created_without_keepalive():
    async def scenario():
        created = (
            "event: response.created\n"
            'data: {"type":"response.created","response":{"status":"in_progress"}}\n\n'
        ).encode()
        callbacks = []

        async def upstream_chunks():
            yield created

        async def emit_keepalive(chunk):
            callbacks.append(chunk)
            return True

        buffered, committed = await _prime_responses_upstream_stream(
            upstream_chunks(),
            precommit_semantic_guard=False,
            precommit_keepalive_callback=emit_keepalive,
        )
        try:
            assert committed is True
            assert b"".join(buffered) == created
            assert callbacks == []
        finally:
            await buffered.clear()

    asyncio.run(scenario())


def test_responses_precommit_guard_is_only_enabled_for_codex_engine():
    assert runtime._responses_precommit_semantic_guard("gpt") is False
    assert runtime._responses_precommit_semantic_guard("codex") is True


def test_responses_guarded_precommit_keeps_response_created_private_and_emits_keepalive():
    async def scenario():
        created = (
            "event: response.created\n"
            'data: {"type":"response.created","response":{"status":"in_progress"}}\n\n'
        ).encode()
        delta = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
        ).encode()
        callbacks = []

        async def upstream_chunks():
            yield created + delta

        async def emit_keepalive(chunk):
            callbacks.append(chunk)
            return True

        buffered, committed = await _prime_responses_upstream_stream(
            upstream_chunks(),
            precommit_semantic_guard=True,
            precommit_keepalive_callback=emit_keepalive,
        )
        try:
            assert committed is True
            assert b"".join(buffered) == created + delta
            assert callbacks == [None]
        finally:
            await buffered.clear()

    asyncio.run(scenario())


def test_responses_precommit_preserves_split_utf8_pending_bytes_after_commit():
    async def scenario():
        first_event = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
        ).encode()
        partial_next = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"你'
        ).encode()
        split_at = len(partial_next) - 1

        async def upstream_chunks():
            yield first_event + partial_next[:split_at]
            yield partial_next[split_at:] + b'"}\n\n'

        buffered, committed = await _prime_responses_upstream_stream(
            upstream_chunks()
        )

        assert committed is True
        combined = b"".join(buffered)
        assert first_event in combined
        assert partial_next[:split_at] in combined

    asyncio.run(scenario())


def test_responses_precommit_preserves_split_crlf_parser_state_after_commit():
    async def scenario():
        committed = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
        ).encode()
        partial_next = b"event: keepalive\r"

        async def upstream_chunks():
            yield committed + partial_next
            yield b"\n\r\n"

        upstream = upstream_chunks()
        buffered, did_commit = await _prime_responses_upstream_stream(upstream)
        assert did_commit is True
        assert b"".join(buffered).endswith(b"\r")

        resumed = IncrementalSSEParser()
        events = []
        for chunk in buffered:
            events.extend(resumed.feed(chunk))
        assert len(events) == 1
        events.extend(resumed.feed(await anext(upstream)))
        assert events == [
            committed.decode().strip(),
            "event: keepalive",
        ]
        assert resumed.finish() == []

    asyncio.run(scenario())


def test_responses_precommit_sampled_request_reports_pipeline_phases():
    async def scenario():
        observed = []
        diagnostics = ResponsesStreamDiagnostics(
            current_info={"upstream_attempts": [{}]},
            attempt_index=0,
            logical_authority="benchmark.invalid",
            proxy_configured=False,
            phase_sampled=True,
            phase_observer=lambda phase, **metrics: observed.append(
                (phase, metrics)
            ),
        )

        async def upstream_chunks():
            yield (
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            ).encode()

        buffered, committed = await _prime_responses_upstream_stream(
            upstream_chunks(),
            diagnostics=diagnostics,
        )
        try:
            assert committed is True
        finally:
            await buffered.clear()
        return observed

    observed = asyncio.run(scenario())
    phases = {phase for phase, _metrics in observed}
    assert {
        "socket_receive",
        "sse_frame",
        "json_parse",
        "observer_hash",
    }.issubset(phases)
    assert all(
        set(metrics) == {"wall_ns", "cpu_ns", "bytes_count", "events"}
        and all(isinstance(value, int) for value in metrics.values())
        for _phase, metrics in observed
    )


def test_responses_queue_and_logging_complete_normal_stream_without_disconnect(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 2)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 64)
        disconnect_event = asyncio.Event()

        async def upstream_body():
            yield b"data: one\n\n"
            yield b"data: two\n\n"

        execution = _ControlledResponsesExecution(
            upstream_body(),
            disconnect_event=disconnect_event,
        )
        inner = await execution._run_stream()
        response = LoggingStreamingResponse(
            inner.body_iterator,
            media_type="text/event-stream",
            current_info=execution.current_info,
            disconnect_event=disconnect_event,
        )
        sent = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

        body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert body == b"data: one\n\ndata: two\n\n"
        assert sent[-1]["more_body"] is False
        assert execution.current_info["stream_outcome"] == "completed"
        assert not disconnect_event.is_set()
        assert runtime.runtime_gauges.snapshot()["stream_queue_active"] == 0

    asyncio.run(scenario())


def test_logging_preserves_local_backpressure_instead_of_misclassifying_disconnect(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 1)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 32)
        monkeypatch.setattr(
            runtime,
            "RESPONSES_STREAM_QUEUE_PUT_TIMEOUT_SECONDS",
            0.01,
        )
        disconnect_event = asyncio.Event()

        async def upstream_body():
            yield b"data: first\n\n"
            yield b"data: second\n\n"

        execution = _ControlledResponsesExecution(
            upstream_body(),
            disconnect_event=disconnect_event,
        )
        inner = await execution._run_stream()
        response = LoggingStreamingResponse(
            inner.body_iterator,
            media_type="text/event-stream",
            current_info=execution.current_info,
            disconnect_event=disconnect_event,
        )
        sent = []
        delayed_body = False

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            nonlocal delayed_body
            sent.append(message)
            if (
                message["type"] == "http.response.body"
                and message.get("body")
                and not delayed_body
            ):
                delayed_body = True
                await asyncio.sleep(0.03)

        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

        body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert b"event: error\n" in body
        assert sent[-1]["more_body"] is False
        assert execution.current_info["stream_outcome"] == "local_backpressure_abort"
        assert execution.current_info["status_code"] == 200
        assert execution.current_info["wire_status_code"] == 200
        assert execution.current_info["stream_error_status_code"] == 503
        assert execution.current_info.get("downstream_disconnected") is not True
        assert not disconnect_event.is_set()
        assert runtime.runtime_gauges.snapshot()["stream_queue_active"] == 0

    asyncio.run(scenario())


def test_logging_records_stream_worker_failure_after_response_started(monkeypatch):
    async def scenario():
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_ITEMS", 4)
        monkeypatch.setattr(runtime, "RESPONSES_STREAM_QUEUE_MAX_BYTES", 128)

        async def upstream_body():
            yield b"data: first\n\n"
            raise RuntimeError("upstream iterator exploded")

        execution = _ControlledResponsesExecution(upstream_body())
        inner = await execution._run_stream()
        response = LoggingStreamingResponse(
            inner.body_iterator,
            media_type="text/event-stream",
            current_info=execution.current_info,
        )
        sent = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )

        body = b"".join(message.get("body", b"") for message in sent)
        assert b"event: error\n" in body
        assert execution.current_info["status_code"] == 200
        assert execution.current_info["wire_status_code"] == 200
        assert execution.current_info["stream_error_status_code"] == 500
        assert execution.current_info["error_type"] == "RuntimeError"
        assert execution.current_info["stream_outcome"] == "stream_worker_error"
        assert execution.current_info["success"] is False

    asyncio.run(scenario())
