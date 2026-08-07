import asyncio
import json
from contextlib import suppress
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse

from uni_api.admission import (
    AdmissionRejected,
    RequestAdmissionController,
    bind_request_admission_lease,
    get_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.middleware.admission import RequestAdmissionMiddleware
from uni_api.middleware.request_decompression import (
    RequestBodyDecompressionMiddleware,
    RequestBodyTooLarge,
)
from uni_api.observability.middleware import StatsMiddleware, StatsMiddlewareDependencies
from uni_api.observability.request_context import (
    get_request_info,
    reset_request_info,
    set_request_info,
)
from uni_api.streaming.logging_response import (
    LoggingStreamingResponse,
    timed_out_io_task_snapshot,
)
from uni_api.streaming.sse import IncrementalSSEParser, SSEProtocolError
from uni_api.upstream.client_pool import UpstreamAdmissionRejected
from uni_api.upstream.responses_errors import responses_failure_error
import uni_api.runtime as runtime
import uni_api.streaming.cleanup as stream_cleanup
import upstream as upstream_module


def _scope(*, method: str = "GET", path: str = "/v1/test") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "state": {},
    }


async def _never_receive() -> dict:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def test_admission_bypass_masks_inherited_released_lease_and_preserves_json_wire():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        stale_lease = await controller.acquire()
        await stale_lease.release()
        token = bind_request_admission_lease(stale_lease)
        payload = b'{"runtime":"ok"}'
        sent = []

        async def body():
            yield payload

        async def send(message):
            sent.append(dict(message))

        async def app(scope, receive, send):
            assert get_request_admission_lease() is None
            response = LoggingStreamingResponse(
                body(),
                media_type="application/json",
                headers={"content-length": str(len(payload))},
                current_info={"start_time": 0},
            )
            await response(scope, receive, send)

        middleware = RequestAdmissionMiddleware(
            app,
            controller=controller,
            bypass=lambda _scope: True,
        )
        try:
            await middleware(
                _scope(path="/v1/observability/runtime"),
                _never_receive,
                send,
            )
        finally:
            reset_request_admission_lease(token)

        body_messages = [
            message
            for message in sent
            if message["type"] == "http.response.body"
        ]
        assert (
            b"".join(message.get("body", b"") for message in body_messages)
            == payload
        )
        assert body_messages[-1].get("more_body") is False

    asyncio.run(scenario())


def test_released_lease_cannot_make_usage_telemetry_truncate_json_wire():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        stale_lease = await controller.acquire()
        await stale_lease.release()
        token = bind_request_admission_lease(stale_lease)
        payload = b'{"runtime":"ok"}'
        sent = []
        current_info = {"start_time": 0}

        async def body():
            yield payload

        async def send(message):
            sent.append(dict(message))

        response = LoggingStreamingResponse(
            body(),
            media_type="application/json",
            headers={"content-length": str(len(payload))},
            current_info=current_info,
        )
        try:
            await response(_scope(), _never_receive, send)
        finally:
            reset_request_admission_lease(token)

        body_messages = [
            message
            for message in sent
            if message["type"] == "http.response.body"
        ]
        assert (
            b"".join(message.get("body", b"") for message in body_messages)
            == payload
        )
        assert body_messages[-1].get("more_body") is False
        assert current_info["usage_parse_error"] == "RuntimeError"

    asyncio.run(scenario())


def test_released_lease_cannot_make_usage_telemetry_replace_sse_wire():
    async def scenario():
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        stale_lease = await controller.acquire()
        await stale_lease.release()
        token = bind_request_admission_lease(stale_lease)
        payload = (
            b'data: {"choices":[],"usage":{"prompt_tokens":1,'
            b'"completion_tokens":1}}\n\n'
        )
        sent = []
        current_info = {"start_time": 0}

        async def body():
            yield payload

        async def send(message):
            sent.append(dict(message))

        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        try:
            await response(_scope(), _never_receive, send)
        finally:
            reset_request_admission_lease(token)

        wire = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert wire == payload
        assert b"event: error" not in wire
        assert current_info["usage_parse_error"] == "RuntimeError"

    asyncio.run(scenario())


def test_logging_stream_disconnect_closes_iterator_and_lifecycle_once():
    async def scenario():
        first_body_sent = asyncio.Event()
        disconnect_event = asyncio.Event()
        body_closed = False
        lifecycle_calls = []
        sent = []

        async def body():
            nonlocal body_closed
            try:
                yield b"data: first\n\n"
                await asyncio.Event().wait()
            finally:
                body_closed = True

        async def receive():
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                first_body_sent.set()

        async def lifecycle_close(info):
            await asyncio.sleep(0)
            lifecycle_calls.append(dict(info))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
            lifecycle_close=lifecycle_close,
            disconnect_event=disconnect_event,
        )
        response_task = asyncio.create_task(response(_scope(), receive, send))
        await asyncio.wait_for(first_body_sent.wait(), timeout=1)
        disconnect_event.set()
        await asyncio.wait_for(response_task, timeout=1)
        await response.close()

        assert body_closed is True
        assert len(lifecycle_calls) == 1
        assert current_info["stream_outcome"] == "downstream_disconnected"
        assert current_info["downstream_disconnected"] is True
        assert current_info["error_type"] == "downstream_disconnect"
        assert current_info["success"] is False
        assert not any(
            message["type"] == "http.response.body"
            and message.get("body") == b""
            and message.get("more_body") is False
            for message in sent
        )

    asyncio.run(scenario())


def test_logging_stream_task_cancellation_closes_iterator_and_lifecycle_once():
    async def scenario():
        first_body_sent = asyncio.Event()
        body_closed = False
        lifecycle_calls = []

        async def body():
            nonlocal body_closed
            try:
                yield b"data: first\n\n"
                await asyncio.Event().wait()
            finally:
                body_closed = True

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body_sent.set()

        async def lifecycle_close(info):
            lifecycle_calls.append(dict(info))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
            lifecycle_close=lifecycle_close,
        )
        response_task = asyncio.create_task(response(_scope(), _never_receive, send))
        await asyncio.wait_for(first_body_sent.wait(), timeout=1)
        response_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response_task

        assert body_closed is True
        assert len(lifecycle_calls) == 1
        assert current_info["stream_outcome"] == "cancelled"
        assert current_info["error_type"] == "CancelledError"
        assert current_info["success"] is False

    asyncio.run(scenario())


def test_logging_segments_large_body_and_preserves_explicit_content_length():
    async def scenario():
        body_closed = False
        sent = []

        async def body():
            nonlocal body_closed
            try:
                yield b"abcdefghij"
                yield b""
            finally:
                body_closed = True

        async def send(message):
            sent.append(message)

        response = LoggingStreamingResponse(
            body(),
            headers={
                "content-type": "application/octet-stream",
                "content-length": "10",
            },
            current_info={"start_time": 0},
            downstream_chunk_bytes=3,
        )
        await response(_scope(), _never_receive, send)

        start = next(
            message
            for message in sent
            if message["type"] == "http.response.start"
        )
        headers = dict(start["headers"])
        assert headers[b"content-length"] == b"10"
        data_messages = [
            message
            for message in sent
            if message["type"] == "http.response.body"
            and message.get("more_body") is True
            and message.get("body")
        ]
        assert [message["body"] for message in data_messages] == [
            b"abc",
            b"def",
            b"ghi",
            b"j",
        ]
        assert b"".join(message["body"] for message in data_messages) == b"abcdefghij"
        assert body_closed is True
        assert sent[-1]["more_body"] is False

    asyncio.run(scenario())


def test_downstream_writer_reuses_one_task_and_preserves_per_frame_backpressure():
    async def scenario():
        timeline = []
        sent = []
        writer_tasks = set()

        async def body():
            timeline.append("yield:a")
            yield b"a"
            timeline.append("yield:b")
            yield b"b"
            timeline.append("body:done")

        async def send(message):
            task = asyncio.current_task()
            assert task is not None
            writer_tasks.add(task)
            sent.append(dict(message))
            if (
                message["type"] == "http.response.body"
                and message.get("more_body")
                and message.get("body")
            ):
                timeline.append(f"send:{message['body'].decode('ascii')}")

        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info={"start_time": 0},
            observe_usage=False,
        )
        await response(_scope(), _never_receive, send)

        assert timeline == [
            "yield:a",
            "send:a",
            "yield:b",
            "send:b",
            "body:done",
        ]
        assert len(writer_tasks) == 1
        assert [
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        ] == [b"a", b"b", b""]

    asyncio.run(scenario())


def test_logging_stream_drops_previous_chunk_before_waiting_for_next_item():
    async def scenario():
        waiting_for_next = asyncio.Event()

        class Body:
            def __init__(self, item):
                self.item = item
                self.first = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.first:
                    self.first = False
                    item = self.item
                    self.item = None
                    return item
                waiting_for_next.set()
                await asyncio.Event().wait()

        chunk = b"x" * (1024 * 1024)
        response = LoggingStreamingResponse(
            Body(chunk),
            media_type="application/octet-stream",
            current_info={"start_time": 0},
        )
        chunk = None

        async def send(_message):
            return None

        task = asyncio.create_task(response._stream_response_body(send))
        await asyncio.wait_for(waiting_for_next.wait(), timeout=1)
        frame = task.get_coro().cr_frame
        assert frame is not None
        assert frame.f_locals.get("chunk") is None
        assert frame.f_locals.get("segment") is None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_downstream_write_timeout_is_bounded_and_does_not_forge_disconnect():
    async def scenario():
        release_stuck_send = asyncio.Event()
        cancellation_seen = asyncio.Event()
        body_closed = False
        lifecycle_calls = []
        nonempty_writes = []
        disconnect_event = asyncio.Event()

        async def body():
            nonlocal body_closed
            try:
                yield b"abcdefgh"
                raise AssertionError("timeout must stop before the next chunk")
            finally:
                body_closed = True

        async def send(message):
            if message["type"] != "http.response.body" or not message.get("body"):
                return
            nonempty_writes.append(message["body"])
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_stuck_send.wait()

        async def lifecycle_close(info):
            lifecycle_calls.append(dict(info))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="application/octet-stream",
            current_info=current_info,
            lifecycle_close=lifecycle_close,
            disconnect_event=disconnect_event,
            downstream_write_timeout_seconds=0.01,
            downstream_chunk_bytes=4,
        )
        await asyncio.wait_for(
            response(_scope(), _never_receive, send),
            timeout=1,
        )

        assert cancellation_seen.is_set()
        assert nonempty_writes == [b"abcd"]
        assert body_closed is True
        assert len(lifecycle_calls) == 1
        assert current_info["stream_outcome"] == "downstream_write_timeout"
        assert current_info["success"] is False
        assert not disconnect_event.is_set()
        assert timed_out_io_task_snapshot()["pending"] >= 1

        release_stuck_send.set()
        async with asyncio.timeout(1):
            while timed_out_io_task_snapshot()["pending"]:
                await asyncio.sleep(0)

    asyncio.run(scenario())


def test_stuck_iterator_close_retains_request_ownership_until_cleanup_finishes(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(
            stream_cleanup,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
        )
        disconnect_event = asyncio.Event()
        release_close = asyncio.Event()
        first_sent = asyncio.Event()
        lifecycle_calls = []

        class StuckCloseBody:
            def __init__(self):
                self.sent = False
                self.retained = bytearray(700)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.sent:
                    self.sent = True
                    return b"data: first\n\n"
                await asyncio.Event().wait()

            async def aclose(self):
                try:
                    await release_close.wait()
                except asyncio.CancelledError:
                    await release_close.wait()

        async def lifecycle_close(info):
            lifecycle_calls.append(dict(info))

        async def app(scope, receive, send):
            request_lease = get_request_admission_lease()
            assert request_lease is not None
            await request_lease.reserve_response_bytes(700)
            response = LoggingStreamingResponse(
                StuckCloseBody(),
                media_type="text/event-stream",
                current_info={"start_time": 0},
                lifecycle_close=lifecycle_close,
                disconnect_event=disconnect_event,
            )
            await response(scope, receive, send)

        middleware = RequestAdmissionMiddleware(app, controller=controller)

        async def receive():
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_sent.set()
                disconnect_event.set()

        task = asyncio.create_task(middleware(_scope(), receive, send))
        await asyncio.wait_for(first_sent.wait(), timeout=1)
        await asyncio.sleep(0.05)
        assert not task.done()
        snapshot = controller.snapshot()
        assert snapshot["active"] == 1
        assert snapshot["reserved_response_bytes"] == 700
        assert snapshot["deferred_memory_requests"] == 0
        assert stream_cleanup.background_stream_cleanup_snapshot()["pending"] == 0

        release_close.set()
        await asyncio.wait_for(task, timeout=1)
        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_retained_bytes"] == 0
        assert snapshot["deferred_memory_requests"] == 0
        assert len(lifecycle_calls) == 1
        assert lifecycle_calls[0]["stream_outcome"] == "downstream_disconnected"

    asyncio.run(scenario())


def test_isolated_transport_cleanup_releases_active_but_defers_request_memory(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(
            stream_cleanup,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        release_cleanup = asyncio.Event()
        cleanup_started = asyncio.Event()
        baseline_pending = stream_cleanup.background_stream_cleanup_snapshot()[
            "pending"
        ]

        async def noncooperative_transport_cleanup():
            retained = bytearray(700)
            cleanup_started.set()
            try:
                await release_cleanup.wait()
            except asyncio.CancelledError:
                await release_cleanup.wait()
            assert len(retained) == 700

        async def app(scope, receive, send):
            request_lease = get_request_admission_lease()
            assert request_lease is not None
            await request_lease.reserve_response_bytes(700)
            completed = await stream_cleanup.await_isolated_transport_cleanup_safely(
                noncooperative_transport_cleanup(),
                label="isolated transport test",
            )
            assert completed is False

        middleware = RequestAdmissionMiddleware(app, controller=controller)
        task = asyncio.create_task(
            middleware(
                _scope(),
                _never_receive,
                lambda _message: asyncio.sleep(0),
            )
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        await asyncio.wait_for(task, timeout=1)

        snapshot = controller.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["reserved_response_bytes"] == 700
        assert snapshot["deferred_memory_requests"] == 1
        assert stream_cleanup.background_stream_cleanup_snapshot()["pending"] == (
            baseline_pending + 1
        )

        second = await controller.try_acquire()
        assert second is not None
        await second.release()

        release_cleanup.set()
        async with asyncio.timeout(1):
            while (
                stream_cleanup.background_stream_cleanup_snapshot()["pending"]
                > baseline_pending
            ):
                await asyncio.sleep(0)
        snapshot = controller.snapshot()
        assert snapshot["reserved_retained_bytes"] == 0
        assert snapshot["deferred_memory_requests"] == 0

    asyncio.run(scenario())


def test_isolated_cleanup_handoff_is_atomic_under_repeated_cancellation(
    monkeypatch,
):
    async def scenario():
        monkeypatch.setattr(
            stream_cleanup,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=1024,
            body_budget_bytes=1024,
            max_response_bytes=1024,
        )
        lease = await controller.acquire()
        await lease.reserve_response_bytes(700)
        token = bind_request_admission_lease(lease)
        release_cleanup = asyncio.Event()
        deferral_started = asyncio.Event()
        allow_deferral = asyncio.Event()
        baseline_pending = stream_cleanup.background_stream_cleanup_snapshot()[
            "pending"
        ]
        original_defer = lease.defer_memory_release

        async def delayed_defer():
            deferral_started.set()
            await allow_deferral.wait()
            return await original_defer()

        monkeypatch.setattr(lease, "defer_memory_release", delayed_defer)

        async def noncooperative_cleanup():
            retained = bytearray(700)
            try:
                await release_cleanup.wait()
            except asyncio.CancelledError:
                await release_cleanup.wait()
            assert len(retained) == 700

        try:
            owner = asyncio.create_task(
                stream_cleanup.await_isolated_transport_cleanup_safely(
                    noncooperative_cleanup(),
                    label="cancelled isolated handoff",
                )
            )
            await asyncio.wait_for(deferral_started.wait(), timeout=1)
            owner.cancel()
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done()
            snapshot = controller.snapshot()
            assert snapshot["active"] == 1
            assert snapshot["reserved_response_bytes"] == 700
            assert (
                stream_cleanup.background_stream_cleanup_snapshot()["pending"]
                == baseline_pending
            )

            allow_deferral.set()
            with pytest.raises(asyncio.CancelledError):
                await owner
            await lease.release()
            snapshot = controller.snapshot()
            assert snapshot["active"] == 0
            assert snapshot["deferred_memory_requests"] == 1
            assert snapshot["reserved_response_bytes"] == 700
            assert (
                stream_cleanup.background_stream_cleanup_snapshot()["pending"]
                == baseline_pending + 1
            )

            release_cleanup.set()
            async with asyncio.timeout(1):
                while (
                    stream_cleanup.background_stream_cleanup_snapshot()[
                        "pending"
                    ]
                    > baseline_pending
                ):
                    await asyncio.sleep(0)
            assert controller.snapshot()["reserved_retained_bytes"] == 0
        finally:
            reset_request_admission_lease(token)

    asyncio.run(scenario())


def test_stats_write_failure_is_visible_to_lifecycle_observability():
    async def scenario():
        lifecycle_calls = []

        async def body():
            yield b"ok"

        async def update_stats(_info):
            return False

        async def lifecycle_close(info):
            lifecycle_calls.append(dict(info))

        response = LoggingStreamingResponse(
            body(),
            media_type="application/json",
            current_info={"start_time": 0},
            update_stats=update_stats,
            lifecycle_close=lifecycle_close,
        )
        await response(_scope(), _never_receive, lambda _message: asyncio.sleep(0))

        assert len(lifecycle_calls) == 1
        assert lifecycle_calls[0]["stats_write_failed"] is True

    asyncio.run(scenario())


def test_legacy_stream_channel_result_is_finalized_only_at_terminal_outcome(
    monkeypatch,
):
    async def scenario():
        recorded = []
        response_attempt_outcomes = []

        class ResponseMemoryLease:
            def finish_response_attempt(self, *, outcome, keep_active=False):
                response_attempt_outcomes.append((outcome, keep_active))

        response_memory_lease = ResponseMemoryLease()
        monkeypatch.setattr(
            upstream_module,
            "get_request_admission_lease",
            lambda: None,
        )

        def record(*args, success, **kwargs):
            recorded.append(success)

        monkeypatch.setattr(runtime, "_schedule_channel_stats_bounded", record)

        async def completed_source():
            yield b"one"
            yield b"two"

        completed_info = {
            "request_id": "completed",
            "api_key": "key",
            "routing_attempts": [{"index": 1, "outcome": "stream_pending"}],
        }
        completed = runtime._track_legacy_stream_outcome(
            completed_source(),
            current_info=completed_info,
            channel_id="provider",
            model="model",
            provider_api_key="provider-key",
            fallback_background_tasks=None,
            response_memory_lease=response_memory_lease,
        )
        assert [chunk async for chunk in completed] == [b"one", b"two"]
        assert recorded == [True]
        assert completed_info["success"] is True
        assert completed_info["routing_attempts"][0]["outcome"] == "stream_completed"
        assert completed_info["routing_attempts"][0]["success"] is True
        assert response_attempt_outcomes[-1] == ("stream_completed", False)

        upstream_module.finalize_latest_routing_attempt(
            {},
            response_memory_lease=response_memory_lease,
            outcome="stream_completed_without_routing_entry",
        )
        assert response_attempt_outcomes[-1] == (
            "stream_completed_without_routing_entry",
            False,
        )

        async def failed_source():
            yield b"first"
            raise RuntimeError("midstream abort")

        failed_info = {
            "request_id": "failed",
            "api_key": "key",
            "routing_attempts": [{"index": 1, "outcome": "stream_pending"}],
        }
        failed = runtime._track_legacy_stream_outcome(
            failed_source(),
            current_info=failed_info,
            channel_id="provider",
            model="model",
            provider_api_key="provider-key",
            fallback_background_tasks=None,
            response_memory_lease=response_memory_lease,
        )
        assert await anext(failed) == b"first"
        with pytest.raises(RuntimeError, match="midstream abort"):
            await anext(failed)
        assert recorded == [True, False]
        assert failed_info["success"] is False
        assert failed_info["stream_outcome"] == "upstream_stream_abort"
        assert failed_info["routing_attempts"][0]["outcome"] == "stream_failed"
        assert failed_info["routing_attempts"][0]["semantic_status_code"] == 502
        assert response_attempt_outcomes[-1] == ("stream_failed", False)

        async def protocol_failed_source():
            yield b"event: image_edit.partial_image\ndata: {}\n\n"
            raise SSEProtocolError("upstream ended before image terminal")

        protocol_info = {
            "request_id": "protocol-failed",
            "api_key": "key",
            "routing_attempts": [{"index": 1, "outcome": "stream_pending"}],
            "image_stream_diagnostics": {
                "contract_version": 1,
                "last_event_type": "image_edit.partial_image",
                "eof": True,
                "terminal_seen": False,
                "synthetic_terminal": False,
            },
        }
        protocol_failed = runtime._track_legacy_stream_outcome(
            protocol_failed_source(),
            current_info=protocol_info,
            channel_id="provider",
            model="model",
            provider_api_key="provider-key",
            fallback_background_tasks=None,
            response_memory_lease=response_memory_lease,
        )
        assert b"image_edit.partial_image" in await anext(protocol_failed)
        protocol_terminal = await anext(protocol_failed)
        assert protocol_terminal.startswith(b"event: error\n")
        assert b'"type":"error"' in protocol_terminal
        assert b'"code":"upstream_sse_protocol_error"' in protocol_terminal
        with pytest.raises(StopAsyncIteration):
            await anext(protocol_failed)
        assert recorded == [True, False, False]
        assert protocol_info["postcommit_sse_protocol_error_isolated"] is True
        assert protocol_info["stream_error_after_response_start"] is True
        assert protocol_info["stream_error_event_type"] == "error"
        assert protocol_info["image_stream_diagnostics"][
            "synthetic_terminal"
        ] is True
        assert protocol_info["image_stream_diagnostics"][
            "synthetic_terminal_type"
        ] == "error"
        assert protocol_info["routing_attempts"][0]["outcome"] == (
            "stream_failed"
        )

        semantic_error = responses_failure_error(
            {
                "type": "error",
                "error": {
                    "message": "context window exceeded",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                },
            },
            event_type="error",
        )
        assert semantic_error is not None

        async def semantic_failed_source():
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise semantic_error

        semantic_info = {
            "request_id": "semantic-failed",
            "api_key": "key",
            "routing_attempts": [
                {
                    "index": 1,
                    "wire_status_code": 200,
                    "outcome": "stream_pending",
                }
            ],
        }
        semantic_failed = runtime._track_legacy_stream_outcome(
            semantic_failed_source(),
            current_info=semantic_info,
            channel_id="provider",
            model="model",
            provider_api_key="provider-key",
            fallback_background_tasks=None,
            response_memory_lease=response_memory_lease,
        )
        assert b"partial" in await anext(semantic_failed)
        semantic_terminal = await anext(semantic_failed)
        assert semantic_terminal.startswith("event: error\n")
        assert '"status_code": 400' in semantic_terminal
        assert '"code": "context_length_exceeded"' in semantic_terminal
        with pytest.raises(StopAsyncIteration):
            await anext(semantic_failed)
        semantic_attempt = semantic_info["routing_attempts"][0]
        assert semantic_attempt["wire_status_code"] == 200
        assert semantic_attempt["semantic_status_code"] == 400
        assert semantic_attempt["terminal_event_type"] == "error"
        assert semantic_attempt["error_code"] == "context_length_exceeded"
        assert semantic_attempt["outcome"] == "semantic_failure_terminal"
        assert recorded == [True, False, False]

        async def abandoned_source():
            yield b"first"
            await asyncio.Event().wait()

        abandoned_info = {
            "request_id": "abandoned",
            "api_key": "key",
            "routing_attempts": [{"index": 1, "outcome": "stream_pending"}],
        }
        abandoned = runtime._track_legacy_stream_outcome(
            abandoned_source(),
            current_info=abandoned_info,
            channel_id="provider",
            model="model",
            provider_api_key="provider-key",
            fallback_background_tasks=None,
            response_memory_lease=response_memory_lease,
        )
        assert await anext(abandoned) == b"first"
        await abandoned.aclose()
        assert recorded == [True, False, False]
        assert abandoned_info["routing_attempts"][0]["outcome"] == (
            "cancelled_or_consumer_closed"
        )
        assert "success" not in abandoned_info["routing_attempts"][0]

    asyncio.run(scenario())


def test_response_start_failure_closes_without_sending_an_error_body():
    async def scenario():
        class Body:
            def __init__(self):
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                return b"data: should-not-start\n\n"

            async def aclose(self):
                self.closed = True

        body = Body()
        lifecycle_calls = []
        send_calls = 0

        async def send(_message):
            nonlocal send_calls
            send_calls += 1
            raise OSError("client already gone")

        async def lifecycle_close(info):
            lifecycle_calls.append(dict(info))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body,
            media_type="text/event-stream",
            current_info=current_info,
            lifecycle_close=lifecycle_close,
        )
        await response(_scope(), _never_receive, send)

        assert send_calls == 1
        assert body.closed is True
        assert len(lifecycle_calls) == 1
        assert current_info["response_committed"] is False
        assert current_info["stream_error_after_response_start"] is False
        assert current_info["stream_outcome"] == "downstream_disconnected"

    asyncio.run(scenario())


def test_upstream_body_oserror_is_not_misclassified_as_downstream_disconnect():
    async def scenario():
        async def body():
            yield b"data: first\n\n"
            raise OSError("upstream source failed")

        sent = []

        async def send(message):
            sent.append(dict(message))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(_scope(), _never_receive, send)

        assert current_info["stream_outcome"] == "error"
        assert current_info["error_type"] == "OSError"
        assert current_info.get("downstream_disconnected") is not True
        bodies = [
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        ]
        assert bodies[0] == b"data: first\n\n"
        assert any(b"event: error" in body for body in bodies)
        assert bodies[-1] == b""

    asyncio.run(scenario())


def test_body_send_cancellation_still_closes_and_finalizes_once():
    async def scenario():
        body_closed = False
        lifecycle_calls = []

        async def body():
            nonlocal body_closed
            try:
                yield b"data: first\n\n"
            finally:
                body_closed = True

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                raise asyncio.CancelledError()

        async def lifecycle_close(info):
            await asyncio.sleep(0)
            lifecycle_calls.append(dict(info))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
            lifecycle_close=lifecycle_close,
        )
        with pytest.raises(asyncio.CancelledError):
            await response(_scope(), _never_receive, send)
        await response.close()

        assert body_closed is True
        assert len(lifecycle_calls) == 1
        assert current_info["stream_outcome"] == "cancelled"
        assert current_info["error_type"] == "CancelledError"

    asyncio.run(scenario())


def test_binary_stream_error_never_injects_an_sse_frame():
    async def scenario():
        sent = []
        lifecycle_calls = []

        async def body():
            yield b"\x00\xffaudio"
            raise RuntimeError("upstream broke")

        async def send(message):
            sent.append(message)

        async def lifecycle_close(info):
            lifecycle_calls.append(dict(info))

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="audio/mpeg",
            current_info=current_info,
            lifecycle_close=lifecycle_close,
        )
        await response(_scope(path="/v1/audio/speech"), _never_receive, send)

        bodies = [message.get("body", b"") for message in sent if message["type"] == "http.response.body"]
        assert bodies == [b"\x00\xffaudio", b""]
        assert len(lifecycle_calls) == 1
        assert current_info["stream_outcome"] == "error"
        assert current_info["error_type"] == "RuntimeError"
        assert current_info["success"] is False

    asyncio.run(scenario())


def test_sse_stream_error_uses_a_well_formed_error_event_and_records_failure():
    async def scenario():
        sent = []

        async def body():
            if False:
                yield b""
            raise RuntimeError("upstream broke")

        async def send(message):
            sent.append(message)

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(_scope(), _never_receive, send)

        bodies = [message.get("body", b"") for message in sent if message["type"] == "http.response.body"]
        assert len(bodies) == 2
        assert bodies[0].startswith(b"event: error\ndata: ")
        error_payload = bodies[0].split(b"data: ", 1)[1].strip()
        assert b'"message": "Streaming error: upstream broke"' in error_payload
        assert b'"type": "stream_error"' in error_payload
        assert bodies[0].endswith(b"\n\n")
        assert bodies[1] == b""
        assert current_info["status_code"] == 200
        assert current_info["stream_outcome"] == "error"
        assert current_info["error_type"] == "RuntimeError"
        assert current_info["success"] is False

    asyncio.run(scenario())


def test_sse_semantic_error_preserves_status_and_error_envelope_after_commit():
    async def scenario():
        sent = []
        semantic_error = responses_failure_error(
            {
                "type": "error",
                "error": {
                    "message": "Your input exceeds the context window of this model.",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "param": "input",
                },
            },
            event_type="error",
        )
        assert semantic_error is not None

        async def body():
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise semantic_error

        async def send(message):
            sent.append(message)

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(_scope(), _never_receive, send)

        starts = [message for message in sent if message["type"] == "http.response.start"]
        assert [message["status"] for message in starts] == [200]
        bodies = [
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        ]
        error_frame = next(body for body in bodies if body.startswith(b"event: error"))
        payload = json.loads(error_frame.split(b"data: ", 1)[1].strip())
        assert payload == {
            "type": "error",
            "error": {
                "message": "Your input exceeds the context window of this model.",
                "status_code": 400,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "input",
            },
        }
        assert b"Streaming error" not in error_frame
        assert b"data: [DONE]" not in b"".join(bodies)
        assert current_info["wire_status_code"] == 200
        assert current_info["stream_error_status_code"] == 400
        assert current_info["stream_error_code"] == "context_length_exceeded"
        assert current_info["stream_outcome"] == "upstream_failure_terminal"
        assert current_info["success"] is False

    asyncio.run(scenario())


def test_huge_stream_exception_has_bounded_sse_frame_and_log(caplog):
    async def scenario():
        sent = []

        async def body():
            yield b"data: first\n\n"
            raise RuntimeError("x" * (5 * 1024 * 1024))

        async def send(message):
            sent.append(message)

        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info={"start_time": 0, "success": True},
        )
        await response(_scope(), _never_receive, send)

        bodies = [
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        ]
        error_frame = next(body for body in bodies if body.startswith(b"event: error"))
        assert len(error_frame) < 32 * 1024
        assert b"[truncated]" in error_frame
        payload = error_frame.split(b"data: ", 1)[1].strip()
        decoded = json.loads(payload)
        assert decoded["type"] == "error"
        assert len(decoded["error"]["message"].encode("utf-8")) <= 4096

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())
    matching = [
        record.getMessage()
        for record in caplog.records
        if "Error in streaming response" in record.getMessage()
    ]
    assert matching
    assert max(len(message.encode("utf-8")) for message in matching) < 8 * 1024


def test_stream_error_event_sanitizes_lone_unicode_surrogate():
    event = runtime._build_responses_stream_error_event(
        500,
        RuntimeError("\ud800"),
    )
    assert event.decode("utf-8")
    assert b"event: error" in event


def test_sse_error_event_write_timeout_is_not_reported_as_disconnect():
    async def scenario():
        async def body():
            yield b"data: first\n\n"
            raise RuntimeError("upstream failed")

        async def send(message):
            if message.get("body", b"").startswith(b"event: error"):
                await asyncio.Event().wait()

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
            downstream_write_timeout_seconds=0.01,
        )
        await response(_scope(), _never_receive, send)

        assert current_info["stream_outcome"] == "downstream_write_timeout"
        assert current_info.get("downstream_disconnected") is not True

    asyncio.run(scenario())


def test_sse_error_event_generic_send_failure_is_not_reported_as_disconnect():
    async def scenario():
        async def body():
            yield b"data: first\n\n"
            raise RuntimeError("upstream failed")

        async def send(message):
            if message.get("body", b"").startswith(b"event: error"):
                raise ValueError("local ASGI adapter failure")

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(_scope(), _never_receive, send)

        assert current_info["stream_outcome"] == "downstream_send_error"
        assert current_info.get("downstream_disconnected") is not True

    asyncio.run(scenario())


def test_partial_sse_frame_never_gets_a_glued_in_band_error_event():
    async def scenario():
        sent = []

        async def body():
            yield b"data: 0123456789"
            raise RuntimeError("queue backpressure aborted the stream")

        async def send(message):
            sent.append(message)

        current_info = {
            "start_time": 0,
            "success": False,
            "stream_outcome": "local_backpressure_abort",
        }
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
            downstream_chunk_bytes=8,
        )
        await response(_scope(), _never_receive, send)

        wire = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert wire == b"data: 0123456789"
        assert b"event: error" not in wire
        parser = IncrementalSSEParser()
        assert parser.feed(wire) == []
        with pytest.raises(SSEProtocolError, match="incomplete trailing event"):
            parser.finish()
        assert current_info["stream_outcome"] == "local_backpressure_abort"
        assert current_info["sse_error_event_suppressed"] == (
            "partial_or_unknown_frame_boundary"
        )

    asyncio.run(scenario())


def test_failed_body_send_marks_sse_boundary_unknown_before_error_recovery():
    async def scenario():
        wire = bytearray()
        nonempty_attempts = 0

        async def body():
            yield b"data: complete\n\n"

        async def send(message):
            nonlocal nonempty_attempts
            payload = message.get("body", b"")
            if payload:
                nonempty_attempts += 1
                if nonempty_attempts == 1:
                    wire.extend(payload[:5])
                    raise ValueError("adapter failed after partial write")
                wire.extend(payload)

        current_info = {"start_time": 0, "success": True}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(_scope(), _never_receive, send)

        assert bytes(wire) == b"data:"
        assert nonempty_attempts == 1
        assert b"event: error" not in wire
        assert current_info["sse_error_event_suppressed"] == (
            "downstream_send_failed_boundary_unknown"
        )
        assert current_info["stream_outcome"] == "downstream_send_error"
        assert current_info["error_type"] == "DownstreamSendError"
        assert current_info.get("downstream_disconnected") is not True

    asyncio.run(scenario())


def test_usage_observation_is_bounded_without_truncating_customer_bytes():
    async def scenario():
        sent = []
        payload = b'{"padding":"' + (b"x" * 64) + b'"}'

        async def body():
            yield payload[:20]
            yield payload[20:]

        async def send(message):
            sent.append(message)

        current_info = {"start_time": 0}
        response = LoggingStreamingResponse(
            body(),
            media_type="application/json",
            current_info=current_info,
            usage_buffer_limit_bytes=32,
        )
        await response(_scope(), _never_receive, send)

        customer_body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert customer_body == payload
        assert current_info["usage_parse_error"] == "ValueError"
        assert current_info["stream_outcome"] == "completed"

    asyncio.run(scenario())


def test_usage_observation_can_be_disabled_for_non_usage_json():
    async def scenario():
        sent = []
        payload = (
            b'{"output":"'
            + (b"x" * (70 * 1024))
            + b'","usage":{"prompt_tokens":7,"completion_tokens":9,'
            + b'"total_tokens":16}}'
        )

        async def body():
            yield payload

        async def send(message):
            sent.append(message)

        diagnostics = {}
        current_info = {
            "start_time": 0,
            "responses_stream_diagnostics": diagnostics,
        }
        response = LoggingStreamingResponse(
            body(),
            media_type="application/json",
            current_info=current_info,
            observe_usage=False,
            usage_buffer_limit_bytes=128 * 1024,
        )
        await response(_scope(), _never_receive, send)

        customer_body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert customer_body == payload
        assert "usage_parse_error" not in current_info
        assert "prompt_tokens" not in current_info
        assert "completion_tokens" not in current_info
        assert "total_tokens" not in current_info
        assert current_info["stream_outcome"] == "completed"
        assert diagnostics["downstream_usage_observer_status"] == "not_applicable"
        assert diagnostics["downstream_usage_observer_reason"] == "disabled_by_policy"

    asyncio.run(scenario())


def test_sse_done_frame_finishes_without_a_false_incomplete_error():
    async def scenario():
        sent = []

        async def body():
            yield b'data: {"usage":{"prompt_tokens":2,"completion_tokens":3}}\n\n'
            yield b"data: [DONE]\n\n"

        async def send(message):
            sent.append(message)

        current_info = {"start_time": 0}
        response = LoggingStreamingResponse(
            body(),
            media_type="text/event-stream",
            current_info=current_info,
        )
        await response(_scope(), _never_receive, send)

        assert current_info["stream_outcome"] == "completed"
        assert "usage_parse_error" not in current_info
        assert current_info["prompt_tokens"] == 2
        assert current_info["completion_tokens"] == 3
        assert current_info["total_tokens"] == 5

    asyncio.run(scenario())


class _Trace:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.span_id = "span"
        self.parent_span_id = None
        self.trace_flags = "01"
        self.tracestate = ""
        self._marks = {}

    def mark(self, name: str):
        self._marks.setdefault(name, len(self._marks) + 1)

    def set_tag(self, name: str, value):
        _ = name, value

    def snapshot(self):
        return dict(self._marks)


class _Gauges:
    def __init__(self):
        self.active = 0
        self.begin_calls = 0
        self.end_calls = 0

    def begin_inflight(self):
        self.active += 1
        self.begin_calls += 1

    def end_inflight(self):
        self.active -= 1
        self.end_calls += 1

    async def record_event_loop_lag(self):
        return None


def _stats_dependencies(
    gauges: _Gauges,
    *,
    parse_request_body,
    monitor_disconnect,
    emitted: list,
    updated: list,
) -> StatsMiddlewareDependencies:
    async def get_api_key(_request):
        return "token"

    async def update_stats(info):
        updated.append(dict(info))

    async def moderation_handler(*_args):
        raise AssertionError("moderation should not run")

    return StatsMiddlewareDependencies(
        app_state=SimpleNamespace(
            api_key_index={"token": 0},
            api_list=["token"],
            config={"api_keys": [{"api": "token", "preferences": {}}]},
            paid_api_keys_states={},
        ),
        database_disabled=False,
        runtime_gauges=gauges,
        trace_factory=lambda **kwargs: _Trace(kwargs["trace_id"]),
        incoming_trace_context=lambda _headers: {
            "trace_id": "trace",
            "parent_span_id": None,
            "trace_flags": "01",
            "tracestate": "",
        },
        get_api_key=get_api_key,
        get_client_ip=lambda _request: "127.0.0.1",
        parse_request_body=parse_request_body,
        message_role_summary=lambda _body: (None, None),
        messages_request_last_text=lambda _body: None,
        is_public_health_request=lambda _request: False,
        is_video_or_asset_request_path=lambda _path: False,
        lingjing_request_model_for_openapi=lambda _body, _params: "",
        video_prompt_from_body=lambda _body: "",
        monitor_disconnect=monitor_disconnect,
        log_debug_request_headers=lambda *_args, **_kwargs: None,
        log_debug_request_body=lambda *_args, **_kwargs: None,
        mask_secret_for_log=lambda value: str(value),
        update_stats=update_stats,
        emit_request_observability=lambda info: emitted.append(dict(info)),
        mark_first_byte_observed=lambda _info: None,
        moderation_handler=moderation_handler,
        logging_response_class=LoggingStreamingResponse,
    )


def test_stats_middleware_holds_inflight_and_disconnect_monitor_until_stream_end():
    async def scenario():
        release = asyncio.Event()
        first_body_sent = asyncio.Event()
        receive_queue: asyncio.Queue[dict] = asyncio.Queue()
        inner_closed = False
        inner_saw_disconnect = False
        monitor_finished = False
        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(request):
            await request.body()
            return {}

        async def monitor_disconnect(request, disconnect_event):
            nonlocal monitor_finished
            try:
                while not disconnect_event.is_set():
                    message = await request.receive()
                    if message.get("type") == "http.disconnect":
                        disconnect_event.set()
                        release.set()
                        return
            finally:
                monitor_finished = True

        async def inner_app(_scope, _receive, send):
            nonlocal inner_closed, inner_saw_disconnect
            cached_body = await _receive()
            assert cached_body["type"] == "http.request"

            async def listen_for_disconnect():
                nonlocal inner_saw_disconnect
                message = await _receive()
                inner_saw_disconnect = message["type"] == "http.disconnect"

            inner_disconnect_listener = asyncio.create_task(listen_for_disconnect())
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            try:
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"data: first\n\n",
                        "more_body": True,
                    }
                )
                await release.wait()
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"data: done\n\n",
                        "more_body": False,
                    }
                )
            finally:
                if not inner_disconnect_listener.done():
                    inner_disconnect_listener.cancel()
                with suppress(asyncio.CancelledError):
                    await inner_disconnect_listener
                inner_closed = True

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(inner_app, dependencies=dependencies)

        async def receive():
            return await receive_queue.get()

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body_sent.set()

        request_scope = _scope(method="POST")
        request_scope["headers"] = [(b"content-type", b"application/json")]
        await receive_queue.put(
            {"type": "http.request", "body": b"{}", "more_body": False}
        )
        request_task = asyncio.create_task(middleware(request_scope, receive, send))
        await asyncio.wait_for(first_body_sent.wait(), timeout=1)

        assert gauges.active == 1
        assert gauges.end_calls == 0
        assert monitor_finished is False

        await receive_queue.put({"type": "http.disconnect"})
        await asyncio.wait_for(request_task, timeout=1)
        for _ in range(10):
            if inner_closed:
                break
            await asyncio.sleep(0)

        assert inner_closed is True
        assert inner_saw_disconnect is True
        assert monitor_finished is True
        assert gauges.active == 0
        assert gauges.begin_calls == 1
        assert gauges.end_calls == 1
        assert len(emitted) == 1
        assert len(updated) == 1
        assert emitted[0]["stream_outcome"] == "downstream_disconnected"
        assert emitted[0]["downstream_disconnected"] is True

    asyncio.run(scenario())


def test_stats_middleware_does_not_swallow_admission_rejections():
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(_request):
            raise AdmissionRejected("body_budget_exhausted", status_code=503)

        async def monitor_disconnect(_request, _event):
            raise AssertionError("monitor should not start")

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(lambda *_args: None, dependencies=dependencies)
        request = Request(_scope(method="POST"), receive=_never_receive)

        async def call_next(_request):
            raise AssertionError("inner app should not be called")

        with pytest.raises(AdmissionRejected) as exc_info:
            await middleware.dispatch(request, call_next)

        assert exc_info.value.reason == "body_budget_exhausted"
        assert gauges.active == 0
        assert gauges.begin_calls == 1
        assert gauges.end_calls == 1
        assert len(emitted) == 1
        assert emitted[0]["status_code"] == 503
        assert emitted[0]["error_type"] == "body_budget_exhausted"
        assert updated == []

    asyncio.run(scenario())


def test_stats_middleware_does_not_swallow_body_hard_limit():
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(_request):
            raise RequestBodyTooLarge()

        async def monitor_disconnect(_request, _event):
            raise AssertionError("monitor should not start")

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(lambda *_args: None, dependencies=dependencies)
        request = Request(_scope(method="POST"), receive=_never_receive)

        async def call_next(_request):
            raise AssertionError("inner app should not be called")

        with pytest.raises(RequestBodyTooLarge):
            await middleware.dispatch(request, call_next)
        assert gauges.active == 0
        assert emitted[0]["status_code"] == 413
        assert emitted[0]["error_type"] == "body_too_large"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("body_budget", "body", "expected_status", "expected_reason"),
    [
        (64, b"123456789", 413, "body_too_large"),
        (4, b"12345", 503, "body_budget_exhausted"),
    ],
)
def test_composed_admission_body_and_stats_stack_preserves_bounded_http_outcome(
    body_budget,
    body,
    expected_status,
    expected_reason,
):
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()
        inner_called = False

        async def parse_request_body(request):
            await request.body()
            return {}

        async def monitor_disconnect(_request, _event):
            raise AssertionError("body rejection should happen before monitoring")

        async def inner_app(_scope, _receive, _send):
            nonlocal inner_called
            inner_called = True

        stats = StatsMiddleware(
            inner_app,
            dependencies=_stats_dependencies(
                gauges,
                parse_request_body=parse_request_body,
                monitor_disconnect=monitor_disconnect,
                emitted=emitted,
                updated=updated,
            ),
        )
        body_middleware = RequestBodyDecompressionMiddleware(
            stats,
            max_identity_body_bytes=8,
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=64,
            body_budget_bytes=body_budget,
        )
        app = RequestAdmissionMiddleware(body_middleware, controller=controller)
        receive_queue = asyncio.Queue()
        await receive_queue.put(
            {"type": "http.request", "body": body, "more_body": False}
        )
        sent = []

        async def receive():
            return await receive_queue.get()

        async def send(message):
            sent.append(message)

        await app(_scope(method="POST"), receive, send)

        start = next(message for message in sent if message["type"] == "http.response.start")
        headers = dict(start["headers"])
        assert start["status"] == expected_status
        assert headers[b"x-uni-api-admission-reason"] == expected_reason.encode()
        assert inner_called is False
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0
        assert controller.snapshot()["rejected"][expected_reason] == 1
        assert emitted[0]["status_code"] == expected_status
        assert emitted[0]["error_type"] == expected_reason

    asyncio.run(scenario())


def test_identity_upload_disconnect_is_499_observability_without_500_response():
    async def scenario():
        emitted = []
        gauges = _Gauges()

        async def parse_request_body(request):
            await request.body()
            return {}

        async def monitor_disconnect(_request, _event):
            raise AssertionError("body disconnect happens before monitor startup")

        stats = StatsMiddleware(
            lambda *_args: None,
            dependencies=_stats_dependencies(
                gauges,
                parse_request_body=parse_request_body,
                monitor_disconnect=monitor_disconnect,
                emitted=emitted,
                updated=[],
            ),
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=64,
            body_budget_bytes=64,
        )
        app = RequestAdmissionMiddleware(
            RequestBodyDecompressionMiddleware(
                stats,
                max_identity_body_bytes=64,
            ),
            controller=controller,
        )
        receive_queue = asyncio.Queue()
        await receive_queue.put(
            {"type": "http.request", "body": b"{", "more_body": True}
        )
        await receive_queue.put({"type": "http.disconnect"})
        sent = []

        async def receive():
            return await receive_queue.get()

        async def send(message):
            sent.append(message)

        await app(_scope(method="POST"), receive, send)

        assert sent == []
        assert emitted[0]["status_code"] == 499
        assert emitted[0]["downstream_disconnected"] is True
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0

    asyncio.run(scenario())


def test_identity_slow_upload_times_out_and_releases_admission_slot():
    async def scenario():
        emitted = []
        gauges = _Gauges()

        async def parse_request_body(request):
            await request.body()
            return {}

        async def monitor_disconnect(_request, _event):
            raise AssertionError("body timeout happens before monitor startup")

        stats = StatsMiddleware(
            lambda *_args: None,
            dependencies=_stats_dependencies(
                gauges,
                parse_request_body=parse_request_body,
                monitor_disconnect=monitor_disconnect,
                emitted=emitted,
                updated=[],
            ),
        )
        controller = RequestAdmissionController(
            capacity=1,
            waiter_limit=0,
            wait_timeout_seconds=1,
            max_body_bytes=64,
            body_budget_bytes=64,
        )
        app = RequestAdmissionMiddleware(
            RequestBodyDecompressionMiddleware(
                stats,
                max_identity_body_bytes=64,
                body_idle_timeout_seconds=0.01,
                body_total_timeout_seconds=1,
            ),
            controller=controller,
        )
        first_message = True

        async def receive():
            nonlocal first_message
            if first_message:
                first_message = False
                return {"type": "http.request", "body": b"{", "more_body": True}
            await asyncio.Event().wait()

        sent = []

        async def send(message):
            sent.append(message)

        await app(_scope(method="POST"), receive, send)

        start = next(message for message in sent if message["type"] == "http.response.start")
        assert start["status"] == 408
        assert emitted[0]["status_code"] == 408
        assert emitted[0]["error_type"] == "request_body_timeout"
        assert controller.snapshot()["active"] == 0
        assert controller.snapshot()["reserved_body_bytes"] == 0

    asyncio.run(scenario())


def test_stats_middleware_maps_direct_upstream_admission_to_retryable_503():
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(_request):
            return None

        async def monitor_disconnect(_request, _event):
            raise AssertionError("GET request should not start a monitor")

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(lambda *_args: None, dependencies=dependencies)
        request = Request(_scope(), receive=_never_receive)

        async def call_next(_request):
            raise UpstreamAdmissionRejected(
                "upstream_wait_timeout",
                retry_after_seconds=5,
                client_key_id="redacted",
            )

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
        assert response.headers["x-uni-api-admission-reason"] == "upstream_wait_timeout"
        assert gauges.active == 0
        assert emitted[0]["status_code"] == 503
        assert emitted[0]["error_type"] == "upstream_wait_timeout"

    asyncio.run(scenario())


def test_rate_limit_only_maps_the_intentional_429_exception():
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(_request):
            return None

        async def monitor_disconnect(_request, _event):
            return None

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(lambda *_args: None, dependencies=dependencies)
        current_info = {}

        class Pool:
            def __init__(self, error):
                self.error = error

            async def next(self, _model):
                raise self.error

        dependencies.app_state.user_api_keys_rate_limit = {
            "key": Pool(HTTPException(status_code=429, detail="limited"))
        }
        response = await middleware._rate_limit_response("key", "model", current_info)
        assert response.status_code == 429

        dependencies.app_state.user_api_keys_rate_limit = {
            "key": Pool(RuntimeError("storage failed"))
        }
        with pytest.raises(RuntimeError, match="storage failed"):
            await middleware._rate_limit_response("key", "model", {})

    asyncio.run(scenario())


def test_stats_middleware_non_stream_finally_releases_gauge_and_context():
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(_request):
            return None

        async def monitor_disconnect(_request, _event):
            raise AssertionError("GET request should not start a disconnect monitor")

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(lambda *_args: None, dependencies=dependencies)
        request = Request(_scope(), receive=_never_receive)

        async def call_next(_request):
            return JSONResponse({"ok": True})

        sentinel = {"request_id": "outer-context"}
        outer_token = set_request_info(sentinel)
        try:
            response = await middleware.dispatch(request, call_next)

            assert response.status_code == 200
            assert gauges.active == 0
            assert gauges.begin_calls == 1
            assert gauges.end_calls == 1
            assert len(emitted) == 1
            assert updated == []
            assert get_request_info() is sentinel
        finally:
            reset_request_info(outer_token)

    asyncio.run(scenario())


def test_stats_stream_context_lives_through_body_then_restores_outer_context():
    async def scenario():
        emitted = []
        updated = []
        gauges = _Gauges()
        context_seen_in_body = []

        async def parse_request_body(_request):
            return None

        async def monitor_disconnect(_request, _event):
            raise AssertionError("GET request should not start a disconnect monitor")

        dependencies = _stats_dependencies(
            gauges,
            parse_request_body=parse_request_body,
            monitor_disconnect=monitor_disconnect,
            emitted=emitted,
            updated=updated,
        )
        middleware = StatsMiddleware(lambda *_args: None, dependencies=dependencies)
        request = Request(_scope(), receive=_never_receive)

        async def body():
            context_seen_in_body.append(dict(get_request_info()))
            yield b"data: [DONE]\n\n"

        async def call_next(_request):
            return StreamingResponse(body(), media_type="text/event-stream")

        async def send(_message):
            return None

        sentinel = {"request_id": "outer-context"}
        outer_token = set_request_info(sentinel)
        try:
            response = await middleware.dispatch(request, call_next)
            request_context = get_request_info()
            assert request_context is not sentinel
            assert request_context["request_id"]
            assert gauges.active == 1

            await response(_scope(), _never_receive, send)

            assert context_seen_in_body[0]["request_id"] == request_context["request_id"]
            assert get_request_info() is sentinel
            assert gauges.active == 0
            assert gauges.end_calls == 1
            assert len(emitted) == 1
            assert len(updated) == 1
        finally:
            if get_request_info() is not sentinel:
                # Preserve isolation if an assertion exposes a lifecycle bug.
                set_request_info(sentinel)
            reset_request_info(outer_token)

    asyncio.run(scenario())


def test_responses_generic_postcommit_error_is_observed_before_base_http_eof(
    monkeypatch,
):
    """A post-commit iterator bug must end as an incomplete stream."""

    from fastapi import BackgroundTasks

    import main
    from core.models import ResponsesRequest
    from test_responses_retry import (
        DummyClientManager,
        DummyStreamingUpstreamResponse,
        _configure_responses_test,
        _responses_sse,
    )

    async def scenario():
        _configure_responses_test(monkeypatch, engine="codex")
        main.app.state.client_manager = DummyClientManager(
            DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse(
                        "response.created",
                        {"type": "response.created"},
                    ),
                    _responses_sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "delta": "partial",
                        },
                    ),
                ],
                stream_error=RuntimeError("unexpected postcommit iterator failure"),
            )
        )

        emitted = []
        updated = []
        gauges = _Gauges()

        async def parse_request_body(request):
            await request.body()
            return {}

        async def monitor_disconnect(_request, disconnect_event):
            await disconnect_event.wait()

        async def inner_app(scope, receive, send):
            current_info = get_request_info()
            response = await main.ResponsesRequestHandler().request_responses(
                http_request=SimpleNamespace(
                    headers={},
                    state=SimpleNamespace(uni_api_request_info=current_info),
                ),
                request_data=ResponsesRequest(
                    model="gpt-5.4",
                    input=[{"role": "user", "content": "hello"}],
                    stream=True,
                ),
                api_index=0,
                background_tasks=BackgroundTasks(),
                endpoint="/v1/responses",
            )
            await response(scope, receive, send)

        app = StatsMiddleware(
            inner_app,
            dependencies=_stats_dependencies(
                gauges,
                parse_request_body=parse_request_body,
                monitor_disconnect=monitor_disconnect,
                emitted=emitted,
                updated=updated,
            ),
        )
        scope = _scope(method="POST", path="/v1/responses")
        scope["headers"] = [
            (b"authorization", b"Bearer token"),
            (b"content-type", b"application/json"),
        ]
        receive_queue = asyncio.Queue()
        await receive_queue.put(
            {"type": "http.request", "body": b"{}", "more_body": False}
        )
        sent = []

        async def receive():
            return await receive_queue.get()

        async def send(message):
            sent.append(dict(message))

        await asyncio.wait_for(app(scope, receive, send), timeout=3)

        body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert b"response.output_text.delta" in body
        assert b"event: error" not in body
        assert b"unexpected postcommit iterator failure" not in body
        assert b"data: [DONE]" not in body

        assert gauges.active == 0
        assert gauges.begin_calls == 1
        assert gauges.end_calls == 1
        assert len(emitted) == 1
        assert len(updated) == 1

        current_info = emitted[0]
        assert current_info["wire_status_code"] == 200
        assert current_info["response_committed"] is True
        assert current_info["stream_outcome"] == "upstream_stream_abort"
        assert current_info["error_type"] == "RuntimeError"
        assert current_info["success"] is False
        assert current_info["stream_error_after_response_start"] is True
        assert current_info["postcommit_stream_terminal_suppressed"] is True

        diagnostics = current_info["responses_stream_diagnostics"]
        assert diagnostics["diagnosis"] == "responses_stream_error"
        assert diagnostics["semantic_status"] == "transport_error"
        assert diagnostics["failure_stage"] == "postcommit"
        assert diagnostics["exception_type"] == "RuntimeError"
        assert diagnostics["error_event_seen"] is False
        assert diagnostics["downstream_terminal_seen"] is False
        assert diagnostics["downstream_terminal_asgi_write_completed"] is False
        assert "terminal_asgi_write_completed_unix_nano" not in diagnostics
        assert diagnostics["downstream_final_body_outcome"] == "completed"

    asyncio.run(scenario())
