import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
import core.response as response_module
from core.utils import collect_openai_chat_completion_from_streaming_sse


async def _mark_first_byte_wrapper_closes_inner_generator():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield ": keepalive\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    wrapped = main._mark_first_byte_on_stream(gen(), {}, skip_keepalive=True)
    first = await wrapped.__anext__()
    assert first == ": keepalive\n\n"

    await wrapped.aclose()
    assert closed


def test_mark_first_byte_wrapper_closes_inner_generator():
    asyncio.run(_mark_first_byte_wrapper_closes_inner_generator())


async def _stream_collector_closes_generator_on_done():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield "data: [DONE]\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    body = await collect_openai_chat_completion_from_streaming_sse(
        gen(),
        model="gpt-5.5",
    )

    assert '"content": "ok"' in body
    assert closed


def test_stream_collector_closes_generator_on_done():
    asyncio.run(_stream_collector_closes_generator_on_done())


async def _logging_response_closes_body_when_final_send_is_cancelled():
    closed = False

    async def body():
        nonlocal closed
        try:
            yield "data: ok\n\n"
        finally:
            closed = True

    async def send(message):
        if message["type"] == "http.response.body" and not message.get("body"):
            raise asyncio.CancelledError()

    response = main.LoggingStreamingResponse(
        body(),
        media_type="text/event-stream",
        current_info={
            "start_time": 0,
            "endpoint": "POST /v1/chat/completions",
            "request_id": "request",
            "trace_id": "trace",
        },
    )

    try:
        await response({}, None, send)
    except asyncio.CancelledError:
        pass

    assert closed


def test_logging_response_closes_body_when_final_send_is_cancelled():
    asyncio.run(_logging_response_closes_body_when_final_send_is_cancelled())


async def _fetch_response_stream_closes_selected_provider_stream(monkeypatch):
    closed = False

    async def provider_stream(*args, **kwargs):
        nonlocal closed
        try:
            yield "data: ok\n\n"
            await asyncio.Event().wait()
        finally:
            closed = True

    monkeypatch.setattr(response_module, "fetch_gpt_response_stream", provider_stream)

    stream = response_module.fetch_response_stream(
        client=None,
        url="http://example.test",
        headers={},
        payload={},
        engine="codex",
        model="gpt-5.5",
        timeout=None,
    )

    first = await stream.__anext__()
    assert first == "data: ok\n\n"

    await stream.aclose()
    for _ in range(10):
        if closed:
            break
        await asyncio.sleep(0)
    assert closed


def test_fetch_response_stream_closes_selected_provider_stream(monkeypatch):
    asyncio.run(_fetch_response_stream_closes_selected_provider_stream(monkeypatch))


class _FakeConnection:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _FakePoolRequest:
    def __init__(self, connection):
        self.connection = connection


class _FakePool:
    def __init__(self, request, connection):
        self._requests = [request]
        self._connections = [connection]
        self._optional_thread_lock = None

    def _assign_requests_to_connections(self):
        return []

    async def _close_connections(self, closing):
        for connection in closing:
            await connection.aclose()


class _FakePoolStream:
    def __init__(self, pool, request):
        self._pool = pool
        self._pool_request = request


async def _force_release_closes_assigned_connection(force_release):
    connection = _FakeConnection()
    request = _FakePoolRequest(connection)
    pool = _FakePool(request, connection)
    stream = _FakePoolStream(pool, request)

    result = await force_release(stream)

    assert result is True
    assert request not in pool._requests
    assert connection not in pool._connections
    assert connection.closed


def test_main_force_release_closes_assigned_connection():
    asyncio.run(_force_release_closes_assigned_connection(main._force_release_httpcore_pool_request_safely))


def test_core_force_release_closes_assigned_connection():
    async def force_release(stream):
        return await response_module._force_release_httpcore_pool_request_safely(
            stream,
            label="test upstream stream",
        )

    asyncio.run(_force_release_closes_assigned_connection(force_release))
