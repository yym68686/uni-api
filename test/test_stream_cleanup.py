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
