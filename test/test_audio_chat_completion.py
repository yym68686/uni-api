import os
import sys
import json
import asyncio
import pytest
from fastapi import HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import error_handling_wrapper


async def _audio_chat_completion_non_stream_not_empty():
    sample = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-audio-2025-08-28",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "audio": {"id": "audio_x", "data": "AAAA", "format": "wav"},
                    "annotations": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    async def gen():
        yield "data: " + json.dumps(sample) + "\n\n"

    wrapped, _ = await error_handling_wrapper(
        gen(),
        channel_id="test",
        engine="gpt",
        stream=False,
        error_triggers=[],
        last_message_role="user",
    )
    first = await wrapped.__anext__()
    assert '"audio"' in first


async def _stream_comment_frame_does_not_raise_empty_response(comment_frame: str):
    sample = {
        "choices": [
            {
                "delta": {
                    "content": "ok",
                }
            }
        ]
    }

    async def gen():
        yield comment_frame
        yield "data: " + json.dumps(sample) + "\n\n"

    wrapped, _ = await error_handling_wrapper(
        gen(),
        channel_id="test",
        engine="codex",
        stream=True,
        error_triggers=[],
        last_message_role="user",
    )
    first = await wrapped.__anext__()
    second = await wrapped.__anext__()
    assert first == comment_frame
    assert '"content": "ok"' in second


async def _stream_invalid_non_comment_frame_still_raises_bad_gateway():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield "oops\n\n"
        finally:
            closed = True

    with pytest.raises(HTTPException) as excinfo:
        await error_handling_wrapper(
            gen(),
            channel_id="test",
            engine="codex",
            stream=True,
            error_triggers=[],
            last_message_role="user",
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "Upstream server returned an empty response."
    assert closed


async def _stream_wrapper_closes_upstream_generator_when_consumer_closes():
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            await asyncio.Event().wait()
        finally:
            closed = True

    wrapped, _ = await error_handling_wrapper(
        gen(),
        channel_id="test",
        engine="codex",
        stream=True,
        error_triggers=[],
        last_message_role="user",
    )
    first = await wrapped.__anext__()
    assert '"content":"ok"' in first

    await wrapped.aclose()
    assert closed


async def _stream_wrapper_cancels_pending_keepalive_read_when_consumer_closes():
    closed = False

    async def gen():
        nonlocal closed
        try:
            await asyncio.Event().wait()
            yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        finally:
            closed = True

    wrapped, _ = await error_handling_wrapper(
        gen(),
        channel_id="test",
        engine="codex",
        stream=True,
        error_triggers=[],
        keepalive_interval=0.001,
        last_message_role="user",
    )
    first = await wrapped.__anext__()
    assert first == ": keepalive\n\n"

    await wrapped.aclose()
    assert closed


def test_audio_chat_completion_non_stream_not_empty():
    asyncio.run(_audio_chat_completion_non_stream_not_empty())


def test_stream_bare_sse_comment_frame_does_not_raise_empty_response():
    asyncio.run(_stream_comment_frame_does_not_raise_empty_response(":\n\n"))


def test_stream_named_sse_comment_frame_does_not_raise_empty_response():
    asyncio.run(_stream_comment_frame_does_not_raise_empty_response(": provider heartbeat\n\n"))


def test_stream_invalid_non_comment_frame_still_raises_bad_gateway():
    asyncio.run(_stream_invalid_non_comment_frame_still_raises_bad_gateway())


def test_stream_wrapper_closes_upstream_generator_when_consumer_closes():
    asyncio.run(_stream_wrapper_closes_upstream_generator_when_consumer_closes())


def test_stream_wrapper_cancels_pending_keepalive_read_when_consumer_closes():
    asyncio.run(_stream_wrapper_cancels_pending_keepalive_read_when_consumer_closes())


if __name__ == "__main__":
    test_audio_chat_completion_non_stream_not_empty()
    test_stream_bare_sse_comment_frame_does_not_raise_empty_response()
    test_stream_named_sse_comment_frame_does_not_raise_empty_response()
    test_stream_invalid_non_comment_frame_still_raises_bad_gateway()
    test_stream_wrapper_closes_upstream_generator_when_consumer_closes()
    test_stream_wrapper_cancels_pending_keepalive_read_when_consumer_closes()
