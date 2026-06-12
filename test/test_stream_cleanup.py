import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
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
