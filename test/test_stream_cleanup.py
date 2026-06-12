import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


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
