from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uni_api.streaming.sse import IncrementalSSEParser
from uni_api.streaming.responses_events import stream_responses_to_chat_completions


def time_sse_parser(events: int, rounds: int) -> list[float]:
    timings = []
    payload = b"".join(
        f'event: response.output_text.delta\ndata: {{"delta": "chunk-{index}"}}\n\n'.encode()
        for index in range(events)
    )
    chunks = [payload[index:index + 37] for index in range(0, len(payload), 37)]
    for _ in range(rounds):
        parser = IncrementalSSEParser()
        count = 0
        started = time.perf_counter()
        for chunk in chunks:
            count += len(parser.feed(chunk))
        elapsed = time.perf_counter() - started
        assert count == events
        timings.append(elapsed)
    return timings


async def time_response_parse(events: int, rounds: int) -> list[float]:
    timings = []

    async def upstream_iter():
        for index in range(events):
            yield f'event: response.output_text.delta\ndata: {{"delta": "chunk-{index}"}}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    for _ in range(rounds):
        count = 0
        started = time.perf_counter()
        async for chunk in stream_responses_to_chat_completions(upstream_iter(), request_model="gpt-5.4"):
            count += len(chunk)
        elapsed = time.perf_counter() - started
        assert count > 0
        timings.append(elapsed)
    return timings


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=5000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    sse_timings = time_sse_parser(args.events, args.rounds)
    response_timings = await time_response_parse(args.events, args.rounds)
    print(
        json.dumps(
            {
                "events": args.events,
                "rounds": args.rounds,
                "sse_parser_seconds_best": min(sse_timings),
                "sse_parser_seconds_mean": statistics.mean(sse_timings),
                "response_parse_seconds_best": min(response_timings),
                "response_parse_seconds_mean": statistics.mean(response_timings),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

