from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import sys
import tracemalloc
from dataclasses import dataclass
from time import perf_counter
from typing import Any

ROOT_DIR = os.getenv("UNI_API_BENCHMARK_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from starlette.types import Message, Receive, Scope, Send

from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.middleware.idempotency import (
    IdempotencyMiddleware,
    InMemoryIdempotencyCoordinator,
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    requests: int
    elapsed_seconds: float
    requests_per_second: float
    peak_python_heap_bytes: int | None
    snapshot: dict[str, Any]


def _payload(size: int) -> bytes:
    prefix = b'{"model":"gpt-test","input":"'
    suffix = b'"}'
    if size < len(prefix) + len(suffix):
        raise ValueError("request-bytes is too small for the benchmark payload")
    return prefix + (b"x" * (size - len(prefix) - len(suffix))) + suffix


def _scope(index: int) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [
            (b"authorization", b"Bearer benchmark"),
            (b"content-type", b"application/json"),
            (b"idempotency-key", f"benchmark-{index}".encode("ascii")),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8000),
        "state": {DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY: asyncio.Event()},
    }


def _receive_body(body: bytes, chunk_size: int) -> Receive:
    offset = 0

    async def receive() -> Message:
        nonlocal offset
        if offset >= len(body):
            await asyncio.Future()
        end = min(len(body), offset + chunk_size)
        chunk = body[offset:end]
        offset = end
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": offset < len(body),
        }

    return receive


async def _run_once(args: argparse.Namespace, *, measure_heap: bool) -> BenchmarkResult:
    request_body = _payload(args.request_bytes)
    response_chunk = b"y" * args.response_chunk_bytes
    full_chunks, trailing = divmod(args.response_bytes, len(response_chunk))

    async def app(_scope: Scope, receive: Receive, send: Send) -> None:
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue
            received += len(message.get("body", b"") or b"")
            if not message.get("more_body", False):
                break
        if received != len(request_body):
            raise RuntimeError("benchmark app received a truncated request")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/octet-stream")],
            }
        )
        for index in range(full_chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": response_chunk,
                    "more_body": index + 1 < full_chunks or trailing > 0,
                }
            )
        if trailing:
            await send(
                {
                    "type": "http.response.body",
                    "body": response_chunk[:trailing],
                    "more_body": False,
                }
            )
        elif full_chunks == 0:
            await send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )

    coordinator = InMemoryIdempotencyCoordinator(
        ttl_seconds=900,
        max_entries=max(args.requests, 4096),
        max_stored_bytes=args.cache_bytes,
        max_response_bytes=max(args.response_bytes, 1),
    )
    middleware = IdempotencyMiddleware(
        app,
        coordinator=coordinator,
        max_request_body_bytes=max(args.request_bytes, 1),
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def discard(_message: Message) -> None:
        return None

    async def execute(index: int) -> None:
        async with semaphore:
            await middleware(
                _scope(index),
                _receive_body(request_body, args.request_chunk_bytes),
                discard,
            )

    gc.collect()
    if measure_heap:
        tracemalloc.start()
    started = perf_counter()
    await asyncio.gather(*(execute(index) for index in range(args.requests)))
    elapsed = perf_counter() - started
    peak_heap = None
    if measure_heap:
        _current, peak_heap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return BenchmarkResult(
        requests=args.requests,
        elapsed_seconds=elapsed,
        requests_per_second=args.requests / elapsed,
        peak_python_heap_bytes=peak_heap,
        snapshot=coordinator.snapshot(),
    )


async def _main(args: argparse.Namespace) -> None:
    throughput_results = [
        await _run_once(args, measure_heap=False) for _ in range(args.rounds)
    ]
    heap_result = await _run_once(args, measure_heap=True)
    payload = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "request_bytes": args.request_bytes,
        "response_bytes": args.response_bytes,
        "cache_bytes": args.cache_bytes,
        "rounds": args.rounds,
        "requests_per_second": {
            "median": statistics.median(
                result.requests_per_second for result in throughput_results
            ),
            "min": min(result.requests_per_second for result in throughput_results),
            "max": max(result.requests_per_second for result in throughput_results),
            "samples": [
                result.requests_per_second for result in throughput_results
            ],
        },
        "heap_measurement": {
            "requests_per_second": heap_result.requests_per_second,
            "peak_python_heap_bytes": heap_result.peak_python_heap_bytes,
        },
        "coordinator": heap_result.snapshot,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark idempotent request/response buffering."
    )
    parser.add_argument("--requests", type=int, default=192)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--request-bytes", type=int, default=256 * 1024)
    parser.add_argument("--request-chunk-bytes", type=int, default=64 * 1024)
    parser.add_argument("--response-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--response-chunk-bytes", type=int, default=64 * 1024)
    parser.add_argument("--cache-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    for name in (
        "requests",
        "concurrency",
        "request_bytes",
        "request_chunk_bytes",
        "response_bytes",
        "response_chunk_bytes",
        "cache_bytes",
        "rounds",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(_main(_args()))
