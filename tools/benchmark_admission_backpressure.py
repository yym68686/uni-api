from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import statistics
import sys
from time import perf_counter

ROOT_DIR = os.getenv("UNI_API_BENCHMARK_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uni_api.admission import RequestAdmissionController
from uni_api.middleware.admission import RequestAdmissionMiddleware
from uni_api.middleware.request_decompression import (
    RequestBodyDecompressionMiddleware,
)

try:
    from uni_api.middleware.request_decompression import (
        mark_request_body_releasable_at_response_start,
    )
except ImportError:

    def mark_request_body_releasable_at_response_start(scope: dict) -> None:
        scope.setdefault("state", {})[
            "uni_api_release_body_at_response_start"
        ] = True


def _body(size: int) -> bytes:
    prefix = b'{"model":"benchmark","input":"'
    suffix = b'"}'
    if size < len(prefix) + len(suffix):
        raise ValueError("body-bytes is too small")
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


def _scope(index: int, body_bytes: int) -> dict:
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
            (b"authorization", f"Bearer tenant-{index % 8}".encode("ascii")),
            (b"content-type", b"application/json"),
            (b"content-length", str(body_bytes).encode("ascii")),
        ],
        "client": ("127.0.0.1", index + 1),
        "server": ("127.0.0.1", 8000),
        "state": {},
    }


def _receive(payload: bytes):
    sent = False
    blocked = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": payload,
                "more_body": False,
            }
        await blocked.wait()
        return {"type": "http.disconnect"}

    return receive


async def _round(args: argparse.Namespace) -> dict:
    payload = _body(args.body_bytes)
    controller_options = {
        "capacity": args.concurrency,
        "waiter_limit": args.concurrency,
        "wait_timeout_seconds": 1.0,
        "max_body_bytes": args.body_budget_bytes,
        "body_budget_bytes": args.body_budget_bytes,
        "max_response_bytes": args.body_budget_bytes,
        "max_retained_bytes_per_request": args.body_budget_bytes,
    }
    supports_backpressure = "body_memory_wait_timeout_seconds" in inspect.signature(
        RequestAdmissionController
    ).parameters
    if supports_backpressure:
        controller_options.update(
            {
                "body_memory_wait_timeout_seconds": args.memory_wait_seconds,
                "body_memory_waiter_limit": args.concurrency,
                "small_body_lane_threshold_bytes": args.body_budget_bytes,
                "small_body_lane_reserve_bytes": 0,
            }
        )
    controller = RequestAdmissionController(**controller_options)

    async def app(scope, receive, send):
        request = await receive()
        if request.get("body") != payload:
            raise RuntimeError("benchmark request body mismatch")
        mark_request_body_releasable_at_response_start(scope)
        await send(
            {"type": "http.response.start", "status": 200, "headers": []}
        )
        await asyncio.sleep(args.stream_hold_ms / 1000.0)
        await send(
            {"type": "http.response.body", "body": b"ok", "more_body": False}
        )

    middleware = RequestAdmissionMiddleware(
        RequestBodyDecompressionMiddleware(
            app,
            max_identity_body_bytes=args.body_budget_bytes,
            json_max_estimated_bytes=args.body_budget_bytes,
        ),
        controller=controller,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute(index: int) -> int:
        messages = []

        async def send(message):
            messages.append(message)

        async with semaphore:
            await middleware(
                _scope(index, len(payload)),
                _receive(payload),
                send,
            )
        start = next(
            message
            for message in messages
            if message["type"] == "http.response.start"
        )
        return int(start["status"])

    started = perf_counter()
    statuses = await asyncio.gather(
        *(execute(index) for index in range(args.requests))
    )
    elapsed = perf_counter() - started
    successes = sum(status == 200 for status in statuses)
    return {
        "supports_backpressure": supports_backpressure,
        "elapsed_seconds": elapsed,
        "successes": successes,
        "rejections": len(statuses) - successes,
        "successful_requests_per_second": successes / elapsed,
        "snapshot": controller.snapshot(),
    }


async def _main(args: argparse.Namespace) -> None:
    rounds = [await _round(args) for _ in range(args.rounds)]
    throughput = [item["successful_requests_per_second"] for item in rounds]
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "body_bytes": args.body_bytes,
                "body_budget_bytes": args.body_budget_bytes,
                "stream_hold_ms": args.stream_hold_ms,
                "supports_backpressure": rounds[-1]["supports_backpressure"],
                "successful_requests_per_second": {
                    "median": statistics.median(throughput),
                    "min": min(throughput),
                    "max": max(throughput),
                    "samples": throughput,
                },
                "successes_per_round": [item["successes"] for item in rounds],
                "rejections_per_round": [item["rejections"] for item in rounds],
                "last_snapshot": rounds[-1]["snapshot"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sustained success under body-memory pressure."
    )
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--body-bytes", type=int, default=64 * 1024)
    parser.add_argument("--body-budget-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--stream-hold-ms", type=float, default=50.0)
    parser.add_argument("--memory-wait-seconds", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if (
        args.requests <= 0
        or args.concurrency <= 0
        or args.body_bytes <= 0
        or args.body_budget_bytes <= 0
        or args.stream_hold_ms < 0
        or args.memory_wait_seconds <= 0
        or args.rounds <= 0
    ):
        parser.error("benchmark dimensions must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(_main(_args()))
