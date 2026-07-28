"""A/B benchmark legacy, metadata, and selective Responses SSE paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uni_api.admission import (
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.admission.json_parsing import ReusableJSONParseWorkspace
from uni_api.observability.responses_stream import ResponsesStreamDiagnostics
from uni_api.streaming.bounded_queue import ObservedStreamChunk
from uni_api.streaming.logging_response import LoggingStreamingResponse
from uni_api.streaming.responses_fast_path import (
    MIN_CANONICAL_RESPONSES_DELTA_FRAME_BYTES,
    can_forward_responses_delta_without_materializing,
    match_canonical_responses_delta_frame,
)
from uni_api.streaming.sse import (
    IncrementalSSEParser,
    parse_owned_sse_event,
    validate_sse_event_type_consistency,
)
from uni_api.streaming.usage import stream_usage_snapshot_from_payload
from uni_api.upstream.responses_normalization import (
    ResponsesCustomToolCallIdNormalizer,
)


def _event(event_type: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def _transport_events(count: int, delta_bytes: int) -> tuple[bytes, ...]:
    padding = "x" * max(0, delta_bytes)
    events = [
        _event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": index,
                "item_id": "item_benchmark_message",
                "delta": f"token-{index}-{padding}",
            },
        )
        for index in range(count)
    ]
    events.append(
        _event(
            "response.completed",
            {
                "type": "response.completed",
                "sequence_number": count,
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 128,
                        "output_tokens": count,
                        "total_tokens": count + 128,
                    },
                },
            },
        )
    )
    return tuple(events)


async def _one_bound_stream(
    transport_chunks: tuple[bytes, ...],
    *,
    variant: str,
    expected_output_tokens: int,
) -> tuple[int, list[float], list[float], int]:
    current_info: dict[str, Any] = {
        "request_id": "benchmark",
        "start_time": time.time(),
    }
    diagnostics = ResponsesStreamDiagnostics(
        current_info=current_info,
        attempt_index=None,
        logical_authority="benchmark.invalid",
        proxy_configured=False,
    )
    parser = IncrementalSSEParser()
    parse_workspace = await ReusableJSONParseWorkspace.create()
    yielded_at: deque[int] = deque()
    available_at: deque[int] = deque()
    frame_latencies_us: list[float] = []
    pipeline_latencies_us: list[float] = []
    sent_bytes = 0
    output_seen = False
    fast_path_events = 0
    custom_tool_call_id_normalizer = (
        ResponsesCustomToolCallIdNormalizer()
        if variant == "fast_normalizer"
        else None
    )
    item_id_requires_full_normalization = (
        custom_tool_call_id_normalizer.requires_item_id_full_normalization
        if custom_tool_call_id_normalizer is not None
        else None
    )

    async def body():
        nonlocal output_seen, fast_path_events
        for chunk in transport_chunks:
            chunk_available_at = time.perf_counter_ns()
            received_at = datetime.now(timezone.utc)
            fast_frame = (
                match_canonical_responses_delta_frame(chunk)
                if (
                    variant in {"fast", "fast_normalizer"}
                    and output_seen
                    and len(chunk)
                    >= MIN_CANONICAL_RESPONSES_DELTA_FRAME_BYTES
                    and chunk.startswith(b"event: response.")
                    and chunk.endswith(b"\n\n")
                    and parser.can_bypass_complete_frame
                )
                else None
            )
            if fast_frame is not None and await (
                can_forward_responses_delta_without_materializing(
                    fast_frame,
                    workspace=parse_workspace,
                    item_id_requires_full_normalization=(
                        item_id_requires_full_normalization
                    ),
                )
            ):
                diagnostics.observe_complete_event(
                    "",
                    has_data_field=True,
                    event_type=fast_frame.event_type,
                    wire_bytes=chunk,
                    received_at=received_at,
                )
                available_at.append(chunk_available_at)
                yielded_at.append(time.perf_counter_ns())
                fast_path_events += 1
                yield ObservedStreamChunk(
                    chunk,
                    event_type=fast_frame.event_type,
                    semantic_outcome="nonterminal",
                    sse_metadata_complete=True,
                )
                continue
            raw_events = parser.feed(chunk)
            try:
                for event_index in range(len(raw_events)):
                    raw_event = raw_events[event_index]
                    owner = await parse_owned_sse_event(
                        raw_event,
                        workspace=parse_workspace,
                    )
                    payload = None
                    try:
                        payload = owner.payload
                        event_type = owner.event_name
                        wire = raw_event.encode("utf-8") + b"\n\n"
                        diagnostics.observe_complete_event(
                            raw_event,
                            has_data_field=owner.has_data_field,
                            event_type=event_type,
                            wire_bytes=wire,
                            received_at=received_at,
                        )
                        validate_sse_event_type_consistency(
                            owner.declared_event_name,
                            payload,
                            protocol_name="Responses",
                            has_event_field=owner.has_event_field,
                            require_event_name=True,
                        )
                        semantic_outcome = (
                            "completed"
                            if event_type == "response.completed"
                            else "nonterminal"
                        )
                        diagnostics.observe_parsed_event(
                            raw_event,
                            event_type,
                            payload,
                            semantic_outcome=semantic_outcome,
                            wire_bytes=wire,
                            received_at=received_at,
                        )
                        if (
                            not output_seen
                            and event_type.endswith(".delta")
                            and isinstance(payload, dict)
                            and payload.get("delta")
                        ):
                            output_seen = True
                        available_at.append(chunk_available_at)
                        yielded_at.append(time.perf_counter_ns())
                        if variant != "reparse":
                            snapshot = stream_usage_snapshot_from_payload(
                                payload
                            )
                            yield ObservedStreamChunk(
                                wire,
                                event_type=event_type,
                                semantic_outcome=semantic_outcome,
                                sse_metadata_complete=True,
                                usage_snapshot=snapshot,
                            )
                        else:
                            yield wire
                    finally:
                        payload = None
                        await owner.aclose()
                        raw_events[event_index] = ""
            finally:
                raw_events.clear()
        parser.finish()

    async def send(message):
        nonlocal sent_bytes
        body_bytes = message.get("body", b"")
        if body_bytes:
            sent_at = time.perf_counter_ns()
            sent_bytes += len(body_bytes)
            frame_latencies_us.append(
                (sent_at - yielded_at.popleft()) / 1000.0
            )
            pipeline_latencies_us.append(
                (sent_at - available_at.popleft()) / 1000.0
            )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    response = LoggingStreamingResponse(
        body(),
        media_type="text/event-stream",
        current_info=current_info,
    )
    try:
        await response(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )
    finally:
        await parse_workspace.aclose()
    if yielded_at:
        raise AssertionError("not every yielded frame reached ASGI send")
    if available_at:
        raise AssertionError("not every available frame reached ASGI send")
    if current_info.get("completion_tokens") != expected_output_tokens:
        raise AssertionError("downstream usage was not preserved")
    return (
        sent_bytes,
        frame_latencies_us,
        pipeline_latencies_us,
        fast_path_events,
    )


async def _one_stream(
    transport_chunks: tuple[bytes, ...],
    *,
    variant: str,
    admission: RequestAdmissionController | None,
    expected_output_tokens: int,
) -> tuple[int, list[float], list[float], int]:
    if admission is None:
        return await _one_bound_stream(
            transport_chunks,
            variant=variant,
            expected_output_tokens=expected_output_tokens,
        )
    lease = await admission.acquire()
    token = bind_request_admission_lease(lease)
    try:
        return await _one_bound_stream(
            transport_chunks,
            variant=variant,
            expected_output_tokens=expected_output_tokens,
        )
    finally:
        reset_request_admission_lease(token)
        await lease.release()


async def _measure(args, *, variant: str) -> dict[str, float | int | str]:
    event_frames = _transport_events(
        args.events_per_stream,
        args.delta_bytes,
    )
    if args.transport_chunk_bytes:
        wire = b"".join(event_frames)
        transport_chunks = tuple(
            wire[offset : offset + args.transport_chunk_bytes]
            for offset in range(0, len(wire), args.transport_chunk_bytes)
        )
    else:
        transport_chunks = event_frames
    total_events = args.concurrency * len(event_frames)
    admission = (
        RequestAdmissionController(
            capacity=args.concurrency,
            waiter_limit=0,
            wait_timeout_seconds=10,
            max_body_bytes=16 * 1024 * 1024,
            body_budget_bytes=1024 * 1024 * 1024,
            max_response_bytes=128 * 1024 * 1024,
        )
        if args.admission
        else None
    )
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    results = await asyncio.gather(
        *(
            _one_stream(
                transport_chunks,
                variant=variant,
                admission=admission,
                expected_output_tokens=args.events_per_stream,
            )
            for _ in range(args.concurrency)
        )
    )
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    latencies = sorted(
        latency
        for (
            _sent_bytes,
            stream_latencies,
            _pipeline_latencies,
            _fast_events,
        ) in results
        for latency in stream_latencies
    )
    pipeline_latencies = sorted(
        latency
        for (
            _sent_bytes,
            _stream_latencies,
            stream_pipeline_latencies,
            _fast_events,
        ) in results
        for latency in stream_pipeline_latencies
    )
    return {
        "variant": variant,
        "events": total_events,
        "bytes": sum(
            sent_bytes
            for (
                sent_bytes,
                _latencies,
                _pipeline_latencies,
                _fast_events,
            ) in results
        ),
        "fast_path_events": sum(
            fast_events
            for (
                _sent_bytes,
                _latencies,
                _pipeline_latencies,
                fast_events,
            ) in results
        ),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "events_per_wall_second": total_events / wall_seconds,
        "cpu_us_per_event": cpu_seconds * 1_000_000 / total_events,
        "frame_latency_us_p50": latencies[(len(latencies) - 1) // 2],
        "frame_latency_us_p95": latencies[
            int((len(latencies) - 1) * 0.95)
        ],
        "pipeline_latency_us_p50": pipeline_latencies[
            (len(pipeline_latencies) - 1) // 2
        ],
        "pipeline_latency_us_p95": pipeline_latencies[
            int((len(pipeline_latencies) - 1) * 0.95)
        ],
    }


async def _benchmark(args) -> None:
    measurements: dict[str, list[dict[str, float | int | str]]] = {
        variant: [] for variant in args.variants
    }
    for round_index in range(args.rounds):
        round_variants = (
            args.variants
            if round_index % 2 == 0
            else tuple(reversed(args.variants))
        )
        for variant in round_variants:
            result = await _measure(args, variant=variant)
            measurements[str(result["variant"])].append(result)

    summaries = {}
    for variant, samples in measurements.items():
        summaries[variant] = {
            "best_events_per_wall_second": max(
                float(item["events_per_wall_second"]) for item in samples
            ),
            "best_cpu_us_per_event": min(
                float(item["cpu_us_per_event"]) for item in samples
            ),
            "median_events_per_wall_second": statistics.median(
                float(item["events_per_wall_second"]) for item in samples
            ),
            "median_cpu_us_per_event": statistics.median(
                float(item["cpu_us_per_event"]) for item in samples
            ),
            "median_frame_latency_us_p50": statistics.median(
                float(item["frame_latency_us_p50"]) for item in samples
            ),
            "median_frame_latency_us_p95": statistics.median(
                float(item["frame_latency_us_p95"]) for item in samples
            ),
            "median_pipeline_latency_us_p50": statistics.median(
                float(item["pipeline_latency_us_p50"]) for item in samples
            ),
            "median_pipeline_latency_us_p95": statistics.median(
                float(item["pipeline_latency_us_p95"]) for item in samples
            ),
            "samples": samples,
        }
    comparison = {}
    if "reparse" in summaries and "metadata" in summaries:
        baseline_cpu = summaries["reparse"]["best_cpu_us_per_event"]
        optimized_cpu = summaries["metadata"]["best_cpu_us_per_event"]
        baseline_throughput = summaries["reparse"][
            "best_events_per_wall_second"
        ]
        optimized_throughput = summaries["metadata"][
            "best_events_per_wall_second"
        ]
        comparison = {
            "cpu_reduction_percent": (
                100.0 * (baseline_cpu - optimized_cpu) / baseline_cpu
            ),
            "throughput_gain_percent": (
                100.0
                * (optimized_throughput - baseline_throughput)
                / baseline_throughput
            ),
        }
    if "metadata" in summaries and "fast" in summaries:
        baseline_cpu = summaries["metadata"]["best_cpu_us_per_event"]
        optimized_cpu = summaries["fast"]["best_cpu_us_per_event"]
        baseline_throughput = summaries["metadata"][
            "best_events_per_wall_second"
        ]
        optimized_throughput = summaries["fast"][
            "best_events_per_wall_second"
        ]
        comparison.update(
            {
                "fast_cpu_reduction_percent_vs_metadata": (
                    100.0 * (baseline_cpu - optimized_cpu) / baseline_cpu
                ),
                "fast_throughput_gain_percent_vs_metadata": (
                    100.0
                    * (optimized_throughput - baseline_throughput)
                    / baseline_throughput
                ),
            }
        )
    if "metadata" in summaries and "fast_normalizer" in summaries:
        baseline_cpu = summaries["metadata"]["best_cpu_us_per_event"]
        optimized_cpu = summaries["fast_normalizer"][
            "best_cpu_us_per_event"
        ]
        baseline_throughput = summaries["metadata"][
            "best_events_per_wall_second"
        ]
        optimized_throughput = summaries["fast_normalizer"][
            "best_events_per_wall_second"
        ]
        median_baseline_cpu = summaries["metadata"][
            "median_cpu_us_per_event"
        ]
        median_optimized_cpu = summaries["fast_normalizer"][
            "median_cpu_us_per_event"
        ]
        median_baseline_throughput = summaries["metadata"][
            "median_events_per_wall_second"
        ]
        median_optimized_throughput = summaries["fast_normalizer"][
            "median_events_per_wall_second"
        ]
        comparison.update(
            {
                "fast_normalizer_cpu_reduction_percent_vs_metadata": (
                    100.0 * (baseline_cpu - optimized_cpu) / baseline_cpu
                ),
                "fast_normalizer_throughput_gain_percent_vs_metadata": (
                    100.0
                    * (optimized_throughput - baseline_throughput)
                    / baseline_throughput
                ),
                "fast_normalizer_median_cpu_reduction_percent_vs_metadata": (
                    100.0
                    * (median_baseline_cpu - median_optimized_cpu)
                    / median_baseline_cpu
                ),
                "fast_normalizer_median_throughput_gain_percent_vs_metadata": (
                    100.0
                    * (
                        median_optimized_throughput
                        - median_baseline_throughput
                    )
                    / median_baseline_throughput
                ),
            }
        )
    print(
        json.dumps(
            {
                "concurrency": args.concurrency,
                "events_per_stream": args.events_per_stream,
                "delta_bytes": args.delta_bytes,
                "transport_chunk_bytes": args.transport_chunk_bytes,
                "rounds": args.rounds,
                "admission": args.admission,
                "summaries": summaries,
                **comparison,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--events-per-stream", type=int, default=500)
    parser.add_argument("--delta-bytes", type=int, default=64)
    parser.add_argument("--transport-chunk-bytes", type=int, default=0)
    parser.add_argument("--admission", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("reparse", "metadata", "fast", "fast_normalizer"),
        default=("reparse", "metadata", "fast", "fast_normalizer"),
    )
    parser.add_argument(
        "--loop",
        choices=("asyncio", "uvloop"),
        default="uvloop",
    )
    args = parser.parse_args()
    if min(
        args.rounds,
        args.concurrency,
        args.events_per_stream,
        args.delta_bytes,
    ) <= 0:
        parser.error("benchmark dimensions must be positive")
    if args.transport_chunk_bytes < 0:
        parser.error("transport-chunk-bytes cannot be negative")
    if args.loop == "uvloop":
        import uvloop

        uvloop.run(_benchmark(args))
    else:
        asyncio.run(_benchmark(args))


if __name__ == "__main__":
    main()
