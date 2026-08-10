"""Compare the native and Python JSON memory guards on repeatable payloads."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uni_api.admission.json_memory import (
    IncrementalJSONMemoryEstimator,
    json_memory_native_available,
)


@dataclass(frozen=True, slots=True)
class Result:
    payload: str
    chunk_bytes: int
    payload_bytes: int
    python_seconds: float
    native_seconds: float
    python_mib_per_second: float
    native_mib_per_second: float
    speedup: float


def _payloads() -> dict[str, bytes]:
    return {
        "long_string": json.dumps(
            {"model": "gpt", "input": "x" * (4 * 1024 * 1024)},
            separators=(",", ":"),
        ).encode(),
        "token_dense": json.dumps(
            [{"a": index, "b": True} for index in range(120_000)],
            separators=(",", ":"),
        ).encode(),
        "mixed_messages": json.dumps(
            {
                "model": "gpt",
                "messages": [
                    {
                        "role": "user" if index % 2 == 0 else "assistant",
                        "content": "x" * 16_384,
                        "metadata": {"i": index, "ok": True},
                    }
                    for index in range(256)
                ],
            },
            separators=(",", ":"),
        ).encode(),
    }


def _scan(payload: bytes, chunk_bytes: int, *, native: bool):
    estimator = IncrementalJSONMemoryEstimator(
        max_estimated_bytes=4 * 1024**3,
    )
    estimator._native_enabled = native
    for offset in range(0, len(payload), chunk_bytes):
        estimator.feed(payload[offset : offset + chunk_bytes])
    return estimator.snapshot()


def _median_seconds(callback, iterations: int) -> float:
    callback()
    samples = []
    for _ in range(iterations):
        gc.collect()
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _benchmark(iterations: int) -> list[Result]:
    if not json_memory_native_available():
        raise SystemExit(
            "native JSON guard is unavailable; build uni_api._uni_api_native first"
        )

    results = []
    for payload_name, payload in _payloads().items():
        for chunk_bytes in (len(payload), 64 * 1024):
            python_snapshot = _scan(payload, chunk_bytes, native=False)
            native_snapshot = _scan(payload, chunk_bytes, native=True)
            if native_snapshot != python_snapshot:
                raise RuntimeError(
                    f"native result differs for {payload_name}/{chunk_bytes}"
                )
            python_seconds = _median_seconds(
                lambda: _scan(payload, chunk_bytes, native=False),
                iterations,
            )
            native_seconds = _median_seconds(
                lambda: _scan(payload, chunk_bytes, native=True),
                iterations,
            )
            payload_mib = len(payload) / 1024 / 1024
            results.append(
                Result(
                    payload=payload_name,
                    chunk_bytes=chunk_bytes,
                    payload_bytes=len(payload),
                    python_seconds=python_seconds,
                    native_seconds=native_seconds,
                    python_mib_per_second=payload_mib / python_seconds,
                    native_mib_per_second=payload_mib / native_seconds,
                    speedup=python_seconds / native_seconds,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    results = _benchmark(args.iterations)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return
    for result in results:
        print(
            f"{result.payload:14} chunk={result.chunk_bytes:7} "
            f"python={result.python_mib_per_second:9.2f} MiB/s "
            f"native={result.native_mib_per_second:9.2f} MiB/s "
            f"speedup={result.speedup:7.2f}x"
        )


if __name__ == "__main__":
    main()
