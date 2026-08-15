#!/usr/bin/env python3
"""Measure Rust container RSS after concurrent large Responses requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, FixtureHandler)
        self.request_count = 0
        self.request_lock = threading.Lock()

    def config(self) -> dict[str, Any]:
        port = self.server_address[1]
        return {
            "providers": [
                {
                    "provider": "allocator-benchmark",
                    "engine": "codex",
                    "base_url": (
                        f"http://host.docker.internal:{port}/v1/responses"
                    ),
                    "api": "sk-upstream",
                    "model": [{"gpt-upstream": "gpt-memory"}],
                    "preferences": {
                        "model_timeout": {"default": 120},
                        "cooldown_period": 0,
                        "api_key_cooldown_period": 0,
                        "post_body_parameter_overrides": {"store": False},
                    },
                }
            ],
            "api_keys": [
                {
                    "api": "sk-benchmark",
                    "model": ["gpt-memory"],
                    "preferences": {"AUTO_RETRY": True},
                }
            ],
            "preferences": {"rate_limit": "999999/min", "cooldown_period": 0},
        }


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path != "/config":
            self.send_error(404)
            return
        body = json.dumps(self.server.config()).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("content-type", "application/yaml")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/responses":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length") or 0)
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        with self.server.request_lock:  # type: ignore[attr-defined]
            self.server.request_count += 1  # type: ignore[attr-defined]
        body = (
            b'event: response.created\ndata: {"type":"response.created",'
            b'"response":{"status":"in_progress"}}\n\n'
            b'event: response.output_text.delta\ndata: {"type":'
            b'"response.output_text.delta","delta":"ok"}\n\n'
            b'event: response.completed\ndata: {"type":"response.completed",'
            b'"response":{"status":"completed","usage":{"input_tokens":1,'
            b'"output_tokens":1,"total_tokens":2}}}\n\n'
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _docker(*args: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["docker", *args],
        check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def _container_memory(container: str) -> int:
    return int(
        _docker(
            "exec",
            container,
            "sh",
            "-c",
            "cat /sys/fs/cgroup/memory.current",
        )
    )


def _container_memory_peak(container: str) -> int | None:
    for path in (
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
    ):
        try:
            return int(_docker("exec", container, "cat", path))
        except (subprocess.CalledProcessError, ValueError):
            continue
    return None


def _process_status(container: str) -> dict[str, int]:
    raw = _docker(
        "exec",
        container,
        "sh",
        "-c",
        "grep -E '^(VmRSS|VmHWM|RssAnon|Threads):' /proc/1/status",
    )
    parsed: dict[str, int] = {}
    for line in raw.splitlines():
        name, value = line.split(":", 1)
        parsed[name] = int(value.strip().split()[0])
    return parsed


def _wait_healthy(port: int) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/healthz", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise RuntimeError("container did not become healthy")


def _post(port: int, body: bytes) -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/responses",
        data=body,
        headers={
            "authorization": "Bearer sk-benchmark",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        wire = response.read()
        if response.status != 200 or b"event: response.completed" not in wire:
            raise RuntimeError(f"unexpected response: {response.status} {wire[:200]!r}")
        return len(wire)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--allocator", choices=("default", "bounded"), default="default")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--request-mib", type=int, default=8)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.concurrency <= 0 or args.rounds <= 0 or args.request_mib <= 0:
        parser.error("concurrency, rounds, and request-mib must be positive")

    fixture = FixtureServer(("0.0.0.0", _free_port()))
    fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    fixture_thread.start()
    container = f"uni-api-allocator-{uuid.uuid4().hex[:10]}"
    port = _free_port()
    command = [
        "run",
        "--detach",
        "--rm",
        "--name",
        container,
        "--memory",
        "2g",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        f"127.0.0.1:{port}:8000",
        "-e",
        f"CONFIG_URL=http://host.docker.internal:{fixture.server_address[1]}/config",
        "-e",
        "DISABLE_DATABASE=true",
        "-e",
        "STDOUT_REQUEST_SUMMARY_LOG_ENABLED=false",
        "-e",
        "FUGUE_OBSERVABILITY_ENABLED=false",
    ]
    if args.allocator == "bounded":
        command.extend(
            [
                "-e",
                "MALLOC_ARENA_MAX=2",
                "-e",
                "MALLOC_MMAP_THRESHOLD_=131072",
                "-e",
                "MALLOC_TRIM_THRESHOLD_=131072",
            ]
        )
    command.append(args.image)

    monitor_stop = threading.Event()
    samples: list[int] = []

    def monitor() -> None:
        while not monitor_stop.wait(0.1):
            try:
                samples.append(_container_memory(container))
            except (subprocess.CalledProcessError, ValueError):
                return

    try:
        _docker(*command)
        _wait_healthy(port)
        baseline = _container_memory(container)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        input_text = "x" * (args.request_mib * 1024 * 1024)
        body = json.dumps(
            {"model": "gpt-memory", "input": input_text, "stream": True},
            separators=(",", ":"),
        ).encode()
        started = time.perf_counter()
        response_bytes = 0
        for _ in range(args.rounds):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                response_bytes += sum(
                    pool.map(lambda _: _post(port, body), range(args.concurrency))
                )
        elapsed = time.perf_counter() - started
        time.sleep(args.idle_seconds)
        idle = _container_memory(container)
        status = _process_status(container)
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        print(
            json.dumps(
                {
                    "image": args.image,
                    "allocator": args.allocator,
                    "requests": args.concurrency * args.rounds,
                    "request_mib": args.request_mib,
                    "elapsed_seconds": elapsed,
                    "response_bytes": response_bytes,
                    "baseline_bytes": baseline,
                    "peak_bytes": max(samples, default=baseline),
                    "kernel_peak_bytes": _container_memory_peak(container),
                    "idle_bytes": idle,
                    "process_status_kib": status,
                    "fixture_requests": fixture.request_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        monitor_stop.set()
        subprocess.run(
            ["docker", "stop", "--time", "2", container],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        fixture.shutdown()
        fixture.server_close()
        fixture_thread.join(timeout=2)


if __name__ == "__main__":
    main()
