#!/usr/bin/env python3
"""Local no-token TCP benchmark for the Rust Responses data plane."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUST_BINARY = (
    ROOT
    / "rust"
    / "uni-api-native"
    / "target"
    / "release"
    / "uni-api-front"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], event_count: int):
        super().__init__(address, FixtureHandler)
        self.event_count = event_count
        self.retry_probe_failed = False
        self.retry_probe_lock = threading.Lock()
        self.request_counts: dict[str, int] = {}
        self.request_counts_lock = threading.Lock()

    def config(self) -> dict[str, Any]:
        port = self.server_address[1]
        return {
            "providers": [
                {
                    "provider": provider,
                    "engine": "codex",
                    "base_url": f"http://127.0.0.1:{port}/v1/responses",
                    "api": f"sk-{provider}",
                    "model": [{"gpt-upstream": "gpt-benchmark"}],
                    "preferences": {
                        "model_timeout": {"default": 120},
                        "cooldown_period": 0,
                        "api_key_cooldown_period": 0,
                        "post_body_parameter_overrides": {
                            "store": False,
                            "__remove__": ["temperature"],
                        },
                    },
                }
                for provider in ("benchmark-a", "benchmark-b")
            ],
            "api_keys": [
                {
                    "api": "sk-benchmark",
                    "model": ["gpt-benchmark"],
                    "preferences": {"AUTO_RETRY": True},
                }
            ],
            "preferences": {
                "rate_limit": "999999/min",
                "cooldown_period": 0,
            },
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
        payload = json.loads(self.rfile.read(length) or b"{}")
        if (
            payload.get("model") != "gpt-upstream"
            or payload.get("store") is not False
            or payload.get("instructions") != ""
        ):
            body = json.dumps(
                {"error": "request-side provider payload compilation mismatch"}
            ).encode()
            self.send_response(422)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        count = (
            100
            if payload.get("input") == "warmup"
            else self.server.event_count  # type: ignore[attr-defined]
        )
        input_value = str(payload.get("input") or "")
        with self.server.request_counts_lock:  # type: ignore[attr-defined]
            request_number = self.server.request_counts.get(input_value, 0) + 1  # type: ignore[attr-defined]
            self.server.request_counts[input_value] = request_number  # type: ignore[attr-defined]
        if input_value == "idempotency-wait-probe":
            time.sleep(0.2)
        if not payload.get("stream"):
            body = json.dumps(
                {
                    "id": "resp-benchmark",
                    "status": "completed",
                    "input": input_value,
                },
                separators=(",", ":"),
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("x-fixture-request-number", str(request_number))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("x-fixture-request-number", str(request_number))
        self.end_headers()
        try:
            if payload.get("input") == "retry-after-keepalive":
                with self.server.retry_probe_lock:  # type: ignore[attr-defined]
                    should_fail = not self.server.retry_probe_failed  # type: ignore[attr-defined]
                    self.server.retry_probe_failed = True  # type: ignore[attr-defined]
                if should_fail:
                    self.wfile.write(
                        b'event: keepalive\ndata: {"type":"keepalive",'
                        b'"sequence_number":0}\n\n'
                    )
                    self.wfile.write(
                        b'event: response.created\ndata: {"type":"response.created",'
                        b'"response":{"status":"in_progress"}}\n\n'
                    )
                    self.wfile.write(
                        b'event: response.failed\ndata: {"type":"response.failed",'
                        b'"response":{"status":"failed","error":{"message":'
                        b'"retry probe","type":"server_error","code":'
                        b'"upstream_unavailable"}}}\n\n'
                    )
                    self.wfile.flush()
                    self.close_connection = True
                    return
            self.wfile.write(
                b'event: response.created\ndata: {"type":"response.created",'
                b'"response":{"status":"in_progress"}}\n\n'
            )
            for _ in range(count):
                self.wfile.write(
                    b'event: response.output_text.delta\ndata: {"type":'
                    b'"response.output_text.delta","delta":"x"}\n\n'
                )
            self.wfile.write(
                b'event: response.completed\ndata: {"type":"response.completed",'
                b'"response":{"status":"completed","usage":{"input_tokens":1,'
                b'"output_tokens":1,"total_tokens":2}}}\n\n'
            )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _request_json(
    url: str,
    *,
    timeout: float = 5.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait_healthy(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 90
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup: {process.returncode}")
        try:
            if _request_json(url).get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.05)
    raise RuntimeError("service did not become healthy")


def _post_stream(
    port: int,
    input_value: str,
    *,
    idempotency_key: str | None = None,
) -> tuple[bytes, dict[str, str], float]:
    body = json.dumps(
        {
            "model": "gpt-benchmark",
            "input": input_value,
            "stream": True,
        }
    ).encode()
    headers = {
        "authorization": "Bearer sk-benchmark",
        "content-type": "application/json",
    }
    if idempotency_key is not None:
        headers["idempotency-key"] = idempotency_key
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/responses",
        data=body,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        wire = response.read()
        headers = {name.lower(): value for name, value in response.headers.items()}
        if response.status != 200:
            raise RuntimeError(f"unexpected response status {response.status}")
    return wire, headers, time.perf_counter() - started


def _post_nonstream(
    port: int,
    input_value: str,
    *,
    idempotency_key: str | None = None,
) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(
        {
            "model": "gpt-benchmark",
            "input": input_value,
            "stream": False,
        }
    ).encode()
    headers = {
        "authorization": "Bearer sk-benchmark",
        "content-type": "application/json",
    }
    if idempotency_key is not None:
        headers["idempotency-key"] = idempotency_key
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/responses",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        wire = response.read()
        response_headers = {
            name.lower(): value for name, value in response.headers.items()
        }
        if response.status != 200:
            raise RuntimeError(f"unexpected response status {response.status}")
    return wire, response_headers


def _worker_cpu_seconds(port: int) -> float | None:
    snapshot = _request_json(
        f"http://127.0.0.1:{port}/v1/observability/runtime",
        headers={"authorization": "Bearer sk-benchmark"},
    )

    def find(value: Any) -> float | None:
        if isinstance(value, dict):
            observed = value.get("worker_cpu_seconds_total")
            if isinstance(observed, (int, float)):
                return float(observed)
            for child in value.values():
                result = find(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = find(child)
                if result is not None:
                    return result
        return None

    return find(snapshot)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_service(
    *,
    mode: str,
    config_url: str,
    requests: int,
    event_count: int,
    large_request_mib: int = 0,
) -> dict[str, Any]:
    public_port = _free_port()
    backend_port = _free_port()
    env = {
        **os.environ,
        "CONFIG_URL": config_url,
        "DISABLE_DATABASE": "true",
        "HOST": "127.0.0.1",
        "PORT": str(public_port),
        "STDOUT_REQUEST_SUMMARY_LOG_ENABLED": "false",
        "FUGUE_OBSERVABILITY_ENABLED": "false",
        "WORKER_RUNTIME_SAMPLE_INTERVAL_SECONDS": "0.1",
    }
    with tempfile.TemporaryDirectory(prefix=f"uni-api-{mode}-") as workdir:
        if mode == "python":
            command = [str(PYTHON), str(ROOT / "main.py")]
        else:
            command = [str(RUST_BINARY)]
            env.update(
                {
                    "UNI_API_PYTHON_EXECUTABLE": str(PYTHON),
                    "UNI_API_PYTHON_MAIN": str(ROOT / "main.py"),
                    "UNI_API_PYTHON_PORT": str(backend_port),
                    "PYTHONPATH": str(ROOT),
                    "RUST_REQUEST_SPOOL_DIRECTORY": str(Path(workdir) / "request-spool"),
                    "RUST_REQUEST_SPOOL_DISK_RESERVE_BPS": "0",
                    "RUST_REQUEST_SPOOL_INODE_RESERVE_BPS": "0",
                }
            )
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_healthy(public_port, process)
            if mode == "rust":
                retry_wire, retry_headers, _ = _post_stream(
                    public_port,
                    "retry-after-keepalive",
                )
                if retry_headers.get("x-uni-api-data-plane") != "rust-native-v2":
                    raise RuntimeError(
                        "retry probe did not use the native Rust control/data plane: "
                        f"{retry_headers.get('x-uni-api-data-plane')}"
                    )
                if retry_wire.count(b"event: keepalive") != 1:
                    raise RuntimeError(
                        "retry probe duplicated or lost the precommit keepalive"
                    )
                if b"event: response.failed" in retry_wire:
                    raise RuntimeError("precommit provider failure leaked downstream")
                if b"event: response.completed" not in retry_wire:
                    raise RuntimeError("retry probe did not reach the fallback provider")
                first, first_headers, _ = _post_stream(
                    public_port,
                    "idempotency-probe",
                    idempotency_key="benchmark-idempotency-1",
                )
                replay, replay_headers, _ = _post_stream(
                    public_port,
                    "idempotency-probe",
                    idempotency_key="benchmark-idempotency-1",
                )
                if first != replay:
                    raise RuntimeError("Rust idempotency replay changed the SSE wire")
                if first_headers.get("x-uni-api-idempotency-status") != "executed":
                    raise RuntimeError("Rust idempotency owner status is missing")
                if replay_headers.get("x-uni-api-idempotency-status") != "replayed":
                    raise RuntimeError("Rust idempotency replay status is missing")
                if replay_headers.get("x-uni-api-data-plane") != "rust-native-v2":
                    raise RuntimeError("Rust idempotency replay lost data-plane provenance")
                if replay_headers.get("x-fixture-request-number") != "1":
                    raise RuntimeError("Rust idempotency replay called the provider twice")
                nonstream, nonstream_headers = _post_nonstream(
                    public_port,
                    "idempotency-nonstream-probe",
                    idempotency_key="benchmark-idempotency-nonstream-1",
                )
                nonstream_replay, nonstream_replay_headers = _post_nonstream(
                    public_port,
                    "idempotency-nonstream-probe",
                    idempotency_key="benchmark-idempotency-nonstream-1",
                )
                if nonstream != nonstream_replay:
                    raise RuntimeError(
                        "Rust non-stream idempotency replay changed the response body"
                    )
                if nonstream_headers.get("x-uni-api-idempotency-status") != "executed":
                    raise RuntimeError("Rust non-stream idempotency owner status is missing")
                if nonstream_headers.get("x-uni-api-data-plane") != "rust-native-v2":
                    raise RuntimeError("Rust non-stream request left the native control plane")
                if nonstream_replay_headers.get("x-uni-api-idempotency-status") != "replayed":
                    raise RuntimeError("Rust non-stream idempotency replay status is missing")
                if nonstream_replay_headers.get("x-fixture-request-number") != "1":
                    raise RuntimeError("Rust non-stream idempotency called the provider twice")
                try:
                    _post_stream(
                        public_port,
                        "idempotency-conflict",
                        idempotency_key="benchmark-idempotency-1",
                    )
                except urllib.error.HTTPError as error:
                    if error.code != 409:
                        raise RuntimeError(
                            f"Rust idempotency conflict returned {error.code}"
                        ) from error
                else:
                    raise RuntimeError("Rust idempotency conflict was not rejected")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    waiting = [
                        pool.submit(
                            _post_stream,
                            public_port,
                            "idempotency-wait-probe",
                            idempotency_key="benchmark-idempotency-wait-1",
                        )
                        for _ in range(2)
                    ]
                    waiting_results = [future.result() for future in waiting]
                if waiting_results[0][0] != waiting_results[1][0]:
                    raise RuntimeError("Rust idempotency waiter replay changed the SSE wire")
                waiter_statuses = sorted(
                    result[1].get("x-uni-api-idempotency-status")
                    for result in waiting_results
                )
                if waiter_statuses != ["executed", "replayed"]:
                    raise RuntimeError(
                        f"Rust idempotency waiter statuses are {waiter_statuses}"
                    )
                if any(
                    result[1].get("x-fixture-request-number") != "1"
                    for result in waiting_results
                ):
                    raise RuntimeError("Rust idempotency waiter called the provider twice")
                if large_request_mib > 0:
                    large_input = "large-request-probe:" + (
                        "x" * (large_request_mib * 1024 * 1024)
                    )
                    large_wire, large_headers, _ = _post_stream(
                        public_port,
                        large_input,
                        idempotency_key="benchmark-large-idempotency-1",
                    )
                    replay_wire, replay_headers, _ = _post_stream(
                        public_port,
                        large_input,
                        idempotency_key="benchmark-large-idempotency-1",
                    )
                    if large_wire != replay_wire:
                        raise RuntimeError("large Rust idempotency replay changed the SSE wire")
                    if large_headers.get("x-uni-api-idempotency-status") != "executed":
                        raise RuntimeError("large Rust idempotency request was not executed")
                    if replay_headers.get("x-uni-api-idempotency-status") != "replayed":
                        raise RuntimeError("large Rust idempotency request was not replayed")
                    if replay_headers.get("x-fixture-request-number") != "1":
                        raise RuntimeError("large Rust idempotency request called the provider twice")
                    direct_large_wire, direct_large_headers, _ = _post_stream(
                        public_port,
                        "large-request-no-idempotency:"
                        + ("y" * (large_request_mib * 1024 * 1024)),
                    )
                    if b"event: response.completed" not in direct_large_wire:
                        raise RuntimeError(
                            "large non-idempotent Rust request lost its terminal event"
                        )
                    if direct_large_headers.get("x-uni-api-data-plane") != "rust-native-v2":
                        raise RuntimeError(
                            "large non-idempotent request left the Rust data plane"
                        )
            warmup, warmup_headers, _ = _post_stream(public_port, "warmup")
            if warmup.count(b"response.output_text.delta") != 200:
                raise RuntimeError(
                    "warmup stream was not preserved: "
                    f"delta_markers={warmup.count(b'response.output_text.delta')} "
                    f"headers={warmup_headers} wire_prefix={warmup[:500]!r}"
                )
            time.sleep(0.25)
            cpu_before = _worker_cpu_seconds(public_port)
            elapsed = 0.0
            wire_bytes = 0
            data_plane = None
            for _ in range(requests):
                wire, headers, request_elapsed = _post_stream(
                    public_port,
                    "benchmark",
                )
                if wire.count(b"response.output_text.delta") != event_count * 2:
                    raise RuntimeError("benchmark stream lost or duplicated delta frames")
                if b"event: response.completed" not in wire:
                    raise RuntimeError("benchmark stream lost the terminal event")
                elapsed += request_elapsed
                wire_bytes += len(wire)
                data_plane = headers.get("x-uni-api-data-plane")
            time.sleep(0.25)
            cpu_after = _worker_cpu_seconds(public_port)
            cpu_seconds = (
                cpu_after - cpu_before
                if cpu_before is not None and cpu_after is not None
                else None
            )
            total_events = requests * (event_count + 2)
            return {
                "mode": mode,
                "requests": requests,
                "events": total_events,
                "wire_bytes": wire_bytes,
                "elapsed_seconds": elapsed,
                "events_per_second": total_events / elapsed,
                "mebibytes_per_second": wire_bytes / elapsed / (1024 * 1024),
                "python_worker_cpu_seconds": cpu_seconds,
                "python_worker_mcpu": (
                    cpu_seconds / elapsed * 1000 if cpu_seconds is not None else None
                ),
                "data_plane": data_plane,
            }
        finally:
            _stop(process)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=20_000)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--large-request-mib", type=int, default=0)
    args = parser.parse_args()
    if args.events <= 0 or args.requests <= 0 or args.large_request_mib < 0:
        parser.error("--events/--requests must be positive and --large-request-mib non-negative")
    if not RUST_BINARY.exists():
        raise SystemExit(
            f"Rust release binary is missing: run cargo build --release in "
            f"{RUST_BINARY.parents[3]}"
        )

    fixture = FixtureServer(("127.0.0.1", 0), event_count=args.events)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    config_url = f"http://127.0.0.1:{fixture.server_address[1]}/config"
    try:
        python_result = _run_service(
            mode="python",
            config_url=config_url,
            requests=args.requests,
            event_count=args.events,
            large_request_mib=0,
        )
        rust_result = _run_service(
            mode="rust",
            config_url=config_url,
            requests=args.requests,
            event_count=args.events,
            large_request_mib=args.large_request_mib,
        )
    finally:
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=5)

    result = {
        "python": python_result,
        "rust": rust_result,
        "throughput_speedup": (
            rust_result["events_per_second"]
            / python_result["events_per_second"]
        ),
        "python_worker_cpu_reduction": (
            1
            - rust_result["python_worker_cpu_seconds"]
            / python_result["python_worker_cpu_seconds"]
            if python_result["python_worker_cpu_seconds"]
            and rust_result["python_worker_cpu_seconds"] is not None
            else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
