"""Offline regressions for hostile cooldowns, body admission and bounded SSE."""
import http.client
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from verify_dispatch_timing import free_port, get_json


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append((self.path, self.headers.get("Authorization")))
        if self.path.startswith("/native/") and self.headers.get("Authorization") == "Bearer first":
            status, content_type = 429, "application/json"
            raw = json.dumps({"error": {"message": "Too many requests, please try again in " + "9" * 80 + " seconds"}}).encode()
        elif payload.get("stream"):
            status, content_type, raw = 200, "text/event-stream", b"data: " + b"x" * 2048
        elif self.path.startswith("/native/"):
            status, content_type = 200, "application/json"
            raw = json.dumps({"id": "resp_fixture", "object": "response", "status": "completed",
                              "model": "fixture", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        else:
            status, content_type = 200, "application/json"
            raw = json.dumps({"id": "chat_fixture", "object": "chat.completion", "model": "fixture",
                              "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def call(port, path, payload, compressed=False, target=None):
    raw = json.dumps(payload).encode()
    headers = {"Authorization": "Bearer fixture-key", "Content-Type": "application/json"}
    if target is not None:
        headers["X-Uni-API-Provider"] = target
    if compressed:
        # Valid single-segment zstd frame with one raw, final block. No optional
        # Python packages or external encoder are needed by this fixture.
        assert len(raw) < 256
        raw = b"\x28\xb5\x2f\xfd\x20" + bytes([len(raw)]) + ((len(raw) << 3) | 1).to_bytes(3, "little") + raw
        headers["Content-Encoding"] = "zstd"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request("POST", path, raw, headers)
    response = connection.getresponse()
    status, body = response.status, response.read()
    connection.close()
    return status, body


def descriptor_limit():
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = 256 if hard == resource.RLIM_INFINITY else min(256, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def verify(binary):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="uni-availability-") as directory:
            root, port = Path(directory), free_port()
            config = {"providers": [
                {"provider": "native", "engine": "gpt", "base_url": f"http://127.0.0.1:{upstream.server_port}/native/v1/responses", "api": ["first", "second"], "model": ["fixture"]},
                {"provider": "generic", "engine": "gpt", "base_url": f"http://127.0.0.1:{upstream.server_port}/generic/v1/chat/completions", "api": "generic", "model": ["generic-fixture"]}],
                "api_keys": [{"api": "fixture-key", "model": ["native/*", "generic/*"], "preferences": {"AUTO_RETRY": True}}],
                "preferences": {"hedging": {"enabled": False}}}
            (root / "api.json").write_text(json.dumps(config))
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("FACTS_", "UNI_API_CONTROL_", "CONFIG_URL", "RUST_", "MEMORY_", "REQUEST_", "ZSTD_", "FUGUE_OBSERVABILITY_"))}
            env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
                       RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot"), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                       RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), NO_PROXY="127.0.0.1,localhost",
                       RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                       RUST_RESOURCE_WAIT_TIMEOUT_SECONDS="0.05", REQUEST_BODY_IDLE_TIMEOUT_SECONDS="0.05",
                       RUST_RESPONSES_MAX_EVENT_BYTES="1024")
            with (root / "log").open("w+") as log:
                process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log,
                                           preexec_fn=descriptor_limit if os.name == "posix" else None)
                try:
                    for _ in range(100):
                        try:
                            get_json(port, "/healthz")
                            break
                        except OSError:
                            time.sleep(0.05)
                    else:
                        raise AssertionError("fixture did not start")

                    if Path("/proc/net/tcp").exists():
                        # Linux exposes the network namespace's TIME_WAIT rows
                        # to the gateway although these are not gateway FDs.
                        for _ in range(300):
                            client = http.client.HTTPConnection("127.0.0.1", upstream.server_port)
                            client.request("HEAD", "/time-wait")
                            client.getresponse().read()
                            client.close()

                    native = {"model": "fixture", "input": "hello", "stream": False}
                    status, body = call(port, "/v1/responses", native, compressed=True)
                    assert status == 200, (status, body)
                    assert ("/native/v1/responses", "Bearer first") in upstream.hits
                    status, body = call(port, "/v1/responses", native, target="native")
                    assert status == 200, (status, body)
                    assert ("/native/v1/responses", "Bearer second") in upstream.hits

                    # Force the generic provider through its explicit channel
                    # target, and exercise the dense DOM path without 3x math.
                    generic = {"model": "generic-fixture", "messages": [{"role": "user", "content": "hello"}], "values": [0] * 65536}
                    status, body = call(port, "/v1/chat/completions", generic)
                    assert status == 200, (status, body)
                    assert any(path.startswith("/generic/") for path, _ in upstream.hits)

                    client = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                    client.putrequest("POST", "/v1/responses")
                    for key, value in {"Authorization": "Bearer fixture-key", "Content-Encoding": "zstd", "Content-Length": "100"}.items():
                        client.putheader(key, value)
                    client.endheaders()
                    response = client.getresponse()
                    assert response.status == 408, (response.status, response.read())
                    client.close()

                    status, body = call(port, "/v1/responses", {**native, "stream": True}, target="native")
                    assert status == 502 and b"configured limit" in body, (status, body)
                    assert get_json(port, "/healthz")["status"] == "ok"
                    for _ in range(100):
                        totals = struct.unpack("<7Q", (root / "ledger").read_bytes())
                        if totals[0] == 0 and not list((root / "spool").glob("*.spool")):
                            break
                        time.sleep(0.01)
                    else:
                        raise AssertionError(("leaked memory/spool", totals, list((root / "spool").iterdir())))
                    print("PASS availability: TIME_WAIT/FD isolation, oversized 429 fallback, governed zstd/JSON, upload idle, bounded SSE, cleanup")
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    finally:
        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
