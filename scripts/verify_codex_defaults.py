"""Verify Codex defaults on the wire using only isolated local upstreams."""

import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from verify_dispatch_timing import free_port


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append((self.path, payload))
        response = {
            "id": "resp_fixture", "object": "response", "status": "completed",
            "model": "upstream-model",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "OK"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        if self.path.endswith("/compact"):
            response["object"] = "response.compaction"
        content_type = "application/json"
        if payload.get("stream"):
            content_type = "text/event-stream"
            events = [
                {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                {"type": "response.output_text.delta", "delta": "OK", "output_index": 0, "content_index": 0},
                {"type": "response.completed", "response": response},
            ]
            raw = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        else:
            raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def verify(binary):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    variants = {
        "defaults": {},
        "legacy": {"post_body_parameter_overrides": {
            "store": False, "__remove__": ["response_format", "temperature"]}},
        "conflicting": {"post_body_parameter_overrides": {
            "store": True, "temperature": 0.2, "response_format": {"type": "text"},
            "public-model": {"__remove__": ["store"], "temperature": 0.9,
                             "response_format": {"type": "json_object"},
                             "metadata": {"model_override": True}}}},
        "gpt": {},
    }
    with tempfile.TemporaryDirectory(prefix="uni-codex-defaults-") as directory:
        root = Path(directory)
        port = free_port()
        config = {
            "providers": [{"provider": name, "engine": "gpt" if name == "gpt" else "codex",
                           "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}/v1/responses",
                           "api": "upstream-fixture", "model": [{"upstream-model": "public-model"}],
                           "preferences": preferences} for name, preferences in variants.items()],
            "api_keys": [{"api": "admin-fixture", "model": ["all"]}]
                        + [{"api": f"{name}-fixture", "model": [f"{name}/*"]} for name in variants],
            "preferences": {"AUTO_RETRY": False},
        }
        config_path = root / "api.json"
        original = json.dumps(config).encode()
        config_path.write_bytes(original)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("FACTS_S3_", "OTEL_", "FUGUE_OTEL_")) and key != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                   NO_PROXY="127.0.0.1,localhost")

        def request(path, payload=None, provider="defaults"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.request("GET" if payload is None else "POST", path,
                                   None if payload is None else json.dumps(payload),
                                   {"Authorization": f"Bearer {provider}-fixture", "Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        with (root / "log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if request("/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.05)
                else:
                    raise AssertionError("fixture did not become healthy")
                checked = 0
                for provider in variants:
                    for path in ["/v1/responses", "/v1/responses/compact", "/v1/chat/completions"]:
                        for stream in ([False] if path.endswith("/compact") else [False, True]):
                            payload = {"model": "public-model", "store": True, "temperature": 0.6,
                                       "response_format": {"type": "json_object"}, "stream": stream}
                            payload.update({"messages": [{"role": "user", "content": "hi"}]}
                                           if path.endswith("/completions") else {"input": "hi"})
                            upstream.hits.clear()
                            status, raw = request(path, payload, provider)
                            assert status == 200, (provider, path, stream, status, raw)
                            assert len(upstream.hits) == 1, upstream.hits
                            upstream_path, body = upstream.hits[0]
                            assert upstream_path.startswith(f"/{provider}/"), upstream_path
                            assert body["model"] == "upstream-model", body
                            if provider != "gpt":
                                assert "temperature" not in body and "response_format" not in body, body
                                if path.endswith("/compact"):
                                    assert "store" not in body, body
                                else:
                                    assert body["store"] is False, body
                                if provider == "conflicting":
                                    assert body["metadata"]["model_override"] is True, body
                            else:
                                assert body["temperature"] == 0.6, body
                                if path == "/v1/responses":
                                    assert body["store"] is True, body
                                    assert body["response_format"] == payload["response_format"], body
                            assert b"OK" in raw, raw
                            checked += 1
                assert config_path.read_bytes() == original
                print(f"PASS Codex defaults: {checked} HTTP cases; Responses, compact, Chat, streaming, "
                      "legacy/conflicting overrides, GPT isolation, config unchanged")
            except Exception:
                print((root / "log").read_text()[-8000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                upstream.shutdown()
                upstream.server_close()


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
