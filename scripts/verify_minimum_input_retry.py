"""Offline HTTP regression for channel minimum-input and model-unavailable failover."""
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


CHINESE = "该令牌不接受输入少于 2000 token 的请求(按请求体大小判定)。"
ENGLISH = ("This key does not accept requests with fewer than 2000 input tokens "
           "(judged by request body size).")
MODEL_UNAVAILABLE = "This model is not available."
UPSTREAM_PROCESSING_FAILURE = "The upstream service could not process this request."


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        channel = self.path.split("/")[1]
        self.server.hits.append(channel)
        if channel != "fallback":
            raw = json.dumps(self.server.error, ensure_ascii=self.server.escape).encode()
            status, content_type = 400, "application/json"
        else:
            status = 200
            if self.path.endswith("/chat/completions"):
                response = {
                    "id": "chat_fixture", "object": "chat.completion", "model": "test-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
                stream = ('data: {"id":"chat_fixture","choices":[{"index":0,'
                          '"delta":{"content":"test"},"finish_reason":null}]}\n\n'
                          'data: {"id":"chat_fixture","choices":[{"index":0,'
                          '"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
            else:
                response = {
                    "id": "resp_fixture", "object": "response", "status": "completed",
                    "model": "test-model",
                    "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "test"}]}],
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                }
                if self.path.endswith("/compact"):
                    response["object"] = "response.compaction"
                events = [
                    {"type": "response.output_text.delta", "delta": "test",
                     "item_id": "msg_fixture", "output_index": 0, "content_index": 0},
                    {"type": "response.completed", "response": response},
                ]
                stream = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                                 for event in events)
            content_type = "text/event-stream" if payload.get("stream") else "application/json"
            raw = stream.encode() if payload.get("stream") else json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def verify(binary, endpoint, hedging):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-minimum-input-") as directory:
        root, port = Path(directory), free_port()
        upstream_endpoint = "/v1/responses" if endpoint.endswith("/compact") else endpoint
        config = {
            "providers": [{
                "provider": name, "engine": "gpt",
                "base_url": f"http://127.0.0.1:{server.server_port}/{name}{upstream_endpoint}",
                "api": "fixture-upstream", "model": ["test-model"],
                "preferences": {"cooldown_period": 0},
            } for name in ["limited", "fallback", "also-limited"]],
            "api_keys": [
                {"api": "retry", "model": ["limited/*", "fallback/*"]},
                {"api": "no-retry", "model": ["limited/*", "fallback/*"],
                 "preferences": {"AUTO_RETRY": False}},
                {"api": "exhausted", "model": ["limited/*", "also-limited/*"]},
            ],
            "preferences": {"hedging": {"enabled": hedging, "max_inflight_attempts": 2}},
        }
        config_path = root / "api.json"
        original = json.dumps(config).encode()
        config_path.write_bytes(original)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("FACTS_S3_", "OTEL_", "FUGUE_OTEL_", "UNI_API_CONTROL_RESTORE_"))
               and key != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                   # Keep repeated fixture cases independent of the accumulated
                   # model circuit breaker; route cooldown has its own Rust test.
                   PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD="1000",
                   NO_PROXY="127.0.0.1,localhost")

        def request(path, payload=None, key="retry"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.request("GET" if payload is None else "POST", path,
                                   None if payload is None else json.dumps(payload),
                                   {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, response.getheader("Content-Type", ""), response.read()
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
                cases = [
                    ("bilingual", CHINESE + ENGLISH, "retry", False, 200, ["limited", "fallback"]),
                    ("escaped-chinese", CHINESE, "retry", True, 200, ["limited", "fallback"]),
                    ("english-new-limit", ENGLISH.replace("2000", "8192"), "retry", False,
                     200, ["limited", "fallback"]),
                    ("ordinary-400", "Missing required parameter: input", "retry", False, 400, ["limited"]),
                    ("input-validation", "Input must contain at least 1 token.", "retry", False, 400, ["limited"]),
                    ("no-retry", CHINESE + ENGLISH, "no-retry", False, 502, ["limited"]),
                    # Existing two-channel retry budget is six total attempts.
                    ("exhausted", CHINESE + ENGLISH, "exhausted", False, 502,
                     ["limited", "also-limited"] * 3),
                    ("model-unavailable", MODEL_UNAVAILABLE, "retry", False,
                     200, ["limited", "fallback"]),
                    ("wrapped-model-unavailable", json.dumps({"error": {
                        "type": "invalid_request_error", "message": MODEL_UNAVAILABLE}}),
                     "retry", False, 200, ["limited", "fallback"]),
                    ("model-no-retry", MODEL_UNAVAILABLE, "no-retry", False, 503, ["limited"]),
                    ("model-exhausted", MODEL_UNAVAILABLE, "exhausted", False, 503,
                     ["limited", "also-limited"] * 3),
                    ("quoted-model-message", f"Invalid input: expected '{MODEL_UNAVAILABLE}'",
                     "retry", False, 400, ["limited"]),
                    ("upstream-processing-failure", UPSTREAM_PROCESSING_FAILURE, "retry", False,
                     200, ["limited", "fallback"]),
                    ("wrapped-upstream-processing-failure", json.dumps({"error": {
                        "type": "invalid_request_error", "message": UPSTREAM_PROCESSING_FAILURE}}),
                     "retry", False, 200, ["limited", "fallback"]),
                    ("upstream-processing-no-retry", UPSTREAM_PROCESSING_FAILURE, "no-retry", False,
                     502, ["limited"]),
                    ("upstream-processing-exhausted", UPSTREAM_PROCESSING_FAILURE, "exhausted", False,
                     502, ["limited", "also-limited"] * 3),
                    ("quoted-upstream-processing-message",
                     f"Invalid input: expected '{UPSTREAM_PROCESSING_FAILURE}'",
                     "retry", False, 400, ["limited"]),
                ]
                checked = 0
                for streaming in ([False] if endpoint.endswith("/compact") else [False, True]):
                    for label, message, key, escape, expected_status, expected_hits in cases:
                        server.hits.clear()
                        server.error = {"error": {"type": "invalid_request_error", "message": message}}
                        server.escape = escape
                        payload = {"model": "test-model", "stream": streaming}
                        payload.update({"messages": [{"role": "user", "content": "say test"}]}
                                       if endpoint.endswith("/completions") else
                                       {"input": [{"role": "user", "content": "say test"}]})
                        status, content_type, raw = request(endpoint, payload, key)
                        context = (endpoint, hedging, streaming, label, status, raw, server.hits)
                        assert status == expected_status, context
                        assert server.hits == expected_hits, context
                        if status == 200:
                            assert b"test" in raw and b"invalid_request_error" not in raw, context
                            if streaming:
                                assert "text/event-stream" in content_type, context
                        elif status == 503:
                            assert b"All configured providers failed for model test-model" in raw, context
                        else:
                            assert b"invalid_request_error" in raw, context
                        checked += 1
                assert config_path.read_bytes() == original
                print(f"PASS provider failure failover {endpoint} hedge={hedging}: {checked} HTTP cases")
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
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    binary = Path(sys.argv[1]).resolve()
    for endpoint in ["/v1/responses", "/v1/chat/completions", "/v1/responses/compact"]:
        for hedging in [False, True]:
            verify(binary, endpoint, hedging)
