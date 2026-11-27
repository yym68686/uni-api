"""Exercise the native Interactions gateway against a local, unpaid upstream."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

from verify_dispatch_timing import free_port


JSON_BODY = b'{"status":"completed","steps":[{"type":"test-opaque"}],"usage":{"total_input_tokens":100,"total_output_tokens":12,"total_thought_tokens":8,"total_tokens":120}}'
SSE_BODY = (
    b'event: step.delta\r\ndata: {"event_type":"step.delta","delta":{"type":"text","text":"hello"}}\r\n\r\n'
    b'event: interaction.completed\r\ndata: {"interaction":' + JSON_BODY + b'}\r\n\r\n'
)
GOOGLE_ERROR = b'{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "fixture"}}'
TEXT_STEPS = [
    {"type": "user_input", "content": [{"type": "text", "text": "你好"}]},
    {"type": "model_output", "content": [{"type": "text", "text": "Hello"}]},
    {"type": "user_input", "content": [{"type": "text", "text": "Continue"}]},
]


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append((self.path, payload, {k.lower(): v for k, v in self.headers.items()}))
        status, media, body = 200, "application/json", JSON_BODY
        if urlsplit(self.path).path.endswith(":generateContent"):
            body = b'{"candidates":[{"content":{"role":"model","parts":[{"text":"legacy"}]},"finishReason":"STOP"}]}'
        elif payload["input"] == "quota":
            status, body = 429, GOOGLE_ERROR
        elif payload["input"] == "redirect":
            status, body = 307, b""
        elif payload["input"] == "invalid":
            body = b"null"
        elif payload["input"] == "wrong-content-type":
            pass
        elif payload.get("stream"):
            media, body = "text/event-stream; charset=utf-8", SSE_BODY
        self.send_response(status)
        self.send_header("Content-Type", media)
        if status == 307:
            self.send_header("Location", "/do-not-follow")
        self.end_headers()
        for offset in range(0, len(body), 13):
            self.wfile.write(body[offset:offset + 13])
            self.wfile.flush()


def verify(binary):
    with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream, tempfile.TemporaryDirectory() as work:
        upstream.hits = []
        worker = threading.Thread(target=upstream.serve_forever, daemon=True)
        worker.start()
        port = free_port()
        root = Path(work)
        base = f"http://127.0.0.1:{upstream.server_port}"
        config = {"api_keys": [{"api": "fixture-client", "model": ["all"], "preferences": {
            "AUTO_RETRY": 3, "SCHEDULING_ALGORITHM": "fixed_priority"
        }}], "providers": [
            *[{"provider": name, "engine": "gemini", "base_url": base + f"/{name}/v1beta",
               "api": "fixture-provider", "model": [{"wire-model": "alias"}, {"gemini-embedding-001": "embedding-alias"}, {"gemini-flash-preview-tts": "audio-alias"}],
               "preferences": {"gemini_interactions_version": "v1", "cooldown_period": 0}}
              for name in ("primary", "fallback")],
            {"provider": "legacy", "engine": "gemini", "base_url": base + "/legacy/v1beta",
             "api": "fixture-provider", "model": ["legacy-model", {"wire-model": "alias"}]},
        ]}
        (root / "config.json").write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("FACTS_", "UNI_API_CONTROL_RESTORE_", "OTEL_", "FUGUE_OTEL_")) and k != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "config.json"),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0", NO_PROXY="127.0.0.1,localhost")

        def call(path, payload=None, key=True, bearer=False):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            headers = {"content-type": "application/json", "cookie": "caller=secret", "idempotency-key": "caller"}
            if key:
                headers["Authorization" if bearer else "x-goog-api-key"] = "Bearer fixture-client" if bearer else "fixture-client"
            try:
                conn.request("GET" if payload is None else "POST", path, None if payload is None else json.dumps(payload), headers)
                res = conn.getresponse()
                return res.status, dict(res.getheaders()), res.read()
            finally:
                conn.close()

        with (root / "gateway.log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(160):
                    try:
                        if call("/healthz", key=False)[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.05)
                else:
                    raise AssertionError("Gateway was not ready")
                path, request = "/v1/interactions", {"model": "alias", "input": "hello"}
                missing_key = call(path, request, key=False)
                assert missing_key[0] == 403, missing_key
                method_status, method_headers, _ = call(path)
                assert method_status == 405 and method_headers.get("allow") == "POST"
                assert call(path + "/fixture")[0] == 404
                for changes, message in [
                    ({"input": TEXT_STEPS, "store": True}, "stored or background"),
                    ({"input": TEXT_STEPS, "previous_interaction_id": "fixture"}, "continuation"),
                    ({"input": TEXT_STEPS, "tool_choice": {"mode": "any"}}, "required tool calls"),
                    ({"input": [{"type": "user_input", "content": [{"type": "image", "uri": "fixture"}]}]}, "requires nonempty text input"),
                ]:
                    result, _, raw = call(path, {**request, **changes})
                    assert result == 400 and message in json.loads(raw)["error"]["message"], (result, raw)
                assert call(path, {**request, "model": "embedding-alias"})[0] >= 400
                assert call(path, {**request, "model": "audio-alias"})[0] >= 400
                catalog_status, _, catalog = call("/v1/model-channels?endpoint=%2Fv1%2Finteractions", bearer=True)
                assert catalog_status == 200
                assert {(row["model"], row["interactions_mode"]) for row in json.loads(catalog)["data"]} == {
                    ("alias", "native"), ("alias", "translated"), ("legacy-model", "translated")
                }, catalog
                assert not upstream.hits
                for changes, status, expected in [
                    ({}, 200, JSON_BODY), ({"stream": True}, 200, SSE_BODY),
                    ({"input": TEXT_STEPS[:1], "store": False, "stream": True, "generation_config": {"thinking_summaries": "auto"}}, 200, SSE_BODY),
                    ({"input": TEXT_STEPS}, 200, JSON_BODY),
                    ({"input": "quota"}, 429, GOOGLE_ERROR), ({"input": "invalid"}, 502, None),
                    ({"input": "wrong-content-type", "stream": True}, 502, None),
                    ({"input": "redirect"}, 307, None),
                ]:
                    upstream.hits.clear()
                    result, headers, raw = call(path + "?key=fixture-client", {**request, **changes}, key=False)
                    assert result == status and (expected is None or raw == expected), (result, raw)
                    assert "location" not in {k.lower() for k in headers}
                    assert len(upstream.hits) == 1, upstream.hits
                    route, body, wire = upstream.hits[0]
                    assert route == "/primary/v1/interactions", route
                    assert body == {**request, **changes, "model": "wire-model", "store": False}, body
                    assert wire["x-goog-api-key"] == "fixture-provider"
                    assert not set(wire) & {"authorization", "cookie", "idempotency-key"}
                upstream.hits.clear()
                status, headers, raw = call(path, {**request, "stream": "not-a-boolean",
                    "tools": [{"type": "function", "name": "optional"}]})
                assert status == 200 and raw == JSON_BODY and len(upstream.hits) == 1, (status, raw)
                assert set(headers["x-uni-api-interactions-omitted"].split(",")) == {"extensions", "tools"}, headers
                assert not set(upstream.hits[0][1]) & {"tools", "stream"}, upstream.hits
                status, _, raw = call("/v1beta/models/legacy-model:generateContent", {"contents": [{"parts": [{"text": "old"}]}]})
                assert status == 200 and b"legacy" in raw
            except Exception:
                log.seek(0)
                print(log.read()[-7000:])
                raise
            finally:
                process.terminate()
                process.wait(timeout=10)
                upstream.shutdown()
    print("Gemini Interactions HTTP matrix passed")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
