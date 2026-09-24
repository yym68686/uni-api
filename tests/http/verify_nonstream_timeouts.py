"""Offline HTTP regression for non-streaming timeout rules and body deadlines."""
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

from verify_dispatch_timing import free_port, get_json


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = payload["model"]
        channel = self.path.split("/")[1]
        case = self.server.cases[model]
        with self.server.lock:
            self.server.hits.setdefault(model, []).append(channel)
        first = channel == "first"
        time.sleep(case.get("headers_delay", 0) if first else 0)
        response = {"id": "resp_fixture", "object": "response", "status": "completed",
                    "model": model, "output": [{"id": "msg_fixture", "type": "message",
                    "role": "assistant", "status": "completed", "content": [
                        {"type": "output_text", "text": channel}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        raw = json.dumps(response).encode()
        content_type = "application/json"
        if payload.get("stream"):
            content_type = "text/event-stream"
            raw = (f'event: response.output_text.delta\ndata: '
                   f'{json.dumps({"type": "response.output_text.delta", "delta": channel})}\n\n'
                   f'event: response.completed\ndata: '
                   f'{json.dumps({"type": "response.completed", "response": response})}\n\n').encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-OAIX-Token-ID", "123" if first else "456")
        self.end_headers()
        self.wfile.flush()
        try:
            time.sleep(case.get("body_delay", 0) if first else 0)
            if first and case.get("drip"):
                # Several chunks keep idle alive while the absolute total expires.
                chunks = [raw[:1], raw[1:2], raw[2:3], raw[3:]]
                for index, chunk in enumerate(chunks):
                    if index:
                        time.sleep(case["drip"])
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary):
    cases = {
        "responses-total-only": {"policy": {"total": 1.2}, "headers_delay": 0.35},
        "compact-total-only": {"policy": {"total": 1.2}, "headers_delay": 0.35,
                               "endpoint": "/v1/responses/compact"},
        "chat-total-only": {"policy": {"total": 1.2}, "headers_delay": 0.35,
                            "endpoint": "/v1/chat/completions"},
        "explicit-first-byte": {"policy": {"first_byte": 0.1, "total": 1.2},
                                "headers_delay": 0.35, "winner": "fallback"},
        "model-fallback": {"policy": {}, "headers_delay": 0.35, "winner": "fallback"},
        "stream-unaffected": {"policy": {"total": 1.2}, "headers_delay": 0.35,
                              "stream": True, "winner": "fallback"},
        "total-before-headers": {"policy": {"total": 0.2}, "headers_delay": 0.6,
                                 "winner": "fallback"},
        "total-body-stall": {"policy": {"total": 0.2}, "body_delay": 0.6,
                             "winner": "fallback"},
        "total-keeps-original-deadline": {"policy": {"total": 0.5, "idle": 0.3},
                                          "headers_delay": 0.25, "drip": 0.15,
                                          "winner": "fallback"},
        "idle-body-stall": {"policy": {"idle": 0.15, "total": 1.2},
                            "body_delay": 0.6, "winner": "fallback"},
        "idle-resets-on-chunks": {"policy": {"idle": 0.3, "total": 1.2}, "drip": 0.1},
        "first-byte-disabled": {"policy": {"first_byte": 0, "total": 1.2},
                                "headers_delay": 0.35},
        "total-disabled": {"policy": {"total": 0}, "headers_delay": 0.35},
        "first-byte-not-body": {"policy": {"first_byte": 0.1}, "body_delay": 0.35},
        "fast": {"policy": {"total": 1.2}},
    }
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.cases, upstream.hits, upstream.lock = cases, {}, threading.Lock()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-nonstream-timeout-") as directory:
        root, port = Path(directory), free_port()
        rules = [{"match": {"model": model, "stream": False}, "timeout": case["policy"]}
                 for model, case in cases.items() if case["policy"]]
        config = {
            "providers": [{"provider": name, "engine": "gpt",
                           "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}/v1/responses",
                           "api": "fixture-upstream", "model": list(cases),
                           "preferences": {"cooldown_period": 0}}
                          for name in ["first", "fallback"]],
            "api_keys": [{"api": "fixture-key", "model": ["first/*", "fallback/*"],
                          "preferences": {"AUTO_RETRY": True}}],
            "preferences": {"model_timeout": {"default": 0.1},
                            "timeout_policy": {"rules": rules},
                            "hedging": {"enabled": False}},
        }
        config_path = root / "api.json"
        config_path.write_text(json.dumps(config))
        original = config_path.read_bytes()
        env = dict(os.environ, PORT=str(port), DISABLE_DATABASE="true",
                   UNI_API_CONFIG_PATH=str(config_path), NO_PROXY="127.0.0.1,localhost",
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "memory-ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0")
        # A developer's production exporter/restore environment must never leak
        # into this fixture. All requests and credentials stay on loopback.
        for key in list(env):
            if key.startswith(("FACTS_S3_", "UNI_API_CONTROL_RESTORE_")):
                env.pop(key)
        with (root / "log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        get_json(port, "/healthz")
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    raise AssertionError("fixture did not become healthy")
                for model, case in cases.items():
                    endpoint = case.get("endpoint", "/v1/responses")
                    body = {"model": model, "stream": case.get("stream", False), "input": "fixture"}
                    if endpoint == "/v1/chat/completions":
                        body.pop("input")
                        body["messages"] = [{"role": "user", "content": "fixture"}]
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
                    started = time.monotonic()
                    conn.request("POST", endpoint, json.dumps(body), {
                        "Authorization": "Bearer fixture-key", "Content-Type": "application/json",
                        "X-Request-ID": model,
                    })
                    response = conn.getresponse()
                    raw, elapsed = response.read(), time.monotonic() - started
                    winner = case.get("winner", "first")
                    hits = upstream.hits[model]
                    context = (model, response.status, raw[:300], hits, elapsed)
                    assert response.status == 200 and winner.encode() in raw, context
                    assert hits == (["first"] if winner == "first" else ["first", "fallback"]), context
                    if endpoint != "/v1/chat/completions":
                        assert response.getheader("X-OAIX-Token-ID") == ("123" if winner == "first" else "456"), context
                    if winner == "fallback":
                        assert elapsed < 1.0, context
                    conn.close()
                    print(f"PASS {model}: winner={winner}, elapsed={elapsed:.3f}s")
                assert config_path.read_bytes() == original
            except Exception:
                log.flush()
                print((root / "log").read_text()[-6000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)
    upstream.shutdown()
    upstream.server_close()


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
