"""Test a fresh process, config publication and edits using only loopback."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from verify_dispatch_timing import free_port

KEY = "fixture-managed-admin"
KID = "key-" + hashlib.sha256(KEY.encode()).hexdigest()
active = None
pending = None
rejected = False
fail_commit = False
fetched = threading.Event()


class Server(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, result, code=200):
        body = json.dumps(result).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.headers["Authorization"] == "Bearer " + KEY
        self.send_json({"revision": active["revision"] if active else 0,
                        "uninitialized": active is None and pending is None and not rejected,
                        "published": active, "pending": pending})
        fetched.set()

    def do_POST(self):
        global active, pending, rejected
        if self.path == "/upstream/v1/responses":
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["model"] == "m"
            events = [
                {"type": "response.created", "response": {"id": "resp_fixture", "status": "in_progress"}},
                {"type": "response.output_text.delta", "delta": "ready", "item_id": "msg_fixture",
                 "output_index": 0, "content_index": 0},
                {"type": "response.completed", "response": {"id": "resp_fixture", "object": "response",
                 "status": "completed", "model": "m", "output": [{"id": "msg_fixture", "type": "message",
                     "role": "assistant", "content": [{"type": "output_text", "text": "ready"}]}],
                 "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}},
            ]
            raw = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        assert self.headers["Authorization"] == "Bearer " + KEY
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["expected_revision"] == (active["revision"] if active else 0)
        if body["action"] == "promote":
            assert body["draft_revision"] == pending["revision"]
            active, pending = pending, None
        elif body["action"] == "reject":
            pending, rejected = None, True
        else:
            assert body["action"] == "commit"
            if fail_commit:
                self.send_json({}, 503)
                return
            active = {"revision": active["revision"] + 1, "bundle": body["bundle"]}
        self.send_json({"revision": active["revision"] if active else 0})


def eventually(check, limit=12):
    until = time.monotonic() + limit
    while time.monotonic() < until:
        try:
            if check():
                return
        except OSError:
            pass
        time.sleep(.07)
    raise AssertionError("condition not reached")


def main():
    global pending, fail_commit
    binary = str(Path(sys.argv[1]).resolve())
    manager = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=manager.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-managed-test-") as dirname:
        root = Path(dirname)
        port = free_port()
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
        env.update(PORT=str(port), DISABLE_DATABASE="true",
                   UNI_API_CONFIG_PATH=str(root / "absent.yaml"),
                   UNI_API_MANAGED_CONFIG_URL=f"http://127.0.0.1:{manager.server_port}/v1/runtime-config/fixture",
                   UNI_API_MANAGED_CONFIG_ADMIN_KEY=KEY,
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"))

        def call(method, path, body=None, key=KEY):
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
            try:
                conn.request(method, path, json.dumps(body) if body is not None else None, headers)
                response = conn.getresponse()
                raw = response.read()
                if "text/event-stream" in (response.getheader("Content-Type") or ""):
                    return response.status, raw.decode()
                try:
                    return response.status, json.loads(raw)
                except ValueError:
                    return response.status, raw.decode(errors="replace")
            finally:
                conn.close()

        def serving(version):
            status, result = call("GET", "/v1/channel-controls")
            return status == 200 and result.get("managed_config_revision") == version

        provider = {"provider": "configured", "engine": "gpt", "model": ["m"],
                    "base_url": f"http://127.0.0.1:{manager.server_port}/upstream/v1/responses",
                    "api": "fixture-upstream"}
        bundle = {"schema": 1, "base": {"providers": [provider],
                                      "api_keys": [{"api": KEY, "model": ["all"], "role": "admin"},
                                                   {"api": "fixture-business", "model": ["all"]}]},
                  "controls": {"version": 2, "rules": [],
                               "temporary_channels": [{"provider": "sub2api-example", "api_key_id": KID,
                                                       "base_url": provider["base_url"], "api_key": "temporary-secret",
                                                       "models": ["m"]}]}}
        with (root / "gateway.log").open("w") as log:
            with subprocess.Popen([binary], cwd=root, env=env, stdout=log, stderr=log) as process:
                try:
                    assert fetched.wait(12) and process.poll() is None, (root / "gateway.log").read_text()
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=.5):
                            raise AssertionError("unconfigured pod became TCP ready")
                    except ConnectionRefusedError:
                        pass
                    pending = {"revision": 1, "bundle": bundle}
                    eventually(lambda: serving(1))
                    assert not (root / "absent.yaml").exists()
                    assert call("GET", "/v1/models", key="fixture-business")[0] == 200
                    code, response = call("POST", "/v1/responses", {"model": "m", "input": "ready", "stream": True},
                                          key="fixture-business")
                    assert code == 200 and "response.completed" in response, (code, response)
                    assert call("GET", "/v1/configuration/export", key="fixture-business")[0] == 403
                    code, exported = call("GET", "/v1/configuration/export")
                    assert code == 200 and exported["bundle"]["controls"]["temporary_channels"][0]["api_key"] == "temporary-secret"
                    pending = {"revision": 2, "bundle": {**bundle, "base": {}}}
                    eventually(lambda: rejected)
                    assert serving(1)
                    controls = call("GET", "/v1/channel-controls")[1]
                    edit = {"revision": controls["revision"], "action": "set", "api_key_id": KID,
                            "model": "m", "order": ["sub2api-example", "configured"], "disabled": []}
                    fail_commit = True
                    assert call("POST", "/v1/channel-controls", edit)[0] == 503
                    assert call("GET", "/v1/channel-controls")[1]["rules"] == []
                    fail_commit = False
                    assert call("POST", "/v1/channel-controls", edit)[0] == 200
                    assert serving(2)
                finally:
                    process.terminate()
                    process.wait(timeout=6)
            with subprocess.Popen([binary], cwd=root, env=env, stdout=log, stderr=log) as replacement:
                try:
                    eventually(lambda: serving(2))
                    assert call("GET", "/v1/channel-controls")[1]["rules"][0]["order"] == edit["order"]
                finally:
                    replacement.terminate()
                    replacement.wait(timeout=6)
    manager.shutdown()
    manager.server_close()
    print("PASS: fresh PVC-free boot, TCP gate, admin export, rejected draft and durable channel edit across replacement")


if __name__ == "__main__":
    main()
