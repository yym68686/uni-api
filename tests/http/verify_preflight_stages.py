"""Real local HTTP waits must remain distinct from gateway processing."""
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

facts = []


def event(kind, **extra):
    return ("event: " + kind + "\ndata: " + json.dumps({"type": kind, **extra}) + "\n\n").encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_PUT(self):
        facts.extend(json.loads(line) for line in self.rfile.read(int(self.headers["Content-Length"])).splitlines())
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        name = self.path.split("/")[1]
        time.sleep(.2)
        if name == "error":
            body = b'{"error":{"message":"PRIVATE_FIXTURE_ERROR"}}'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            time.sleep(.3)
            self.wfile.write(body)
            return
        prefix = event("response.created", response={"id": "resp_fixture", "model": "m", "output": []})
        suffix = event("response.output_text.delta", delta="test", item_id="msg_fixture", output_index=0, content_index=0)
        suffix += event("response.completed", response={"id": "resp_fixture", "status": "completed", "model": "m",
                        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "test"}]}]})
        if name == "eof":
            suffix = b""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(prefix)
        self.wfile.flush()
        time.sleep(.3)
        self.wfile.write(suffix)
        self.wfile.flush()


server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    with tempfile.TemporaryDirectory(prefix="uni-preflight-stages-") as directory:
        root = Path(directory)
        port = free_port()
        config = {
            "providers": [{"provider": name, "engine": "codex", "api": "fixture-upstream",
                           "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses", "model": ["m"]}
                          for name in ["primer", "error", "eof"]],
            "api_keys": [{"api": "fixture-admin", "model": ["all"]}],
        }
        (root / "api.json").write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "request-spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0", NO_PROXY="127.0.0.1,localhost",
                   FACTS_S3_ENDPOINT=f"http://127.0.0.1:{server.server_port}", FACTS_S3_BUCKET="fixture",
                   FACTS_S3_ACCESS_KEY_ID="fixture", FACTS_S3_SECRET_ACCESS_KEY="fixture",
                   FACTS_S3_SPOOL_DIR=str(root / "facts-spool"))

        def call(method, path, name=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                headers = {"Authorization": "Bearer fixture-admin", "Content-Type": "application/json"}
                if name:
                    headers["x-uni-api-provider"] = name
                connection.request(method, path, json.dumps({"model": "m", "input": "fixture", "stream": True}) if name else None, headers)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        with (root / "runtime.log").open("w") as log:
            process = subprocess.Popen([str(Path(sys.argv[1]).resolve())], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if call("GET", "/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                status, wire = call("POST", "/v1/responses", "primer")
                assert status == 200, (status, wire)
                events = [json.loads(line[5:]) for line in wire.splitlines() if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"]
                assert any(e.get("type") == "response.output_text.delta" and e.get("delta") == "test" for e in events), events
                assert call("POST", "/v1/responses", "error")[0] >= 400
                assert call("POST", "/v1/responses", "eof")[0] >= 400
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    attempts = {f["provider"]: f for f in facts if f["kind"] == "attempt"}
                    if all(name in attempts for name in ("primer", "error", "eof")):
                        break
                    time.sleep(.1)
                for name in ("primer", "error", "eof"):
                    timing = attempts[name]["transport_timing"]
                    assert timing["headers_received_ms"] >= 175, timing
                    assert timing["network_write_measured"] is False
                    if name == "error":
                        assert timing["error_body_read_ms"] >= 250, timing
                        assert timing["first_upstream_chunk_ms"] is None
                        assert timing["first_wire_prepared_ms"] is None
                    else:
                        assert timing["preflight_read_wait_ms"] >= 250, timing
                        assert timing["preflight_read_calls"] >= 2, timing
                        assert timing["preflight_process_ms"] < timing["preflight_read_wait_ms"], timing
                        assert timing["error_body_read_ms"] is None
                        if name == "primer":
                            assert timing["public_stream_ready_ms"] - timing["first_upstream_chunk_ms"] >= 250, timing
                        else:
                            assert timing["public_stream_ready_ms"] is None, timing
                    assert "PRIVATE_FIXTURE_ERROR" not in json.dumps(attempts[name])
                    print("PASS", name, json.dumps(timing))
            finally:
                process.terminate()
                process.wait(timeout=5)
finally:
    server.shutdown()
    server.server_close()
