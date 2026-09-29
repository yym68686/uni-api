"""Request trace regression: local gateway, local upstreams and local S3 only."""
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

class Stub(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_PUT(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        facts.extend(json.loads(line) for line in body.splitlines() if line)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        rejected = self.path.startswith("/reject/")
        body = json.dumps({"code": "INSUFFICIENT_BALANCE", "message": "Insufficient account balance", "debug": "private-prompt"} if rejected else {
            "id": "resp_fixture", "object": "response", "status": "completed", "model": "m",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        self.send_response(403 if rejected else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=server.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory(prefix="uni-request-trace-") as directory:
    root = Path(directory)
    port = free_port()
    config = {"providers": [{"provider": name, "engine": "gpt", "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses",
        "api": "fixture-upstream-secret", "model": ["m"]} for name in ["reject", "success"]],
        "api_keys": [{"api": "fixture-admin", "model": ["all"]}]}
    (root / "api.json").write_text(json.dumps(config))
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
    env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
        RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
        RUST_REQUEST_SPOOL_DIRECTORY=str(root / "body-spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
        NO_PROXY="127.0.0.1,localhost", FACTS_S3_ENDPOINT=f"http://127.0.0.1:{server.server_port}", FACTS_S3_BUCKET="fixture",
        FACTS_S3_ACCESS_KEY_ID="fixture", FACTS_S3_SECRET_ACCESS_KEY="fixture", FACTS_S3_SPOOL_DIR=str(root / "facts-spool"))

    def call(path, request_id=None, authorized=True):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
        headers = {"Content-Type": "application/json"}
        if authorized: headers["Authorization"] = "Bearer fixture-admin"
        if request_id: headers["X-Request-ID"] = request_id
        conn.request("POST" if request_id else "GET", path, json.dumps({"model": "m", "input": "private-prompt", "stream": False}) if request_id else None, headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, raw

    with (root / "gateway.log").open("w") as log:
        process = subprocess.Popen([str(Path(sys.argv[1]).resolve())], cwd=root, env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    if call("/healthz")[0] == 200: break
                except OSError: pass
                time.sleep(.05)
            status, body = call("/v1/responses", "trace-retry")
            assert status == 200 and b"done" in body, (status, body)
            status, body = call("/v1/responses", "trace-auth", False)
            assert status in (401, 403), status
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if any(f.get("stage") == "response_body_finished" and f["request_id"] == "trace-auth" for f in facts): break
                time.sleep(.1)
            selected = [f for f in facts if f["request_id"] == "trace-retry"]
            for stage in ["request_received", "response_headers", "response_body_finished"]:
                assert len([f for f in selected if f.get("stage") == stage]) == 1, (stage, selected)
            for provider in ["reject", "success"]:
                assert any(f["kind"] == "dispatch" and f["provider"] == provider for f in selected), selected
            assert len([f for f in selected if f["kind"] == "request"]) == 1
            assert next(f for f in selected if f["kind"] == "request")["outcome"] == "success"
            assert any(f.get("trace_detail", {}).get("error", {}).get("error_code") == "INSUFFICIENT_BALANCE" for f in selected)
            assert any(f.get("stage") == "routing_attempt" for f in selected)
            assert any(f.get("stage") == "request_received" and f["request_id"] == "trace-auth" for f in facts)
            encoded = json.dumps(facts)
            for secret in ["fixture-admin", "fixture-upstream-secret", "private-prompt"]: assert secret not in encoded, secret
            print("PASS request trace: retry, final success, admission rejection, stages, errors, no credentials or prompt")
        except Exception:
            print((root / "gateway.log").read_text()[-8000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            process.wait(timeout=5)
server.shutdown()
server.server_close()
