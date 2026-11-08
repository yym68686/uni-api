"""Hermetic compaction failover, including already-forwarded progress/body."""
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
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        provider = self.path.split("/")[1]
        case = body.get("metadata", {}).get("case", "missing")
        self.server.hits.append(provider)
        compact = self.path.endswith("/compact") or any(
            item.get("type") == "compaction_trigger" for item in body.get("input", []))
        item = {"type": "compaction", "encrypted_content": provider + "-private-state"}
        final = {"id": "resp_" + provider, "object": "response", "status": "completed",
                 "output": [item], "usage": {"input_tokens": 1, "output_tokens": 1}}
        bad = compact and (provider == "bad" or case == "exhausted")
        if bad and case == "unsupported":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"unsupported compaction_trigger"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
        self.send_header("Connection", "close")
        self.end_headers()
        if not body.get("stream"):
            if bad:
                final["output"] = []
            self.wfile.write(json.dumps(final).encode())
            return

        def event(kind, **data):
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(dict(type=kind, **data))}\n\n".encode())
            self.wfile.flush()

        try:
            event("response.created", response={"id": "resp_" + provider, "status": "in_progress"})
            event("response.output_text.delta", delta="ordinary-progress")
            if case == "gated" and provider == "bad" and self.server.gated:
                self.server.progress.wait(5)
            if bad and case in ("missing", "gated", "exhausted"):
                event("response.completed", response=final)  # No done: old probe falsely accepted.
            elif bad and case == "eof":
                event("response.output_item.done", item=item, output_index=0)
            elif bad and case == "sse-unsupported":
                event("response.failed", response={"status":"failed", "error":{"message":"compaction not supported"}})
            elif bad and case == "duplicate":
                event("response.output_item.done", item=item, output_index=0)
                event("response.output_item.done", item=item, output_index=0)
                event("response.completed", response=final)
            elif bad and case == "mismatch":
                event("response.output_item.done", item=item, output_index=0)
                final["output"] = [{**item, "encrypted_content": "different-private-state"}]
                event("response.completed", response=final)
            else:
                event("response.output_item.done", item=item, output_index=0)
                event("response.completed", response=final)
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary, mode):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits, upstream.progress = [], threading.Event()
    upstream.gated = mode == "native"
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-compact-retry-") as directory:
        root, port = Path(directory), free_port()
        config = {"providers": [{"provider": p, "engine": "gpt", "api": "fixture",
                    "base_url": f"http://127.0.0.1:{upstream.server_port}/{p}/v1/responses",
                    "model": ["model-a"]} for p in ("bad", "good")],
                  "api_keys": [{"api": "fixture-admin", "role": "admin", "model": ["all"]},
                               {"api":"fixture-disabled", "model":["all"], "preferences":{"AUTO_RETRY":False}}],
                  "preferences": {"AUTO_RETRY": True, "MAX_RETRIES": 8, "SCHEDULING_ALGORITHM": "fixed_priority"}}
        (root / "api.json").write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG")}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                   NO_PROXY="127.0.0.1,localhost", UNI_API_RUST_RESPONSES_DATA_PLANE=str(mode == "native").lower())

        def call(path, payload=None, key="fixture-admin", target=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
            if target: headers["x-uni-api-provider"] = target
            conn.request("POST" if payload else "GET", path,
                         json.dumps(payload) if payload else None,
                         headers)
            return conn, conn.getresponse()

        with (root / "runtime.log").open("w") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        conn, response = call("/healthz")
                        response.read(); conn.close()
                        if response.status == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError("not ready")
                for stream in (True, False):
                    for case in ("missing", "duplicate", "mismatch", "eof", "unsupported", "sse-unsupported", "exhausted", "gated"):
                        upstream.hits.clear(); upstream.progress.clear()
                        payload = {"model": "model-a", "input": [{"type": "compaction_trigger"}],
                                   "stream": stream, "metadata": {"case": case}}
                        conn, response = call("/v1/responses", payload)
                        prefix = b""
                        if mode == "native" and stream and case == "gated":
                            while b"ordinary-progress" not in prefix:
                                prefix += response.readline()
                            assert upstream.hits == ["bad"], upstream.hits
                            upstream.progress.set()
                        raw = prefix + response.read(); conn.close()
                        assert upstream.hits == ["bad", "good"], (mode, stream, case, upstream.hits, raw)
                        assert b"bad-private-state" not in raw and b"different-private-state" not in raw, raw
                        if case == "exhausted":
                            assert b"good-private-state" not in raw and (response.status == 502 or b"event: error" in raw), raw
                        else:
                            assert response.status == 200 and b"good-private-state" in raw, raw
                            if stream:
                                events = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
                                assert sum(e["type"] == "response.completed" for e in events) == 1, events
                                assert sum(e["type"] == "response.output_item.done" for e in events) == 1, events
                # Explicit target and disabled AUTO_RETRY each remain one attempt.
                for key, target in (("fixture-disabled",None), ("fixture-admin","bad")):
                    for stream in (False,True):
                        upstream.hits.clear()
                        conn,response = call("/v1/responses", {"model":"model-a", "input":[{"type":"compaction_trigger"}], "stream":stream}, key, target)
                        raw=response.read(); conn.close()
                        assert upstream.hits==["bad"] and b"good-private-state" not in raw, (upstream.hits, raw)
                if mode == "native":
                    upstream.hits.clear(); upstream.progress.clear()
                    conn,response = call("/v1/responses", {"model":"model-a", "input":[{"type":"compaction_trigger"}], "stream":True, "metadata":{"case":"gated"}})
                    prefix=b""
                    while b"ordinary-progress" not in prefix: prefix += response.readline()
                    response.close(); conn.close()
                    time.sleep(.2)
                    upstream.progress.set()
                    time.sleep(.2)
                    assert upstream.hits==["bad"], upstream.hits
                # Failed compression must not cool the route for normal traffic.
                upstream.hits.clear()
                normal_input = [{"type":"message", "role":"user", "content":[{"type":"input_text", "text":"fixture"}]}]
                conn, response = call("/v1/responses", {"model":"model-a", "input":normal_input, "stream":False})
                raw = response.read(); conn.close()
                assert upstream.hits == ["bad"], (response.status, upstream.hits, raw)
                for endpoint in ("/v1/responses/compact", "/v1/responses/compact/"):
                    upstream.hits.clear()
                    conn, response = call(endpoint, {"model":"model-a", "input":normal_input, "stream":False})
                    raw = response.read(); conn.close()
                    assert response.status == 200 and b"good-private-state" in raw and upstream.hits == ["bad","good"], raw
            finally:
                upstream.progress.set()
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        # These local logs contain fixture values only; don't print production logs.
        events = []
        for line in (root / "runtime.log").read_text().splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
        assert any(e.get("event") == "upstream_attempt" and e.get("attempt_success") is False for e in events)
        if mode == "native":
            reasons = {e.get("failure_reason") for e in events}
            assert {"compaction_invalid_output","compaction_stream_failed","compaction_unsupported"} <= reasons, reasons
    upstream.shutdown(); upstream.server_close()
    print(f"PASS compaction retry {mode}: invalid/missing/duplicate/mismatch/EOF/capability, exhausted budget, normal-route isolation, compact JSON")


if __name__ == "__main__":
    for mode in sys.argv[2:] or ("native", "generic"):
        verify(Path(sys.argv[1]).resolve(), mode)
