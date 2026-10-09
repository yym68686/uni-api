"""Guarded Responses heartbeat visibility, private buffering and retry contracts."""
import http.client
from concurrent.futures import ThreadPoolExecutor
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


KEEPALIVE = b'event: keepalive\ndata: {"type":"keepalive","sequence_number":0}\n\n'
CASES = ["success", "normalized", "canonical", "noncanonical", "retry", "same-chunk",
         "eof", "malformed", "first-timeout", "idle-timeout", "total-timeout",
         "overflow", "empty-terminal", "exhausted", "postcommit"]
RETRIES = {"retry", "same-chunk", "eof", "malformed", "first-timeout",
           "idle-timeout", "total-timeout", "overflow"}


def event(kind, **extra):
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **extra}) + "\n\n").encode()


def failure():
    return event("response.failed", response={"status": "failed", "error": {
        "code": "rate_limit_exceeded", "message": "fixture retryable failure"}})


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_PUT(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.facts.extend(json.loads(line) for line in raw.splitlines() if line)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        case = body["model"]
        role = self.path.split("/")[1]
        self.server.hits[case].append(role)
        response_id = f"resp_{role}_{case}"
        prefix = event("response.created", response={"id": response_id, "status": "in_progress", "output": []})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Client-Request-ID", f"receipt-{role}")
        self.end_headers()
        try:
            if role == "first":
                time.sleep(.08)
                if case == "canonical":
                    prefix = KEEPALIVE + prefix
                if case == "noncanonical":
                    prefix = event("keepalive", sequence_number=9) + prefix
                if case == "same-chunk":
                    prefix += failure()
                if case in ("success", "normalized"):
                    prefix += KEEPALIVE + event("response.in_progress", response={"status": "in_progress"})
                    prefix += event("response.output_text.delta", delta=" ")
                self.wfile.write(prefix)
                self.wfile.flush()
                if case == "same-chunk":
                    return
                if not self.server.release[case].wait(3):
                    return
                if case in ("retry", "exhausted"):
                    self.wfile.write(failure())
                    return
                if case == "eof":
                    return
                if case == "malformed":
                    self.wfile.write(b"event: broken\ndata: not-json\n\n")
                    return
                if case in ("first-timeout", "idle-timeout", "total-timeout"):
                    time.sleep(.8)
                    return
                if case == "overflow":
                    self.wfile.write(event("response.in_progress", response={"status": "in_progress"}) * 150)
                    return
            else:
                if not self.server.release[case].wait(3):
                    return
                self.wfile.write(prefix + KEEPALIVE)
            if case == "empty-terminal":
                self.wfile.write(event("response.completed", response={"id": response_id, "status": "completed", "output": []}))
                return
            self.wfile.write(event("response.output_text.delta", delta="OK"))
            if case == "postcommit":
                self.wfile.write(failure())
                return
            self.wfile.write(event("response.completed", response={"id": response_id, "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "oaix_settlement_receipt": {"payload": f"final-{role}", "signatures": ["fixture"]}}}))
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary, hedging):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.facts = []
    server.hits = {case: [] for case in CASES}
    server.release = {case: threading.Event() for case in CASES}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="uni-responses-keepalive-") as directory:
            root, port = Path(directory), free_port()
            providers = []
            for case in CASES:
                for role in ("first", "fallback") if case in RETRIES or case == "postcommit" else ("first",):
                    policy = {"first_byte": 2, "total": 3}
                    if role == "first" and case.endswith("timeout"):
                        policy[{"first-timeout": "first_byte", "idle-timeout": "idle", "total-timeout": "total"}[case]] = .35
                    providers.append({"provider": f"{role}-{case}", "engine": "codex", "api": "fixture",
                        "model": [case], "base_url": f"http://127.0.0.1:{server.server_port}/{role}/v1/responses",
                        "preferences": {"cooldown_period": 0, "timeout_policy": {"default": policy},
                                        "normalize_responses_custom_tool_call_ids": case == "normalized"}})
            config = {"providers": providers, "api_keys": [{"api": "fixture-key", "model": [p["provider"] + "/*" for p in providers],
                "preferences": {"AUTO_RETRY": True, "SCHEDULING_ALGORITHM": "fixed_priority"}}],
                "preferences": {"hedging": {"enabled": hedging}}}
            (root / "api.json").write_text(json.dumps(config))
            env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
            env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
                RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot"), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                NO_PROXY="127.0.0.1,localhost", FACTS_S3_ENDPOINT=f"http://127.0.0.1:{server.server_port}", FACTS_S3_BUCKET="fixture",
                FACTS_S3_ACCESS_KEY_ID="fixture", FACTS_S3_SECRET_ACCESS_KEY="fixture", FACTS_S3_SPOOL_DIR=str(root / "facts-spool"))
            with (root / "runtime.log").open("w") as log:
                process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
                try:
                    for _ in range(100):
                        try:
                            get_json(port, "/healthz")
                            break
                        except OSError:
                            time.sleep(.05)
                    else:
                        raise RuntimeError("fixture gateway did not start")
                    for case in CASES:
                        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                        started = time.monotonic()
                        conn.request("POST", "/v1/responses", json.dumps({"model": case, "input": "fixture", "stream": True}),
                            {"Authorization": "Bearer fixture-key", "Content-Type": "application/json", "X-Request-ID": case})
                        response = conn.getresponse()
                        first = response.read1(65536)
                        elapsed = time.monotonic() - started
                        assert response.status == 200 and first == KEEPALIVE, (hedging, case, response.status, first)
                        assert elapsed < .5 and not server.release[case].is_set(), (case, elapsed)
                        with ThreadPoolExecutor(max_workers=1) as reader:
                            pending = reader.submit(response.read1, 65536)
                            time.sleep(.05)
                            assert not pending.done(), (case, "business events leaked before output")
                            server.release[case].set()
                            next_chunk = pending.result(timeout=2)
                        try:
                            rest = next_chunk + response.read()
                        except http.client.IncompleteRead as error:
                            assert case == "postcommit", (case, error)
                            rest = next_chunk + error.partial
                        conn.close()
                        wire = first + rest
                        values = [json.loads(line[5:]) for line in wire.splitlines()
                                  if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"]
                        assert sum(v.get("type") == "keepalive" and v.get("sequence_number") == 0 for v in values) == 1, (case, wire)
                        assert response.getheader("X-Client-Request-ID") == "receipt-first"
                        hits = server.hits[case]
                        expected_hits = ["first", "fallback"] if case in RETRIES else ["first"]
                        if case == "exhausted":
                            expected_hits = ["first"] * 3
                        assert hits == expected_hits, (case, hits)
                        if case in RETRIES:
                            assert f"resp_first_{case}".encode() not in rest, (case, rest)
                            assert f"resp_fallback_{case}".encode() in rest, (case, rest)
                        if case == "exhausted":
                            assert any(v.get("type") == "error" for v in values), wire
                            assert b"resp_first" not in rest and b"response.completed" not in rest, rest
                        elif case == "postcommit":
                            assert b"OK" in rest and b"response.completed" not in rest, rest
                        else:
                            assert values[-1]["type"] == "response.completed", (case, values)
                            if case != "empty-terminal":
                                assert values[-1]["response"]["usage"]["oaix_settlement_receipt"]["payload"] == (
                                    "final-fallback" if case in RETRIES else "final-first")
                        print(f"PASS hedging={hedging} {case} heartbeat={elapsed:.3f}s hits={hits}", flush=True)
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and sum(f["kind"] == "request" for f in server.facts) < len(CASES):
                        time.sleep(.1)
                    attempts = [f for f in server.facts if f["kind"] == "attempt"]
                    for case in CASES:
                        first_fact = next(f for f in attempts if f["request_id"] == case and f["provider"] == f"first-{case}")
                        timing = first_fact["transport_timing"]
                        assert timing["public_stream_ready_ms"] - first_fact["response_created_ms"] < 100, first_fact
                        if case in RETRIES or case in ("exhausted", "empty-terminal"):
                            assert first_fact["first_output_ms"] is None, first_fact
                        if case in RETRIES:
                            assert first_fact["outcome"] == "failed", first_fact
                            assert next(f for f in attempts if f["request_id"] == case and f["provider"] == f"fallback-{case}")["outcome"] == "success"
                    print(f"PASS hedging={hedging}: heartbeat is not semantic output; private retry events and final receipt preserved")
                except Exception:
                    print((root / "runtime.log").read_text()[-12000:], file=sys.stderr)
                    raise
                finally:
                    for gate in server.release.values():
                        gate.set()
                    process.terminate()
                    process.wait(timeout=5)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    for enabled in (False, True):
        verify(Path(sys.argv[1]).resolve(), enabled)
