"""Verify completion statistics against real SSE and immutable exported facts."""
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
wire_by_case = {}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_PUT(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        facts.extend(json.loads(line) for line in raw.splitlines() if line)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        case = payload["input"]
        events = [{"type": "response.created", "response": {"status": "in_progress"}}]
        if case != "tool":
            events.append({"type": "response.output_text.delta", "delta": " " if case == "blank-incomplete" else "partial answer"})
        if case in ("completed", "tool", "blank-incomplete", "partial-incomplete"):
            complete = case in ("completed", "tool")
            output = [{"type": "function_call", "name": "echo", "call_id": "call_fixture", "arguments": "{}"}] if case == "tool" else []
            event = {"type": "response.completed" if complete else "response.incomplete", "response": {
                "id": "resp_fixture", "status": "completed" if complete else "incomplete", "output": output,
                "incomplete_details": None if complete else {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}
            events.append(event)
        raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        if case == "done-without-completed":
            raw += b"data: [DONE]\n\n"
        wire_by_case[case] = raw
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        # Separate the first body chunk from termination so failures exercise
        # an already-committed HTTP 200 stream, not only precommit rejection.
        split = raw.find(b"\n\n", raw.find(b"\n\n") + 2) + 2
        self.wfile.write(raw[:split])
        self.wfile.flush()
        time.sleep(.1)
        self.wfile.write(raw[split:])


def main():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    variants = [("gpt", "gpt"), ("codex", "codex"), ("normalized", "codex")]
    cases = ["completed", "tool", "blank-incomplete", "partial-incomplete", "eof", "done-without-completed"]
    with tempfile.TemporaryDirectory(prefix="uni-completion-") as directory:
        root = Path(directory)
        port = free_port()
        cfg = {"providers": [{"provider": name, "engine": engine,
            "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}/v1/responses", "api": "fixture",
            "model": ["gpt-6-astra"], "preferences": {"normalize_responses_custom_tool_call_ids": name == "normalized"}}
            for name, engine in variants], "api_keys": [{"api": "fixture-admin", "model": ["all"]}],
            "preferences": {"AUTO_RETRY": False}}
        (root / "api.json").write_text(json.dumps(cfg))
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root / "body-spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
            NO_PROXY="127.0.0.1,localhost", FACTS_S3_ENDPOINT=f"http://127.0.0.1:{upstream.server_port}", FACTS_S3_BUCKET="fixture",
            FACTS_S3_ACCESS_KEY_ID="fixture", FACTS_S3_SECRET_ACCESS_KEY="fixture", FACTS_S3_SPOOL_DIR=str(root / "facts-spool"))

        def call(method, path, body=None, provider=None, request_id=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
            headers = {"Authorization": "Bearer fixture-admin", "Content-Type": "application/json"}
            if provider: headers["X-Uni-API-Provider"] = provider
            if request_id: headers["X-Request-ID"] = request_id
            conn.request(method, path, json.dumps(body) if body else None, headers)
            response = conn.getresponse()
            try: raw = response.read()
            except http.client.IncompleteRead as exc: raw = exc.partial
            conn.close()
            return response.status, raw

        with (root / "runtime.log").open("w") as log:
            process = subprocess.Popen([str(Path(sys.argv[1]).resolve())], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if call("GET", "/healthz")[0] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                for provider, _ in variants:
                    # This endpoint is a current-minute live gauge, not a
                    # rolling history query. Keep this sub-second six-request
                    # fixture within one bucket; immutable facts below still
                    # verify every request regardless of wall-clock time.
                    minute_offset = time.time() % 60
                    if minute_offset > 55:
                        time.sleep(60 - minute_offset + .01)
                    for case in cases:
                        status, raw = call("POST", "/v1/responses", {"model": "gpt-6-astra", "input": case, "stream": True}, provider, provider + "-" + case)
                        assert status == 200, (provider, case, status, raw)
                        if "incomplete" in case:
                            assert b"response.incomplete" in raw and b"response.completed" not in raw
                        if provider == "gpt" and case in ("completed", "tool", "blank-incomplete", "partial-incomplete"):
                            assert raw == wire_by_case[case], (provider, case, "wire changed")
                    status, raw = call("GET", "/v1/channel-metrics?model=gpt-6-astra&endpoint=/v1/responses&stream=true")
                    assert status == 200
                    row = next(r for r in json.loads(raw)["data"] if r["provider"] == provider)
                    stats = row["stats"]
                    assert stats["success"] == 2 and stats["failed"] == 4, row
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and sum(f["kind"] == "request" for f in facts) < len(variants) * len(cases): time.sleep(.1)
                for provider, _ in variants:
                    for case in cases:
                        complete = case in ("completed", "tool")
                        for kind in ("request", "attempt"):
                            rows = [f for f in facts if f["kind"] == kind and f["request_id"] == provider + "-" + case]
                            assert len(rows) == 1, (provider, case, kind, rows)
                            fact = rows[0]
                            assert fact["response_completed"] is complete, fact
                            assert fact["outcome"] == ("success" if complete else "failed"), fact
                            if "incomplete" in case:
                                assert fact["terminal_kind"] == "incomplete", fact
                                assert fact["failure_reason"] == "responses_max_output_tokens", fact
                                if kind == "request": assert fact["status"] == 200, fact
                print("PASS: 18 real SSE cases; HTTP 200, tool completion, incomplete, EOF, [DONE], live metrics, S3 request/attempt facts, unchanged passthrough")
            except Exception:
                print((root / "runtime.log").read_text()[-18000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                process.wait(timeout=5)
    upstream.shutdown()
    upstream.server_close()


if __name__ == "__main__":
    main()
