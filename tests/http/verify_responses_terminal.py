"""A Responses terminal wins over later EOF, timeouts and client cancellation."""
from fact_sink import accept_facts
from verify_dispatch_timing import free_port, get_json
import hashlib
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


def event(kind, **fields):
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path == "/facts": return accept_facts(self, self.server.facts)
        case = json.loads(self.rfile.read(int(self.headers["Content-Length"])))['input']
        self.server.hits.append(case)
        prefix = event("response.created", response={"id": "resp_fixture", "status": "in_progress"})
        prefix += event("response.output_text.delta", delta="OK")
        kind = case.split("-", 1)[0]
        if kind in ("completed", "incomplete", "failed"):
            response = {"id": "resp_fixture", "status": kind, "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
            if kind == "failed": response['error'] = {"code": "server_error", "message": "synthetic failure"}
            if kind == "incomplete": response['incomplete_details'] = {"reason": "max_output_tokens"}
            terminal = event("response." + kind, response=response)
        else:
            terminal = {"eof": b"", "done": b"data: [DONE]\n\n", "malformed": b"data: not-json\n\n"}[kind]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(prefix)
            self.wfile.flush()
            time.sleep(.03)
            # Exercise a terminal that spans HTTP body chunks too.
            for piece in (terminal[:17], terminal[17:]):
                self.wfile.write(piece)
                self.wfile.flush()
                time.sleep(.01)
            if case.endswith(("held", "close")): time.sleep(1.1)
        except (BrokenPipeError, ConnectionResetError): pass


def verify(binary, mixed):
    cases = [f"{kind}-{end}" for kind in ("completed", "incomplete", "failed") for end in ("eof", "held", "close")]
    cases += ["eof", "done", "malformed"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits, server.facts = [], []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="uni-terminal-") as directory:
            root, port = Path(directory), free_port()
            providers = [{"provider": "fixture", "engine": "gpt", "api": "fixture", "model": ["fixture"],
                "base_url": f"http://127.0.0.1:{server.server_port}/v1/responses", "preferences": {
                    "keepalive_interval": .02, "cooldown_period": 0,
                    "timeout_policy": {"default": {"first_byte": .3, "total": .7}}}}]
            if mixed:
                providers.append({"provider": "unused-chat", "engine": "gpt", "api": "fixture", "model": ["fixture"],
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1/chat/completions"})
            config = {"providers": providers, "api_keys": [{"api": "fixture-key",
                "model": [p['provider'] + "/*" for p in providers],
                "preferences": {"AUTO_RETRY": True, "SCHEDULING_ALGORITHM": "fixed_priority"}}]}
            (root / "api.json").write_text(json.dumps(config))
            env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
            env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
                RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot"), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                FACTS_INGEST_URL=f"http://127.0.0.1:{server.server_port}/facts", FACTS_INGEST_TOKEN="fixture",
                FACTS_SPOOL_DIR=str(root / "facts"), NO_PROXY="127.0.0.1,localhost")
            with (root / "runtime.log").open("w") as log:
                process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
                try:
                    for _ in range(100):
                        try: get_json(port, "/healthz"); break
                        except OSError: time.sleep(.05)
                    else: raise AssertionError("gateway did not start")
                    for case in cases:
                        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                        conn.request("POST", "/v1/responses", json.dumps({"model": "fixture", "stream": True, "input": case}),
                            {"Authorization": "Bearer fixture-key", "Content-Type": "application/json", "X-Request-ID": case})
                        response = conn.getresponse()
                        assert response.status == 200, (case, response.status)
                        wire = b""
                        if case.endswith("close"):
                            kind = case.split("-", 1)[0].encode()
                            while b"event: response." + kind not in wire or not wire.endswith(b"\n\n"):
                                part = response.read(1)
                                if not part:
                                    # Native raw mode filters an upstream failure terminal.
                                    assert not mixed and case.startswith("failed-"), (case, wire)
                                    break
                                wire += part
                        else:
                            wire = response.read()
                        response.close()
                        conn.close()
                        assert server.hits.count(case) == 1, (case, server.hits)
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and sum(f.get('kind') == 'request' for f in server.facts) < len(cases): time.sleep(.1)
                    for case in cases:
                        kind = case.split("-", 1)[0]
                        for fact_kind in ("request", "attempt"):
                            fact = next(f for f in server.facts if f.get('request_id') == case and f.get('kind') == fact_kind)
                            success = kind == "completed"
                            assert fact['outcome'] == ("success" if success else "failed"), (mixed, case, fact)
                            assert fact['response_completed'] is success, (mixed, case, fact)
                            assert fact['first_output_ms'] is not None and fact['first_text_ms'] is not None, fact
                            terminal = {"completed": "completed", "incomplete": "incomplete", "failed": "semantic_failure"}.get(kind, "protocol_error")
                            assert fact['terminal_kind'] == terminal, (case, fact)
                            if kind == "incomplete": assert fact['failure_reason'] == 'responses_max_output_tokens', fact
                            if kind == "failed": assert fact['failure_reason'] == 'upstream_response_failed', fact
                            if fact_kind == "request":
                                assert fact['status'] == (200 if kind in ("completed", "incomplete") else 500 if kind == "failed" else 502), fact
                    if mixed:
                        traces = [f for f in server.facts if f.get('kind') == 'trace' and f.get('stage') == 'upstream_attempt']
                        for case in ("eof", "done", "malformed", "failed-close"):
                            trace = next(f for f in traces if f.get('request_id') == case)
                            detail = trace['trace_detail']
                            assert detail['error_sha256'] and detail['response_completed'] is False, trace
                            assert detail['failure_reason'] and detail['first_output_ms'] is not None, trace
                            assert 'synthetic failure' not in json.dumps(detail) and 'not-json' not in json.dumps(detail), trace
                        eof = next(f for f in traces if f.get('request_id') == 'eof')['trace_detail']
                        assert eof['error_sha256'] == hashlib.sha256(b'Responses upstream ended without a terminal response event').hexdigest(), eof
                        for case, code in (("eof", "eof_without_terminal"), ("done", "done_without_terminal"), ("malformed", "invalid_event_json")):
                            trace = next(f for f in traces if f.get('request_id') == case)
                            assert trace['trace_detail']['stream_error_code'] == code, trace
                    print(f"PASS mixed={mixed}: {len(cases)} terminal, cancellation, timeout, no-replay and diagnostic contracts", flush=True)
                except Exception:
                    print((root / "runtime.log").read_text()[-12000:], file=sys.stderr)
                    raise
                finally:
                    process.terminate()
                    process.wait(timeout=5)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    for mixed in (False, True): verify(Path(sys.argv[1]).resolve(), mixed)
