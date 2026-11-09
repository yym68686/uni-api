"""Responses failover must survive mixed candidate fallback and HTTP/SSE chunking."""
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

RETRY = {"failed", "coalesced", "eof", "whitespace-failed", "empty-completed",
         "whitespace-completed", "empty-incomplete", "malformed", "wrong-type",
         "wrong-content-type", "done-only", "multiline-failed", "timeout", "overflow", "flat-error"}
NO_RETRY = {"text-failed", "tool-failed", "custom-tool-failed", "reasoning-failed", "partial-incomplete", "flat-error-after-output"}
SUCCESS = {"tool-only", "terminal-only", "whitespace-success", "opaque-reasoning", "refusal"}
CASES = sorted(RETRY | NO_RETRY | SUCCESS | {"semantic-400", "flat-400", "exhausted"})
VARIANTS = [("gpt", False, 0, False), ("codex", False, 0, True),
            ("gpt", True, 0, False), ("gpt", True, 3, False), ("codex", True, 3, False)]


def event(kind, **fields):
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


def fixture(case, role):
    rid = f"resp_{role}_{case}"
    created = event("response.created", response={"id": rid, "status": "in_progress", "output": []})
    text = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}
    tool = {"type": "function_call", "name": "lookup", "call_id": "call_fixture", "arguments": "{}"}
    custom = {"type": "custom_tool_call", "name": "lookup", "call_id": "call_fixture", "input": "fixture"}
    reasoning = {"type": "reasoning", "summary": [], "encrypted_content": "opaque-fixture"}
    failure = event("response.failed", response={"id": rid, "status": "failed", "error": {
        "type": "invalid_request_error" if case == "semantic-400" else "upstream_error",
        "code": "fixture_error", "message": "synthetic failure"}})
    complete = lambda items: event("response.completed", response={"id": rid, "status": "completed", "output": items,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}})
    ok = created + event("response.output_text.delta", delta="OK") + complete([text])
    if role != "first" and case != "exhausted":
        return [ok]
    if case in ("flat-error", "flat-400", "flat-error-after-output"):
        flat = event("error", code="invalid_request_error" if case == "flat-400" else "rate_limit_exceeded",
                     message="synthetic flat error", param=None, sequence_number=2)
        prefix = created + event("response.output_text.delta", delta="OK") if case == "flat-error-after-output" else created
        return [prefix, flat]
    if case in ("failed", "semantic-400", "exhausted", "wrong-content-type"):
        return [created, failure]
    if case == "coalesced": return [created + failure]
    if case == "multiline-failed":
        return [created, failure.replace(b', "response":', b',\r\ndata: "response":').replace(b"\n\n", b"\r\n\r\n")]
    if case in ("eof", "timeout"): return [created]
    if case == "whitespace-failed": return [created + event("response.output_text.delta", delta=" \n\t"), failure]
    if case == "empty-completed": return [created, complete([])]
    if case == "whitespace-completed":
        return [created + event("response.output_text.delta", delta=" \n\t"), complete([
            {"type": "message", "content": [{"type": "output_text", "text": "  "}]}])]
    if case == "empty-incomplete": return [created, event("response.incomplete", response={"id": rid, "status": "incomplete", "output": []})]
    if case == "malformed": return [created, b"event: response.in_progress\ndata: not-json\n\n"]
    if case == "wrong-type": return [created, b'event: response.completed\ndata: {"type":"response.in_progress"}\n\n']
    if case == "done-only": return [created, b"data: [DONE]\n\n"]
    if case == "overflow": return [created, event("response.in_progress", response={"status": "in_progress"}) * 140]
    if case == "text-failed": return [created + event("response.output_text.delta", delta="OK") + failure]
    if case == "partial-incomplete": return [created + event("response.output_text.delta", delta="OK"),
        event("response.incomplete", response={"id": rid, "status": "incomplete", "output": [text],
              "incomplete_details": {"reason": "max_output_tokens"}})]
    if case in ("tool-failed", "custom-tool-failed", "reasoning-failed"):
        item = {"tool-failed": tool, "custom-tool-failed": custom, "reasoning-failed": reasoning}[case]
        return [created + event("response.output_item.added", item=item), failure]
    if case == "tool-only": return [created + event("response.output_item.added", item=tool), complete([tool])]
    if case == "opaque-reasoning": return [created, complete([reasoning])]
    if case == "refusal": return [created, complete([{"type": "message", "content": [{"type": "refusal", "refusal": "fixture refusal"}]}])]
    if case == "terminal-only": return [created, complete([text])]
    if case == "whitespace-success": return [created + event("response.output_text.delta", delta=" \n\t"), event("response.output_text.delta", delta="OK") + complete([text])]
    raise AssertionError(case)


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path == "/facts": return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        case, role = json.loads(raw)["model"], self.path.split("/")[1]
        self.server.hits.append((case, role, hashlib.sha256(raw).hexdigest()))
        chunks = fixture(case, role)
        if self.path.endswith("/chat/completions"):
            def chat(**fields): return ("data: " + json.dumps(fields) + "\n\n").encode()
            delta = lambda **fields: chat(choices=[{"index": 0, "delta": fields, "finish_reason": None}])
            finish = chat(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]) + b"data: [DONE]\n\n"
            error = chat(error={"status_code": 400 if case == "semantic-400" else 503, "message": "synthetic failure"})
            chunks = {
                "failed": [error], "semantic-400": [error], "exhausted": [error],
                "eof": [delta(role="assistant")],
                "empty-completed": [finish],
                "whitespace-completed": [delta(content=" \n\t"), finish],
                "whitespace-failed": [delta(content=" \n\t"), error],
                "text-failed": [delta(content="OK"), error],
                "tool-only": [delta(tool_calls=[{"index": 0, "id": "call_fixture", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}}]), finish],
            }[case]
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if case == "wrong-content-type" and role == "first" else "text/event-stream")
        self.send_header("X-Oaix-Fixture", role)
        self.end_headers()
        try:
            for index, chunk in enumerate(chunks):
                if index: time.sleep(.045)
                # Exercise JSON, UTF-8 and SSE separators across reads.
                if case == "multiline-failed":
                    for offset in range(0, len(chunk), 7):
                        self.wfile.write(chunk[offset:offset + 7]); self.wfile.flush()
                else:
                    self.wfile.write(chunk); self.wfile.flush()
            if case == "timeout" and role == "first": time.sleep(.5)
        except (BrokenPipeError, ConnectionResetError): pass


def verify(binary, engine, mixed, budget, hedging, converted=False):
    label = f"{engine}-mixed={mixed}-400={budget}-hedging={hedging}-converted={converted}"
    cases = (["failed", "semantic-400", "eof", "empty-completed", "whitespace-completed",
              "whitespace-failed", "text-failed", "tool-only", "exhausted"] if converted else CASES)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits, server.facts = [], []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="uni-responses-precommit-") as directory:
            root, port = Path(directory), free_port()
            providers = []
            for role in ("first", "fallback"):
                endpoint = "chat/completions" if converted and role == "first" else "responses"
                providers.append({"provider": role, "engine": engine, "api": "fixture",
                    "base_url": f"http://127.0.0.1:{server.server_port}/{role}/v1/{endpoint}", "model": cases,
                    "preferences": {"cooldown_period": 0, "api_key_cooldown_period": 0,
                        "keepalive_interval": .02,
                        "timeout_policy": {"default": {"first_byte": .3, "total": 2}}}})
            if mixed:
                providers.append({"provider": "unused-chat", "engine": "gpt", "api": "fixture", "model": CASES,
                    "base_url": f"http://127.0.0.1:{server.server_port}/unused/v1/chat/completions"})
            config = {"providers": providers, "api_keys": [{"api": "fixture-key", "model": [p["provider"] + "/*" for p in providers],
                "preferences": {"AUTO_RETRY": 1, "SCHEDULING_ALGORITHM": "fixed_priority"}}],
                "preferences": {"unknown_upstream_400_retries": budget, "hedging": {"enabled": hedging}}}
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
                        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
                        conn.request("POST", "/v1/responses", json.dumps({"model": case, "stream": True, "input": "fixture"}),
                            {"Authorization": "Bearer fixture-key", "Content-Type": "application/json", "X-Request-ID": case})
                        response = conn.getresponse()
                        try: wire = response.read()
                        except http.client.IncompleteRead as error: wire = error.partial
                        conn.close()
                        hits = [role for model, role, _ in server.hits if model == case]
                        retried = case in RETRY or case == "exhausted" or case in ("semantic-400", "flat-400") and budget > 0
                        assert hits == (["first", "fallback"] if retried else ["first"]), (label, case, hits, wire)
                        if not converted:
                            assert len({sha for model, _, sha in server.hits if model == case}) == 1, (label, case, "request mutated")
                        completed = b"response.completed" in wire
                        expected_success = case in SUCCESS or retried and case != "exhausted"
                        assert completed == expected_success, (label, case, response.status, wire)
                        if retried:
                            assert f"resp_first_{case}".encode() not in wire, (label, case, "failed metadata leaked", wire)
                        if mixed and case in SUCCESS and not converted:
                            # Gateway comments may be added, while upstream event bytes stay exact.
                            expected = b"".join(fixture(case, "first"))
                            business = b"".join(frame + b"\n\n" for frame in wire.split(b"\n\n")
                                                if frame and not frame.startswith(b": keepalive"))
                            assert business == expected, (label, case, wire)
                        if mixed and case in ("exhausted", "semantic-400") and not expected_success:
                            values = [json.loads(line[5:]) for line in wire.splitlines() if line.startswith(b"data:")]
                            assert any(v.get("type") == "error" for v in values), (label, case, values)
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and sum(f.get("kind") == "request" for f in server.facts) < len(cases): time.sleep(.1)
                    for case in ("whitespace-failed", "whitespace-completed", "empty-completed", "failed"):
                        fact = next(f for f in server.facts if f.get("kind") == "attempt" and f.get("request_id") == case and f.get("provider") == "first")
                        assert fact.get("first_output_ms") is None and fact.get("first_text_ms") is None, (label, case, fact)
                    if not converted:
                        fact = next(f for f in server.facts if f.get("kind") == "request" and f.get("request_id") == "partial-incomplete")
                        assert fact["outcome"] == "failed" and fact["failure_reason"] == "responses_max_output_tokens", (label, fact)
                        for case in ("flat-error", "flat-400", "flat-error-after-output"):
                            fact = next(f for f in server.facts if f.get("kind") == "attempt" and f.get("request_id") == case and f.get("provider") == "first")
                            assert fact["failure_reason"] in ("upstream_response_failed", "upstream_400_sse"), (label, case, fact)
                    print(f"PASS {label}: {len(cases)} failover, commit, wire, timing and budget contracts", flush=True)
                except Exception:
                    print((root / "runtime.log").read_text()[-10000:], file=sys.stderr)
                    raise
                finally:
                    process.terminate(); process.wait(timeout=5)
    finally:
        server.shutdown(); server.server_close()


if __name__ == "__main__":
    for variant in VARIANTS:
        verify(Path(sys.argv[1]).resolve(), *variant)
    for budget in (0, 3):
        verify(Path(sys.argv[1]).resolve(), "gpt", False, budget, False, converted=True)
