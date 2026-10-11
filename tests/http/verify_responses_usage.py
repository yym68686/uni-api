"""Responses usage conflicts across native/conversion, SSE/JSON and stream modes."""
import copy
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

MODEL = "deepseek-v4.1-flash"
NATIVE = {
    "input_tokens": 11476, "output_tokens": 107, "total_tokens": 11583,
    "input_tokens_details": {"cached_tokens": 11264, "cache_write_tokens": 32},
    "output_tokens_details": {"reasoning_tokens": 67},
    "oaix_settlement_receipt": "fixture-receipt",
}
CONFLICT = NATIVE | {
    "prompt_tokens": 0, "completion_tokens": 0,
    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}
CASES = {
    "conflict": CONFLICT,
    "native": NATIVE,
    "zero": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
             "prompt_tokens": 9, "completion_tokens": 8},
    "alias": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
    "partial": {"input_tokens": 6},
    "total": {"total_tokens": 7},
    "null": None,
    "missing": None,
}
facts = []
source_wires = {}


def case_name(value):
    if isinstance(value, str) and value.startswith("fixture:"):
        return value.split(":", 1)[1]
    children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return next((name for child in children if (name := case_name(child))), None)


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
        name = case_name(payload)
        assert name in CASES, payload
        response = {
            "id": "resp_fixture", "object": "response", "model": MODEL, "status": "completed",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "test"}]}],
        }
        if name != "missing":
            response["usage"] = copy.deepcopy(CASES[name])
        if payload.get("stream"):
            events = [
                {"type": "response.created", "response": {"status": "in_progress"}},
                {"type": "response.output_text.delta", "delta": "test"},
                {"type": "response.completed", "sequence_number": 42, "response": response},
            ]
            prefix = b"".join(f"event: {e['type']}\r\ndata: {json.dumps(e)}\r\n\r\n".encode() for e in events[:2])
            terminal = f"id: fixture-terminal\r\nretry: 123\r\n: usage-check\r\nevent: response.completed\r\ndata: {json.dumps(events[-1])}\r\n\r\n".encode()
            raw = prefix + terminal
            content_type = "text/event-stream"
        else:
            raw = json.dumps(response, indent=2).encode()
            content_type = "application/json"
        source_wires[(name, bool(payload.get("stream")))] = raw
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if payload.get("stream"):
            self.wfile.write(prefix)
            self.wfile.flush()
            time.sleep(.01)  # Exercise normalization after the guarded commit too.
            for offset in range(0, len(terminal), 23):
                self.wfile.write(terminal[offset:offset + 23])
        else:
            self.wfile.write(raw)


def extract_usage(raw, stream, chat):
    if not stream:
        return json.loads(raw).get("usage")
    values = []
    for line in raw.decode().splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            values.append(json.loads(line[5:]))
    if chat:
        reported = [v["usage"] for v in values if isinstance(v.get("usage"), dict)]
        return reported[-1] if reported else None
    terminal = [v for v in values if v.get("type") == "response.completed"]
    assert len(terminal) == 1, values
    assert terminal[0]["sequence_number"] == 42
    return terminal[0]["response"].get("usage")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    variants = [("gpt", "gpt"), ("codex", "codex"), ("normalized", "codex")]
    with tempfile.TemporaryDirectory(prefix="uni-responses-usage-") as directory:
        root = Path(directory)
        port = free_port()
        config = {
            "providers": [{"provider": name, "engine": engine, "api": "fixture-upstream",
                "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses", "model": [{MODEL: f"{name}-model"}],
                "preferences": {"normalize_responses_custom_tool_call_ids": name == "normalized"}}
                for name, engine in variants],
            "api_keys": [{"api": "fixture-admin", "model": ["all"]}],
            "preferences": {"AUTO_RETRY": False},
        }
        (root / "api.json").write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
            UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
            RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0", NO_PROXY="127.0.0.1,localhost",
            FACTS_S3_ENDPOINT=f"http://127.0.0.1:{server.server_port}", FACTS_S3_BUCKET="fixture",
            FACTS_S3_ACCESS_KEY_ID="fixture", FACTS_S3_SECRET_ACCESS_KEY="fixture",
            FACTS_S3_SPOOL_DIR=str(root / "facts-spool"))

        def call(path, body=None, provider=None, request_id=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            headers = {"Authorization": "Bearer fixture-admin", "Content-Type": "application/json"}
            if provider and path == "/v1/responses":
                headers["X-Uni-API-Provider"] = provider
            if request_id:
                headers["X-Request-ID"] = request_id
            conn.request("POST" if body else "GET", path, json.dumps(body) if body else None, headers)
            response = conn.getresponse()
            raw = response.read()
            conn.close()
            return response.status, raw

        with (root / "runtime.log").open("w") as log:
            process = subprocess.Popen([str(Path(sys.argv[1]).resolve())], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if call("/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                checked = []
                for provider, _ in variants:
                    for stream in [False, True]:
                        for chat in [False, True]:
                            for name in CASES:
                                request_id = f"usage-{provider}-{stream}-{chat}-{name}"
                                path = "/v1/chat/completions" if chat else "/v1/responses"
                                body = {"model": f"{provider}-model", "stream": stream}
                                body.update({"messages": [{"role": "user", "content": f"fixture:{name}"}]} if chat else {"input": f"fixture:{name}"})
                                if stream and chat:
                                    body["stream_options"] = {"include_usage": True}
                                status, raw = call(path, body, provider, request_id)
                                assert status == 200, (request_id, status, raw)
                                usage = extract_usage(raw, stream, chat)
                                if name in ("missing", "null"):
                                    assert usage is None, (request_id, usage)
                                elif name == "total":
                                    assert usage == {"total_tokens": 7}, (request_id, usage)
                                elif name == "partial":
                                    assert usage == {"prompt_tokens" if chat else "input_tokens": 6}, (request_id, usage)
                                else:
                                    in_key = "prompt_tokens" if chat or name == "alias" else "input_tokens"
                                    out_key = "completion_tokens" if chat or name == "alias" else "output_tokens"
                                    expected = (0, 0, 0) if name == "zero" else (6, 1, 7) if name == "alias" else (11476, 107, 11583)
                                    assert (usage[in_key], usage[out_key], usage["total_tokens"]) == expected, (request_id, usage)
                                    if name in ("native", "conflict"):
                                        assert usage["prompt_tokens_details" if chat else "input_tokens_details"]["cached_tokens"] == 11264
                                        assert usage["completion_tokens_details" if chat else "output_tokens_details"]["reasoning_tokens"] == 67
                                        assert usage["oaix_settlement_receipt"] == "fixture-receipt"
                                    if not chat and name in ("conflict", "zero"):
                                        assert usage["prompt_tokens"] == usage["input_tokens"]
                                        assert usage["completion_tokens"] == usage["output_tokens"]
                                if not chat and name not in ("conflict", "zero") and provider == "gpt":
                                    assert raw == source_wires[(name, stream)], (request_id, "unchanged wire was rewritten")
                                if not chat and stream and provider in ("gpt", "codex"):
                                    assert b"id: fixture-terminal\r\nretry: 123\r\n: usage-check\r\n" in raw
                                checked.append(request_id)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and sum(f["kind"] == "request" for f in facts) < len(checked):
                    time.sleep(.1)
                for request_id in checked:
                    if request_id.endswith(("-conflict", "-native")):
                        row = next(f for f in facts if f["kind"] == "request" and f["request_id"] == request_id)
                        assert row["input_tokens"] == 11476, row
                        assert row["output_tokens"] == 107, row
                        assert row["cache_read_tokens"] == 11264, row
                print(f"PASS: {len(checked)} usage HTTP cases; native/converted, SSE/JSON, three stream modes, cache/reasoning/receipt, absent/zero, preserved wire and exported facts")
            except Exception:
                print((root / "runtime.log").read_text()[-6000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                process.wait(timeout=10)
    server.shutdown()
    server.server_close()


if __name__ == "__main__":
    main()
