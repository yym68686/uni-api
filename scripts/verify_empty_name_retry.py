"""Offline HTTP regressions for empty-name repair, fallback and retry boundaries."""
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


ERROR = {"error": {
    "code": "empty_string", "type": "invalid_request_error", "param": "input[82].name",
    "message": "Invalid 'input[82].name': empty string. Expected a string with minimum length 1, but got an empty string instead.",
}}


def pair(call_id):
    return [{"type": "function_call", "name": "", "call_id": call_id, "arguments": '{"ok":true}'},
            {"type": "function_call_output", "call_id": call_id, "output": "unsupported call: "}]


def payload(stream):
    return {"model": "test-model", "stream": stream, "input": [
        {"role": "user", "content": "say OK"},
        {"type": "function_call", "name": "real", "call_id": "real-call", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "real-call", "output": "actual result"},
        {"type": "reasoning", "encrypted_content": "preserve"},
        *pair("bad-one"), *pair("bad-two"),
    ], "tools": [{"type": "function", "name": "real", "parameters": {"type": "object"}}]}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        channel = self.path.split("/")[1]
        with self.server.lock:
            attempt = sum(h["channel"] == channel for h in self.server.hits) + 1
            self.server.hits.append({"channel": channel, "body": body,
                                     "key": self.headers.get("Authorization"),
                                     "attempt_id": self.headers.get("X-OAIX-Routing-Attempt-ID")})
        label = self.server.case
        repaired = body["input"][4].get("type") == "message"
        if label == "hedge-race":
            time.sleep(0.15 if channel == "first" and attempt == 1 else
                       0.5 if channel == "fallback" else 0)
        error = copy.deepcopy(ERROR)
        if label == "wrong-code":
            error["error"]["code"] = "invalid_type"
        if label == "wrong-param":
            error["error"]["param"] = "tools[82].name"
        if label == "quoted-message":
            error["error"]["message"] = "Invalid request: " + error["error"]["message"]
        if label == "wrapped":
            error = {"error": {"message": json.dumps(error)}}
        if label == "tokensfather":
            error = {"error": {"type": "upstream_error", "message": ERROR["error"]["message"]}}
        if label in ("oaix-json", "oaix-sse", "wrapped-oaix-sse"):
            error = {"error": {"code": "oaix_gateway_error", "type": "gateway_error",
                               "status": 400, "message": ERROR["error"]["message"]}}
        raw = json.dumps(error).encode()
        if label in ("oaix-sse", "wrapped-oaix-sse"):
            raw = b"event: error\ndata: " + raw + b"\n\n"
            if label == "wrapped-oaix-sse":
                raw = json.dumps({"error": {"message": raw.decode()}}).encode()
        response = {"id": "resp_fixture", "object": "response", "status": "completed",
                    "model": "upstream-model", "output": [{"id": "msg_fixture", "type": "message",
                    "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}]}
        status, content_type = 400, "application/json"
        if label == "postcommit" and channel == "first":
            status, content_type = 200, "text/event-stream"
            events = [{"type": "response.output_text.delta", "delta": "started"},
                      {"type": "response.failed", "response": {"status": "failed", "error": ERROR["error"]}}]
            raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        elif repaired and label in ("repair-then-400", "repair-then-503"):
            status = 400 if label == "repair-then-400" else 503
            raw = json.dumps({"error": {"message": "fixture failure", "type": "invalid_request_error"}}).encode()
        elif (repaired and label not in ("repeat-error", "exhausted")) or label == "already-success" or (
            channel == "fallback" and label not in ("repeat-error", "exhausted")) or (
            channel == "third" and label != "exhausted"):
            status = 200
            if body.get("stream"):
                content_type = "text/event-stream"
                events = [{"type": "response.output_text.delta", "delta": "OK"},
                          {"type": "response.completed", "response": response}]
                raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
            else:
                raw = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary, hedging):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits, server.lock = [], threading.Lock()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-empty-name-") as directory:
        root, port = Path(directory), free_port()
        config = {"providers": [{"provider": name, "engine": "gpt",
            "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses",
            "api": ["fixture-key-1", "fixture-key-2"], "model": [{"upstream-model": "test-model"}],
            "preferences": {"cooldown_period": 0, "oaix_routing_attempt_id": True,
                            "model_timeout": {"default": 0.05}}}
            for name in ("first", "fallback", "third")],
            "api_keys": [
                {"api": "retry", "model": ["first/*", "fallback/*", "third/*"]},
                {"api": "no-retry", "model": ["first/*", "fallback/*", "third/*"],
                 "preferences": {"AUTO_RETRY": False}}],
            "preferences": {"hedging": {"enabled": hedging, "max_inflight_attempts": 2}}}
        config_path = root / "api.json"
        original = json.dumps(config).encode()
        config_path.write_bytes(original)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("FACTS_S3_", "OTEL_", "FUGUE_OTEL_", "UNI_API_CONTROL_RESTORE_"))
               and k != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                   NO_PROXY="127.0.0.1,localhost")

        def request(body=None, key="retry", target=False, endpoint="/v1/responses"):
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            if target:
                headers["X-Uni-API-Provider"] = "first"
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.request("GET" if body is None else "POST", "/healthz" if body is None else endpoint,
                             None if body is None else json.dumps(body), headers)
                response = conn.getresponse()
                return response.status, response.read()
            finally:
                conn.close()

        with (root / "log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if request()[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.05)
                else:
                    raise AssertionError("fixture startup failed")
                count, expected_repairs = 0, 0
                for stream in (False, True):
                    cases = ["repair", "wrapped", "tokensfather", "oaix-json", "oaix-sse", "wrapped-oaix-sse",
                             "repair-then-400", "repair-then-503", "repeat-error", "exhausted",
                             "wrong-code", "wrong-param", "quoted-message", "named-call", "executed-call",
                             "duplicate-id", "no-retry", "targeted", "compact", "already-success"]
                    if stream:
                        cases += ["postcommit"]
                        if hedging:
                            cases += ["hedge-race"]
                    for label in cases:
                        server.case, server.hits = label, []
                        body = payload(stream)
                        if label == "named-call":
                            body["input"][4]["name"] = body["input"][6]["name"] = "real"
                        if label == "executed-call":
                            body["input"][5]["output"] = body["input"][7]["output"] = "actual result"
                        if label == "duplicate-id":
                            body["input"] += [body["input"][5], body["input"][7]]
                        if label == "compact":
                            body["stream"] = False
                        status, raw = request(body, "no-retry" if label == "no-retry" else "retry",
                                              label == "targeted", "/v1/responses/compact" if label == "compact" else "/v1/responses")
                        no_repair = label in ("wrong-code", "wrong-param", "quoted-message", "named-call",
                                             "executed-call", "duplicate-id", "no-retry", "targeted", "compact",
                                             "already-success", "postcommit")
                        expected = (["first"] if no_repair else
                                    ["first", "fallback", "first"] if label == "hedge-race" else
                                    ["first", "first", "fallback"] if label in ("repair-then-400", "repair-then-503") else
                                    ["first", "first", "fallback", "third"] if label == "repeat-error" else
                                    ["first", "first"])
                        success = not no_repair or label in ("already-success", "postcommit")
                        if label == "exhausted":
                            # Three providers + configured key budget (capped 10), plus one repair.
                            expected = ["first", "first"] + ["fallback", "third", "first"] * 4
                            success = False
                        context = (hedging, stream, label, status, raw[:500], server.hits)
                        assert status == (200 if success else 400), context
                        assert [h["channel"] for h in server.hits] == expected, context
                        assert server.hits[0]["body"]["input"] == body["input"], context
                        fixed_hits = [h for h in server.hits if h["body"]["input"][4].get("type") == "message"]
                        assert len(fixed_hits) == (0 if no_repair else 1), context
                        if fixed_hits:
                            expected_repairs += 1
                            first, fixed = server.hits[0], fixed_hits[0]
                            assert first["key"] == fixed["key"], context
                            assert first["attempt_id"] != fixed["attempt_id"], context
                            assert fixed["channel"] == "first", context
                            assert fixed["body"]["input"][:4] == body["input"][:4], context
                            for index in (4, 5, 6, 7):
                                item = fixed["body"]["input"][index]
                                assert item["role"] == ("assistant" if index % 2 == 0 else "user"), context
                                assert json.loads(item["content"][0]["text"].split(": ", 1)[1]) == body["input"][index], context
                            assert {k: v for k, v in first["body"].items() if k != "input"} == {
                                k: v for k, v in fixed["body"].items() if k != "input"}, context
                        assert all(h["body"]["input"] == body["input"] for h in server.hits
                                   if h not in fixed_hits), context
                        if success and label != "postcommit":
                            assert b"OK" in raw and b"empty_string" not in raw, context
                        if label == "hedge-race":
                            time.sleep(0.6)
                        count += 1
                assert config_path.read_bytes() == original
                log.flush()
                events = []
                for line in (root / "log").read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass
                repairs = [e for e in events if e.get("event") == "responses_empty_name_repair"]
                assert len(repairs) == expected_repairs, repairs
                assert all(e["tool_pairs_converted"] == 2 for e in repairs), repairs
                for repair in repairs:
                    attempts = [e for e in events if e.get("event") == "upstream_attempt"
                                and e.get("request_id") == repair["request_id"]]
                    assert attempts[0]["attempt_status_code"] == 400, attempts
                    assert len({e["attempt_id"] for e in attempts}) == len(attempts), attempts
                print(f"PASS empty-name repair hedge={hedging}: {count} HTTP cases, {len(repairs)} repairs")
            except Exception:
                print((root / "log").read_text()[-8000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    for hedging in (False, True):
        verify(Path(sys.argv[1]).resolve(), hedging)
