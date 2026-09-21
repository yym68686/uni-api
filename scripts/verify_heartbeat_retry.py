"""Offline HTTP regressions for reactive, same-channel heartbeat context repair."""
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
    "code": "function_call_output_not_found", "type": "invalid_request_error",
    "param": "input",
    "message": "The tool output does not match a previous tool call. Resend the tool-call context.",
}}
HEARTBEAT = {"type": "function_call_output", "name": "automation_update",
             "namespace": "codex_app", "output": "<heartbeat>preserve context</heartbeat>\n"}


def payload(stream):
    return {"model": "test-model", "stream": stream, "input": [
        {"role": "user", "content": "say OK"},
        {"type": "function_call", "name": "automation_update", "call_id": "real-call",
         "arguments": "{}"},
        {**HEARTBEAT, "call_id": "real-call"},
        {"type": "reasoning", "encrypted_content": "preserve"},
        *[copy.deepcopy(HEARTBEAT) for _ in range(16)],
    ]}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        channel = self.path.split("/")[1]
        with self.server.lock:
            attempt = sum(hit["channel"] == channel for hit in self.server.hits) + 1
            self.server.hits.append({"channel": channel, "body": body,
                                     "key": self.headers.get("Authorization"),
                                     "attempt_id": self.headers.get("X-OAIX-Routing-Attempt-ID")})
        label = self.server.case
        repaired = body["input"][-1].get("type") == "message"
        if label == "hedge-race":
            time.sleep(0.15 if channel == "first" and attempt == 1 else
                       0.5 if channel == "fallback" else 0)
        error = copy.deepcopy(ERROR)
        if label == "wrong-code":
            error["error"]["code"] = "another_error"
        if label == "quoted-message":
            error["error"]["message"] = "Invalid input: " + error["error"]["message"]
        if label == "wrapped":
            error = {"error": {"message": json.dumps(error)}}
        response = {"id": "resp_fixture", "object": "response", "status": "completed",
                    "model": "upstream-model", "output": [{"id": "msg_fixture",
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "OK"}]}],
                    "usage": {"input_tokens": 20, "output_tokens": 1, "total_tokens": 21}}
        status, content_type = 200, "application/json"
        success = channel == "fallback" or label == "already-success" or repaired
        if label == "repeat-error":
            success = False
        if channel == "first" and repaired and label == "repair-then-503":
            status, raw = 503, json.dumps({"error": {"message": "fixture unavailable"}}).encode()
        elif label == "postcommit" and channel == "first":
            events = [{"type": "response.output_text.delta", "delta": "started"},
                      {"type": "response.failed", "response": {"status": "failed",
                                                                  "error": ERROR["error"]}}]
            content_type = "text/event-stream"
            raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        elif not success:
            status, raw = 400, json.dumps(error).encode()
        elif body.get("stream"):
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
    with tempfile.TemporaryDirectory(prefix="uni-heartbeat-retry-") as directory:
        root, port = Path(directory), free_port()
        config = {
            "providers": [{
                "provider": name, "engine": "gpt",
                "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses",
                "api": ["fixture-key-1", "fixture-key-2"],
                "model": [{"upstream-model": "test-model"}],
                "preferences": {"cooldown_period": 0, "oaix_routing_attempt_id": True,
                                "model_timeout": {"default": 0.05}},
            } for name in ["first", "fallback"]],
            "api_keys": [
                {"api": "retry", "model": ["first/*", "fallback/*"]},
                {"api": "no-retry", "model": ["first/*", "fallback/*"],
                 "preferences": {"AUTO_RETRY": False}},
            ],
            "preferences": {"hedging": {"enabled": hedging, "max_inflight_attempts": 2}},
        }
        config_path = root / "api.json"
        original = json.dumps(config).encode()
        config_path.write_bytes(original)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("FACTS_S3_", "OTEL_", "FUGUE_OTEL_", "UNI_API_CONTROL_RESTORE_"))
               and key != "CONFIG_URL"}
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
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.request("GET" if body is None else "POST",
                                   "/healthz" if body is None else endpoint,
                                   None if body is None else json.dumps(body), headers)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

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
                    raise AssertionError("fixture did not become healthy")
                checked = 0
                for stream in [False, True]:
                    cases = ["repair", "wrapped", "repeat-error", "wrong-code", "quoted-message",
                             "real-call-id", "no-heartbeat", "no-retry", "targeted", "compact",
                             "already-success", "repair-then-503"]
                    if stream:
                        cases += ["postcommit"]
                        if hedging:
                            cases += ["hedge-race"]
                    for label in cases:
                        server.case, server.hits = label, []
                        body = payload(stream)
                        if label == "real-call-id":
                            for item in body["input"][4:]:
                                item["call_id"] = "real-call"
                        if label == "no-heartbeat":
                            body["input"] = body["input"][:4]
                        endpoint = "/v1/responses/compact" if label == "compact" else "/v1/responses"
                        if label == "compact":
                            body["stream"] = False
                        status, raw = request(body, "no-retry" if label == "no-retry" else "retry",
                                              target=label == "targeted", endpoint=endpoint)
                        success = label in ["repair", "wrapped", "already-success",
                                            "repair-then-503", "hedge-race", "postcommit"]
                        expected_hits = (["first", "fallback", "first"] if label == "hedge-race" else
                                         ["first", "first", "fallback"] if label == "repair-then-503" else
                                         ["first", "first"] if label in ["repair", "wrapped", "repeat-error"] else
                                         ["first"])
                        context = (hedging, stream, label, status, raw[:500], server.hits)
                        assert status == (200 if success else 400), context
                        assert [hit["channel"] for hit in server.hits] == expected_hits, context
                        first = server.hits[0]
                        assert first["body"]["input"] == body["input"], context
                        assert first["body"]["model"] == "upstream-model", context
                        if expected_hits.count("first") == 2:
                            second = [hit for hit in server.hits if hit["channel"] == "first"][1]
                            assert first["key"] == second["key"], context
                            assert first["attempt_id"] != second["attempt_id"], context
                            assert second["body"]["input"][:4] == body["input"][:4], context
                            expected = copy.deepcopy(first["body"])
                            expected["input"][4:] = [{"type": "message", "role": "user", "content": [{
                                "type": "input_text", "text": "Historical scheduled heartbeat context:\n" +
                                item["output"]}]} for item in body["input"][4:]]
                            assert second["body"] == expected, context
                        if success and label != "postcommit":
                            assert b"OK" in raw and b"function_call_output_not_found" not in raw, context
                        if label == "hedge-race":
                            time.sleep(0.6)
                        checked += 1
                assert config_path.read_bytes() == original
                log.flush()
                events = []
                for line in (root / "log").read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass
                repairs = [e for e in events if e.get("event") == "responses_heartbeat_repair"]
                assert len(repairs) == 8 + int(hedging), repairs
                assert all(e["heartbeat_items_converted"] == 16 for e in repairs), repairs
                for repair in repairs:
                    attempts = [e for e in events if e.get("event") == "upstream_attempt"
                                and e.get("request_id") == repair["request_id"]]
                    assert attempts[0]["attempt_status_code"] == 400, attempts
                    assert len({e["attempt_id"] for e in attempts}) == len(attempts), attempts
                print(f"PASS heartbeat repair hedge={hedging}: {checked} HTTP cases, {len(repairs)} repairs")
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
    for hedging in [False, True]:
        verify(Path(sys.argv[1]).resolve(), hedging)
