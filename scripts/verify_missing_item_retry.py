"""Offline HTTP regressions for missing-reasoning repair and bounded fallback."""
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


ITEM_ID = "rs_c65ee145dd863116153ca83b"
ERROR = {"error": {
    "code": None, "type": "invalid_request_error", "param": "input",
    "message": "Item with id '" + ITEM_ID + "' not found. Items are not persisted when `store` is set to false. Try again with `store` set to true, or remove this item from your input.",
}}
EMPTY_NAME = {"error": {"code": "empty_string", "type": "invalid_request_error",
    "param": "input[6].name", "message": "Invalid 'input[6].name': empty string. Expected a string with minimum length 1, but got an empty string instead."}}


def payload(stream):
    return {"model": "test-model", "stream": stream, "store": False, "input": [
        {"role": "user", "content": "say OK"},
        {"type": "function_call", "name": "real", "call_id": "real-call", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "real-call", "output": "actual result"},
        {"type": "reasoning", "id": "rs_keep", "encrypted_content": "preserve"},
        {"type": "reasoning", "id": ITEM_ID, "content": None, "encrypted_content": None,
         "summary": [{"type": "summary_text", "text": "Keep the first summary.\n"},
                     {"type": "summary_text", "text": "Keep the second summary."}]},
        {"type": "reasoning", "id": "rs_other", "summary": [{"type": "summary_text", "text": "preserve other"}]},
        {"type": "function_call", "name": "", "call_id": "bad-call", "arguments": '{"ok":true}'},
        {"type": "function_call_output", "call_id": "bad-call", "output": "unsupported call: "},
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
        if label == "nested-wrapped":
            error = {"error": {"message": json.dumps({"error": {"message": json.dumps(error)}})}}
        if label == "wrong-id":
            error["error"]["message"] = error["error"]["message"].replace(ITEM_ID, "rs_absent")
        if label == "absent-code":
            error["error"].pop("code")
        raw = json.dumps(error).encode()
        response = {"id": "resp_fixture", "object": "response", "status": "completed",
                    "model": "upstream-model", "output": [{"id": "msg_fixture", "type": "message",
                    "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}]}
        status, content_type = 404, "application/json"
        if label == "postcommit" and channel == "first":
            status, content_type = 200, "text/event-stream"
            events = [{"type": "response.output_text.delta", "delta": "started"},
                      {"type": "response.failed", "response": {"status": "failed", "error": ERROR["error"]}}]
            raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        elif repaired and label in ("repair-then-400", "repair-then-404", "repair-then-503", "repair-then-empty-name"):
            status = 404 if label == "repair-then-404" else 503 if label == "repair-then-503" else 400
            raw = json.dumps(EMPTY_NAME if label == "repair-then-empty-name" else {
                "error": {"message": "fixture failure", "type": "invalid_request_error"}}).encode()
        elif channel == "fallback" and label == "later-unrelated-400":
            status, raw = 400, b'{"error":{"message":"unrelated bad input","type":"invalid_request_error"}}'
        elif (repaired and label not in ("repeat-error", "exhausted", "later-unrelated-400")) or label == "already-success" or (
            channel == "fallback" and label not in ("repeat-error", "exhausted", "later-unrelated-400")) or (
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
    with tempfile.TemporaryDirectory(prefix="uni-missing-item-") as directory:
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
                    cases = ["repair", "wrapped", "nested-wrapped", "without-null-fields",
                             "repair-then-400", "repair-then-404", "repair-then-503", "repair-then-empty-name",
                             "repeat-error", "exhausted", "later-unrelated-400", "wrong-code", "absent-code",
                             "wrong-param", "quoted-message", "wrong-id", "encrypted", "raw-content",
                             "empty-summary", "unknown-summary", "store-true", "store-missing",
                             "bare-reference", "duplicate-id", "no-retry", "targeted", "compact", "already-success"]
                    if stream:
                        cases += ["postcommit"]
                        if hedging:
                            cases += ["hedge-race"]
                    for label in cases:
                        server.case, server.hits = label, []
                        body = payload(stream)
                        if label == "without-null-fields":
                            body["input"][4].pop("content")
                            body["input"][4].pop("encrypted_content")
                        if label == "encrypted":
                            body["input"][4]["encrypted_content"] = "opaque"
                        if label == "raw-content":
                            body["input"][4]["content"] = [{"type": "reasoning_text", "text": "raw"}]
                        if label == "empty-summary":
                            body["input"][4]["summary"] = []
                        if label == "unknown-summary":
                            body["input"][4]["summary"][0]["type"] = "unknown"
                        if label == "store-true":
                            body["store"] = True
                        if label == "store-missing":
                            body.pop("store")
                        if label == "bare-reference":
                            body["input"][4] = {"type": "item_reference", "id": ITEM_ID}
                        if label == "duplicate-id":
                            body["input"].append({"type": "item_reference", "id": ITEM_ID})
                        if label == "compact":
                            body["stream"] = False
                        status, raw = request(body, "no-retry" if label == "no-retry" else "retry",
                                              label == "targeted", "/v1/responses/compact" if label == "compact" else "/v1/responses")
                        no_repair = label in ("wrong-code", "absent-code", "wrong-param", "quoted-message",
                                             "wrong-id", "encrypted", "raw-content", "empty-summary",
                                             "unknown-summary", "store-true", "store-missing", "bare-reference",
                                             "duplicate-id", "no-retry", "targeted", "compact", "already-success", "postcommit")
                        expected = (["first"] if no_repair else
                                    ["first", "fallback", "first"] if label == "hedge-race" else
                                    ["first", "first", "fallback"] if label in ("repair-then-400", "repair-then-404",
                                        "repair-then-503", "repair-then-empty-name", "later-unrelated-400") else
                                    ["first", "first", "fallback", "third"] if label == "repeat-error" else
                                    ["first", "first"])
                        success = not no_repair or label in ("already-success", "postcommit")
                        expected_status = 200 if success else 404
                        if label == "later-unrelated-400":
                            success, expected_status = False, 400
                        if label == "exhausted":
                            # Three providers + configured key budget (capped 10), plus one repair.
                            expected = ["first", "first"] + ["fallback", "third", "first"] * 4
                            success, expected_status = False, 404
                        context = (hedging, stream, label, status, raw[:500], server.hits)
                        assert status == expected_status, context
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
                            item = fixed["body"]["input"][4]
                            assert item == {"type": "message", "role": "assistant", "content": [{
                                "type": "output_text", "text": "Historical reasoning summary:\n"
                                + "\n\n".join(p["text"] for p in body["input"][4]["summary"])}]}, context
                            assert fixed["body"]["input"][5:] == body["input"][5:], context
                            assert {k: v for k, v in first["body"].items() if k != "input"} == {
                                k: v for k, v in fixed["body"].items() if k != "input"}, context
                        assert all(h["body"]["input"] == body["input"] for h in server.hits
                                   if h not in fixed_hits), context
                        if success and label != "postcommit":
                            assert b"OK" in raw and b"not found" not in raw, context
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
                repairs = [e for e in events if e.get("event") == "responses_missing_item_repair"]
                assert not any(e.get("event") in ("responses_empty_name_repair", "responses_heartbeat_repair") for e in events), events
                assert len(repairs) == expected_repairs, repairs
                assert all(e["reasoning_items_converted"] == 1 for e in repairs), repairs
                for repair in repairs:
                    attempts = [e for e in events if e.get("event") == "upstream_attempt"
                                and e.get("request_id") == repair["request_id"]]
                    assert attempts[0]["attempt_status_code"] == 404, attempts
                    assert len({e["attempt_id"] for e in attempts}) == len(attempts), attempts
                print(f"PASS missing-item repair hedge={hedging}: {count} HTTP cases, {len(repairs)} repairs")
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
