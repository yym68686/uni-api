"""Isolated HTTP regressions for exact encrypted-envelope repair and fallback."""
import base64
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

ITEM = "rs_c1bc50c20fa1492e929b32afd1ccef85"
ERROR = {"error": {"code": "invalid_encrypted_content", "type": "invalid_request_error", "param": None,
    "message": f"The encrypted content for item {ITEM} could not be verified. Reason: Encrypted content could not be decrypted or parsed."}}
EMPTY = {"error": {"code": "empty_string", "type": "invalid_request_error", "param": "input[8].name",
    "message": "Invalid 'input[8].name': empty string. Expected a string with minimum length 1, but got an empty string instead."}}
MISSING = {"error": {"code": None, "type": "invalid_request_error", "param": "input",
    "message": "Item with id 'rs_missing' not found. Items are not persisted when `store` is set to false. Try again with `store` set to true, or remove this item from your input."}}
HEARTBEAT = {"error": {"code": "function_call_output_not_found", "type": "invalid_request_error", "param": "input",
    "message": "The tool output does not match a previous tool call. Resend the tool-call context."}}


def envelope(text, signature="fixture-signature-do-not-log"):
    return "cursor-sand-v1:" + base64.b64encode(json.dumps({"signature": signature, "text": text}).encode()).decode()


def item(item_id):
    parts = ["Sensitive fixture summary one.\n", "Sensitive fixture summary two 中文."]
    return {"type": "reasoning", "id": item_id, "content": None,
            "summary": [{"type": "summary_text", "text": p} for p in parts],
            "encrypted_content": envelope("\n\n".join(parts))}


def payload(stream):
    return {"model": "test-model", "stream": stream, "store": False, "input": [
        {"role": "user", "content": "say OK"},
        {"type": "function_call", "name": "real", "call_id": "real-call", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "real-call", "output": "actual result"},
        {"type": "reasoning", "id": "rs_native", "encrypted_content": "gAAAApreserve"},
        item(ITEM), item("rs_second"), item("rs_third"),
        {"type": "reasoning", "id": "rs_missing", "summary": [{"type": "summary_text", "text": "Other missing summary"}]},
        {"type": "function_call", "name": "", "call_id": "bad-call", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "bad-call", "output": "unsupported call: "},
        {"type": "function_call_output", "name": "automation_update", "namespace": "codex_app", "output": "<heartbeat>fixture</heartbeat>"},
    ], "tools": [{"type": "function", "name": "real", "parameters": {"type": "object"}}],
        "reasoning": {"effort": "low"}, "metadata": {"caller": "retained"}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        channel = self.path.split("/")[1]
        with self.server.lock:
            attempt = sum(h["channel"] == channel for h in self.server.hits) + 1
            self.server.hits.append({"channel": channel, "body": body, "key": self.headers.get("Authorization"),
                                     "attempt_id": self.headers.get("X-OAIX-Routing-Attempt-ID")})
        label = self.server.case
        repaired = body["input"][4].get("type") == "message"
        if label == "hedge-race":
            time.sleep(0.15 if channel == "first" and attempt == 1 else 0.5 if channel == "fallback" else 0)
        error = copy.deepcopy(ERROR)
        if label == "wrong-code": error["error"]["code"] = "other"
        if label == "wrong-type": error["error"]["type"] = "other"
        if label == "wrong-param": error["error"]["param"] = "input"
        if label == "quoted-message": error["error"]["message"] = "Quoted: " + error["error"]["message"]
        if label == "wrong-id": error["error"]["message"] = error["error"]["message"].replace(ITEM, "rs_absent")
        if label in ("absent-code", "absent-param"): error["error"].pop(label.split("-")[1])
        if label in ("wrapped", "nested-wrapped", "too-deep", "typed-wrapper"):
            depth = {"wrapped": 1, "nested-wrapped": 2, "too-deep": 3, "typed-wrapper": 1}[label]
            for _ in range(depth): error = {"error": {"message": json.dumps(error)}}
            if label == "typed-wrapper": error["error"]["type"] = "gateway_error"
        status, content_type, raw = 400, "application/json", json.dumps(error).encode()
        response = {"id": "resp_fixture", "object": "response", "status": "completed", "model": "upstream-model",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}]}
        if label == "postcommit" and channel == "first":
            status, content_type = 200, "text/event-stream"
            events = [{"type": "response.output_text.delta", "delta": "started"},
                      {"type": "response.failed", "response": {"status": "failed", "error": ERROR["error"]}}]
            raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        elif label in ("empty-first", "missing-first", "heartbeat-first"):
            if channel == "first" and attempt == 1:
                error = {"empty-first": EMPTY, "missing-first": MISSING, "heartbeat-first": HEARTBEAT}[label]
                status = 404 if label == "missing-first" else 400
                raw = json.dumps(error).encode()
        elif repaired and label.startswith("repair-then-"):
            failure = label.removeprefix("repair-then-")
            status = {"400": 400, "404": 404, "503": 503, "empty": 400, "missing": 404, "heartbeat": 400}[failure]
            raw = json.dumps({"empty": EMPTY, "missing": MISSING, "heartbeat": HEARTBEAT}.get(failure,
                {"error": {"message": "fixture failure", "type": "invalid_request_error"}})).encode()
        elif channel == "fallback" and label == "later-unrelated-400":
            raw = b'{"error":{"message":"unrelated parameter","type":"invalid_request_error"}}'
        elif (repaired and label not in ("repeat-error", "exhausted", "later-unrelated-400")) or label == "already-success" or (
            channel == "fallback" and label not in ("repeat-error", "exhausted", "later-unrelated-400")) or (
            channel == "third" and label != "exhausted"):
            status = 200
            if body.get("stream"):
                content_type = "text/event-stream"
                events = [{"type": "response.output_text.delta", "delta": "OK"}, {"type": "response.completed", "response": response}]
                raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
            else:
                raw = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try: self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError): pass


def verify(binary, hedging):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits, server.lock = [], threading.Lock()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-encrypted-repair-") as directory:
        root, port = Path(directory), free_port()
        providers = [{"provider": name, "engine": "gpt", "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses",
            "api": ["fixture-key-1", "fixture-key-2"], "model": [{"upstream-model": "test-model"}],
            "preferences": {"cooldown_period": 0, "oaix_routing_attempt_id": True, "model_timeout": {"default": 0.05},
                "post_body_parameter_overrides": {"metadata": {"channel": name}}}} for name in ("first", "fallback", "third")]
        config = {"providers": providers, "api_keys": [
            {"api": "retry", "model": ["first/*", "fallback/*", "third/*"]},
            {"api": "no-retry", "model": ["first/*", "fallback/*", "third/*"], "preferences": {"AUTO_RETRY": False}}],
            "preferences": {"hedging": {"enabled": hedging, "max_inflight_attempts": 2}}}
        config_path = root / "api.json"
        original_config = json.dumps(config).encode(); config_path.write_bytes(original_config)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("FACTS_S3_", "OTEL_", "FUGUE_OTEL_", "UNI_API_CONTROL_RESTORE_")) and k != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_path),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
            RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0", NO_PROXY="127.0.0.1,localhost")

        def request(body=None, key="retry", target=False, endpoint="/v1/responses"):
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            if target: headers["X-Uni-API-Provider"] = "first"
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.request("GET" if body is None else "POST", "/healthz" if body is None else endpoint,
                             None if body is None else json.dumps(body), headers)
                response = conn.getresponse(); return response.status, response.read()
            finally: conn.close()

        with (root / "log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if request()[0] == 200: break
                    except OSError: pass
                    time.sleep(0.05)
                else: raise AssertionError("fixture startup failed")
                count, expected_repairs, expected_other = 0, 0, 0
                for stream in (False, True):
                    negative = ["wrong-code", "wrong-type", "wrong-param", "quoted-message", "wrong-id", "absent-code", "absent-param",
                        "too-deep", "typed-wrapper", "native-encrypted", "unknown-prefix", "bad-base64", "summary-mismatch",
                        "unknown-envelope-field", "empty-signature", "raw-content", "empty-summary", "unknown-summary",
                        "store-true", "store-missing", "bare-reference", "duplicate-target", "no-retry", "targeted", "compact"]
                    cases = ["repair", "wrapped", "nested-wrapped", "without-content", "unknown-sibling", "duplicate-sibling",
                        "repair-then-400", "repair-then-404", "repair-then-503", "repair-then-empty", "repair-then-missing", "repair-then-heartbeat",
                        "repeat-error", "exhausted", "later-unrelated-400", "empty-first", "missing-first", "heartbeat-first",
                        "already-success", *negative]
                    if stream: cases += ["postcommit"] + (["hedge-race"] if hedging else [])
                    for label in cases:
                        server.case, server.hits = label, []
                        body = payload(stream)
                        target_item = body["input"][4]
                        if label == "without-content": target_item.pop("content")
                        if label == "native-encrypted": target_item["encrypted_content"] = "gAAAAopaque"
                        if label == "unknown-prefix": target_item["encrypted_content"] = "other:opaque"
                        if label == "bad-base64": target_item["encrypted_content"] = "cursor-sand-v1:bad-base64"
                        if label == "summary-mismatch": target_item["encrypted_content"] = envelope("different")
                        if label == "empty-signature": target_item["encrypted_content"] = envelope("\n\n".join(p["text"] for p in target_item["summary"]), "")
                        if label == "unknown-envelope-field": target_item["encrypted_content"] = "cursor-sand-v1:" + base64.b64encode(b'{"signature":"sig","text":"x","extra":"keep"}').decode()
                        if label == "raw-content": target_item["content"] = [{"type": "reasoning_text", "text": "keep"}]
                        if label == "empty-summary": target_item["summary"] = []
                        if label == "unknown-summary": target_item["summary"][0]["type"] = "unknown"
                        if label == "unknown-sibling": body["input"][5]["encrypted_content"] = "unknown:keep"
                        if label == "store-true": body["store"] = True
                        if label == "store-missing": body.pop("store")
                        if label == "bare-reference": body["input"][4] = {"type": "item_reference", "id": ITEM}
                        if label in ("duplicate-target", "duplicate-sibling"):
                            body["input"].append({"type": "item_reference", "id": ITEM if label == "duplicate-target" else "rs_second"})
                        if label == "compact": body["stream"] = False
                        status, raw = request(body, "no-retry" if label == "no-retry" else "retry", label == "targeted",
                            "/v1/responses/compact" if label == "compact" else "/v1/responses")
                        other = label in ("empty-first", "missing-first", "heartbeat-first")
                        no_repair = label in negative or label in ("already-success", "postcommit") or other
                        expected = ["first"] if no_repair else ["first", "first"]
                        expected_status = 400 if label in negative else 200
                        if label.startswith("repair-then-") or label == "later-unrelated-400": expected += ["fallback"]
                        if label == "later-unrelated-400": expected_status = 400
                        if label == "repeat-error": expected += ["fallback", "third"]
                        if label == "exhausted": expected, expected_status = ["first", "first"] + ["fallback", "third", "first"] * 4, 400
                        if label == "hedge-race": expected = ["first", "fallback", "first"]
                        if other: expected, expected_status = ["first", "first"], 400
                        if label in ("empty-first", "missing-first"): expected += ["fallback"]
                        context = (hedging, stream, label, status, raw[:400],
                                   [{"channel": h["channel"], "attempt_id": h["attempt_id"]} for h in server.hits])
                        assert status == expected_status, context
                        assert [h["channel"] for h in server.hits] == expected, context
                        assert server.hits[0]["body"]["input"] == body["input"], context
                        fixed_hits = [h for h in server.hits if h["body"]["input"][4].get("type") == "message"]
                        assert len(fixed_hits) == (0 if no_repair else 1), context
                        if fixed_hits:
                            expected_repairs += 1
                            first, fixed = server.hits[0], fixed_hits[0]
                            assert first["key"] == fixed["key"] and first["attempt_id"] != fixed["attempt_id"], context
                            assert fixed["channel"] == "first", context
                            expected_input = copy.deepcopy(body["input"])
                            indices = [4, 6] if label in ("unknown-sibling", "duplicate-sibling") else [4, 5, 6]
                            for index in indices:
                                expected_input[index] = {"type": "message", "role": "assistant", "content": [{"type": "output_text",
                                    "text": "Historical reasoning summary:\n" + "\n\n".join(p["text"] for p in body["input"][index]["summary"])}]}
                            assert fixed["body"]["input"] == expected_input, context
                            assert {k: v for k, v in first["body"].items() if k != "input"} == {k: v for k, v in fixed["body"].items() if k != "input"}, context
                        if not other:
                            assert all(h["body"]["input"] == body["input"] for h in server.hits if h not in fixed_hits), context
                        else: expected_other += 1
                        for hit in server.hits:
                            assert hit["body"]["metadata"]["channel"] == hit["channel"], context
                        if expected_status == 200 and label != "postcommit": assert b"OK" in raw and b"invalid_encrypted_content" not in raw, context
                        if label == "hedge-race": time.sleep(0.6)
                        count += 1
                assert config_path.read_bytes() == original_config
                log.flush(); log_text = (root / "log").read_text()
                events = []
                for line in log_text.splitlines():
                    try: events.append(json.loads(line))
                    except ValueError: pass
                repairs = [e for e in events if e.get("event") == "responses_encrypted_content_repair"]
                assert len(repairs) == expected_repairs, repairs
                assert sum(e.get("event") in ("responses_empty_name_repair", "responses_missing_item_repair", "responses_heartbeat_repair") for e in events) == expected_other
                for repair in repairs:
                    assert repair["reasoning_items_converted"] in (2, 3), repair
                    attempts = [e for e in events if e.get("event") == "upstream_attempt" and e.get("request_id") == repair["request_id"]]
                    original = next(e for e in attempts if e["attempt_id"] == repair["original_attempt_id"])
                    assert original["attempt_status_code"] == 400, attempts
                    assert len({e["attempt_id"] for e in attempts}) == len(attempts), attempts
                for private in ("Sensitive fixture summary", "fixture-signature-do-not-log", "cursor-sand-v1:"):
                    assert private not in log_text, private
                print(f"PASS encrypted-content repair hedge={hedging}: {count} HTTP cases, {len(repairs)} repairs")
            except Exception:
                print((root / "log").read_text()[-8000:]); raise
            finally:
                process.terminate()
                try: process.wait(timeout=4)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                server.shutdown(); server.server_close()


if __name__ == "__main__":
    for hedge in (False, True): verify(Path(sys.argv[1]).resolve(), hedge)
