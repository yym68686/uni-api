"""Exercise Codex remote compaction across generation wires and process restarts."""
import base64
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
from contextlib import contextmanager

from verify_dispatch_timing import free_port


SUMMARY = "Résumé 中文: keep the regression marker and pending task."
GEMINI = ["gemini-4-neon", "gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon"]
TRIGGER = {"type": "compaction_trigger"}
NATIVE = {"type": "compaction", "encrypted_content": "upstream-opaque-state"}


def message(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def assistant(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def write_result(self, value, events=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if events is not None else "application/json")
        self.end_headers()
        if events is None:
            self.wfile.write(json.dumps(value).encode())
        else:
            for item in events:
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode())

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append((self.path, payload))
        source = self.path.split("/")[1]
        stream = payload.get("stream", False) or ":streamGenerateContent" in self.path
        if source in ("gemini", "broken"):
            usage = {"promptTokenCount": 21, "candidatesTokenCount": 8, "cachedContentTokenCount": 4}
            parts = [{"thought": True, "text": "Never send internal thoughts"}, {"text": SUMMARY}]
            value = {"candidates": [{"content": {"role": "model", "parts": parts}}], "usageMetadata": usage}
            if source == "gemini":
                value["candidates"][0]["finishReason"] = "STOP"
            events = [value] if stream else None
            return self.write_result(value, events)
        if source in ("claude", "claude-truncated"):
            value = {"type": "message", "stop_reason": "end_turn", "content": [{"type": "text", "text": SUMMARY}],
                     "usage": {"input_tokens": 21, "output_tokens": 8, "cache_read_input_tokens": 4}}
            events = None
            if stream:
                events = [{"type": "message_start", "message": {"usage": {"input_tokens": 21}}},
                          {"type": "content_block_delta", "delta": {"type": "text_delta", "text": SUMMARY}},
                          {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}}]
                if source == "claude":
                    events.append({"type": "message_stop"})
            return self.write_result(value, events)
        if source in ("chat", "blank", "short", "false-tool", "no-usage"):
            text = "  " if source == "blank" else SUMMARY
            finish = "length" if source == "short" else "stop"
            msg = {"role": "assistant", "content": text}
            if source == "false-tool":
                msg["tool_calls"] = [{"type": "function"}]
            usage = {"prompt_tokens": 21, "completion_tokens": 8}
            value = {"choices": [{"finish_reason": finish, "message": msg}]}
            if source != "no-usage":
                value["usage"] = usage
            events = [{"choices": [{"delta": msg}]}, {"choices": [{"finish_reason": finish}]}] if stream else None
            if events is not None and source != "no-usage":
                events.append({"choices": [], "usage": usage})
            return self.write_result(value, events)
        is_native = source in ("gpt", "native-override")
        is_compact = self.path.endswith("/compact") or TRIGGER in payload.get("input", [])
        item = NATIVE if is_native and is_compact else assistant(SUMMARY)
        value = {"id": "resp_fixture", "status": "completed", "output": [item],
                 "usage": {"input_tokens": 21, "output_tokens": 8}}
        events = [{"type": "response.output_item.done", "item": item, "output_index": 0},
                  {"type": "response.completed", "response": value}] if stream else None
        return self.write_result(value, events)


def sse_output(raw):
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]


def verify(binary, native):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="stateless-compaction-") as directory:
            root, port = Path(directory), free_port()
            host = f"http://127.0.0.1:{server.server_port}"

            def provider(name, suffix, model, engine, prefs=None):
                if isinstance(model, str):
                    model = [model]
                return {"provider": name, "base_url": f"{host}/{name}{suffix}", "engine": engine,
                        "api": "fixture", "model": model, "preferences": prefs or {}}

            providers = [
                provider("broken", "/v1beta", GEMINI[0], "gemini"),
                provider("gemini", "/v1beta", GEMINI, "gemini", {"post_body_parameter_overrides": {
                    "tools": [{"googleSearch": {}}], "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                }}),
                provider("claude", "/v1/messages", "claude-test", "claude"),
                provider("claude-truncated", "/v1/messages", "claude-truncated", "claude"),
                provider("chat", "/v1/chat/completions", "arbitrary-chat", "gpt"),
                provider("response", "/v1/responses", [{"vendor/arbitrary-text": "public-text"}], "codex",
                         {"upstream_stream_mode": "stream_only"}),
                provider("gpt", "/v1/responses", [{"gpt-6-example": "native-alias"}], "gpt"),
                provider("native-override", "/v1/responses", "vendor-native", "gpt",
                         {"compaction_mode": {"models": {"vendor-native": "native"}}}),
                *[provider(name, "/v1/chat/completions", name, "gpt")
                  for name in ("blank", "short", "false-tool", "no-usage")],
            ]
            config = root / "api.json"
            config.write_text(json.dumps({"providers": providers, "api_keys": [{"api": "fixture-admin", "role": "admin", "model": ["all"]}],
                                           "preferences": {"SCHEDULING_ALGORITHM": "fixed_priority", "AUTO_RETRY": True}}))

            def call(path, payload=None, **headers):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                try:
                    conn.request("POST" if payload is not None else "GET", path,
                                 json.dumps(payload) if payload is not None else None,
                                 {"Authorization": "Bearer fixture-admin", "Content-Type": "application/json", **headers})
                    response = conn.getresponse()
                    return response.status, response.read().decode()
                finally:
                    conn.close()

            @contextmanager
            def process(generation):
                # Fresh files as well as a fresh process: no checkpoint storage survives.
                runtime = root / generation
                runtime.mkdir()
                env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG")}
                env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config),
                           RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(runtime / "snapshot.json"),
                           UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(runtime / "ledger"),
                           RUST_REQUEST_SPOOL_DIRECTORY=str(runtime / "spool"),
                           RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                           NO_PROXY="127.0.0.1,localhost", UNI_API_RUST_RESPONSES_DATA_PLANE=str(native).lower())
                with (runtime / "runtime.log").open("w") as log:
                    proc = subprocess.Popen([str(binary)], cwd=runtime, env=env, stdout=log, stderr=log)
                    try:
                        for _ in range(100):
                            try:
                                if call("/healthz")[0] == 200:
                                    break
                            except OSError:
                                pass
                            time.sleep(.05)
                        else:
                            raise AssertionError("gateway did not become ready")
                        yield
                    finally:
                        proc.terminate()
                        proc.wait(timeout=10)

            def compact(model, items=None, stream=True, path="/v1/responses"):
                history = items or [message("regression marker"), {"type": "custom_tool_call", "name": "patch", "input": "pending diff"}]
                body = {"model": model, "stream": stream, "input": history + ([] if path.endswith("/compact") else [TRIGGER]),
                        "tools": [{"type": "web_search"}], "text": {"format": {"type": "json_schema"}}}
                status, raw = call(path, body)
                assert status == 200, (model, status, raw)
                if stream and not path.endswith("/compact"):
                    events = sse_output(raw)
                    final = [ev["response"] for ev in events if ev["type"] == "response.completed"]
                    done = [ev["item"] for ev in events if ev["type"] == "response.output_item.done"]
                    assert len(final) == 1 and done == final[0]["output"] and len(done) == 1, raw
                    assert "response.output_text.delta" not in raw, raw
                    result = final[0]
                else:
                    result = json.loads(raw)
                checkpoint, = result["output"]
                assert checkpoint["type"] == "compaction", result
                value = checkpoint["encrypted_content"]
                if model in ("native-alias", "vendor-native"):
                    assert value == NATIVE["encrypted_content"], result
                else:
                    assert SUMMARY not in raw and "Never send" not in raw, raw
                    prefix, encoded = value.split(":", 1)
                    assert prefix == "compact-v1"
                    assert base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode() == SUMMARY
                    if model == "no-usage":
                        assert "usage" not in result, result
                    else:
                        assert result["usage"]["output_tokens"] == 8, result
                return checkpoint

            with process("first"):
                checkpoints = {model: compact(model) for model in [*GEMINI, "arbitrary-chat", "claude-test", "public-text", "no-usage"]}
                assert any(path.startswith("/broken/") for path, _ in server.hits)
                for path, body in server.hits:
                    text = json.dumps(body, ensure_ascii=False)
                    assert "compaction_trigger" not in text and "pending diff" in text, (path, body)
                    assert "tools" not in body and "toolConfig" not in body and "text" not in body, body
                for model in ["native-alias", "vendor-native"]:
                    compact(model)
                    assert TRIGGER in server.hits[-1][1]["input"]
                compact(GEMINI[1], stream=False, path="/v1/responses/compact")
                assert ":generateContent" in server.hits[-1][0]
                compact("public-text", stream=False, path="/v1/responses/compact")
                assert server.hits[-1][0].endswith("/v1/responses") and server.hits[-1][1]["stream"]
                for model in ("blank", "short", "false-tool", "claude-truncated"):
                    status, raw = call("/v1/responses", {"model": model, "stream": True, "input": [message("test"), TRIGGER]})
                    assert "response.completed" not in raw and "compact-v1:" not in raw, (model, status, raw)
                    assert status == 502 or '"type":"error"' in raw, (status, raw)

            with process("second"):
                for model, saved in checkpoints.items():
                    history = [saved, message("Follow-up after process restart")]
                    status, raw = call("/v1/responses", {"model": model, "input": history})
                    assert status == 200, (model, status, raw)
                    wire = json.dumps(server.hits[-1][1], ensure_ascii=False)
                    assert SUMMARY in wire and "Follow-up after process restart" in wire and "compact-v1:" not in wire, wire
                    compact(model, history)
                    assert SUMMARY in json.dumps(server.hits[-1][1], ensure_ascii=False)
                for model in ("native-alias", "arbitrary-chat"):
                    old_count = len(server.hits)
                    for bad in ("compact-v1:!", "compact-v1:" + "a" * 45000):
                        status, raw = call("/v1/responses", {"model": model, "input": [{**NATIVE, "encrypted_content": bad}, message("safe")]})
                        assert status == 400 and "Invalid compact-v1 checkpoint" in raw and len(server.hits) == old_count, (status, raw)
                status, _ = call("/v1/responses", {"model": "arbitrary-chat", "input": [NATIVE, message("follow-up")]})
                assert status == 400
                status, _ = call("/v1/responses", {"model": "native-alias", "input": [NATIVE, checkpoints["arbitrary-chat"], message("follow-up")]})
                assert status == 200
                history = server.hits[-1][1]["input"]
                assert history[0] == NATIVE and history[1]["role"] == "user"
                assert SUMMARY in history[1]["content"][0]["text"]
                status, _ = call("/v1/responses", {"model": "native-alias", "previous_response_id": "resp_fixture",
                                                    "input": [checkpoints["arbitrary-chat"], message("follow-up")]})
                assert status == 400
        print(f"PASS stateless compaction (native data plane={native}): protocols, strict completion, restart, replay, native isolation")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    for data_plane in (True, False):
        verify(Path(sys.argv[1]).resolve(), data_plane)
