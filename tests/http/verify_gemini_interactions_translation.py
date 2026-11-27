"""Stateless Interactions through existing Chat, Responses and Gemini channels."""
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
from urllib.parse import urlsplit

from verify_dispatch_timing import free_port


USAGE = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
         "prompt_tokens_details": {"cached_tokens": 2},
         "completion_tokens_details": {"reasoning_tokens": 3}}
RESPONSES_USAGE = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20,
                   "input_tokens_details": {"cached_tokens": 2},
                   "output_tokens_details": {"reasoning_tokens": 3}}
GEMINI_USAGE = {"promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 20,
                "cachedContentTokenCount": 2, "thoughtsTokenCount": 3}
INTERACTIONS_USAGE = {"total_input_tokens": 12, "total_output_tokens": 5, "total_tokens": 20,
                      "total_cached_tokens": 2, "total_thought_tokens": 3}
OMITTED = "x-uni-api-interactions-omitted"
SUMMARY = "verified public summary"


def encode_event(value):
    return b"data: " + json.dumps(value).encode() + b"\r\n\r\n"


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        route = urlsplit(self.path).path
        texts = json.dumps(payload)
        summary_fixture = "summary-trigger" in texts
        self.server.hits.append((route, payload, {k.lower(): v for k, v in self.headers.items()}))
        result, status, media = None, 200, "application/json"
        if "quota-trigger" in texts:
            result, status = {"error": {"message": "fixture quota"}}, 429
        elif "redirect-trigger" in texts:
            result, status = {}, 307
        elif route.endswith("/messages"):
            result = {"id": "msg_fixture", "type": "message", "role": "assistant",
                      "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
                      "usage": {"input_tokens": 10, "cache_read_input_tokens": 2, "output_tokens": 5}}
            if payload.get("stream"):
                events = [
                    {"type": "message_start", "message": {**result, "content": [], "usage":
                        {"input_tokens": 10, "cache_read_input_tokens": 2}}},
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
                ]
                if "messages-eof" not in texts:
                    events.append({"type": "message_stop"})
                media, result = "text/event-stream", b"".join(map(encode_event, events))
        elif route.endswith("/chat/completions"):
            reason = "length" if "length-trigger" in texts else "stop"
            if payload.get("stream"):
                first = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
                if summary_fixture:
                    first["choices"][0]["delta"]["reasoning_content"] = "private chat reasoning"
                if "eof-trigger" in texts:
                    result = encode_event(first)
                else:
                    finish = {"choices": [{"delta": {}, "finish_reason": reason}]}
                    result = b"".join(map(encode_event, [first, finish, {"choices": [], "usage": USAGE}]))
                    result += b"data: [DONE]\r\n\r\n"
                media = "text/event-stream"
            else:
                result = {"choices": [{"message": {"content": "hello"}, "finish_reason": reason}],
                          "usage": None if "usage-unknown" in texts else USAGE}
                if summary_fixture:
                    result["choices"][0]["message"]["reasoning_content"] = "private chat reasoning"
        elif route.endswith("/responses"):
            result = {"id": "resp_fixture", "status": "completed", "usage": RESPONSES_USAGE,
                      "output": [{"type": "message", "role": "assistant",
                                  "content": [{"type": "output_text", "text": "hello"}]}]}
            if summary_fixture:
                result["output"].insert(0, {"type": "reasoning", "summary": [
                    {"type": "summary_text", "text": SUMMARY}],
                    "content": [{"type": "reasoning_text", "text": "private responses reasoning"}]})
            if "length-trigger" in texts:
                result["status"] = "incomplete"
                result["incomplete_details"] = {"reason": "max_output_tokens"}
            if payload.get("stream"):
                media = "text/event-stream"
                terminal = {**result, "output": result["output"][:1] if summary_fixture else []}
                result = b""
                if summary_fixture:
                    result += encode_event({"type": "response.reasoning_text.delta", "delta": "private responses reasoning"})
                    if "snapshot-only" not in texts:
                        result += encode_event({"type": "response.reasoning_summary_text.delta", "delta": SUMMARY})
                result += encode_event({"type": "response.output_text.delta", "delta": "hello"})
                result += encode_event({"type": "response." + terminal["status"], "response": terminal})
        elif ":generateContent" in route or ":streamGenerateContent" in route:
            result = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]},
                                      "finishReason": "STOP"}], "usageMetadata": GEMINI_USAGE}
            if summary_fixture:
                result["candidates"][0]["content"]["parts"].insert(0, {
                    "text": SUMMARY, "thought": True, "thoughtSignature": "unmapped-signature"})
            if ":streamGenerateContent" in route:
                media, result = "text/event-stream", encode_event(result)
        else:
            status, result = 404, {}
        if not isinstance(result, bytes):
            result = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", media)
        if status == 307:
            self.send_header("Location", "/must-not-follow")
        self.end_headers()
        for offset in range(0, len(result), 17):
            self.wfile.write(result[offset:offset + 17])
            self.wfile.flush()


def check_stream(raw, status="completed", has_summary=False):
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith(b"data: ")]
    assert [event["event_type"] for event in events] == [
        "interaction.created", *(["step.start", "step.delta", "step.stop"] if has_summary else []),
        "step.start", "step.delta", "step.stop", "interaction.completed"
    ], events
    if has_summary:
        assert events[1]["step"]["type"] == "thought", events
        assert events[2]["delta"] == {"type": "thought_summary", "content": {"type": "text", "text": SUMMARY}}, events
        assert all(e["index"] == 0 for e in events[1:4]), events
    assert all(e["index"] == int(has_summary) for e in events[-4:-1]), events
    assert events[-3]["delta"] == {"type": "text", "text": "hello"}, events
    created, terminal = events[0]["interaction"], events[-1]["interaction"]
    assert created["id"] == terminal["id"] and created["status"] == "in_progress", events
    assert terminal["status"] == status and terminal["usage"] == INTERACTIONS_USAGE, terminal


def verify(binary):
    with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream, tempfile.TemporaryDirectory() as temp:
        upstream.hits = []
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        root, port = Path(temp), free_port()
        base = f"http://127.0.0.1:{upstream.server_port}"
        models = [("gemini-4-hydrogen-chat", "gpt", "/chat/v1/chat/completions"),
                  ("gemini-4-hydrogen-responses", "codex", "/responses/v1/responses"),
                  ("gemini-4-hydrogen", "gemini", "/gemini/v1beta")]
        providers = [{"provider": engine, "engine": engine, "base_url": base + path,
                      "api": "fixture-provider",
                      "preferences": {"interactions_thinking_summaries": True} if engine == "codex" else {},
                      "model": [{model if engine == "gemini" else "wire-hydrogen": model}]}
                     for model, engine, path in models]
        providers.append({"provider": "forced", "engine": "gpt", "base_url": base + "/forced/v1/chat/completions",
                          "api": "fixture-provider", "model": [{"wire-hydrogen": "forced"}],
                          "preferences": {"post_body_parameter_overrides":
                              {"stream": True, "store": True, "n": 2, "max_tokens": 500,
                               "tools": [{"type": "function"}]}}})
        providers.append({"provider": "forced-responses", "engine": "codex", "base_url": base + "/forced/v1/responses",
                          "api": "fixture-provider", "model": [{"wire-hydrogen": "forced-responses"}],
                          "preferences": {"upstream_stream_mode": "stream_only", "interactions_thinking_summaries": True}})
        providers.append({"provider": "messages", "engine": "claude", "base_url": base + "/claude/v1/messages",
                          "api": "fixture-provider", "model": [{"wire-claude": "messages"}]})
        config = {"api_keys": [{"api": "fixture-client", "model": ["all"],
                               "preferences": {"AUTO_RETRY": 3}}],
                  "providers": providers}
        (root / "config.json").write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ("FACTS_", "UNI_API_CONTROL_RESTORE_", "OTEL_", "FUGUE_OTEL_")) and k != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "config.json"),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0", NO_PROXY="127.0.0.1,localhost")

        def call(path, payload=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            headers = {"content-type": "application/json", "authorization": "Bearer fixture-client",
                       "cookie": "caller=private", "idempotency-key": "caller"}
            try:
                conn.request("GET" if payload is None else "POST", path,
                             None if payload is None else json.dumps(payload), headers)
                response = conn.getresponse()
                return response.status, dict(response.getheaders()), response.read()
            finally:
                conn.close()

        with (root / "gateway.log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(160):
                    try:
                        if call("/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.05)
                else:
                    raise AssertionError("Gateway was not ready")
                endpoint = "/v1/interactions"
                code, _, raw = call("/v1/model-channels?endpoint=%2Fv1%2Finteractions")
                assert code == 200 and {r["interactions_mode"] for r in json.loads(raw)["data"]} == {"translated"}
                for model, engine, route in models:
                    for stream in (False, True):
                        request = {"model": model, "input": "hi", "store": False, "stream": stream,
                                   "system_instruction": "Be concise",
                                   "generation_config": {"max_output_tokens": 17}}
                        code, headers, raw = call(endpoint, request)
                        assert code == 200, (model, stream, code, raw)
                        wire_route, body, wire_headers = upstream.hits[-1]
                        assert not {"cookie", "idempotency-key"} & wire_headers.keys()
                        if engine == "gemini":
                            method = "streamGenerateContent" if stream else "generateContent"
                            assert wire_route == route + f"/models/{model}:{method}", wire_route
                            assert body["generationConfig"]["maxOutputTokens"] == 17, body
                            assert "Be concise" in json.dumps(body["systemInstruction"]), body
                        else:
                            assert wire_route == route and body["model"] == "wire-hydrogen", (wire_route, body)
                            assert wire_headers["authorization"] == "Bearer fixture-provider"
                            if engine == "codex":
                                assert body["store"] is False and body["max_output_tokens"] == 17, body
                            else:
                                assert body["max_tokens"] == 17, body
                                if stream:
                                    assert body["stream_options"]["include_usage"] is True, body
                        if stream:
                            assert headers["content-type"].startswith("text/event-stream"), headers
                            check_stream(raw)
                        else:
                            response = json.loads(raw)
                            assert response["model"] == model and response["status"] == "completed", response
                            assert response["steps"][0]["content"] == [{"type": "text", "text": "hello"}], response
                            assert response["usage"] == INTERACTIONS_USAGE, response
                request = {"model": models[0][0], "store": False}
                for text, stream, expected in [
                    ("length-trigger", True, 200), ("eof-trigger", True, 200),
                    ("usage-unknown", False, 200), ("quota-trigger", False, 429),
                    ("redirect-trigger", False, 307),
                ]:
                    before = len(upstream.hits)
                    code, headers, raw = call(endpoint, {**request, "input": text, "stream": stream})
                    assert code == expected and len(upstream.hits) == before + 1, (code, raw)
                    assert "location" not in headers, headers
                    if text == "length-trigger":
                        check_stream(raw, "incomplete")
                    elif text == "eof-trigger":
                        assert b"step.delta" in raw and b"event: error" in raw and b"interaction.completed" not in raw, raw
                    elif text == "usage-unknown":
                        assert "usage" not in json.loads(raw), raw
                before = len(upstream.hits)
                for stream in (False, True):
                    code, _, raw = call(endpoint, {**request, "model": models[1][0],
                                                    "input": "length-trigger", "stream": stream})
                    assert code == 200, (code, raw)
                    if stream:
                        check_stream(raw, "incomplete")
                    else:
                        result = json.loads(raw)
                        assert result["status"] == "incomplete" and result["usage"] == INTERACTIONS_USAGE, raw
                assert len(upstream.hits) == before + 2
                for model, engine, _ in models:
                    for stream in (False, True):
                        for setting in ("auto", "none", None, {"future": True}):
                            before = len(upstream.hits)
                            payload = {"model": model, "stream": stream, "store": False,
                                       "future_top_level": {"not_wire": True},
                                       "input": [{"type": "user_input", "future_step": True,
                                                  "content": [{"type": "text", "text": "summary-trigger",
                                                               "future_part": True}]}],
                                       "service_tier": "priority",
                                       "generation_config": {"future_setting": True, "temperature": "invalid",
                                                             "thinking_summaries": setting,
                                                             "thinking_level": "xhigh"}}
                            code, headers, raw = call(endpoint, payload)
                            assert code == 200 and len(upstream.hits) == before + 1, (code, raw, upstream.hits)
                            body = upstream.hits[-1][1]
                            assert not any(word in json.dumps(body) for word in ("future_", "invalid", "priority")), body
                            assert b"private" not in raw and b"unmapped-signature" not in raw, raw
                            emits_summary = (setting == "auto" and engine in ("gemini", "codex"))
                            if engine == "gemini":
                                thinking = body["generationConfig"]["thinkingConfig"]
                                assert thinking["thinkingLevel"] == "XHIGH", body
                                assert thinking.get("includeThoughts") == (True if emits_summary else None), body
                            elif engine == "codex":
                                assert body.get("reasoning", {}).get("summary") == ("auto" if emits_summary else None), body
                                assert body["store"] is False
                            else:
                                assert "reasoning" not in body and "reasoning_effort" not in body, body
                            omitted = set(headers[OMITTED].split(","))
                            assert {"extensions", "input_metadata", "generation_config", "service_tier"} <= omitted, headers
                            assert ("thinking_summaries" in omitted) == (isinstance(setting, dict) or (engine == "gpt" and setting == "auto"))
                            if stream:
                                check_stream(raw, has_summary=emits_summary)
                            else:
                                result = json.loads(raw)
                                assert result["steps"][-1]["content"] == [{"type": "text", "text": "hello"}], result
                                assert result["usage"] == INTERACTIONS_USAGE, result
                                assert len(result["steps"]) == 1 + emits_summary, result
                                if emits_summary:
                                    assert result["steps"][0] == {"type": "thought",
                                        "summary": [{"type": "text", "text": SUMMARY}]}, result
                code, _, raw = call(endpoint, {"model": models[1][0], "stream": True, "input": "summary-trigger snapshot-only",
                                               "generation_config": {"thinking_summaries": "auto"}})
                assert code == 200 and raw.count(SUMMARY.encode()) == 1 and b"private" not in raw, raw
                before = len(upstream.hits)
                code, headers, raw = call(endpoint, {"model": "forced", "input": "hi", "stream": False,
                                                   "generation_config": {"max_output_tokens": 19}})
                assert code == 200 and len(upstream.hits) == before + 1, (code, raw)
                body = upstream.hits[-1][1]
                assert body["stream"] is True and not set(body) & {"tools", "n", "store"}, body
                assert body["max_tokens"] == 19, body
                assert "channel_defaults" in headers[OMITTED], headers
                assert headers["content-type"].startswith("application/json") and json.loads(raw)["usage"] == INTERACTIONS_USAGE, raw
                before = len(upstream.hits)
                code, _, raw = call(endpoint, {"model": "forced", "input": "eof-trigger"})
                assert code == 502 and len(upstream.hits) == before + 1, (code, raw)
                code, _, raw = call(endpoint, {"model": "forced-responses", "input": "summary-trigger",
                                              "generation_config": {"thinking_summaries": "auto"}})
                assert code == 200 and upstream.hits[-1][1]["stream"] is True, (code, raw)
                assert json.loads(raw)["steps"][0]["summary"][0]["text"] == SUMMARY, raw
                assert json.loads(raw)["usage"] == INTERACTIONS_USAGE, raw
                for stream in (False, True):
                    code, headers, raw = call(endpoint, {"model": "messages", "input": "hello", "stream": stream,
                        "generation_config": {"thinking_summaries": "auto", "temperature": 1.5},
                        "tools": [{"type": "function", "name": "optional"}]})
                    assert code == 200 and {"thinking_summaries", "generation_config", "tools"} <= set(headers[OMITTED].split(",")), (code, raw)
                    route, body, _ = upstream.hits[-1]
                    assert route == "/claude/v1/messages" and body["stream"] == stream and "temperature" not in body and "tools" not in body, body
                    result = json.loads(raw) if not stream else [json.loads(line[6:]) for line in raw.splitlines()
                        if line.startswith(b"data: ")][-1]["interaction"]
                    assert result["status"] == "completed" and result["usage"] == {
                        "total_input_tokens": 12, "total_output_tokens": 5, "total_cached_tokens": 2}, result
                code, _, raw = call(endpoint, {"model": "messages", "input": "messages-eof", "stream": True})
                assert code == 200 and b"event: error" in raw and b"interaction.completed" not in raw, (code, raw)
                before = len(upstream.hits)
                for extra, expected in [
                    ({"store": True}, 400),
                    ({"previous_interaction_id": "not-a-stateless-call"}, 400),
                    ({"tool_choice": "required"}, 400),
                    ({"response_format": {"type": "text", "mime_type": "application/json",
                                          "schema": {"type": "object"}}}, 422),
                ]:
                    code, _, raw = call(endpoint, {"model": "forced", "input": "hi", **extra})
                    assert code == expected and b"capability unavailable" in raw, (code, raw)
                assert len(upstream.hits) == before
            except Exception:
                log.seek(0)
                print(log.read()[-7000:])
                raise
            finally:
                process.terminate()
                process.wait(timeout=10)
                upstream.shutdown()
    print("Gemini Interactions translated HTTP matrix passed")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
