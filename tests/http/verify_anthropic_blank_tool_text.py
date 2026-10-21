"""Offline strict Bedrock validation regression; no real provider calls."""
import copy
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
from verify_dispatch_timing import free_port

ERROR = {"type": "<nil>", "message": (
    "InvokeModelWithResponseStream: operation error Bedrock Runtime: "
    "InvokeModelWithResponseStream, https response error StatusCode: 400, "
    "RequestID: fixture-request, ValidationException: messages: text content "
    "blocks must contain non-whitespace text")}
THINKING = {"type": "thinking", "thinking": "fixture reasoning", "signature": "preserve-exactly"}
TOOL = {"type": "tool_use", "id": "tool_1", "name": "read", "input": {"text": " "}}
RESULT = {"type": "tool_result", "tool_use_id": "tool_1", "content": " "}
MESSAGE = {"role": "assistant", "content": [THINKING, {"type": "text", "text": " \n\t"}, TOOL]}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        channel = self.path.split('/')[1]
        self.server.hits.append((channel, payload))
        blank = any(isinstance(m.get('content'), list) and any(
            b.get('type') == 'text' and isinstance(b.get('text'), str) and not b['text'].strip()
            for b in m['content']) for m in payload.get('messages', []))
        if self.server.fail_first and channel == 'first':
            status, content_type, raw = 503, 'application/json', b'{"error":{"message":"temporarily unavailable"}}'
        elif blank and channel != 'gpt':
            status, content_type, raw = 400, 'application/json', json.dumps(ERROR).encode()
        else:
            status = 200
            if channel == 'gpt':
                response = {"id": "chat_fixture", "object": "chat.completion", "model": "test-model",
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": "test"}, "finish_reason": "stop"}]}
                content_type, raw = 'application/json', json.dumps(response).encode()
            else:
                response = {"id": "msg_fixture", "type": "message", "role": "assistant", "model": "test-model",
                            "content": [{"type": "text", "text": "test"}], "stop_reason": "end_turn",
                            "usage": {"input_tokens": 10, "output_tokens": 1}}
                events = [
                    {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "test"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ]
                content_type = 'text/event-stream' if payload.get('stream') else 'application/json'
                raw = (''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                       if payload.get('stream') else json.dumps(response).encode())
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits, server.fail_first = [], False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='anthropic-blank-') as directory:
        root, port = Path(directory), free_port()
        providers = [{"provider": n, "engine": "claude", "base_url": f"http://127.0.0.1:{server.server_port}/{n}/v1/messages",
                      "api": "fixture", "model": ["test-model"], "preferences": {"cooldown_period": 0}}
                     for n in ['first', 'fallback']]
        providers.append({"provider": "gpt", "engine": "gpt", "base_url": f"http://127.0.0.1:{server.server_port}/gpt/v1/chat/completions",
                          "api": "fixture", "model": ["test-model"]})
        config = {"providers": providers, "api_keys": [
            {"api": "retry", "model": ["first/*", "fallback/*"]},
            {"api": "no-retry", "model": ["first/*", "fallback/*"], "preferences": {"AUTO_RETRY": False}},
            {"api": "gpt-only", "model": ["gpt/*"]}], "preferences": {"hedging": {"enabled": False}}}
        config_path = root / 'api.json'
        config_path.write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('FACTS_', 'OTEL_', 'FUGUE_OTEL_', 'UNI_API_CONTROL_RESTORE_')) and k != 'CONFIG_URL'}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD='1000', NO_PROXY='127.0.0.1,localhost')
        def request(path, body=None, key='retry'):
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            try:
                c.request('POST' if body is not None else 'GET', path, json.dumps(body) if body is not None else None,
                          {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
                r = c.getresponse()
                return r.status, r.read()
            finally:
                c.close()
        with (root/'gateway.log').open('w') as log:
            proc = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if request('/healthz')[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError('gateway did not start')
                count = 0
                for streaming in [False, True]:
                    for key in ['retry', 'no-retry']:
                        for fail_first in ([False, True] if key == 'retry' else [False]):
                            body = {"model": "test-model", "max_tokens": 100, "stream": streaming, "messages": [
                                {"role": "user", "content": "fixture"}, copy.deepcopy(MESSAGE), {"role": "user", "content": [RESULT]}]}
                            before = copy.deepcopy(body)
                            server.hits, server.fail_first = [], fail_first
                            status, raw = request('/v1/messages', body, key)
                            assert status == 200 and b'test' in raw, (status, raw)
                            assert [n for n, _ in server.hits] == (['first', 'fallback'] if fail_first else ['first'])
                            for _, sent in server.hits:
                                assert sent['messages'][1]['content'] == [THINKING, TOOL]
                                assert sent['messages'][2]['content'] == [RESULT]
                                assert sent['messages'][0] == before['messages'][0]
                            assert body == before
                            if streaming:
                                assert b'message_stop' in raw
                            count += 1
                    server.fail_first = False
                    for content in [[{"type": "text", "text": " "}], [THINKING, {"type": "text", "text": " "}],
                                    [THINKING, {"type": "text", "text": " ", "cache_control": {"type": "ephemeral"}}, TOOL]]:
                        body = {"model": "test-model", "max_tokens": 100, "stream": streaming,
                                "messages": [{"role": "assistant", "content": content}]}
                        server.hits = []
                        status, raw = request('/v1/messages', body)
                        assert status == 400 and b'ValidationException' in raw, (status, raw)
                        assert len(server.hits) == 1 and server.hits[0][1]['messages'] == body['messages']
                        count += 1
                # Converted OpenAI tool turns are cleaned only at the Claude wire boundary.
                for streaming in [False, True]:
                    server.hits, server.fail_first = [], False
                    body = {"model": "test-model", "stream": streaming, "messages": [
                        {"role": "assistant", "content": " ", "tool_calls": [{"id": "tool_1", "type": "function",
                         "function": {"name": "read", "arguments": '{"text":" "}'}}]},
                        {"role": "tool", "tool_call_id": "tool_1", "content": " "}]}
                    status, raw = request('/v1/chat/completions', body)
                    assert status == 200 and b'test' in raw, (status, raw)
                    assert server.hits[0][1]['messages'][0]['content'] == [TOOL]
                    count += 1
                server.hits = []
                body = {"model": "test-model", "messages": [copy.deepcopy(MESSAGE)]}
                status, _ = request('/v1/chat/completions', body, 'gpt-only')
                assert status == 200 and server.hits[0][1]['messages'] == body['messages']
                count += 1
                print(f'PASS Anthropic blank tool text: {count} HTTP cases, signed/tool content preserved, strict validation and fallback')
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
