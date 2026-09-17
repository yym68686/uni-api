"""Issue #171: isolated HTTP fixtures for heartbeats, failover and cooldown."""
import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from verify_dispatch_timing import free_port

TIMEOUT = 0.25
KEEPALIVE = 0.04
STALL = 0.65


def chat_frame(text):
    return ('data: ' + json.dumps({'id': 'fixture', 'choices': [
        {'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}) + '\n\n').encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.server.hits.append(self.path)
        bad = (self.headers.get('Authorization') == 'Bearer fixture-bad-key' if self.server.single_provider
               else self.path.startswith('/bad/')) or self.server.all_fail
        scenario = self.server.scenario if bad else 'success'
        try:
            if scenario == 'headers_delay':
                time.sleep(STALL)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.flush()
            if scenario in ('empty', 'done', 'error'):
                if scenario == 'done':
                    self.wfile.write(b'data: [DONE]\n\n')
                elif scenario == 'error':
                    self.wfile.write(b'data: {"error":{"message":"rate limit exceeded","status_code":429}}\n\n')
                return
            if scenario in ('body_stall', 'role_stall', 'postcommit_stall', 'slow_tail'):
                if scenario == 'role_stall':
                    self.wfile.write(b'data: {"id":"discard-me","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n')
                if scenario in ('postcommit_stall', 'slow_tail'):
                    self.wfile.write(chat_frame('FIRST'))
                self.wfile.flush()
                time.sleep(STALL)
            if scenario == 'heartbeat_stall':
                deadline = time.monotonic() + STALL
                while time.monotonic() < deadline:
                    self.wfile.write(b': upstream heartbeat\n\n')
                    self.wfile.flush()
                    time.sleep(.03)
            if self.path.endswith('/responses'):
                complete = {'id': 'resp_fixture', 'object': 'response', 'status': 'completed', 'model': 'm',
                            'output': [{'id': 'msg_fixture', 'type': 'message', 'role': 'assistant',
                            'status': 'completed', 'content': [{'type': 'output_text', 'text': 'OK'}]}]}
                frames = [{'type': 'response.output_text.delta', 'delta': 'OK', 'item_id': 'msg_fixture',
                           'output_index': 0, 'content_index': 0},
                          {'type': 'response.completed', 'response': complete}]
                raw = ''.join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()
            elif self.path.endswith('/messages'):
                frames = [{'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'OK'}},
                          {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
                          {'type': 'message_stop'}]
                raw = ''.join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()
            else:
                raw = chat_frame('OK') + b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary, kind, scenario, keepalive=KEEPALIVE, all_fail=False, global_interval=False, single_provider=False):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream.hits = []
    upstream.scenario = scenario
    upstream.all_fail = all_fail
    upstream.single_provider = single_provider
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    suffix = {'chat': '/v1/chat/completions', 'responses': '/v1/responses', 'claude': '/v1/messages'}[kind]
    with tempfile.TemporaryDirectory(prefix='uni-chat-keepalive-') as directory:
        root = Path(directory)
        preferences = {'model_timeout': {'default': TIMEOUT}, 'cooldown_period': 30}
        if scenario == 'postcommit_stall':
            preferences['timeout_policy'] = {'default': {'idle': TIMEOUT}}
        if not global_interval:
            preferences['keepalive_interval'] = {'default': keepalive}
        config = {
            'providers': [{'provider': name, 'engine': 'claude' if kind == 'claude' else 'gpt',
                           'api': ['fixture-bad-key', 'fixture-good-key'] if single_provider else 'fixture-upstream-key', 'model': ['m'],
                           'base_url': f'http://127.0.0.1:{upstream.server_port}/{name}{suffix}',
                           'preferences': preferences} for name in (['bad'] if single_provider else ['bad', 'good'])],
            'api_keys': [{'api': 'fixture-key', 'model': ['bad/*'] if single_provider else ['bad/*', 'good/*'],
                          'preferences': {'AUTO_RETRY': True, 'SCHEDULING_ALGORITHM': 'fixed_priority'}}],
            'preferences': {'hedging': {'enabled': False},
                            'keepalive_interval': {'default': keepalive if global_interval else 0}}
        }
        (root/'api.json').write_text(json.dumps(config))
        port = free_port()
        env = {k: v for k, v in os.environ.items() if k in ['PATH', 'HOME', 'TMPDIR', 'LANG', 'SSL_CERT_FILE']}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'api.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), NO_PROXY='127.0.0.1,localhost',
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0')
        with (root/'log').open('w+') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        conn.request('GET', '/healthz')
                        response = conn.getresponse()
                        response.read()
                        conn.close()
                        if response.status == 200:
                            break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise RuntimeError('fixture did not start')
                for number in range(1 if all_fail else 2):
                    before = len(upstream.hits)
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=4)
                    started = time.monotonic()
                    conn.request('POST', '/v1/chat/completions',
                                 json.dumps({'model': 'm', 'stream': True, 'messages': [{'role': 'user', 'content': 'test'}]}),
                                 {'Authorization': 'Bearer fixture-key', 'Content-Type': 'application/json'})
                    response = conn.getresponse()
                    header_time = time.monotonic() - started
                    parts, error = [], None
                    try:
                        while chunk := response.read1(65536):
                            parts.append(chunk)
                    except (http.client.HTTPException, OSError) as exc:
                        error = exc
                    conn.close()
                    body = b''.join(parts).decode()
                    hits = [path.split('/')[1] for path in upstream.hits[before:]]
                    diagnostic = (kind, scenario, keepalive, number, response.status, hits, body, error)
                    heartbeat_enabled = 0 < keepalive <= TIMEOUT
                    if all_fail:
                        assert hits == ['bad', 'good'], diagnostic
                        assert '"error"' in body and '[DONE]' not in body and error is None, diagnostic
                        assert response.status == (200 if heartbeat_enabled else 504), diagnostic
                        if heartbeat_enabled:
                            assert '"status_code":504' in body, diagnostic
                    elif scenario == 'postcommit_stall' and number == 0:
                        assert hits == ['bad'] and 'FIRST' in body and 'OK' not in body, diagnostic
                        assert error is not None or '"error"' in body, diagnostic
                    else:
                        expected = ['bad'] if scenario == 'slow_tail' else ['bad', 'good'] if number == 0 else ['good']
                        if single_provider:
                            expected = ['bad', 'bad']
                        assert hits == expected and error is None, diagnostic
                        assert response.status == 200 and 'OK' in body and '[DONE]' in body, diagnostic
                        assert 'discard-me' not in body, diagnostic
                    if number == 0 and keepalive != TIMEOUT and scenario in ('headers_delay', 'body_stall', 'role_stall', 'heartbeat_stall'):
                        if heartbeat_enabled:
                            assert header_time < TIMEOUT and body.count(': keepalive') >= 2, diagnostic
                        else:
                            assert header_time >= TIMEOUT * .85 and ': keepalive' not in body, diagnostic
                    if scenario == 'slow_tail':
                        assert body.count(': keepalive') >= 2, diagnostic
                    time.sleep(.1)  # Let terminal accounting apply cooldown before the next request.
                print(f'PASS {kind} {scenario} keepalive={keepalive} all_fail={all_fail} global={global_interval} multi_key={single_provider}', flush=True)
            except Exception:
                print((root/'log').read_text()[-6000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                upstream.shutdown()
                upstream.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    binary = parser.parse_args().binary.resolve()
    for kind in ['chat', 'responses', 'claude']:
        for scenario in ['headers_delay', 'body_stall', 'heartbeat_stall', 'empty']:
            verify(binary, kind, scenario)
    for scenario in ['role_stall', 'done', 'error', 'postcommit_stall', 'slow_tail']:
        verify(binary, 'chat', scenario)
    for interval in [0, 1, TIMEOUT]:
        verify(binary, 'chat', 'body_stall', interval)
    for interval in [KEEPALIVE, 1]:
        verify(binary, 'chat', 'body_stall', interval, all_fail=True)
    verify(binary, 'chat', 'body_stall', global_interval=True)

    verify(binary, 'chat', 'body_stall', single_provider=True)
