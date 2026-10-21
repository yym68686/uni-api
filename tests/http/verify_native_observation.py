"""Native/translated Claude and Chat clocks + HTTP failure facts, local fixtures only."""
from fact_sink import accept_facts
import http.client
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from verify_dispatch_timing import free_port

facts = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path == "/facts": return accept_facts(self, facts)
        self.rfile.read(int(self.headers['Content-Length']))
        name = self.path.split('/')[1]
        time.sleep(.1)
        if name == 'http-error':
            body = b'{"error":{"message":"fixture validation failure"}}'
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if name == 'chat':
            events = [
                {'choices': [{'delta': {'role': 'assistant'}}]},
                {'choices': [{'delta': {'content': 'hello'}}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            ]
        else:
            events = [
                {'type': 'message_start', 'message': {'role': 'assistant', 'usage': {'input_tokens': 1}}},
                {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'hello'}},
                {'type': 'message_stop'},
            ]
        chunks = [('data: ' + json.dumps(event) + '\n\n').encode() for event in events]
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(sum(map(len, chunks))))
        self.end_headers()
        for index, chunk in enumerate(chunks):
            if index == 1:
                time.sleep(.2)
            self.wfile.write(chunk)
            self.wfile.flush()


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory(prefix='uni-native-facts-') as path:
    root = pathlib.Path(path)
    port = free_port()
    providers = [('claude-translated', 'claude'), ('claude-native', 'claude'), ('chat', 'gpt'), ('http-error', 'gpt')]
    config = {
        'providers': [{'provider': name, 'engine': engine, 'base_url': f'http://127.0.0.1:{server.server_port}/{name}/v1/' + ('messages' if engine == 'claude' else 'chat/completions'), 'api': 'fixture-upstream', 'model': [{'m': name}]} for name, engine in providers],
        'api_keys': [{'api': 'fixture-admin', 'model': ['all']}],
    }
    (root / 'api.json').write_text(json.dumps(config))
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
    env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root / 'api.json'), RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / 'snapshot.json'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / 'ledger'), RUST_REQUEST_SPOOL_DIRECTORY=str(root / 'body-spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', NO_PROXY='127.0.0.1,localhost', FACTS_INGEST_URL=f'http://127.0.0.1:{server.server_port}'+"/facts", FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root / 'facts-spool'))

    def call(method, endpoint, payload=None, provider=None):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        headers = {'Authorization': 'Bearer fixture-admin', 'Content-Type': 'application/json'}
        conn.request(method, endpoint, json.dumps(payload) if payload else None, headers)
        response = conn.getresponse()
        status, body = response.status, response.read()
        conn.close()
        return status, body

    with (root / 'runtime.log').open('w') as log:
        process = subprocess.Popen([str(pathlib.Path(sys.argv[1]).resolve())], cwd=root, env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    if call('GET', '/healthz')[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(.05)
            for name, _ in providers:
                endpoint = '/v1/messages' if name == 'claude-native' else '/v1/chat/completions'
                status, body = call('POST', endpoint, {'model': name, 'messages': [{'role': 'user', 'content': 'fixture'}], 'max_tokens': 16, 'stream': True}, name)
                assert status == (400 if name == 'http-error' else 200), (name, status, body)
                if name != 'http-error':
                    assert b'hello' in body, (name, body)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and sum(f['kind'] == 'request' for f in facts) < len(providers):
                time.sleep(.1)
            for name, _ in providers:
                attempt = next(f for f in facts if f['kind'] == 'attempt' and f['provider'] == name)
                request = next(f for f in facts if f['kind'] == 'request' and f['request_id'] == attempt['request_id'])
                for fact in (attempt, request):
                    if name == 'http-error':
                        assert fact['outcome'] == 'failed', fact
                        assert fact['failure_reason'] == ('upstream_http_400' if fact['kind'] == 'attempt' else 'no_successful_channel'), fact
                    else:
                        assert fact['outcome'] == 'success', fact
                        assert fact.get('first_output_ms', 0) >= 275, fact
                        assert fact.get('first_text_ms', 0) >= 275, fact
                        assert fact.get('response_created_ms') is None, fact
                        assert fact.get('failure_reason') is None, fact
                print(json.dumps({'provider': name, 'outcome': attempt['outcome'], 'first_text_ms': attempt.get('first_text_ms'), 'failure_reason': attempt.get('failure_reason')}))
        except Exception:
            print((root / 'runtime.log').read_text()[-5000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            process.wait(timeout=10)
server.shutdown()
server.server_close()
