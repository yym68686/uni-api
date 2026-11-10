"""Route preparation failures must be visible without becoming upstream sends."""
from fact_sink import accept_facts
from verify_dispatch_timing import free_port
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        self.rfile.read(int(self.headers['Content-Length']))
        role = self.path.split('/')[1]
        self.server.hits.append(role)
        status = 403 if role == 'r16' else 401
        raw = json.dumps({'error': {'message': 'fixture quota' if status == 403 else 'Invalid token.',
                                   'type': 'upstream_error'}}).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def verify(binary, stream, hedging, local_only=False):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Mock)
    server.facts, server.hits = [], []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-route-skips-') as directory:
        root, port = Path(directory), free_port()
        providers = []
        for index in ([1] if local_only else range(1, 27)):
            engine = 'codex' if index == 16 else 'gpt' if index == 19 else 'gemini'
            endpoint = 'responses' if engine == 'codex' else 'chat/completions' if engine == 'gpt' else 'models'
            providers.append({'provider': f'r{index}', 'engine': engine, 'api': 'fixture-upstream-secret',
                'base_url': f'http://127.0.0.1:{server.server_port}/r{index}/v1/{endpoint}',
                'model': ['m'], 'preferences': {'cooldown_period': 0, 'api_key_cooldown_period': 0}})
        config = {'providers': providers, 'api_keys': [{'api': 'fixture-client', 'model': ['all'],
            'preferences': {'AUTO_RETRY': False if local_only else 25,
                           'SCHEDULING_ALGORITHM': 'fixed_priority'}}],
            'preferences': {'unknown_upstream_400_retries': 0,
                            'hedging': {'enabled': hedging, 'max_inflight_attempts': 2}}}
        (root/'config.json').write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config.json'),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
            FACTS_INGEST_URL=f'http://127.0.0.1:{server.server_port}/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'),
            NO_PROXY='127.0.0.1,localhost')
        def call(body=None):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
            conn.request('GET' if body is None else 'POST', '/healthz' if body is None else '/v1/chat/completions',
                         None if body is None else json.dumps(body),
                         {'Authorization': 'Bearer fixture-client', 'Content-Type': 'application/json', 'X-Request-ID': 'route-skips'})
            response = conn.getresponse()
            status, raw = response.status, response.read()
            conn.close()
            return status, raw
        with (root/'runtime.log').open('w') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if call()[0] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('local gateway did not become ready')
                body = {'model': 'm', 'stream': stream, 'messages': [
                    {'role': 'assistant', 'tool_calls': [{'id': 'call_g2_invalid.1', 'type': 'function',
                        'function': {'name': 'fixture', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'call_g2_invalid.1', 'content': 'private fixture result'}]}
                status, raw = call(body)
                assert status == (400 if local_only else 401), (status, raw)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if any(f.get('stage') == 'response_body_finished' for f in server.facts): break
                    time.sleep(.05)
                facts = server.facts
                requests = [f for f in facts if f['kind'] == 'request']
                assert len(requests) == 1 and requests[0]['status'] == status, requests
                execution = next(f for f in facts if f.get('stage') == 'request_final_facts')
                final = execution['trace_detail']['final_facts']
                assert final['downstream_http_status'] == status, final
                assert final['upstream_http_status'] == (None if local_only else 401), final
                assert requests[0]['provider'] == ('' if local_only else 'r19'), requests
                assert server.hits == ([] if local_only else ['r16', 'r19']), server.hits
                assert len([f for f in facts if f['kind'] == 'dispatch']) == (0 if local_only else 2)
                assert len([f for f in facts if f['kind'] == 'attempt']) == (0 if local_only else 2)
                skips = [f for f in facts if f.get('trace_detail', {}).get('attempt_outcome') == 'skipped']
                assert len(skips) == (1 if local_only else 24), skips
                for skip in skips:
                    detail = skip['trace_detail']
                    assert detail['phase'] == 'request_conversion' and detail['skip_reason'] == 'attempt_build_failed', detail
                    assert detail['error']['error_code'] == 'gemini_tool_history', detail
                    assert detail['error']['error_param'] == 'messages[0].tool_calls[0]', detail
                    assert skip['stream'] == stream, skip
                for secret in ['fixture-client', 'fixture-upstream-secret', 'private fixture result', 'call_g2_invalid.1']:
                    assert secret not in json.dumps(facts), secret
                print(f'PASS route skips: stream={stream}, hedging={hedging}, local_only={local_only}; HTTP={status}, sends={len(server.hits)}, skips={len(skips)}')
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    server.shutdown(); server.server_close()


if __name__ == '__main__':
    binary = Path(sys.argv[1]).resolve()
    for stream, hedging in [(True, False), (False, False), (False, True)]:
        verify(binary, stream, hedging)
        verify(binary, stream, hedging, local_only=True)
