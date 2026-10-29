"""Exact AI Studio examples rejection: unchanged fallback and raw billing, loopback only."""
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
from fact_sink import accept_facts
from verify_dispatch_timing import free_port

MESSAGE = ('AI Studio 请求参数无效: 编码 function declaration 0: parameters: ***.properties.duration_min: '
           'AI Studio 协议能力尚无成功现场证据: JSON schema 字段 examples')
ERROR = {'error': {'message': MESSAGE, 'type': 'invalid_request_error', 'param': '', 'code': 'invalid_request'}}
FUNCTION = {'name': 'lookup', 'description': 'fixture', 'parameters': {'type': 'object',
            'properties': {'duration_min': {'type': 'integer', 'minimum': 1, 'examples': [30, 365, 730]}}}}
SUCCESS = {'id': 'resp_fixture', 'object': 'response', 'status': 'completed',
           'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'OK'}]}],
           'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        channel = self.path.split('/')[1]
        self.server.hits.append((channel, raw, body))
        reject = channel != 'fallback' or self.server.mode == 'exhausted'
        status = self.server.status if reject else 200
        value = self.server.error if reject else copy.deepcopy(SUCCESS)
        if self.path.endswith('/compact') and not reject: value['object'] = 'response.compaction'
        content_type = 'application/json'
        if reject and self.server.mode == 'postcommit':
            status, content_type = 200, 'text/event-stream'
            frames = [{'type': 'response.output_text.delta', 'delta': 'started'},
                      {'type': 'response.failed', 'response': {'status': 'failed', 'error': ERROR['error']}}]
            raw = ''.join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()
        elif not reject and body.get('stream'):
            content_type = 'text/event-stream'
            frames = [{'type': 'response.output_text.delta', 'delta': 'OK'},
                      {'type': 'response.completed', 'response': value}]
            raw = ''.join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames).encode()
        else:
            raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try: self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError): pass


def payload(path, stream):
    shared = {'model': 'fixture-model', 'stream': stream, 'temperature': .3}
    if path == '/v1/chat/completions':
        return {**shared, 'messages': [{'role': 'system', 'content': 'private-system'},
                {'role': 'user', 'content': 'private-query'}], 'max_completion_tokens': 128,
                'tools': [{'type': 'function', 'function': copy.deepcopy(FUNCTION)}]}
    return {**shared, 'input': [{'role': 'system', 'content': 'private-system'},
            {'role': 'user', 'content': 'private-query'}], 'max_output_tokens': 128,
            'tools': [{'type': 'function', **copy.deepcopy(FUNCTION)}]}


def verify(binary, engine):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits, server.facts, server.error, server.mode, server.status = [], [], ERROR, 'retry', 400
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-ai-studio-examples-') as td:
        root, port = Path(td), free_port()
        base = f'http://127.0.0.1:{server.server_port}'
        providers = [{'provider': n, 'base_url': base+'/'+n+'/v1/responses', 'api': 'fixture-upstream-secret',
                      'model': ['fixture-model'], 'preferences': {'cooldown_period': 300}}
                     for n in ['first', 'fallback', 'also-reject']]
        if engine != 'auto':
            for p in providers: p['engine'] = engine
        config = {'providers': providers, 'api_keys': [
            {'api': 'retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': 1}},
            {'api': 'exhausted', 'model': ['first/*', 'also-reject/*'], 'preferences': {'AUTO_RETRY': 1}},
            {'api': 'no-retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': False}}],
            'preferences': {'hedging': {'enabled': False}}}
        config_bytes = json.dumps(config).encode()
        (root/'config.json').write_bytes(config_bytes)
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config.json'),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
            FACTS_INGEST_URL=base+'/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'), NO_PROXY='127.0.0.1,localhost')
        count, expected_facts = 0, []
        def call(body=None, key='retry', path='/v1/responses', targeted=False):
            rid = f'examples-{engine}-{count}'
            headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer '+key, 'X-Request-ID': rid}
            if targeted: headers['X-Uni-API-Provider'] = 'first'
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
            try:
                c.request('GET' if body is None else 'POST', '/healthz' if body is None else path,
                          None if body is None else json.dumps(body), headers)
                r = c.getresponse()
                return rid, r.status, r.read()
            finally: c.close()
        with (root/'log').open('w') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[1] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('gateway fixture not ready')
                paths = [('/v1/chat/completions', False), ('/v1/chat/completions', True),
                         ('/v1/responses', False), ('/v1/responses', True), ('/v1/responses/compact', False)]
                def check(path, stream, key='retry', want=200, names=('first','fallback'), targeted=False):
                    nonlocal count
                    body = payload(path, stream)
                    original = copy.deepcopy(body)
                    server.hits = []
                    rid, status, raw = call(body, key, path, targeted)
                    assert status == want, (engine, count, path, key, status, raw[:300])
                    assert [n for n,_,_ in server.hits] == list(names), (engine, count, [n for n,_,_ in server.hits])
                    assert all(w == server.hits[0][1] for _,w,_ in server.hits), 'wire bytes changed on fallback'
                    for _,_,wire in server.hits:
                        assert wire['tools'][0]['parameters'] == FUNCTION['parameters'], 'examples or constraints changed'
                    assert body == original
                    if want == 200: assert b'OK' in raw
                    count += 1
                    return rid
                for path, stream in paths:
                    for key, want, names in [('retry',200,('first','fallback')), ('no-retry',502,('first',)),
                            ('exhausted',502,('first','also-reject')), ('retry',200,('first','fallback'))]:
                        rid = check(path, stream, key, want, names)
                        expected_facts.append((rid,want))
                    if path == '/v1/responses':
                        rid = check(path, stream, want=502, names=('first',), targeted=True)
                        expected_facts.append((rid,502))
                for path, stream in [('/v1/chat/completions', False), ('/v1/responses', False)]:
                    for param in [None, 'absent']:
                        server.error = copy.deepcopy(ERROR)
                        if param == 'absent': del server.error['error']['param']
                        else: server.error['error']['param'] = param
                        check(path, stream)
                    for patch in [{'type':'upstream_error'}, {'code':'invalid_type'}, {'code':400}, {'code':None},
                                  {'param':'tools'}, {'message':MESSAGE+'.'}, {'message':'Quoted: '+MESSAGE},
                                  {'message':MESSAGE.replace('字段 examples','字段 default')},
                                  {'message':MESSAGE.replace('尚无成功现场证据','校验失败')},
                                  {'message':'examples must be an array'},
                                  {'message':MESSAGE.replace('declaration 0:', 'declaration -1:')}]:
                        server.error = {'error': {**ERROR['error'], **patch}}
                        check(path, stream, want=400, names=('first',))
                    for error in [{'detail':ERROR['error']}, {'error': {'message':json.dumps(ERROR)}},
                                  {'error': {'message':'invalid input'}, 'debug':ERROR}]:
                        server.error = error
                        check(path, stream, want=400, names=('first',))
                server.error = ERROR
                for path in ['/v1/chat/completions', '/v1/responses']:
                    server.mode = 'postcommit'
                    server.hits = []
                    _, status, raw = call(payload(path, True), path=path)
                    assert status == 200 and b'started' in raw and len(server.hits) == 1
                    count += 1
                server.mode, server.status = 'retry', 403
                check('/v1/responses', False, key='no-retry', want=403, names=('first',))
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    if all(any(f.get('kind') == 'request' and f.get('request_id') == rid for f in server.facts) for rid,_ in expected_facts): break
                    time.sleep(.1)
                for rid, want in expected_facts:
                    facts = [f for f in server.facts if f.get('request_id') == rid]
                    assert any(f.get('kind') == 'billing' and f.get('status') == 400 for f in facts), rid
                    assert any(f.get('stage') == 'upstream_attempt' and f.get('trace_detail',{}).get('attempt_status_code') == 400 for f in facts), rid
                    assert any(f.get('stage') == 'ai_studio_examples_fallback' and f.get('trace_detail',{}).get('routing_status_code') == 502 for f in facts), rid
                    assert any(f.get('kind') == 'request' and f.get('status') == want for f in facts), rid
                    if want == 200:
                        assert any(f.get('kind') == 'billing' and f.get('status') == 200 for f in facts), rid
                captured = json.dumps(server.facts)+(root/'log').read_text()
                for secret in ['fixture-upstream-secret', 'private-system', 'private-query']:
                    assert secret not in captured
                assert (root/'config.json').read_bytes() == config_bytes
                print(f'PASS AI Studio examples fallback engine={engine}: {count} HTTP cases, unchanged wire, raw400/routing502, no cooldown/post-output retry')
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    for engine in ['auto', 'gpt', 'codex']:
        verify(Path(sys.argv[1]).resolve(), engine)
