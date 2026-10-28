"""Opaque encrypted-history channel rejection: unchanged fallback, loopback only."""
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

ERROR = {'error': {'code': 'invalid_encrypted_content', 'type': 'invalid_request_error',
                   'message': 'Upstream encrypted content error (details redacted)'}}
RESPONSE = {'id': 'resp_fixture', 'object': 'response', 'status': 'completed', 'model': 'test-model',
            'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'OK'}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw); channel = self.path.split('/')[1]
        self.server.hits.append((channel, raw, body))
        reject = channel != 'fallback' or self.server.mode == 'exhausted'
        status, content_type = (self.server.status if reject else 200), 'application/json'
        value = self.server.error if reject else copy.deepcopy(RESPONSE)
        if self.path.endswith('/compact') and not reject: value['object'] = 'response.compaction'
        if self.server.mode == 'postcommit' and reject:
            status, content_type = 200, 'text/event-stream'
            events = [{'type': 'response.output_text.delta', 'delta': 'started'},
                      {'type': 'response.failed', 'response': {'status': 'failed', 'error': ERROR['error']}}]
            raw = ''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        elif body.get('stream') and not reject:
            content_type = 'text/event-stream'
            events = [{'type': 'response.output_text.delta', 'delta': 'OK'},
                      {'type': 'response.completed', 'response': value}]
            raw = ''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        else: raw = json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw))); self.end_headers()
        try: self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError): pass


def payload(stream):
    return {'model': 'test-model', 'stream': stream, 'store': False,
            'input': [{'role': 'user', 'content': 'private-history'},
                      {'type': 'reasoning', 'encrypted_content': 'gAAAAopaque-do-not-log', 'summary': []},
                      {'type': 'compaction', 'encrypted_content': 'c2c:opaque-do-not-log'},
                      {'type': 'function_call', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{}'},
                      {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'private-result'},
                      {'role': 'user', 'content': 'continue'}],
            'tools': [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}],
            'reasoning': {'effort': 'high'}, 'include': ['reasoning.encrypted_content']}


def verify(binary, engine):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits, server.facts, server.error, server.mode, server.status = [], [], ERROR, 'reject', 400
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-encrypted-fallback-') as td:
        root, port = Path(td), free_port(); base = f'http://127.0.0.1:{server.server_port}'
        config = {'providers': [{'provider': name, 'engine': engine, 'base_url': base+'/'+name+'/v1/responses',
                   'api': 'fixture-upstream-secret', 'model': ['test-model'], 'preferences': {'cooldown_period': 300}}
                   for name in ['first', 'fallback']],
                  'api_keys': [{'api': 'retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': 1}},
                               {'api': 'no-retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': False}}],
                  'preferences': {'hedging': {'enabled': False}}}
        config_bytes = json.dumps(config).encode(); (root/'api.json').write_bytes(config_bytes)
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'api.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   FACTS_INGEST_URL=base+'/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'), NO_PROXY='127.0.0.1,localhost')
        def call(body=None, key='retry', path='/v1/responses', target=False):
            rid = f'{engine}-{count}'
            headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json', 'X-Request-ID': rid}
            if target: headers['X-Uni-API-Provider'] = 'first'
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
            try:
                conn.request('GET' if body is None else 'POST', '/healthz' if body is None else path,
                             None if body is None else json.dumps(body), headers)
                res = conn.getresponse(); return rid, res.status, res.read()
            finally: conn.close()
        count, billed = 0, []
        with (root/'log').open('w') as log:
            proc = subprocess.Popen([str(binary)], env=env, cwd=root, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[1] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('fixture startup failed')
                for path, stream in [('/v1/responses', False), ('/v1/responses', True), ('/v1/responses/compact', False)]:
                    cases = [('retry','retry',200,['first','fallback']),
                             ('no-retry','no-retry',502,['first']),
                             ('exhausted','retry',502,['first','fallback']),
                             ('retry','retry',200,['first','fallback'])]
                    if path == '/v1/responses':
                        cases.insert(2, ('targeted','retry',502,['first']))
                    for case, key, expected, names in cases:
                        server.hits, server.mode = [], case
                        rid, status, raw = call(payload(stream), key, path, case=='targeted')
                        assert status == expected, (engine, path, case, status, raw[:300])
                        assert [n for n,_,_ in server.hits] == names, (engine, path, case, server.hits)
                        # Both channels have identical settings. Even byte serialization must match.
                        assert all(w == server.hits[0][1] for _,w,_ in server.hits), 'wire body changed on fallback'
                        assert all(w['input'] == payload(stream)['input'] for _,_,w in server.hits), 'history changed'
                        if expected == 200:
                            assert b'OK' in raw; billed.append((rid, path))
                        count += 1
                # No encrypted history, malformed fields, unrelated errors/statuses keep ordinary policy.
                negatives = [('no-history', None), ('empty-encrypted', None), ('invalid-encrypted-type', None),
                             ('wrong-item-type', None), ('nonempty-param', {'param':'input'}), ('empty-param', {'param':''}),
                             ('wrong-code', {'code':'invalid_type'}), ('wrong-type', {'type':'upstream_error'}),
                             ('missing-code', {'code':None}), ('message-prefix', {'message':'Quoted: '+ERROR['error']['message']}),
                             ('message-suffix', {'message':ERROR['error']['message']+' extra'})]
                for label, patch in negatives:
                    body = payload(False); error = copy.deepcopy(ERROR)
                    if label == 'no-history': body['input'] = [{'role':'user','content':'continue'}]
                    if label == 'empty-encrypted': body['input'][1]['encrypted_content'] = ''
                    if label == 'invalid-encrypted-type': body['input'][2]['encrypted_content'] = 42
                    if label == 'wrong-item-type': body['input'] = [{'type':'message','role':'user','content':'plain','encrypted_content':'echo'}]
                    if patch: error['error'].update(patch)
                    server.hits, server.mode, server.error = [], 'reject', error
                    _, status, _ = call(body)
                    assert status == 400 and len(server.hits) == 1, (engine, label, status)
                    count += 1
                server.status, server.mode, server.hits = 400, 'postcommit', []
                server.error = ERROR
                _, status, raw = call(payload(True))
                assert status == 200 and b'started' in raw and len(server.hits) == 1, (status, raw)
                count += 1
                # This ordinary auth failure can cool the channel; run it last.
                server.status, server.mode, server.hits = 403, 'reject', []
                _, status, _ = call(payload(False), 'no-retry')
                assert status == 403 and len(server.hits) == 1, status
                count += 1
                deadline = time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='billing' and f.get('request_id')==rid and f.get('status')==200 for f in server.facts) for rid,_ in billed): break
                    time.sleep(.1)
                for rid, path in billed:
                    facts = [f for f in server.facts if f.get('request_id')==rid]
                    assert any(f.get('kind')=='billing' and f.get('status')==400 for f in facts), (rid, path, facts)
                    assert any(f.get('kind')=='billing' and f.get('status')==200 for f in facts), (rid, path, facts)
                    assert any(f.get('stage')=='responses_encrypted_content_fallback' and f.get('trace_detail',{}).get('routing_status_code')==502 for f in facts), (rid, facts)
                    assert any(f.get('stage')=='upstream_attempt' and f.get('trace_detail',{}).get('attempt_status_code')==400 for f in facts), (rid, facts)
                encoded = json.dumps(server.facts)+(root/'log').read_text()
                for secret in ['gAAAAopaque-do-not-log', 'c2c:opaque-do-not-log', 'fixture-upstream-secret', 'private-history', 'private-result']:
                    assert secret not in encoded, secret
                assert (root/'api.json').read_bytes() == config_bytes
                print(f'PASS encrypted channel fallback engine={engine}: {count} HTTP cases; raw400/routing502, no cooldown, unchanged wire')
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                server.shutdown(); server.server_close()


if __name__ == '__main__':
    for engine in ['gpt', 'codex']: verify(Path(sys.argv[1]).resolve(), engine)
