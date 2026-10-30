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
COMPACTION_ERROR = {'error': {'code': 'invalid_compaction_state', 'type': 'invalid_request_error',
    'message': '压缩状态无效或不属于当前调用方，请使用原调用凭据，或携带完整历史重试。'}}
ENCRYPTED_COMPACTION_ID = 'cmp_01234567-89ab-cdef-0123-456789abcdef'
ENCRYPTED_COMPACTION_ERROR = {'error': {'type': 'upstream_error', 'message':
    f'The encrypted content for item {ENCRYPTED_COMPACTION_ID} could not be verified. '
    'Reason: Encrypted content could not be decrypted or parsed.'}}
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
                      {'type': 'response.failed', 'response': {'status': 'failed', 'error': self.server.error['error']}}]
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


def payload(stream, identified_compaction=False):
    compaction = {'type': 'compaction', 'encrypted_content': 'c2c:opaque-do-not-log'}
    if identified_compaction: compaction['id'] = ENCRYPTED_COMPACTION_ID
    return {'model': 'test-model', 'stream': stream, 'store': False,
            'input': [{'role': 'user', 'content': 'private-history'},
                      {'type': 'reasoning', 'encrypted_content': 'gAAAAopaque-do-not-log', 'summary': []},
                      compaction,
                      {'type': 'function_call', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{}'},
                      {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'private-result'},
                      {'role': 'user', 'content': 'continue'}],
            'tools': [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}],
            'reasoning': {'effort': 'high'}, 'include': ['reasoning.encrypted_content']}


def verify(binary, engine, compaction=False, identified_compaction=False):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    selected_error = ENCRYPTED_COMPACTION_ERROR if identified_compaction else COMPACTION_ERROR if compaction else ERROR
    server.hits, server.facts, server.error, server.mode, server.status = [], [], selected_error, 'reject', 400
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-encrypted-fallback-') as td:
        root, port = Path(td), free_port(); base = f'http://127.0.0.1:{server.server_port}'
        config = {'providers': [{'provider': name, 'engine': engine, 'base_url': base+'/'+name+'/v1/responses',
                   'api': 'fixture-upstream-secret', 'model': ['test-model'], 'preferences': {'cooldown_period': 300}}
                   for name in ['first', 'fallback']] + [{'provider': 'auth', 'engine': engine, 'base_url': base+'/auth/v1/responses',
                   'api': 'fixture-upstream-secret', 'model': ['auth-model']}],
                  'api_keys': [{'api': 'retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': 1}},
                               {'api': 'no-retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': False}},
                               {'api': 'auth-check', 'model': ['auth/*'], 'preferences': {'AUTO_RETRY': False}}],
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
                else: raise AssertionError('fixture startup failed: '+(root/'log').read_text()[-1000:])
                for path, stream in [('/v1/responses', False), ('/v1/responses', True), ('/v1/responses/compact', False)]:
                    cases = [('retry','retry',200,['first','fallback']),
                             ('no-retry','no-retry',502,['first']),
                             ('exhausted','retry',502,['first','fallback']),
                             ('retry','retry',200,['first','fallback'])]
                    if path == '/v1/responses':
                        cases.insert(2, ('targeted','retry',502,['first']))
                    if identified_compaction: cases.append(('nulls','retry',200,['first','fallback']))
                    for case, key, expected, names in cases:
                        server.hits, server.mode = [], case
                        server.error = copy.deepcopy(selected_error)
                        if case == 'nulls': server.error['error'].update(code=None, param=None)
                        rid, status, raw = call(payload(stream, identified_compaction), key, path, case=='targeted')
                        assert status == expected, (engine, path, case, status, raw[:300])
                        assert [n for n,_,_ in server.hits] == names, (engine, path, case, server.hits)
                        # Both channels have identical settings. Even byte serialization must match.
                        assert all(w == server.hits[0][1] for _,w,_ in server.hits), 'wire body changed on fallback'
                        assert all(w['input'] == payload(stream, identified_compaction)['input'] for _,_,w in server.hits), 'history changed'
                        if expected == 200:
                            assert b'OK' in raw; billed.append((rid, path))
                        count += 1
                # No encrypted history, malformed fields, unrelated errors/statuses keep ordinary policy.
                negatives = ([('no-compaction', None), ('empty-compaction', None), ('invalid-compaction-type', None),
                             ('duplicate-invalid-compaction', None), ('wrong-item-type', None),
                             ('nonempty-param', {'param':'input'}), ('empty-param', {'param':''}),
                             ('wrong-code', {'code':'invalid_type'}), ('wrong-type', {'type':'upstream_error'}),
                             ('missing-code', {'code':None}), ('message-prefix', {'message':'Quoted: '+selected_error['error']['message']}),
                             ('message-suffix', {'message':selected_error['error']['message']+' extra'})] if compaction else
                            [('no-history', None), ('empty-encrypted', None), ('invalid-encrypted-type', None),
                             ('wrong-item-type', None), ('nonempty-param', {'param':'input'}), ('empty-param', {'param':''}),
                             ('wrong-code', {'code':'invalid_type'}), ('wrong-type', {'type':'upstream_error'}),
                             ('missing-code', {'code':None}), ('message-prefix', {'message':'Quoted: '+selected_error['error']['message']}),
                             ('message-suffix', {'message':selected_error['error']['message']+' extra'})])
                if identified_compaction:
                    negatives = [('no-compaction', None), ('empty-compaction', None), ('invalid-compaction-type', None),
                                 ('missing-encrypted', None), ('whitespace-encrypted', None), ('null-encrypted', None),
                                 ('missing-id', None), ('wrong-id', None), ('wrong-item-type', None),
                                 ('duplicate-id', None), ('duplicate-reference', None), ('invalid-other-compaction', None),
                                 ('nonempty-param', {'param':'input'}), ('empty-param', {'param':''}),
                                 ('concrete-code', {'code':'invalid_encrypted_content'}), ('empty-code', {'code':''}),
                                 ('wrong-type', {'type':'invalid_request_error'}), ('null-type', {'type':None}),
                                 ('message-prefix', {'message':'Quoted: '+selected_error['error']['message']}),
                                 ('message-suffix', {'message':selected_error['error']['message']+' extra'}),
                                 ('missing-target', None), ('reasoning-target', None), ('bad-uuid', None),
                                 ('nested-error', None), ('bare-error', None)]
                for label, patch in negatives:
                    body = payload(False, identified_compaction); error = copy.deepcopy(selected_error)
                    if label in ('no-history', 'no-compaction'): body['input'] = [{'role':'user','content':'continue'}]
                    if label in ('empty-encrypted', 'empty-compaction'): body['input'][2 if compaction or identified_compaction else 1]['encrypted_content'] = ''
                    if label in ('invalid-encrypted-type', 'invalid-compaction-type'): body['input'][2 if compaction or identified_compaction else 1]['encrypted_content'] = 42
                    if label == 'duplicate-invalid-compaction': body['input'].append({'type':'compaction','encrypted_content':' '})
                    if label == 'wrong-item-type': body['input'] = [{'type':'message','role':'user','content':'plain','encrypted_content':'echo'}]
                    if label == 'missing-encrypted': del body['input'][2]['encrypted_content']
                    if label == 'whitespace-encrypted': body['input'][2]['encrypted_content'] = ' \n'
                    if label == 'null-encrypted': body['input'][2]['encrypted_content'] = None
                    if label == 'missing-id': del body['input'][2]['id']
                    if label in ('wrong-id', 'missing-target'): body['input'][2]['id'] = 'cmp_11111111-2222-3333-4444-555555555555'
                    if label == 'reasoning-target': body['input'][2]['type'] = 'reasoning'
                    if label == 'duplicate-id': body['input'].append(copy.deepcopy(body['input'][2]))
                    if label == 'duplicate-reference': body['input'].append({'type':'item_reference','id':ENCRYPTED_COMPACTION_ID})
                    if label == 'invalid-other-compaction': body['input'].append({'type':'compaction','encrypted_content':None})
                    if label == 'bad-uuid':
                        bad_id = ENCRYPTED_COMPACTION_ID.replace('-', '_')
                        body['input'][2]['id'] = bad_id
                        error['error']['message'] = error['error']['message'].replace(ENCRYPTED_COMPACTION_ID, bad_id)
                    if patch: error['error'].update(patch)
                    if label == 'nested-error': error = {'error':{'message':json.dumps(error)}}
                    if label == 'bare-error': error = error['error']
                    server.hits, server.mode, server.error = [], 'reject', error
                    _, status, _ = call(body)
                    assert status == 400 and len(server.hits) == 1, (engine, label, status)
                    count += 1
                server.status, server.mode, server.hits = 400, 'postcommit', []
                server.error = selected_error
                _, status, raw = call(payload(True, identified_compaction))
                assert status == 200 and b'started' in raw and len(server.hits) == 1, (status, raw)
                count += 1
                # A committed upstream_error may cool the first channel under ordinary policy.
                # Isolate the auth status check so it still reaches an upstream.
                server.status, server.mode, server.hits = 403, 'reject', []
                auth_body = payload(False)
                auth_body['model'] = 'auth-model'
                _, status, _ = call(auth_body, 'auth-check')
                assert status == 403 and [n for n,_,_ in server.hits] == ['auth'], (status, server.hits)
                count += 1
                deadline = time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='billing' and f.get('request_id')==rid and f.get('status')==200 for f in server.facts) for rid,_ in billed): break
                    time.sleep(.1)
                for rid, path in billed:
                    facts = [f for f in server.facts if f.get('request_id')==rid]
                    assert any(f.get('kind')=='billing' and f.get('status')==400 for f in facts), (rid, path, facts)
                    assert any(f.get('kind')=='billing' and f.get('status')==200 for f in facts), (rid, path, facts)
                    expected_event = 'responses_compaction_state_fallback' if compaction or identified_compaction else 'responses_encrypted_content_fallback'
                    assert any(f.get('stage')==expected_event and f.get('trace_detail',{}).get('routing_status_code')==502 for f in facts), (rid, facts)
                    assert any(f.get('stage')=='upstream_attempt' and f.get('trace_detail',{}).get('attempt_status_code')==400 for f in facts), (rid, facts)
                encoded = json.dumps(server.facts)+(root/'log').read_text()
                for secret in ['gAAAAopaque-do-not-log', 'c2c:opaque-do-not-log', 'fixture-upstream-secret', 'private-history', 'private-result']:
                    assert secret not in encoded, secret
                assert (root/'api.json').read_bytes() == config_bytes
                kind = 'identified encrypted compaction' if identified_compaction else 'compaction state' if compaction else 'encrypted content'
                print(f'PASS {kind} fallback engine={engine}: {count} HTTP cases; raw400/routing502, no cooldown, unchanged wire')
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                server.shutdown(); server.server_close()


if __name__ == '__main__':
    for engine in ['gpt', 'codex']:
        verify(Path(sys.argv[1]).resolve(), engine)
        verify(Path(sys.argv[1]).resolve(), engine, compaction=True)
        verify(Path(sys.argv[1]).resolve(), engine, identified_compaction=True)
