"""Empty assistant-tool text normalization and exact Gemini data fallback; loopback only."""
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

MESSAGE = "* ***.contents[1].parts[0].data: required oneof field 'data' must have one initialized field\n"
ERROR = {'error': {'type': 'upstream_error', 'code': 400, 'param': '', 'message': MESSAGE}}
NATIVE = {'contents': [
    {'role': 'user', 'parts': [{'text': 'start'}]},
    {'role': 'model', 'parts': [{'text': ''}, {'functionCall': {'name': 'lookup', 'args': {'text': ''}}, 'thoughtSignature': 'opaque-signature'}]},
    {'role': 'user', 'parts': [{'functionResponse': {'name': 'lookup', 'response': {'result': ''}}}]}],
    'generationConfig': {'maxOutputTokens': 77, 'temperature': .2},
    'tools': [{'functionDeclarations': [{'name': 'lookup', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}]}]}
RESPONSE = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'fixture complete'}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        channel = self.path.split('/')[1]
        self.server.hits.append((channel, body))
        empty = any(p == {'text': ''} for c in body.get('contents', []) if c.get('role') == 'model' for p in c.get('parts', []))
        reject = channel != 'fallback' and (self.server.mode == 'reject' or (self.server.mode == 'strict' and empty))
        stream = ':streamGenerateContent' in self.path
        value = self.server.error if reject else RESPONSE
        status = 400 if reject else 200
        if self.server.mode == 'postcommit' and channel != 'fallback':
            partial = copy.deepcopy(RESPONSE); partial['candidates'][0].pop('finishReason')
            events = [partial, self.server.error]
            raw = ''.join('data: ' + json.dumps(v) + '\n\n' for v in events).encode()
        elif stream and not reject:
            raw = ('data: ' + json.dumps(value) + '\n\n').encode()
        else:
            raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/event-stream' if stream and not reject else 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def body_for(endpoint, stream):
    common = {'model': 'gemini-test', 'stream': stream, 'temperature': .2}
    if endpoint == '/v1/chat/completions':
        return {**common, 'max_tokens': 77, 'messages': [
            {'role': 'user', 'content': 'start'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'id': 'call-1', 'function': {'name': 'lookup', 'arguments': '{"text":""}'}}]},
            {'role': 'tool', 'tool_call_id': 'call-1', 'content': ''}],
            'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}}]}
    if endpoint == '/v1/responses':
        return {**common, 'max_output_tokens': 77, 'input': [
            {'role': 'user', 'content': 'start'},
            {'type': 'function_call', 'call_id': 'call-1', 'name': 'lookup', 'arguments': '{"text":""}'},
            {'type': 'function_call_output', 'call_id': 'call-1', 'output': ''}],
            'tools': [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}]}
    return {**common, 'max_tokens': 77, 'messages': [
        {'role': 'user', 'content': 'start'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': ''}, {'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {'text': ''}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': ''}]}],
        'tools': [{'name': 'lookup', 'input_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}]}


def verify(binary):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream.hits, upstream.facts, upstream.error, upstream.mode = [], [], ERROR, 'strict'
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='gemini-empty-tool-') as td:
        root, port = Path(td), free_port()
        base = f'http://127.0.0.1:{upstream.server_port}'
        providers = [{'provider': n, 'engine': 'gemini', 'base_url': base + '/' + n + '/v1beta', 'api': 'fixture-upstream',
                      'model': ['gemini-test'], 'preferences': {'cooldown_period': 300}} for n in ['first', 'fallback', 'also-reject']]
        providers += [{**providers[0], 'provider': 'vertex', 'engine': 'vertex', 'base_url': base + '/vertex', 'api': ['ab.fixture']},
                      {**providers[0], 'provider': 'gpt', 'engine': 'gpt', 'base_url': base + '/gpt/v1/chat/completions'},
                      {**providers[0], 'provider': 'override', 'base_url': base + '/override/v1beta',
                       'preferences': {'post_body_parameter_overrides': {'contents': NATIVE['contents']}}}]
        keys = [{'api': 'retry', 'model': ['first/*', 'fallback/*']},
                {'api': 'no-retry', 'model': ['first/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': False}},
                {'api': 'exhausted', 'model': ['first/*', 'also-reject/*'], 'preferences': {'AUTO_RETRY': 1}}]
        keys += [{'api': n+'-key', 'model': [n+'/*'], 'preferences': {'AUTO_RETRY': False}} for n in ['vertex', 'gpt', 'override']]
        (root/'config.json').write_text(json.dumps({'providers': providers, 'api_keys': keys, 'preferences': {'hedging': {'enabled': False}}}))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   FACTS_INGEST_URL=base+'/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'), NO_PROXY='127.0.0.1,localhost')
        def call(path='/healthz', body=None, key='retry', rid='fixture'):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
            try:
                conn.request('GET' if body is None else 'POST', path, None if body is None else json.dumps(body),
                             {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json', 'X-Request-ID': rid})
                res = conn.getresponse(); return res.status, res.read()
            finally: conn.close()
        count, billed_rids = 0, []
        with (root/'log').open('w') as log:
            proc = subprocess.Popen([str(binary)], env=env, cwd=root, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[0] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('fixture did not become ready: ' + (root/'log').read_text()[-3000:])
                for endpoint in ['/v1/chat/completions', '/v1/responses']:
                    for stream in [False, True]:
                        body = body_for(endpoint, stream)
                        for key in ['retry', 'vertex-key']:
                            upstream.mode, upstream.hits = 'strict', []
                            status, raw = call(endpoint, body, key, f'clean-{count}')
                            assert status == 200 and len(upstream.hits) == 1, (endpoint, key, status, raw)
                            wire = upstream.hits[0][1]
                            assert 'contents' in wire, (endpoint, key, upstream.hits)
                            parts = wire['contents'][1]['parts']
                            assert parts == [{'functionCall': {'id': 'call-1', 'name': 'lookup', 'args': {'text': ''}}}], parts
                            assert wire['generationConfig']['maxOutputTokens'] == 77
                            assert wire['generationConfig']['temperature'] == .2
                            assert wire['contents'][2]['parts'][0]['functionResponse']['response']['result'] == ''
                            count += 1
                        for key, want, names in [('retry', 200, ['first','fallback']), ('no-retry', 502, ['first']),
                                                 ('exhausted', 502, ['first','also-reject']), ('retry', 200, ['first','fallback'])]:
                            upstream.mode, upstream.hits = 'reject', []
                            rid = f'fallback-{count}'; status, raw = call(endpoint, body, key, rid)
                            assert status == want, (endpoint, key, status, raw)
                            assert [c for c,_ in upstream.hits] == names, upstream.hits
                            assert all(w == upstream.hits[0][1] for _,w in upstream.hits), 'body changed on retry'
                            if want == 200: billed_rids.append(rid)
                            count += 1
                        upstream.mode, upstream.hits = 'postcommit', []
                        if stream:
                            status, raw = call(endpoint, body, rid=f'post-output-{count}')
                            assert status == 200 and len(upstream.hits) == 1 and b'fixture complete' in raw, (status, raw, upstream.hits)
                            count += 1
                # Native Gemini retains its payload and signature, but can use the exact fallback.
                for stream in [False, True]:
                    action = 'streamGenerateContent?alt=sse' if stream else 'generateContent'
                    path = '/v1beta/models/gemini-test:'+action
                    upstream.mode, upstream.hits = 'reject', []
                    status, raw = call(path, NATIVE, rid=f'native-{count}')
                    assert status == 200 and [n for n,_ in upstream.hits] == ['first','fallback'], (status,raw)
                    assert all(w == NATIVE for _,w in upstream.hits)
                    count += 1
                # Explicit wire overrides remain authoritative, and other engines retain their text.
                upstream.mode, upstream.hits = 'accept', []
                status, _ = call('/v1/chat/completions', body_for('/v1/chat/completions', False), 'override-key')
                assert status == 200 and upstream.hits[0][1]['contents'] == NATIVE['contents']
                count += 1
                upstream.mode, upstream.hits = 'reject', []
                status, _ = call('/v1/chat/completions', body_for('/v1/chat/completions', False), 'gpt-key')
                assert status == 400 and len(upstream.hits) == 1 and upstream.hits[0][1]['messages'][1]['content'] == ''
                count += 1
                upstream.hits = []
                status, _ = call('/v1/messages', body_for('/v1/messages', False))
                assert status == 400 and len(upstream.hits) == 1 and 'contents' not in upstream.hits[0][1]
                count += 1
                for patch in [{'code':'400'}, {'code':'invalid_type'}, {'type':'invalid_request_error'}, {'param':'contents'},
                              {'message':MESSAGE+'invalid arguments'}, {'message':MESSAGE.replace('contents[1]','contents[99]')},
                              {'message':MESSAGE.replace('contents[1]','contents[0]')}]:
                    upstream.error, upstream.hits = {'error': {**ERROR['error'], **patch}}, []
                    status, _ = call('/v1/chat/completions', body_for('/v1/chat/completions', False), rid=f'negative-{count}')
                    assert status == 400 and len(upstream.hits) == 1
                    count += 1
                upstream.error = ERROR
                for part in [{}, {'thoughtSignature':'opaque-signature'}, {'text': None}, {'text':''},
                             {'functionCall':{'name':'lookup','args':[]}}]:
                    native = copy.deepcopy(NATIVE); native['contents'][1]['parts'] = [part]
                    upstream.hits = []
                    status, _ = call('/v1beta/models/gemini-test:generateContent', native, rid=f'malformed-{count}')
                    assert status == 400 and len(upstream.hits) == 1
                    count += 1
                deadline = time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='billing' and f.get('request_id')==rid for f in upstream.facts) for rid in billed_rids): break
                    time.sleep(.1)
                for rid in billed_rids:
                    facts = [f for f in upstream.facts if f.get('request_id')==rid]
                    assert any(f.get('kind')=='billing' and f.get('status')==400 for f in facts), (rid, facts)
                    assert any(f.get('stage')=='upstream_attempt' and f.get('trace_detail',{}).get('attempt_status_code')==400 for f in facts)
                    assert any(f.get('stage')=='gemini_tool_part_fallback' and f.get('trace_detail',{}).get('routing_status_code')==502 for f in facts)
                    assert any(f.get('kind')=='billing' and f.get('status')==200 for f in facts)
                encoded = json.dumps(upstream.facts)
                for secret in ['opaque-signature', 'fixture-upstream']: assert secret not in encoded
                print(f'PASS Gemini empty tool text: {count} HTTP cases, raw400/routing502, payload/signatures preserved on fallback')
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                upstream.shutdown(); upstream.server_close()


if __name__ == '__main__': verify(Path(sys.argv[1]).resolve())
