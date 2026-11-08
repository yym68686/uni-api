"""Gemini schema conversion, protocol isolation and unchanged retry bodies; loopback only."""
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

LITERAL = {'$schema': 'data', 'const': 'data', 'default': 1, 'additionalProperties': False}
SCHEMA = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object',
          'properties': {'mode': {'anyOf': [{'type': 'string', 'const': 'off'}, {'type': 'string', 'const': 'worktree'}]},
                         '$schema': {'type': 'string'}, 'const': {'type': 'string', 'const': 'reserved'},
                         'default': {'type': 'boolean'}, 'additionalProperties': {'type': 'boolean'},
                         'args': {'description': 'Arbitrary JSON'}, 'data': {'enum': [LITERAL], 'examples': [LITERAL]}}}
NORMALIZED = copy.deepcopy(SCHEMA)
NORMALIZED.pop('$schema')
NORMALIZED['properties']['mode']['anyOf'] = [{'type': 'string', 'enum': ['off']}, {'type': 'string', 'enum': ['worktree']}]
NORMALIZED['properties']['const'] = {'type': 'string', 'enum': ['reserved']}
GEMINI = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'fixture complete'}]}, 'finishReason': 'STOP'}],
          'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12}}
ERROR = {'error': {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'fixture schema rejection'}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        channel = self.path.split('/')[1]
        self.server.hits.append((channel, body))
        if channel == 'reject':
            status, frames, stream = 400, [ERROR], False
        elif '/models/' in self.path:
            status, frames, stream = 200, [GEMINI], ':streamGenerateContent' in self.path
        elif self.path.endswith('/responses'):
            value = {'id': 'resp_fixture', 'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
                     'content': [{'type': 'output_text', 'text': 'fixture complete'}]}]}
            status, stream = 200, body.get('stream', False)
            frames = [{'type': 'response.output_text.delta', 'delta': 'fixture complete'}, {'type': 'response.completed', 'response': value}] if stream else [value]
        else:
            status, stream = 200, body.get('stream', False)
            frames = [{'choices': [{'index': 0, 'delta': {'content': 'fixture complete'}, 'finish_reason': None}]},
                      {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}] if stream else [
                      {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'fixture complete'}, 'finish_reason': 'stop'}]}]
        raw = ''.join('data: '+json.dumps(f)+'\n\n' for f in frames).encode() if stream else json.dumps(frames[0]).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/event-stream' if stream else 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def request(endpoint, stream):
    common = {'model': 'gemini-test', 'stream': stream}
    if endpoint == '/v1/responses':
        return {**common, 'input': [{'role': 'user', 'content': 'start'},
                {'type': 'function_call', 'call_id': 'call-fixture', 'name': 'lookup', 'arguments': json.dumps(LITERAL)},
                {'type': 'function_call_output', 'call_id': 'call-fixture', 'output': json.dumps(LITERAL)}],
                'tools': [{'type': 'function', 'name': 'lookup', 'parameters': copy.deepcopy(SCHEMA)}]}
    return {**common, 'messages': [{'role': 'user', 'content': 'start'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'id': 'call-fixture',
             'function': {'name': 'lookup', 'arguments': json.dumps(LITERAL)}}]},
            {'role': 'tool', 'tool_call_id': 'call-fixture', 'content': json.dumps(LITERAL)}],
            'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': copy.deepcopy(SCHEMA)}}]}


def verify(binary):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='gemini-schema-') as td:
        root, port = Path(td), free_port()
        base = f'http://127.0.0.1:{upstream.server_port}'
        providers = [{'provider': name, 'engine': 'gemini', 'base_url': base+'/'+name+'/v1beta',
                      'api': 'fixture-upstream', 'model': ['gemini-test']} for name in ['gemini', 'reject', 'fallback', 'override']]
        providers += [{'provider': 'vertex', 'engine': 'vertex', 'base_url': base+'/vertex', 'api': ['ab.fixture'], 'model': ['gemini-test']},
                      {'provider': 'gpt', 'engine': 'gpt', 'base_url': base+'/gpt/v1/chat/completions', 'api': 'fixture-upstream', 'model': ['gemini-test']},
                      {'provider': 'responses', 'engine': 'gpt', 'base_url': base+'/responses/v1/responses', 'api': 'fixture-upstream', 'model': ['gemini-test']}]
        providers[3]['preferences'] = {'post_body_parameter_overrides': {'tools': [{'functionDeclarations': [{'name': 'lookup', 'parameters': SCHEMA}]}]}}
        keys = [{'api': name+'-key', 'model': [name+'/*'], 'preferences': {'AUTO_RETRY': False}} for name in ['gemini', 'vertex', 'gpt', 'responses', 'override']]
        keys.append({'api': 'retry-key', 'model': ['reject/*', 'fallback/*'], 'preferences': {'AUTO_RETRY': False}})
        (root/'config.json').write_text(json.dumps({'providers': providers, 'api_keys': keys, 'preferences': {'unknown_upstream_400_retries': 1, 'hedging': {'enabled': False}}}))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', NO_PROXY='127.0.0.1,localhost')
        def call(path='/healthz', body=None, key='gemini-key'):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
            try:
                conn.request('GET' if body is None else 'POST', path, None if body is None else json.dumps(body),
                             {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
                res = conn.getresponse(); return res.status, res.read()
            finally: conn.close()
        count = 0
        with (root/'log').open('w') as log:
            proc = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[0] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('fixture startup failed')
                for endpoint in ['/v1/chat/completions', '/v1/responses']:
                    for stream in [False, True]:
                        body = request(endpoint, stream)
                        for key in ['gemini-key', 'vertex-key', 'retry-key']:
                            upstream.hits = []
                            status, raw = call(endpoint, body, key)
                            assert status == 200 and b'fixture complete' in raw, (status, raw)
                            assert len(upstream.hits) == (2 if key == 'retry-key' else 1)
                            for _, wire in upstream.hits:
                                assert wire['tools'][0]['functionDeclarations'][0]['parameters'] == NORMALIZED
                                assert wire['contents'][1]['parts'][0]['functionCall']['args'] == LITERAL
                                assert wire['contents'][2]['parts'][0]['functionResponse']['response']['result'] == json.dumps(LITERAL)
                            if key == 'retry-key':
                                assert [name for name, _ in upstream.hits] == ['reject', 'fallback']
                                assert upstream.hits[0][1] == upstream.hits[1][1]
                            count += 1
                        key = 'responses-key' if endpoint == '/v1/responses' else 'gpt-key'
                        upstream.hits = []
                        status, raw = call(endpoint, body, key)
                        assert status == 200 and len(upstream.hits) == 1, (status, raw)
                        assert upstream.hits[0][1]['tools'] == body['tools']
                        assert body == request(endpoint, stream)
                        count += 1
                        upstream.hits = []
                        status, raw = call(endpoint, body, 'override-key')
                        assert status == 200 and len(upstream.hits) == 1, (status, raw)
                        assert upstream.hits[0][1]['tools'][0]['functionDeclarations'][0]['parameters'] == SCHEMA
                        count += 1
                native = {'contents': [{'role': 'user', 'parts': [{'text': 'start'}]}],
                          'tools': [{'functionDeclarations': [{'name': 'lookup', 'parameters': SCHEMA}]}]}
                for key in ['gemini-key', 'vertex-key']:
                    for action in ['generateContent', 'streamGenerateContent?alt=sse']:
                        upstream.hits = []
                        status, raw = call('/v1beta/models/gemini-test:'+action, native, key)
                        assert status == 200 and len(upstream.hits) == 1, (status, raw)
                        assert upstream.hits[0][1] == native
                        count += 1
                print(f'PASS Gemini schema: {count} HTTP cases; Chat/Responses conversion, native/other engines/overrides preserved, identical retry bodies')
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                upstream.shutdown(); upstream.server_close()


if __name__ == '__main__': verify(Path(sys.argv[1]).resolve())
