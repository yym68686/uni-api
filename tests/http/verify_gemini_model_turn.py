"""Gemini continuation compatibility at the HTTP boundary; loopback only."""
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
from verify_dispatch_timing import free_port


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append((self.path, body))
        bad = body['contents'][-1]['role'] == 'model'
        value = {'error': {'code': 400, 'message': 'Requests ending with a model turn are not supported.', 'status': 'INVALID_ARGUMENT'}} if bad else {
            'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'fixture complete'}]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 2, 'totalTokenCount': 12},
        }
        stream = ':streamGenerateContent' in self.path and not bad
        raw = ('data: ' + json.dumps(value) + '\n\n').encode() if stream else json.dumps(value).encode()
        self.send_response(400 if bad else 200)
        self.send_header('Content-Type', 'text/event-stream' if stream else 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def verify(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cases = 0
    with tempfile.TemporaryDirectory(prefix='gemini-model-turn-') as directory:
        root = Path(directory)
        port = free_port()
        base = f'http://127.0.0.1:{server.server_port}'
        providers = [
            {'provider': 'gemini', 'engine': 'gemini', 'base_url': base + '/v1beta', 'api': 'fixture', 'model': [{'gemini-4-argon': 'gemini'}]},
            {'provider': 'vertex', 'engine': 'vertex', 'base_url': base, 'api': ['ab.fixture'], 'model': [{'gemini-4-argon': 'vertex'}]},
            {'provider': 'override', 'engine': 'gemini', 'base_url': base + '/v1beta', 'api': 'fixture', 'model': ['override'],
             'preferences': {'post_body_parameter_overrides': {'contents': [{'role': 'model', 'parts': [{'text': 'explicit override'}]}]}}},
        ]
        config = {'providers': providers, 'api_keys': [{'api': 'fixture-client', 'model': ['all'], 'preferences': {'AUTO_RETRY': False}}]}
        (root / 'config.json').write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('FACTS_', 'UNI_API_CONTROL_RESTORE_', 'OTEL_', 'FUGUE_OTEL_')) and k != 'CONFIG_URL'}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root / 'config.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / 'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / 'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / 'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', NO_PROXY='127.0.0.1,localhost', PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD='1000')

        def call(path, body=None):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            try:
                conn.request('GET' if body is None else 'POST', path, None if body is None else json.dumps(body),
                             {'Authorization': 'Bearer fixture-client', 'Content-Type': 'application/json'})
                response = conn.getresponse()
                return response.status, response.read()
            finally:
                conn.close()

        with (root / 'log').open('w') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call('/healthz')[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError('fixture gateway did not become healthy')

                for model in ['gemini', 'vertex']:
                    for streaming in [False, True]:
                        for endpoint in ['/v1/chat/completions', '/v1/responses']:
                            chat = endpoint.endswith('completions')
                            tail = {'role': 'assistant', 'content': 'I will continue.'} if chat else {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'I will continue.'}]}
                            start = {'role': 'user', 'content': 'start'}
                            fields = {'model': model, 'stream': streaming, 'temperature': .2}
                            fields['max_tokens' if chat else 'max_output_tokens'] = 77
                            collection = 'messages' if chat else 'input'
                            for tool_history in [False, True]:
                                items = [start]
                                if tool_history:
                                    if chat:
                                        items += [
                                            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': i, 'type': 'function', 'function': {'name': n, 'arguments': '{"x":1}'}} for i, n in [('a', 'one'), ('b', 'two')]]},
                                            {'role': 'tool', 'tool_call_id': 'b', 'content': 'result b'},
                                            {'role': 'tool', 'tool_call_id': 'a', 'content': 'result a'},
                                        ]
                                    else:
                                        items += [
                                            {'type': 'function_call', 'call_id': i, 'name': n, 'arguments': '{"x":1}'} for i, n in [('a', 'one'), ('b', 'two')]]
                                        items += [{'type': 'function_call_output', 'call_id': i, 'output': 'result ' + i} for i in ['b', 'a']]
                                body = dict(fields, **{collection: items + [tail]})
                                before = len(server.hits)
                                status, raw = call(endpoint, body)
                                assert status == 200, (endpoint, model, status, raw)
                                assert len(server.hits) == before + 1
                                path, wire = server.hits[-1]
                                assert '/models/gemini-4-argon:' in path, path
                                assert wire['contents'][-2] == {'role': 'model', 'parts': [{'text': 'I will continue.'}]}
                                assert wire['contents'][-1] == {'role': 'user', 'parts': [{'text': 'Continue.'}]}
                                assert wire['generationConfig']['maxOutputTokens'] == 77
                                assert wire['generationConfig']['temperature'] == .2
                                parts = [p for c in wire['contents'] for p in c['parts']]
                                assert sum('functionCall' in p for p in parts) == (2 if tool_history else 0)
                                assert sum('functionResponse' in p for p in parts) == (2 if tool_history else 0)
                                if not chat:
                                    if streaming:
                                        assert b'response.completed' in raw
                                    else:
                                        assert json.loads(raw)['status'] == 'completed'
                                # Equivalent explicit user continuation must produce exactly the same wire body.
                                explicit = copy.deepcopy(body)
                                explicit[collection].append({'role': 'user', 'content': 'Continue.'})
                                assert call(endpoint, explicit)[0] == 200
                                assert server.hits[-1][1] == wire
                                cases += 2

                            # An unfinished batch followed by text must not be made successful by a fabricated user turn.
                            call_item = {'role': 'assistant', 'tool_calls': [{'id': 'pending', 'type': 'function', 'function': {'name': 'one', 'arguments': '{}'}}]} if chat else {'type': 'function_call', 'call_id': 'pending', 'name': 'one', 'arguments': '{}'}
                            bad = dict(fields, **{collection: [start, call_item, tail]})
                            status, _ = call(endpoint, bad)
                            assert status == 400
                            assert server.hits[-1][1]['contents'][-1]['role'] == 'model'
                            cases += 1

                        # Native Gemini requests bypass the conversion and retain signatures and last role.
                        native = {'contents': [{'role': 'model', 'parts': [{'text': 'native', 'thoughtSignature': 'native-signature'}]}],
                                  'generationConfig': {'maxOutputTokens': 77}}
                        action = 'streamGenerateContent?alt=sse' if streaming else 'generateContent'
                        status, _ = call(f'/v1beta/models/{model}:{action}', native)
                        assert status == 400
                        assert server.hits[-1][1] == native
                        cases += 1

                # Explicit wire overrides retain their authority even when they end in a model turn.
                status, _ = call('/v1/responses', {'model': 'override', 'input': [{'role': 'assistant', 'content': 'continue'}]})
                assert status == 400
                assert server.hits[-1][1]['contents'] == [{'role': 'model', 'parts': [{'text': 'explicit override'}]}]
                cases += 1
                print(f'PASS Gemini model-turn HTTP contracts: {cases} cases')
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    verify(Path(sys.argv[1]).resolve())
