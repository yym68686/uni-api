"""Synthetic Gemini tool identity/parallel-history contracts; loopback only."""
import base64
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

SIG = 'fixture-real-opaque-signature'
ENDPOINTS = ['/v1/chat/completions', '/v1/responses']


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append(body)
        if self.server.mode == 'error':
            value = {'error': {'message': 'fixture rejection', 'type': 'invalid_request_error',
                               'code': 'specific_fixture_code', 'param': 'contents[1]'}}
            status = 400
        elif self.server.mode == 'calls':
            value = {'candidates': [{'content': {'parts': [
                {'functionCall': {'id': f'native-{i}', 'name': 'read', 'args': {'path': f'{i}.md'}},
                 **({'thoughtSignature': SIG + str(i)} if i != 1 else {})}
                for i in range(3)]}, 'finishReason': 'STOP'}]}
            status = 200
        else:
            value = {'candidates': [{'content': {'parts': [{'text': 'fixture complete'}]}, 'finishReason': 'STOP'}]}
            status = 200
        streaming = ':streamGenerateContent' in self.path and status == 200
        values = [value]
        if streaming and self.server.mode == 'calls':
            # A native-ID call can receive its signature in a later frame.
            parts = value['candidates'][0]['content']['parts']
            first = copy.deepcopy(parts[0]); first.pop('thoughtSignature')
            first['functionCall']['args'] = {}
            values = [{'candidates': [{'content': {'parts': [first]}}]}, value]
        if streaming and self.server.mode == 'postcommit':
            values = [{'candidates': [{'content': {'parts': [{'text': 'partial text'}]}}]},
                      {'error': {'message': 'fixture rejection', 'type': 'invalid_request_error',
                                 'code': 'specific_fixture_code', 'param': 'contents[1]'}}]
        raw = (''.join('data: ' + json.dumps(v) + '\n\n' for v in values)
               if streaming else json.dumps(value)).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/event-stream' if streaming else 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        # Exercise byte fragmentation as well as logical SSE frame boundaries.
        for pos in range(0, len(raw), 17): self.wfile.write(raw[pos:pos+17])


def calls_from(raw, endpoint, stream):
    if not stream:
        value = json.loads(raw)
        if endpoint.endswith('responses'):
            return [v for v in value['output'] if v['type'] == 'function_call']
        return value['choices'][0]['message']['tool_calls']
    events = [json.loads(line[6:]) for line in raw.decode().splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    if endpoint.endswith('responses'):
        final = next(v['response'] for v in events if v.get('type') == 'response.completed')
        return [v for v in final['output'] if v['type'] == 'function_call']
    out = {}
    for event in events:
        for choice in event.get('choices', []):
            for call in choice.get('delta', {}).get('tool_calls', []):
                target = out.setdefault(call['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if call.get('id'):
                    assert not target['id'], 'tool ID must be emitted exactly once'
                    target['id'] = call['id']
                for field in ['name', 'arguments']:
                    target['function'][field] += call.get('function', {}).get(field, '')
    return list(out.values())


def history(endpoint, calls, reverse=True):
    start = {'role': 'user', 'content': 'read fixtures'}
    if endpoint.endswith('responses'):
        results = [{'type': 'function_call_output', 'call_id': c['call_id'], 'output': f'result-{i}'}
                   for i, c in enumerate(calls)]
        return {'input': [start, *calls, *(results[::-1] if reverse else results)]}
    results = [{'role': 'tool', 'tool_call_id': c['id'], 'content': f'result-{i}'} for i, c in enumerate(calls)]
    return {'messages': [start, {'role': 'assistant', 'content': '', 'tool_calls': calls},
                         *(results[::-1] if reverse else results)]}


def verify(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits, server.mode = [], 'calls'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cases = 0
    with tempfile.TemporaryDirectory(prefix='gemini-tool-history-') as directory:
        root, port = Path(directory), free_port()
        config = {'providers': [{'provider': 'fixture', 'engine': 'gemini',
                                'base_url': f'http://127.0.0.1:{server.server_port}/v1beta',
                                'api': 'fixture', 'model': ['gemini-fixture']}],
                  'api_keys': [{'api': 'fixture-client', 'model': ['all'], 'preferences': {'AUTO_RETRY': False}}]}
        config['providers'].append({'provider': 'vertex-fixture', 'engine': 'vertex',
                                    'base_url': f'http://127.0.0.1:{server.server_port}',
                                    'api': ['ab.fixture'], 'model': ['vertex-fixture']})
        (root / 'config.json').write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('FACTS_', 'UNI_API_CONTROL_RESTORE_', 'OTEL_', 'FUGUE_OTEL_')) and k != 'CONFIG_URL'}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root / 'config.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / 'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / 'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / 'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', NO_PROXY='127.0.0.1,localhost')

        def call(endpoint, body=None):
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            try:
                conn.request('GET' if body is None else 'POST', endpoint,
                             None if body is None else json.dumps({'model': 'gemini-fixture', **body}),
                             {'Authorization': 'Bearer fixture-client', 'Content-Type': 'application/json'})
                response = conn.getresponse()
                return response.status, response.read()
            finally: conn.close()

        with (root / 'log').open('w') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call('/healthz')[0] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('fixture gateway did not become healthy')
                for endpoint in ENDPOINTS:
                    collection = 'input' if endpoint.endswith('responses') else 'messages'
                    for stream in [False, True]:
                        server.mode = 'calls'
                        status, raw = call(endpoint, {'stream': stream, collection: [{'role': 'user', 'content': 'start'}]})
                        assert status == 200, (status, raw)
                        calls = calls_from(raw, endpoint, stream)
                        assert len(calls) == 3, calls
                        original = history(endpoint, calls)
                        server.mode = 'text'
                        status, raw = call(endpoint, {**original, 'stream': stream})
                        assert status == 200, (status, raw)
                        wire = server.hits[-1]['contents']
                        assert len(wire) == 3, wire
                        assert [len(c['parts']) for c in wire] == [1, 3, 3], wire
                        for i in range(3):
                            part = wire[1]['parts'][i]
                            assert part['functionCall'] == {'id': f'native-{i}', 'name': 'read', 'args': {'path': f'{i}.md'}}, part
                            assert part.get('thoughtSignature') == (SIG + str(i) if i != 1 else None), part
                            assert wire[2]['parts'][i]['functionResponse'] == {
                                'id': f'native-{i}', 'name': 'read', 'response': {'result': f'result-{i}'}}, wire
                        assert history(endpoint, calls) == original
                        cases += 1
                        # Multiple turns remain separate; reverse results are reordered only inside each batch.
                        multi = copy.deepcopy(original)
                        multi[collection] += [{'role': 'user', 'content': 'next round'}, *copy.deepcopy(original[collection][1:])]
                        assert call(endpoint, {**multi, 'stream': stream})[0] == 200
                        multiwire = server.hits[-1]['contents']
                        assert [len(c['parts']) for c in multiwire] == [1, 3, 3, 1, 3, 3], multiwire
                        assert multiwire[:3] == wire and multiwire[4:] == wire[1:]
                        cases += 1
                        for content in [None, [{'type': 'text', 'text': ''}], ' ', 'visible']:
                            if collection == 'messages':
                                changed = copy.deepcopy(original); changed[collection][1]['content'] = content
                            else:
                                continue
                            assert call(endpoint, {**changed, 'stream': stream, 'model': 'vertex-fixture'})[0] == 200
                            parts = server.hits[-1]['contents'][1]['parts']
                            assert len(parts) == (4 if isinstance(content, str) else 3), parts
                            if isinstance(content, str): assert parts[0] == {'text': content}
                            cases += 1
                        for mutation in ['missing', 'duplicate', 'orphan', 'interrupted']:
                            bad = copy.deepcopy(original)
                            if mutation == 'missing': bad[collection].pop()
                            elif mutation == 'duplicate': bad[collection].append(copy.deepcopy(bad[collection][-1]))
                            elif mutation == 'orphan':
                                bad[collection][-1]['call_id' if collection == 'input' else 'tool_call_id'] = 'orphan'
                            else: bad[collection].insert(-1, {'role': 'user', 'content': 'do not cross this turn'})
                            before = len(server.hits)
                            status, raw = call(endpoint, {**bad, 'stream': stream})
                            assert status == 400, (mutation, status, raw)
                            assert len(server.hits) == before, mutation
                            assert b'gemini_tool_history' in raw, raw
                            err = json.loads(raw)['error']
                            assert err['code'] == 'gemini_tool_history' and err['param'].startswith('messages'), err
                            cases += 1
                        server.mode = 'error'
                        status, raw = call(endpoint, {collection: [{'role': 'user', 'content': 'start'}]})
                        assert status == 400 and json.loads(raw)['error']['param'] == 'contents[1]', raw
                        cases += 1
                        server.mode = 'postcommit'
                        before = len(server.hits)
                        status, raw = call(endpoint, {'stream': True, collection: [{'role': 'user', 'content': 'start'}]})
                        assert status == 200 and b'partial text' in raw and b'specific_fixture_code' in raw and b'contents[1]' in raw, raw
                        assert len(server.hits) == before + 1
                        if collection == 'input': assert b'response.failed' in raw and b'response.completed' not in raw
                        cases += 1
                # Legacy IDs remain readable; unsigned IDs must not become fabricated signatures.
                for cid in ['call_1', 'call_' + base64.urlsafe_b64encode(SIG.encode()).decode().rstrip('=') + '.1']:
                    c = {'id': cid, 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}
                    server.mode = 'text'
                    assert call(ENDPOINTS[0], history(ENDPOINTS[0], [c]))[0] == 200
                    assert server.hits[-1]['contents'][1]['parts'][0].get('thoughtSignature') == (None if cid == 'call_1' else SIG)
                    cases += 1
                for value in ['', 'data:image/png;base64,', 'data:image/png;base64,not base64']:
                    before = len(server.hits)
                    status, raw = call(ENDPOINTS[0], {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': value}}]}]})
                    assert status == 400 and len(server.hits) == before, (value, status, raw)
                    cases += 1
                for endpoint in ENDPOINTS:
                    cid = 'explicit-signature-call'
                    c = {'id': cid, 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'},
                         'extra_content': {'google': {'thought_signature': SIG}}}
                    if endpoint.endswith('responses'):
                        c = {'type': 'function_call', 'call_id': cid, 'name': 'read', 'arguments': '{}',
                             'extra_content': c['extra_content']}
                    server.mode = 'text'
                    assert call(endpoint, history(endpoint, [c]))[0] == 200
                    assert server.hits[-1]['contents'][1]['parts'][0]['thoughtSignature'] == SIG
                    assert not any('inlineData' in p for turn in server.hits[-1]['contents'] for p in turn['parts'])
                    cases += 1
                print(f'PASS Gemini tool-history HTTP contracts: {cases} cases (local mock only)')
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                server.shutdown(); server.server_close()


if __name__ == '__main__': verify(Path(sys.argv[1]).resolve())
