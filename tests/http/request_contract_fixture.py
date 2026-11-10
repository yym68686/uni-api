"""Read-only architecture audit: shared request contract, synthetic loopback peers."""
import contextlib
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tempfile

SOURCE = Path(__file__).resolve().parents[2]
BINARY = Path(sys.argv[1]).resolve()
TEMP = tempfile.TemporaryDirectory(prefix='uni-request-contract-')
ROOT = Path(TEMP.name)
sys.path.insert(0, str(SOURCE / 'tests/http'))
from fact_sink import accept_facts
from verify_dispatch_timing import free_port

os.umask(0o077)
CASES = ['ok', 'incomplete', 'done-only', 'tool-start-fail', 'rules', 'codex-fields', 'missing-item']


def event(kind, **fields):
    return (f'event: {kind}\ndata: ' + json.dumps(dict(type=kind, **fields)) + '\n\n').encode()


def completed(role):
    return {'id': 'resp_' + role, 'object': 'response', 'status': 'completed',
            'output': [{'id': 'msg_' + role, 'type': 'message', 'role': 'assistant',
                        'status': 'completed', 'content': [{'type': 'output_text', 'text': 'OK'}]}],
            'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path == '/facts':
            return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        case = body['model']
        role = self.path.split('/')[1]
        self.server.hits.append({'case': case, 'role': role, 'body': body, 'headers': dict(self.headers),
                                 'header_names': sorted(self.headers.keys())})
        hook = getattr(self.server, 'on_request', None)
        if hook: hook(role, case)
        if case == 'missing-item' and role == 'first':
            raw = json.dumps({'error': {'code': None, 'type': 'invalid_request_error', 'param': 'input',
                'message': "Item with id 'rs_fixture' not found. Items are not persisted when `store` is set to false. Try again with `store` set to true, or remove this item from your input."}}).encode()
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if case == 'omni-moderation-latest':
            chunks, kind = [json.dumps({'results': [{'flagged': False}]}).encode()], 'application/json'
        elif not body.get('stream'):
            payload = ({'id': 'chat_'+role, 'object': 'chat.completion', 'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}]}
                if self.path.endswith('/chat/completions') else completed(role))
            chunks, kind = [json.dumps(payload).encode()], 'application/json'
        else:
            created = event('response.created', response={'id': 'resp_' + role, 'status': 'in_progress', 'output': []})
            delta = event('response.output_text.delta', item_id='msg_' + role, output_index=0, content_index=0, delta='OK')
            failure = event('response.failed', response={'id': 'resp_' + role, 'status': 'failed',
                'error': {'code': 'server_error', 'message': 'synthetic capacity failure'}})
            final = event('response.completed', response=completed(role))
            if role == 'first' and case == 'incomplete':
                chunks = [created, event('response.incomplete', response={'id': 'resp_first',
                    'status': 'incomplete', 'output': [], 'incomplete_details': {'reason': 'max_output_tokens'}})]
            elif role == 'first' and case == 'done-only':
                chunks = [created, b'data: [DONE]\n\n']
            elif role == 'first' and case == 'tool-start-fail':
                chunks = [created, event('response.output_item.added', output_index=0,
                    item={'id': 'fc_first', 'type': 'function_call', 'name': 'fixture',
                          'call_id': 'call_fixture', 'arguments': ''}), failure]
            else:
                chunks = [created, delta, final]
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(sum(map(len, chunks))))
        self.end_headers()
        try:
            for index, chunk in enumerate(chunks):
                if index:
                    time.sleep(.03)
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextlib.contextmanager
def gateway(name, engine='gpt', mixed=False, moderation=False, nested=False, algorithm='fixed_priority', mismatch=False, provider_preferences=None, wire_chat=False):
    root = ROOT / name
    root.mkdir(exist_ok=True)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    peer.hits, peer.facts = [], []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    providers = [{'provider': role, 'engine': engine, 'api': 'fixture-upstream', 'model': CASES,
        'base_url': f'http://127.0.0.1:{peer.server_port}/{role}/v1/' + ('chat/completions' if wire_chat else 'responses'),
        'preferences': {'cooldown_period': 0, 'api_key_cooldown_period': 0},
        'exclude_request_rules': ([{'match': {'request_model': 'rules', 'reasoning_effort': 'high'}}] if role == 'first' else [])}
        for role in ['first', 'fallback']]
    if mixed:
        providers.append({'provider': 'unused', 'engine': 'gpt', 'api': 'fixture-upstream',
            'model': CASES, 'base_url': f'http://127.0.0.1:{peer.server_port}/unused/v1/chat/completions'})
    chain = [p['provider'] + '/*' for p in providers]
    for provider in providers:
        provider.setdefault('preferences', {}).update(provider_preferences or {})
    if mismatch:
        providers[0]['preferences']['post_body_parameter_overrides'] = {'stream': False}
    if moderation:
        providers.append({'provider': 'moderator', 'engine': 'gpt', 'api': 'fixture-upstream',
            'model': ['omni-moderation-latest'], 'base_url': f'http://127.0.0.1:{peer.server_port}/moderator/v1/moderations'})
        chain.append('moderator/*')
    pref = {'AUTO_RETRY': 2, 'SCHEDULING_ALGORITHM': algorithm, 'ENABLE_MODERATION': moderation}
    keys = [{'api': 'fixture-key', 'model': chain, 'preferences': pref}]
    if nested:
        keys = [{'api': 'fixture-key', 'model': ['fixture-child/*'], 'preferences': pref},
                {'api': 'fixture-child', 'model': chain, 'preferences': {'rate_limit': '1/min'}}]
    config = {'providers': providers, 'api_keys': keys,
              'preferences': {'unknown_upstream_400_retries': 3, 'hedging': {'enabled': False}}}
    (root / 'config.json').write_text(json.dumps(config))
    port = free_port()
    env = {k: v for k, v in os.environ.items() if k in ['PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH']}
    env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root / 'config.json'),
        RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / 'snapshot'),
        UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / 'ledger'),
        RUST_REQUEST_SPOOL_DIRECTORY=str(root / 'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',
        RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', FACTS_INGEST_URL=f'http://127.0.0.1:{peer.server_port}/facts',
        FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root / 'facts'), NO_PROXY='127.0.0.1,localhost')
    sequence = 0

    def call(case, endpoint='/v1/responses', extra=None, extra_headers=None):
        nonlocal sequence
        sequence += 1
        rid = f'{name}-{sequence}'
        body = {'model': case, 'stream': True}
        body['messages' if endpoint == '/v1/chat/completions' else 'input'] = (
            [{'role': 'user', 'content': 'offline fixture'}])
        body.update(extra or {})
        before = len(peer.hits)
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=8)
        headers = {'Authorization': 'Bearer fixture-key',
                   'Content-Type': 'application/json', 'X-Request-ID': rid}
        headers.update(extra_headers or {})
        conn.request('POST', endpoint, json.dumps(body), headers)
        response = conn.getresponse()
        try:
            raw = response.read()
        except http.client.IncompleteRead as exc:
            raw = exc.partial
        conn.close()
        time.sleep(.02)
        return {'variant': name, 'case': case, 'endpoint': endpoint, 'request_id': rid,
                'status': response.status, 'hits': peer.hits[before:],
                'wire_completed': b'response.completed' in raw,
                'wire_incomplete': b'response.incomplete' in raw,
                'wire_sha256': hashlib.sha256(raw).hexdigest()}

    process = None
    try:
        with (root / 'runtime.log').open('w') as log:
            process = subprocess.Popen([str(BINARY)], cwd=root, env=env, stdout=log, stderr=log)
            for _ in range(100):
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=.5)
                    conn.request('GET', '/healthz')
                    status = conn.getresponse().status
                    conn.close()
                    if status == 200:
                        break
                except OSError:
                    pass
                time.sleep(.05)
            else:
                raise RuntimeError(f'{name}: gateway did not start: ' + (root/'runtime.log').read_text()[-2000:])
            peer.root, peer.port = root, port
            yield call, peer
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and sum(f.get('kind') == 'request' for f in peer.facts) < sequence:
                time.sleep(.05)
    finally:
        if process:
            process.terminate()
            process.wait(timeout=5)
        peer.shutdown()
        peer.server_close()
