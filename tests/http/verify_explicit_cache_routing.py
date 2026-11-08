"""Offline HTTP regression: passive, prefix-scoped cache routing and availability."""
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
from fact_sink import accept_facts


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path == '/facts':
            return accept_facts(self, self.server.facts)
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append(payload)
        signature = json.dumps(payload, sort_keys=True)
        repeated = signature in self.server.seen
        self.server.seen.add(signature)
        read, write = ((4750, 0) if repeated or self.server.reuses else (650, 4100))
        response = {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant',
                    'model': payload['model'], 'content': [{'type': 'text', 'text': self.server.name}],
                    'stop_reason': 'end_turn', 'usage': {'input_tokens': 10, 'output_tokens': 1,
                    'cache_read_input_tokens': read, 'cache_creation_input_tokens': write}}
        events = [
            {'type': 'message_start', 'message': {**response, 'content': [], 'stop_reason': None}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': self.server.name}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
            {'type': 'message_stop'},
        ]
        raw = (''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
               if payload.get('stream') else json.dumps(response).encode())
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if payload.get('stream') else 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def body(question, prefix='reference ', stream=False, marked=True):
    reference = {'type': 'text', 'text': prefix * 1100}
    if marked:
        reference['cache_control'] = {'type': 'ephemeral', 'ttl': '5m'}
    return {'model': 'client-alias', 'max_tokens': 24, 'stream': stream,
            'messages': [{'role': 'user', 'content': [reference, {'type': 'text', 'text': question}]}]}


def main(binary):
    servers = []
    for name in ['broken', 'healthy']:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        server.name, server.hits, server.seen, server.reuses = name, [], set(), name == 'healthy'
        server.facts = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    broken, healthy = servers
    with tempfile.TemporaryDirectory(prefix='explicit-cache-routing-') as temporary:
        root, port = Path(temporary), free_port()
        providers = [{'provider': s.name, 'engine': 'claude',
                      'base_url': f'http://127.0.0.1:{s.server_port}/v1/messages',
                      'api': 'fixture-upstream', 'model': [{'claude-opus-5-5': 'client-alias'}],
                      'preferences': {'cooldown_period': 0}} for s in servers]
        providers += [{**p, 'provider': 'injected-'+p['provider'],
                       'model': [{'claude-opus-5-5': 'injected-alias'}],
                       'preferences': {'cooldown_period': 0, 'post_body_parameter_overrides': {
                           'system': [{'type': 'text', 'text': 'injected reference ' * 1100,
                                       'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]}}}
                      for p in list(providers)]
        config = {'providers': providers, 'api_keys': [
            {'api': key, 'model': [p['provider']+'/*' for p in providers],
             'preferences': {'AUTO_RETRY': False, 'SCHEDULING_ALGORITHM': 'fixed_priority'}}
            for key in ['fixture-client', 'different-client']]}
        config_path = root / 'config.json'
        config_path.write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('FACTS_', 'OTEL_', 'FUGUE_OTEL_', 'UNI_API_CONTROL_RESTORE_'))
            and k not in ['CONFIG_URL', 'UNI_API_EXPLICIT_CACHE_GUARD']}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   FACTS_INGEST_URL=f'http://127.0.0.1:{broken.server_port}/facts',
                   FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts-spool'),
                   NO_PROXY='127.0.0.1,localhost')

        def request(payload=None, key='fixture-client'):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            try:
                connection.request('GET' if payload is None else 'POST', '/healthz' if payload is None else '/v1/messages',
                                   None if payload is None else json.dumps(payload),
                                   {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
                response = connection.getresponse()
                status, raw = response.status, response.read()
                assert status == 200, (status, raw)
                if payload is None:
                    return None
                if payload.get('stream'):
                    assert b'event: message_stop' in raw, raw
                    texts = [json.loads(line[6:]).get('delta', {}).get('text')
                             for line in raw.decode().splitlines() if line.startswith('data: ')]
                    return ''.join(t for t in texts if t)
                return json.loads(raw)['content'][0]['text']
            finally:
                connection.close()

        def send(question, expected='broken', **options):
            payload = body(question, **options)
            before = [len(s.hits) for s in servers]
            assert request(payload) == expected
            after = [len(s.hits) for s in servers]
            # Every successful client request is sent exactly once; no paid replay.
            assert sum(after)-sum(before) == 1, (before, after)
            target = next(s for s in servers if s.name == expected)
            assert target.hits[-1]['messages'] == payload['messages'], 'cache blocks were rewritten'

        with (root/'log').open('w') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        request()
                        break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise AssertionError('gateway did not start')
                send('A')
                send('A')  # An exact repeat hits but says nothing about a changed suffix.
                send('B')
                send('C')  # Both observed rewrites return successfully, without retry.
                send('D', 'healthy')  # Skips do not consume AUTO_RETRY=False's dispatch budget.
                send('A')  # Known, still-warm repeats are allowed.
                send('E', 'healthy')  # The preceding repeat did not clear negative evidence.
                send('ordinary A', marked=False)
                send('ordinary B', marked=False)
                send('different', prefix='unrelated reference ')
                assert request(body('different caller'), key='different-client') == 'broken'
                multi = body('multiple')
                multi['messages'][0]['content'][1]['cache_control'] = {'type': 'ephemeral'}
                assert request(multi) == 'broken'
                # Both candidates can become unsuitable; fallback still serves the request.
                healthy.reuses = False
                send('F', 'healthy')
                send('G', 'healthy')
                send('H', 'broken')
                healthy.reuses = True
                # Verify streaming observations with an independent prefix and completion wait.
                for question in ['stream A', 'stream B', 'stream C']:
                    observations = (root/'log').read_text().count('cache_guard_action')
                    send(question, prefix='stream reference ', stream=True)
                    deadline = time.monotonic() + 5
                    while True:
                        # Successful completion is recorded asynchronously after the final SSE bytes.
                        text = (root/'log').read_text()
                        if text.count('cache_guard_action') > observations:
                            break
                        assert time.monotonic() < deadline, text[-3000:]
                        time.sleep(.02)
                send('stream D', 'healthy', prefix='stream reference ', stream=True)
                # Classify the final outgoing request, including operator overrides.
                # Incoming requests here have no markers; scanning must still reach
                # the next provider after an observed cache skip with AUTO_RETRY=False.
                for question, expected in [('A', 'broken'), ('B', 'broken'), ('C', 'broken'), ('D', 'healthy')]:
                    assert request({'model': 'injected-alias', 'max_tokens': 24, 'stream': False,
                                    'messages': [{'role': 'user', 'content': question}]}) == expected
                text = (root/'log').read_text()
                assert 'explicit_cache_prefix_rewrite' in text and 'availability_fallback' in text, text[-4000:]
                deadline = time.monotonic() + 8
                while True:
                    avoided = [f for f in broken.facts if f.get('trace_detail', {}).get('skip_reason') == 'explicit_cache_prefix_rewrite']
                    if avoided:
                        break
                    assert time.monotonic() < deadline, 'cache skip evidence was not exported to request traces'
                    time.sleep(.05)
                detail = avoided[0]['trace_detail']
                assert detail['cache_rewrite_count'] >= 2 and detail['cache_evidence_request_id'], detail
                assert detail['cache_breakpoint'] == 'messages[0].content[0]' and detail['cache_ttl_seconds'] == 300, detail
                encoded = json.dumps(broken.facts)
                for secret in ['fixture-client', 'fixture-upstream', 'reference reference']:
                    assert secret not in encoded, 'private prompt or credential leaked into cache evidence'
                print('explicit cache routing: all HTTP, streaming, scope and availability checks passed')
            except Exception:
                print((root/'log').read_text()[-5000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                process.wait(timeout=10)
    for server in servers:
        server.shutdown()


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
