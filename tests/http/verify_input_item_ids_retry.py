"""Exact Responses item-ID capability fallback; no customer requests or secrets."""
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

from fact_sink import accept_facts
from verify_dispatch_timing import free_port
from verify_encrypted_content_fallback import RESPONSE

MESSAGE = 'This request does not support user-supplied input item IDs.'
ERROR = {'error': {'type': 'invalid_request_error', 'code': 'unsupported_parameter',
                   'param': 'input', 'message': MESSAGE}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path == '/facts':
            return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        name = self.path.split('/')[1]
        self.server.hits.append((name, raw))
        rejects = self.server.mode == 'exhausted' or name == 'first'
        error = self.server.error
        if rejects:
            status, content_type, data = 400, 'application/json', json.dumps(error).encode()
        elif json.loads(raw).get('stream'):
            events = [
                {'type': 'response.output_text.delta', 'delta': 'OK'},
                {'type': 'response.completed', 'response': RESPONSE},
            ]
            status, content_type = 200, 'text/event-stream'
            data = b''.join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                for event in events
            )
        else:
            status, content_type, data = 200, 'application/json', json.dumps(RESPONSE).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits, server.facts, server.mode, server.error = [], [], 'retry', ERROR
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-item-ids-') as td:
        root, port = Path(td), free_port()
        upstream = f'http://127.0.0.1:{server.server_port}'
        providers = [
            {'provider': name, 'engine': 'gpt', 'api': 'fixture-upstream-secret',
             'base_url': f'{upstream}/{name}/v1/responses', 'model': ['test-model'],
             'preferences': {'cooldown_period': 300}}
            for name in ['first', 'fallback', 'third', 'fourth']
        ]
        keys = [
            {'api': 'retry-three', 'model': ['all'],
             'preferences': {'AUTO_RETRY': 3, 'SCHEDULING_ALGORITHM': 'fixed_priority'}},
            {'api': 'retry-one', 'model': ['all'],
             'preferences': {'AUTO_RETRY': 1, 'SCHEDULING_ALGORITHM': 'fixed_priority'}},
            {'api': 'no-retry', 'model': ['all'],
             'preferences': {'AUTO_RETRY': False, 'SCHEDULING_ALGORITHM': 'fixed_priority'}},
        ]
        config = {'providers': providers, 'api_keys': keys,
                  'preferences': {'unknown_upstream_400_retries': 3}}
        original_config = json.dumps(config).encode()
        (root / 'config').write_bytes(original_config)
        env = {k: v for k, v in os.environ.items()
               if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
        env.update(
            PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root / 'config'),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / 'snapshot'),
            UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / 'ledger'),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root / 'spool'),
            RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
            FACTS_INGEST_URL=f'{upstream}/facts', FACTS_INGEST_TOKEN='fixture',
            FACTS_SPOOL_DIR=str(root / 'facts'), NO_PROXY='127.0.0.1,localhost',
        )
        count = 0
        successful_rids = []

        def call(payload=None, key='retry-three', targeted=False):
            nonlocal count
            rid = f'item-ids-{count}'
            headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                       'X-Request-ID': rid}
            if targeted:
                headers['X-Uni-API-Provider'] = 'first'
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
            try:
                conn.request('GET' if payload is None else 'POST',
                             '/healthz' if payload is None else '/v1/responses',
                             None if payload is None else json.dumps(payload), headers)
                response = conn.getresponse()
                return rid, response.status, response.read()
            finally:
                conn.close()

        body = {'model': 'test-model', 'store': False, 'stream': False,
                'input': [{'type': 'message', 'id': 'msg_client_supplied_123', 'role': 'user',
                           'content': [{'type': 'input_text', 'text': 'keep unchanged'}]}]}
        with (root / 'gateway.log').open('w') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[1] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError('fixture gateway not ready')

                server.hits = []
                rid, status, raw = call(body)
                assert status == 200, (status, raw[:200])
                assert [name for name, _ in server.hits] == ['first', 'fallback']
                assert server.hits[0][1] == server.hits[1][1], 'wire bytes changed on retry'
                assert json.loads(server.hits[0][1]) == body
                successful_rids.append(rid)
                count += 1

                # Stream requests also fail over before any bytes are committed.
                server.hits = []
                rid_stream, status, raw = call({**body, 'stream': True})
                assert status == 200, (status, raw[:200])
                assert [name for name, _ in server.hits] == ['first', 'fallback']
                assert server.hits[0][1] == server.hits[1][1], 'stream wire bytes changed'
                successful_rids.append(rid_stream)
                count += 1

                # Normal API-key retry count remains authoritative (one extra try).
                server.hits = []
                server.mode = 'exhausted'
                rid_one, status, _ = call(body, key='retry-one')
                assert status == 502
                assert [name for name, _ in server.hits] == ['first', 'fallback']
                assert server.hits[0][1] == server.hits[1][1]
                count += 1

                # Disabled/targeted requests do not fail over.
                server.mode = 'retry'
                server.hits = []
                _, status, _ = call(body, key='no-retry')
                assert status == 502 and [name for name, _ in server.hits] == ['first']
                count += 1
                server.hits = []
                _, status, _ = call(body, targeted=True)
                assert status == 502 and [name for name, _ in server.hits] == ['first']
                count += 1

                # Exact envelope without an actual input ID is ordinary 400.
                server.hits = []
                no_id = {'model': 'test-model', 'store': False, 'stream': False,
                         'input': [{'type': 'message', 'role': 'user',
                                    'content': [{'type': 'input_text', 'text': 'keep'}]}]}
                _, status, _ = call(no_id)
                assert status == 400 and [name for name, _ in server.hits] == ['first']
                count += 1

                # Near misses do not acquire fallback behavior.
                for patch in ({'code': 'invalid_value'}, {'param': 'messages'},
                              {'message': MESSAGE + ' extra'}, {'type': 'upstream_error'}):
                    server.error = {'error': {**ERROR['error'], **patch}}
                    server.hits = []
                    _, status, _ = call(body)
                    assert status == 400 and [name for name, _ in server.hits] == ['first']
                    count += 1

                # Prove exhaustion uses the existing total AUTO_RETRY budget.
                server.error = ERROR
                server.mode = 'exhausted'
                server.hits = []
                _, status, _ = call(body)
                assert status == 502
                assert [name for name, _ in server.hits] == ['first', 'fallback', 'third', 'fourth']
                assert all(wire == server.hits[0][1] for _, wire in server.hits)
                count += 1

                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if all(any(f.get('kind') == 'request' and f.get('request_id') == success_rid
                               for f in server.facts) for success_rid in successful_rids):
                        break
                    time.sleep(.1)
                for success_rid in successful_rids:
                    events = [f for f in server.facts
                              if f.get('stage') == 'responses_item_ids_fallback'
                              and f.get('request_id') == success_rid]
                    billings = [f for f in server.facts if f.get('kind') == 'billing'
                                and f.get('request_id') == success_rid]
                    assert len(events) == 1, success_rid
                    assert [f.get('status') for f in billings] == [400, 200], success_rid
                    assert any(f.get('kind') == 'request' and f.get('request_id') == success_rid
                               and f.get('status') == 200 for f in server.facts)
                    assert events[0].get('trace_detail', {}).get('routing_status_code') == 502
                    assert events[0].get('trace_detail', {}).get('attempt_status_code') == 400
                encoded = json.dumps(server.facts) + (root / 'gateway.log').read_text()
                assert 'fixture-upstream-secret' not in encoded
                assert 'keep unchanged' not in encoded
                assert (root / 'config').read_bytes() == original_config
                print('PASS Responses user-item-ID fallback: exact 400 -> request-scoped 502 -> next-channel 200; wire unchanged; stream, disabled, targeted, near-miss, no-ID and exhaustion cases preserved')
            finally:
                process.terminate()
                process.wait(timeout=10)
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    verify(Path(sys.argv[1]).resolve())
