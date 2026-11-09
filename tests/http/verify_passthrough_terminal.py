"""Native Chat/Messages terminals end streams without waiting for HTTP EOF."""
from fact_sink import accept_facts
from verify_dispatch_timing import free_port, get_json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def event(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path == "/facts": return accept_facts(self, self.server.facts)
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        case = payload['messages'][0]['content']
        claude = self.path.endswith('/messages')
        self.server.hits.append(case)
        if claude:
            prefix = event({'type': 'message_start', 'message': {'usage': {'input_tokens': 7}}})
            prefix += event({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'OK'}})
            usage = event({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 3}})
            terminal = event({'type': 'message_stop'})
        else:
            prefix = event({'choices': [{'delta': {'role': 'assistant'}}]})
            prefix += event({'choices': [{'delta': {'content': 'OK'}}]})
            usage = event({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
            terminal = event({'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}) + b'data: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        try:
            self.wfile.write(prefix)
            self.wfile.flush()
            time.sleep(.03)
            if '-cancel' in case:
                time.sleep(1.1)
                return
            self.wfile.write(usage)
            self.wfile.flush()
            time.sleep(.03)
            if '-error' in case:
                terminal = event({'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'fixture failure'}})
            # Split the terminal across reads; stop only after a complete SSE event.
            for part in (terminal[:13], terminal[13:]):
                self.wfile.write(part)
                self.wfile.flush()
                time.sleep(.01)
            if case.endswith(('held', 'close')): time.sleep(1.1)
        except (BrokenPipeError, ConnectionResetError): pass


def verify(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.facts, server.hits = [], []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cases = [f'{kind}-{result}-{end}' for kind in ('claude', 'chat') for result in ('completed', 'error') for end in ('held', 'close', 'eof')]
    cases += ['claude-cancel', 'chat-cancel']
    try:
        with tempfile.TemporaryDirectory(prefix='uni-passthrough-terminal-') as directory:
            root, port = Path(directory), free_port()
            providers = [{'provider': kind, 'engine': 'claude' if kind == 'claude' else 'gpt', 'api': 'fixture',
                'model': [{'m': kind}], 'base_url': f'http://127.0.0.1:{server.server_port}/v1/' + ('messages' if kind == 'claude' else 'chat/completions'),
                'preferences': {'cooldown_period': 0, 'timeout_policy': {'default': {'first_byte': .3, 'total': .7}}}}
                for kind in ('claude', 'chat')]
            config = {'providers': providers, 'api_keys': [{'api': 'fixture-key', 'model': ['all'],
                'preferences': {'AUTO_RETRY': True, 'SCHEDULING_ALGORITHM': 'fixed_priority'}}]}
            (root / 'api.json').write_text(json.dumps(config))
            env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'DYLD_LIBRARY_PATH')}
            env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root / 'api.json'),
                RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / 'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / 'ledger'),
                RUST_REQUEST_SPOOL_DIRECTORY=str(root / 'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                FACTS_INGEST_URL=f'http://127.0.0.1:{server.server_port}/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root / 'facts'), NO_PROXY='127.0.0.1,localhost')
            with (root / 'runtime.log').open('w') as log:
                process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
                try:
                    for _ in range(100):
                        try: get_json(port, '/healthz'); break
                        except OSError: time.sleep(.05)
                    else: raise AssertionError('gateway did not start')
                    for case in cases:
                        kind = case.split('-')[0]
                        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                        conn.request('POST', '/v1/messages' if kind == 'claude' else '/v1/chat/completions', json.dumps({
                            'model': kind, 'stream': True, 'messages': [{'role': 'user', 'content': case}], 'max_tokens': 16}),
                            {'Authorization': 'Bearer fixture-key', 'Content-Type': 'application/json', 'X-Request-ID': case})
                        response = conn.getresponse()
                        assert response.status == 200, (case, response.status)
                        wire = b''
                        if case.endswith(('close', 'cancel')):
                            marker = b'OK' if case.endswith('cancel') else b'fixture failure' if '-error-' in case else b'message_stop' if kind == 'claude' else b'[DONE]'
                            while marker not in wire or not wire.endswith(b'\n\n'):
                                part = response.read(1)
                                if not part: break
                                wire += part
                            assert marker in wire, (case, wire)
                        else: wire = response.read()
                        response.close(); conn.close()
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and sum(f.get('kind') == 'request' for f in server.facts) < len(cases): time.sleep(.1)
                    for case in cases:
                        assert server.hits.count(case) == 1, (case, server.hits)
                        completed = '-completed-' in case
                        for fact_kind in ('request', 'attempt'):
                            fact = next(f for f in server.facts if f.get('kind') == fact_kind and f.get('request_id') == case)
                            assert fact['outcome'] == ('success' if completed else 'failed'), (case, fact)
                            assert fact['first_output_ms'] is not None, (case, fact)
                            if completed:
                                assert fact['terminal_kind'] == 'completed' and fact['response_completed'] is True, (case, fact)
                                if fact_kind == 'request': assert fact['input_tokens'] == 7 and fact['output_tokens'] == 3, (case, fact)
                            elif '-error-' in case:
                                assert fact['failure_reason'] == 'upstream_response_failed', (case, fact)
                            else:
                                assert fact['failure_reason'] == 'downstream_disconnected', (case, fact)
                            if fact_kind == 'request': assert fact['status'] == (200 if completed else 499 if case.endswith('cancel') else 502), (case, fact)
                    print(f'PASS {len(cases)} native Chat/Messages terminal, usage, error, cancellation and no-replay contracts')
                except Exception:
                    print((root / 'runtime.log').read_text()[-7000:], file=sys.stderr)
                    raise
                finally:
                    process.terminate(); process.wait(timeout=5)
    finally:
        server.shutdown(); server.server_close()


if __name__ == '__main__': verify(Path(sys.argv[1]).resolve())
