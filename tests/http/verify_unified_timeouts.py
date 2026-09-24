"""Loopback contracts for native/translated streaming and non-streaming deadlines."""
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

from verify_dispatch_timing import free_port, get_json

# Keep the ordering assertions meaningful under CI scheduling jitter. The
# original 50 ms header/first-byte margin could test the wrong timeout, and
# a fixed 380 ms wall limit failed even when the correct stream was cut.
TIME_SCALE = 3


def event(value):
    kind = value.get('type')
    return ((f'event: {kind}\n' if kind else '') + f'data: {json.dumps(value)}\n\n').encode()


def completion(model, protocol, text):
    if protocol == 'responses':
        return {'id': 'resp_fixture', 'object': 'response', 'status': 'completed', 'model': model,
                'output': [{'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': text}]}],
                'usage': {'input_tokens': 1, 'output_tokens': 1}}
    if protocol == 'messages':
        return {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'model': model,
                'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
                'usage': {'input_tokens': 1, 'output_tokens': 1}}
    return {'id': 'chat_fixture', 'object': 'chat.completion', 'model': model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}


def delta(protocol, text):
    if protocol == 'responses':
        return event({'type': 'response.output_text.delta', 'delta': text, 'item_id': 'msg_fixture', 'output_index': 0, 'content_index': 0})
    if protocol == 'messages':
        return event({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})
    return event({'id': 'chat_fixture', 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        model = payload['model']
        case = self.server.cases[model]
        fallback = self.path.startswith('/fallback-')
        self.server.hits.setdefault(model, []).append('fallback' if fallback else 'first')
        timing = {} if fallback else case
        protocol = self.path.rsplit('/v1/', 1)[-1]
        try:
            time.sleep(timing.get('headers', 0))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if payload.get('stream') else 'application/json')
            self.end_headers()
            self.wfile.flush()
            time.sleep(timing.get('body', 0))
            if not payload.get('stream'):
                self.wfile.write(json.dumps(completion(model, protocol, 'FINAL_OK')).encode())
                return
            self.wfile.write(delta(protocol, 'FIRST_OK'))
            self.wfile.flush()
            for _ in range(timing.get('chunks', 1)):
                time.sleep(timing.get('gap', 0))
                self.wfile.write(b': upstream ping\n\n')
                self.wfile.flush()
            self.wfile.write(delta(protocol, 'FINAL_OK'))
            if protocol == 'responses':
                self.wfile.write(event({'type': 'response.completed', 'response': completion(model, protocol, 'FIRST_OKFINAL_OK')}))
            elif protocol == 'messages':
                self.wfile.write(event({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}}))
                self.wfile.write(event({'type': 'message_stop'}))
            else:
                self.wfile.write(event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}) + b'data: [DONE]\n\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


def verify(binary):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.cases, server.hits = {}, {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    variants = [
        ('messages', '/v1/messages', 'responses', 'gpt'),
        ('chat', '/v1/chat/completions', 'chat/completions', 'gpt'),
        ('responses-to-chat', '/v1/chat/completions', 'responses', 'gpt'),
        ('claude-to-chat', '/v1/chat/completions', 'messages', 'claude'),
        ('responses', '/v1/responses', 'responses', 'gpt'),
        ('codex-responses', '/v1/responses', 'responses', 'codex'),
        ('chat-to-responses', '/v1/responses', 'chat/completions', 'gpt'),
    ]
    scenarios = {
        'first-only-long-tail': {'gap': .36},
        'total-zero-long-tail': {'policy': {'total': 0}, 'gap': .36},
        'first-zero-late-headers': {'policy': {'first_byte': 0, 'total': 0}, 'headers': .36},
        'idle-resets': {'policy': {'idle': .2, 'total': 0}, 'gap': .06, 'chunks': 6},
        'idle-cuts-after-output': {'policy': {'idle': .15, 'total': 0}, 'gap': .45, 'cut': True},
        'total-cuts-active-stream': {'policy': {'total': .2, 'idle': .3}, 'gap': .06, 'chunks': 8, 'cut': True},
        'total-includes-header-wait': {'policy': {'total': .25, 'first_byte': 1}, 'headers': .15, 'gap': .06, 'chunks': 8, 'cut': True},
        'first-byte-failover': {'policy': {'first_byte': .15}, 'headers': .4, 'fallback': True},
        'nonstream-body-not-first-byte': {'stream': False, 'body': .36},
        'nonstream-idle-failover': {'stream': False, 'policy': {'idle': .15, 'total': 0}, 'body': .4, 'fallback': True},
        'nonstream-zero-total': {'stream': False, 'policy': {'total': 0}, 'headers': .36},
    }
    providers, routes = [], []
    for timing in scenarios.values():
        for field in ('headers', 'body', 'gap'):
            if field in timing:
                timing[field] *= TIME_SCALE
        if 'policy' in timing:
            timing['policy'] = {key: value * TIME_SCALE for key, value in timing['policy'].items()}
    for variant, endpoint, upstream, engine in variants:
        for name, timing in scenarios.items():
            model = variant + '-' + name
            case = dict(timing, endpoint=endpoint, stream=timing.get('stream', True))
            server.cases[model] = case
            for role in ['first', 'fallback']:
                provider = role + '-' + model
                preferences = {'model_timeout': .2 * TIME_SCALE, 'cooldown_period': 0, 'api_key_cooldown_period': 0}
                if case.get('policy'):
                    preferences['timeout_policy'] = {'default': case['policy']}
                providers.append({'provider': provider, 'engine': engine,
                                  'base_url': f'http://127.0.0.1:{server.server_port}/{provider}/v1/{upstream}',
                                  'api': 'fixture-upstream', 'model': [model], 'preferences': preferences})
                routes.append(provider + '/*')
    # Native Responses preflight must enforce idle even before a committing event.
    model = 'responses-precommit-idle'
    server.cases[model] = {'endpoint': '/v1/responses', 'stream': True, 'body': .4 * TIME_SCALE, 'fallback': True}
    for role in ['first', 'fallback']:
        provider = role + '-' + model
        providers.append({'provider': provider, 'engine': 'codex', 'api': 'fixture-upstream', 'model': [model],
                          'base_url': f'http://127.0.0.1:{server.server_port}/{provider}/v1/responses',
                          'preferences': {'timeout_policy': {'default': {'first_byte': TIME_SCALE, 'idle': .15 * TIME_SCALE, 'total': 0}}}})
        routes.append(provider + '/*')
    with tempfile.TemporaryDirectory(prefix='uni-unified-timeouts-') as directory:
        root, port = Path(directory), free_port()
        config = {'providers': providers, 'api_keys': [{'api': 'fixture-key', 'model': routes,
                  'preferences': {'AUTO_RETRY': True, 'SCHEDULING_ALGORITHM': 'fixed_priority'}}],
                  'preferences': {'model_timeout': {'default': 2000}, 'hedging': {'enabled': False},
                    'timeout_policy': {'rules': [{'match': {'model': [m for m in server.cases if m.endswith('total-zero-long-tail')]},
                                                  'timeout': {'total': .15 * TIME_SCALE}}]}}}
        path = root / 'api.json'
        path.write_text(json.dumps(config))
        original = path.read_bytes()
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'SSL_CERT_FILE')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(path), NO_PROXY='127.0.0.1,localhost',
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0')
        with (root/'log').open('w+') as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        get_json(port, '/healthz')
                        break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise RuntimeError('fixture startup failed')
                for model, case in server.cases.items():
                    body = {'model': model, 'stream': case['stream'], 'max_tokens': 16}
                    body.update({'input': 'fixture'} if case['endpoint'] == '/v1/responses' else
                                {'messages': [{'role': 'user', 'content': 'fixture'}]})
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=4)
                    started = time.monotonic()
                    conn.request('POST', case['endpoint'], json.dumps(body),
                                 {'Authorization': 'Bearer fixture-key', 'Content-Type': 'application/json', 'X-Request-ID': model})
                    response = conn.getresponse()
                    parts, error = [], None
                    try:
                        while part := response.read1(65536):
                            parts.append(part)
                    except (http.client.HTTPException, OSError) as exc:
                        error = type(exc).__name__
                    finally:
                        conn.close()
                    raw = b''.join(parts)
                    elapsed = time.monotonic() - started
                    hits = server.hits.get(model, [])
                    context = (model, response.status, elapsed, hits, error, raw[:1500])
                    if case.get('cut'):
                        assert b'FIRST_OK' in raw and b'FINAL_OK' not in raw, context
                        assert error or b'error' in raw or b'failed' in raw, context
                        assert hits == ['first'] and elapsed < .38 * TIME_SCALE, context
                    else:
                        assert response.status == 200 and error is None and b'FINAL_OK' in raw, context
                        assert hits == (['first', 'fallback'] if case.get('fallback') else ['first']), context
                        if case.get('fallback'):
                            assert elapsed < .35 * TIME_SCALE, context
                    print(f'PASS {model} elapsed={elapsed:.3f}s hits={hits}', flush=True)
                # The control surface must accept disabling each limit, preserve
                # explicit zeros, and return the same policy that execution uses.
                def control(method, endpoint, body=None):
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=4)
                    conn.request(method, endpoint, None if body is None else json.dumps(body),
                                 {'Authorization': 'Bearer fixture-key', 'Content-Type': 'application/json'})
                    response = conn.getresponse(); data = json.loads(response.read()); conn.close()
                    assert response.status == 200, (response.status, data)
                    return data
                before = control('GET', '/v1/channel-controls')
                zeros = dict.fromkeys(['connect', 'write', 'pool', 'first_byte', 'idle', 'total'], 0)
                preview = control('POST', '/v1/channel-settings/validate', {'revision': before['revision'],
                    'operation_id': 'zero-timeouts-preview', 'changes': [{'provider': 'first-messages-first-only-long-tail',
                    'set': {'/preferences/timeout_policy': {'default': zeros}}}],
                    'sample': {'model': 'messages-first-only-long-tail', 'endpoint': '/v1/messages', 'stream': True}})
                assert preview['previews'][0]['sample']['timeouts'] == zeros, preview
                assert control('GET', '/v1/channel-controls')['revision'] == before['revision']
                applied = control('PATCH', '/v1/channel-settings', {'revision': before['revision'],
                    'operation_id': 'zero-timeouts-apply', 'changes': [{'provider': 'first-messages-first-only-long-tail',
                    'set': {'/preferences/timeout_policy': {'default': zeros}}}],
                    'sample': {'model': 'messages-first-only-long-tail', 'endpoint': '/v1/messages', 'stream': True}})
                assert applied['previews'][0]['sample']['timeouts'] == zeros, applied
                exported = control('GET', '/v1/channel-settings/export')
                stored = exported['channel_settings']['first-messages-first-only-long-tail']['set']
                assert all(stored.get('/preferences/timeout_policy/default/'+field) == 0 for field in zeros), stored
                view = control('GET', '/v1/channel-settings?provider=first-messages-first-only-long-tail')
                assert view['effective']['preferences']['timeout_policy']['default'] == zeros, view
                for invalid in (-1, '0'):
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=4)
                    conn.request('POST', '/v1/channel-settings/validate', json.dumps({'revision': view['revision'],
                        'operation_id': 'invalid-timeout', 'changes': [{'provider': 'first-messages-first-only-long-tail',
                        'set': {'/preferences/timeout_policy': {'default': {'total': invalid}}}}]}),
                        {'Authorization': 'Bearer fixture-key', 'Content-Type': 'application/json'})
                    response = conn.getresponse(); response.read(); conn.close()
                    assert response.status == 400, (invalid, response.status)
                assert path.read_bytes() == original
                print('PASS zero timeouts validate, apply and export losslessly; preview is read-only; invalid values rejected', flush=True)
            except Exception:
                log.flush(); print((root/'log').read_text()[-6000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=4)
    server.shutdown(); server.server_close()


if __name__ == '__main__':
    verify(Path(sys.argv[1]).resolve())
