"""Native/compat contract: dispatch, admission, wire, outcome and pinned config."""
import http.client
import hashlib
import json
import time
from request_contract_fixture import gateway


def roles(row):
    return [hit['role'] for hit in row['hits']]


def control(peer, method, path, body=None):
    conn = http.client.HTTPConnection('127.0.0.1', peer.port, timeout=5)
    conn.request(method, path, json.dumps(body) if body else None,
                 {'Authorization': 'Bearer fixture-key', 'Content-Type': 'application/json'})
    response = conn.getresponse()
    status, data = response.status, json.loads(response.read())
    conn.close()
    assert status == 200, (status, data)
    return data


def verify():
    wires = {}
    for mixed in (False, True):
        prefix = 'generic' if mixed else 'native'
        for engine in ('gpt', 'codex'):
            with gateway(prefix+'-'+engine, mixed=mixed, engine=engine) as (call, peer):
                assert roles(call('rules', extra={'reasoning': {'effort': 'high'}})) == ['fallback']
                assert roles(call('rules', extra={'reasoning_effort': 'high'})) == ['fallback']
                assert roles(call('done-only')) == ['first', 'fallback']
                assert roles(call('incomplete')) == ['first', 'fallback']
                assert roles(call('tool-start-fail')) == ['first']
                coerced = call('ok', extra={'stream': 'true'})
                assert coerced['wire_completed'], coerced
                codex = call('codex-fields', extra={'previous_response_id': 'resp_fixture',
                    'prompt_cache_retention': '24h', 'safety_identifier': 'fixture',
                    'max_output_tokens': 10, 'top_p': .5, 'truncation': 'auto'})
                wires[(mixed, engine)] = codex['hits'][0]
            completed_facts = [row for row in peer.facts if row.get('kind') in ('request', 'attempt')
                               and row.get('request_id', '').endswith('-7')]
            assert {row['kind'] for row in completed_facts} == {'request', 'attempt'}, completed_facts
            for row in completed_facts:
                assert row['terminal_kind'] == 'completed', row
                assert row['response_completed'] is True, row
            final = [row for row in peer.facts if row.get('kind') == 'trace'
                     and row.get('stage') == 'request_final_facts' and row.get('request_id', '').endswith('-7')]
            assert len(final) == 1 and final[0]['trace_detail']['final_facts']['terminal'] == 'completed', final
            with gateway(prefix+'-'+engine+'-policy', mixed=mixed, engine=engine,
                         provider_preferences={'responses_stream_commit_policy': 'completed_usage'}) as (call, _):
                row = call('tool-start-fail')
                assert roles(row) == ['first', 'fallback'], row
                assert row['wire_completed'], row
        with gateway(prefix+'-moderation', mixed=mixed, moderation=True) as (call, _):
            assert roles(call('ok')) == ['moderator', 'first']
        with gateway(prefix+'-nested', mixed=mixed, nested=True) as (call, _):
            assert call('ok')['status'] == 200
            denied = call('ok')
            assert denied['status'] == 429 and not denied['hits'], denied
        with gateway(prefix+'-idempotency', mixed=mixed) as (call, _):
            first = call('ok', extra_headers={'Idempotency-Key': 'fixture-success'})
            second = call('ok', extra_headers={'Idempotency-Key': 'fixture-success'})
            assert [first['status'], second['status']] == [200, 200], second
            assert len(first['hits']) == 1 and not second['hits'], (first, second)
            assert first['wire_sha256'] == second['wire_sha256'], (first, second)
            failed = call('tool-start-fail', extra_headers={'Idempotency-Key': 'fixture-failed'})
            repeat = call('tool-start-fail', extra_headers={'Idempotency-Key': 'fixture-failed'})
            assert repeat['status'] == 409 and not repeat['hits'], (failed, repeat)
        # Replace configuration during the first upstream attempt. The fallback
        # must retain old intent while the next request observes new intent.
        with gateway(prefix+'-snapshot', mixed=mixed,
                     provider_preferences={'headers': {'X-Config-Marker': 'before'}}) as (call, peer):
            changed = []
            def reload(role, case):
                if role != 'first' or changed:
                    return
                changed.append(True)
                path = peer.root/'snapshot'
                cfg = json.loads(path.read_text())
                for p in cfg['providers']:
                    p.setdefault('preferences', {}).setdefault('headers', {})['X-Config-Marker'] = 'after'
                cfg['revision'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
                path.write_text(json.dumps(cfg))
                time.sleep(2)
            peer.on_request = reload
            row = call('done-only')
            assert roles(row) == ['first', 'fallback'], row
            headers = [{k.lower(): v for k, v in hit['headers'].items()} for hit in row['hits']]
            assert [h['x-config-marker'] for h in headers] == ['before', 'before'], headers
            row = call('ok')
            assert {k.lower(): v for k, v in row['hits'][0]['headers'].items()}['x-config-marker'] == 'after', row
        # Runtime disable remains dynamic even though configuration is pinned.
        with gateway(prefix+'-reenable', mixed=mixed) as (call, peer):
            state = control(peer, 'GET', '/v1/channel-controls')
            control(peer, 'POST', '/v1/channel-controls', {'revision': state['revision'],
                'action': 'set', 'disabled': ['fallback'] + (['unused'] if mixed else []), 'order': [], 'api_key_id': '', 'model': ''})
            def reenable(role, case):
                if role == 'first':
                    state = control(peer, 'GET', '/v1/channel-controls')
                    control(peer, 'POST', '/v1/channel-controls', {'revision': state['revision'], 'action': 'reset'})
            peer.on_request = reenable
            row = call('done-only')
            assert roles(row) == ['first', 'fallback'] and row['wire_completed'], row
        with gateway(prefix+'-disable', mixed=mixed) as (call, peer):
            def disable(role, case):
                if role == 'first':
                    state = control(peer, 'GET', '/v1/channel-controls')
                    control(peer, 'POST', '/v1/channel-controls', {'revision': state['revision'],
                        'action': 'set', 'disabled': ['fallback'] + (['unused'] if mixed else []), 'order': [], 'api_key_id': '', 'model': ''})
            peer.on_request = disable
            row = call('done-only')
            assert set(roles(row)) == {'first'}, row
            assert not row['wire_completed'], row
    for engine in ('gpt', 'codex'):
        native, generic = wires[(False, engine)], wires[(True, engine)]
        assert native['body'] == generic['body'], (native['body'], generic['body'])
        assert native['header_names'] == generic['header_names'], (native['header_names'], generic['header_names'])
    with gateway('round-robin', algorithm='round_robin', mismatch=True) as (call, _):
        assert [roles(call('ok')) for _ in range(6)] == [['first'], ['fallback']]*3
    with gateway('synthetic-replay', mismatch=True, wire_chat=True) as (call, _):
        first = call('ok', extra_headers={'Idempotency-Key': 'fixture-synthetic'})
        repeat = call('ok', extra_headers={'Idempotency-Key': 'fixture-synthetic'})
        assert first['wire_completed'] and repeat['wire_completed'], (first, repeat)
        assert len(first['hits']) == 1 and not repeat['hits'], (first, repeat)
        assert first['wire_sha256'] == repeat['wire_sha256'], (first, repeat)
    print('PASS shared request contract: routing, reasoning, admission, moderation, wire, commit policy, replay, snapshot and live disable')


if __name__ == '__main__':
    verify()
