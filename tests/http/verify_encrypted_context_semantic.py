"""Exact encrypted-continuation SSE rejection, entirely on loopback."""
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
from verify_encrypted_content_fallback import CONTINUATION_ERROR, RESPONSE, payload


def frame(kind, **fields):
    return f'event: {kind}\ndata: {json.dumps(dict(type=kind, **fields))}\n\n'.encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        name = self.path.split('/')[1]
        self.server.hits.append((name, raw))
        reject = name == 'first' or self.server.mode == 'exhausted'
        if reject:
            error = self.server.error['error']
            if self.server.event == 'error': data = frame('error', error=error)
            else: data = frame('response.failed', response={'status':'failed','error':error})
            if self.server.mode == 'postcommit':
                data = frame('response.output_text.delta', delta='started') + data
            if self.server.mode == 'keepalive':
                data = frame('response.created', response={'id':'resp_first','status':'in_progress','output':[]}) + data
        else:
            data = frame('response.output_text.delta', delta='OK') + frame('response.completed', response=RESPONSE)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try: self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass


def verify(binary, engine, collect, hedging=False):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    server.hits, server.facts, server.mode, server.event = [], [], 'reject', 'error'
    server.error = CONTINUATION_ERROR
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-encrypted-semantic-') as td:
        root, port = Path(td), free_port()
        base = f'http://127.0.0.1:{server.server_port}'
        preferences = {'cooldown_period':300}
        # GPT's opaque streaming contract commits HTTP headers immediately.
        # Exercise its existing opt-in precommit path; Codex guards by default.
        if engine == 'gpt' and not collect:
            preferences['normalize_responses_custom_tool_call_ids'] = True
        if collect: preferences['upstream_stream_mode'] = 'stream_only'
        config = {'providers':[{'provider':name, 'engine':engine, 'api':'fixture-secret',
            'base_url':base+'/'+name+'/v1/responses', 'model':['test-model'], 'preferences':preferences}
            for name in ['first','fallback']],
            'api_keys':[{'api':key,'model':['first/*','fallback/*'], 'preferences':{'AUTO_RETRY':retry}}
                for key,retry in [('retry',1),('no-retry',False)]],
            'preferences':{'hedging':{'enabled':hedging,'max_inflight_attempts':2}}}
        original_config = json.dumps(config).encode()
        (root/'api.json').write_bytes(original_config)
        env = {k:v for k,v in os.environ.items() if k in ('PATH','HOME','TMPDIR','LANG','DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'api.json'),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
            FACTS_INGEST_URL=base+'/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'), NO_PROXY='127.0.0.1,localhost')
        count, successful = 0, []

        def call(body=None, key='retry', target=False):
            rid = f'semantic-{engine}-{collect}-{count}'
            headers = {'Authorization':'Bearer '+key, 'Content-Type':'application/json', 'X-Request-ID':rid}
            if target: headers['X-Uni-API-Provider'] = 'first'
            c = http.client.HTTPConnection('127.0.0.1',port,timeout=15)
            try:
                c.request('GET' if body is None else 'POST','/healthz' if body is None else '/v1/responses',
                    None if body is None else json.dumps(body),headers)
                r = c.getresponse()
                return rid,r.status,r.read()
            finally: c.close()

        with (root/'log').open('w') as log:
            process = subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[1] == 200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('loopback gateway startup failed')
                for event in ['error','response.failed']:
                    server.event = event
                    cases = [
                        ('reject','retry',200,['first','fallback']),
                        ('reject','no-retry',502,['first']),
                        ('targeted','retry',502,['first']),
                        ('exhausted','retry',502,['first','fallback']),
                        ('reject','retry',200,['first','fallback'])]
                    if engine == 'codex' or collect:
                        cases.append(('keepalive','retry',200,['first','fallback']))
                    for mode,key,expected,names in cases:
                        server.hits,server.mode,server.error = [],mode,copy.deepcopy(CONTINUATION_ERROR)
                        body = payload(not collect)
                        rid,status,raw = call(body,key,mode=='targeted')
                        assert status == expected,(engine,collect,event,mode,status,raw[:300])
                        assert [n for n,_ in server.hits] == names,(engine,collect,event,mode,status)
                        assert all(w == server.hits[0][1] for _,w in server.hits),'wire changed on fallback'
                        assert all(json.loads(w)['input'] == body['input'] for _,w in server.hits),'encrypted history changed'
                        if expected == 200:
                            assert b'OK' in raw
                            successful.append(rid)
                        count += 1
                    # Ordinary validation and malformed history must still fail without fallback.
                    for field,value in [('code','other'),('param',None),
                        ('param',''),('param','input[1]'),('message','Quoted: '+CONTINUATION_ERROR['error']['message'])]:
                        server.error = copy.deepcopy(CONTINUATION_ERROR)
                        server.error['error'][field] = value
                        server.mode,server.hits = 'reject',[]
                        _,status,_ = call(payload(not collect))
                        assert status == 400 and len(server.hits) == 1,(event,field,status)
                        count += 1
                    # Missing/wrong type can map to an ordinary retryable 500;
                    # strict type matching is covered by the classifier units.
                    for field in ['code','param','message']:
                        server.error = copy.deepcopy(CONTINUATION_ERROR)
                        del server.error['error'][field]
                        server.hits = []
                        _,status,_ = call(payload(not collect))
                        # Missing message is still invalid_request_error, not a recognized fallback.
                        assert status == 400 and len(server.hits) == 1,(event,field,status)
                        count += 1
                    server.error = copy.deepcopy(CONTINUATION_ERROR)
                    for items in [[{'role':'user','content':'plain'}],
                        [{'type':'reasoning','encrypted_content':''}],
                        [{'type':'compaction','encrypted_content':None}],
                        [{'type':'reasoning','encrypted_content':'opaque'},{'type':'compaction'}]]:
                        body = payload(not collect); body['input'] = items
                        server.hits = []
                        _,status,_ = call(body)
                        assert status == 400 and len(server.hits) == 1,(event,status)
                        count += 1
                    server.hits,server.mode = [],'postcommit'
                    _,status,raw = call(payload(not collect))
                    assert len(server.hits) == 1 and status == (400 if collect else 200),(status,raw[:200])
                    if not collect: assert b'started' in raw
                    count += 1
                deadline = time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(sum(f.get('kind')=='billing' and f.get('request_id')==rid for f in server.facts)>=2 for rid in successful): break
                    time.sleep(.1)
                for rid in successful:
                    facts = [f for f in server.facts if f.get('request_id')==rid]
                    bill = [f['status'] for f in facts if f.get('kind')=='billing']
                    assert bill == [200,200],(rid,bill)
                    assert any(f.get('stage')=='responses_encrypted_content_fallback'
                        and f.get('trace_detail',{}).get('attempt_status_code')==200
                        and f.get('trace_detail',{}).get('semantic_status_code')==400
                        and f.get('trace_detail',{}).get('routing_status_code')==502 for f in facts),(rid,facts)
                encoded = json.dumps(server.facts)+(root/'log').read_text()
                for secret in ['gAAAAopaque-do-not-log','c2c:opaque-do-not-log','fixture-secret','private-history','private-result']:
                    assert secret not in encoded,secret
                assert (root/'api.json').read_bytes() == original_config
                print(f'PASS exact continuation SSE engine={engine} collect={collect} hedging={hedging}: {count} HTTP cases; raw200/semantic400/routing502, unchanged body, no cooldown')
            finally:
                process.terminate(); process.wait(timeout=15)
                server.shutdown(); server.server_close()


if __name__ == '__main__':
    for engine in ['gpt','codex']:
        for collect in [False,True]: verify(Path(sys.argv[1]).resolve(),engine,collect)
    verify(Path(sys.argv[1]).resolve(),'codex',False,True)
