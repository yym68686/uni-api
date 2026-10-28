"""Text-only alpha/search relay failures: exact matching and unchanged failover."""
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
from fact_sink import accept_facts
from verify_dispatch_timing import free_port

MESSAGE = 'Search input supports text messages, not encrypted history or tool items'
ERROR = {'error': {'message': MESSAGE, 'type': 'invalid_request_error', 'code': 'bad_request'}}
BODY = {'model': 'search-model', 'id': 'search-session', 'max_output_tokens': 10000,
        'input': [{'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'fixture query'}, {'type': 'input_text', 'text': 'fixture context'}],
            'internal_chat_message_metadata_passthrough': {'content_item_kinds': ['user.image'], 'turn_id': 'fixture'}}],
        'commands': {'search_query': [{'q': 'fixture query'}], 'response_length': 'long'},
        'settings': {'allowed_callers': ['direct'], 'external_web_access': False}}
SUCCESS = {'output': 'fixture search result', 'results': [{'type': 'text_result', 'ref_id': 'fixture', 'title': 'fixture', 'url': 'https://example.invalid/'}]}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        channel = self.path.split('/')[1]
        self.server.hits.append((channel, self.path, raw))
        rejected = channel != 'fallback'
        value = self.server.error if rejected else SUCCESS
        data = json.dumps(value).encode()
        self.send_response(self.server.status if rejected else 200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def verify(binary):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream.hits, upstream.facts, upstream.error, upstream.status = [], [], ERROR, 400
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='alpha-search-retry-') as td:
        root, port = Path(td), free_port()
        base = f'http://127.0.0.1:{upstream.server_port}'
        providers = [{'provider': n, 'engine': engine, 'base_url': base+'/'+n+'/v1/responses', 'api': 'fixture-upstream',
                      'model': ['search-model'], 'preferences': {'cooldown_period': 300}}
                     for n, engine in [('first','gpt'), ('fallback','gpt'), ('also-reject','gpt'), ('codex','codex')]]
        keys = [{'api': 'retry', 'model': ['first/*','fallback/*']},
                {'api': 'codex-key', 'model': ['codex/*','fallback/*']},
                {'api': 'no-retry', 'model': ['first/*','fallback/*'], 'preferences': {'AUTO_RETRY': False}},
                {'api': 'exhausted', 'model': ['first/*','also-reject/*'], 'preferences': {'AUTO_RETRY': 1}}]
        (root/'config.json').write_text(json.dumps({'providers': providers, 'api_keys': keys, 'preferences': {'hedging': {'enabled': False}}}))
        env = {k:v for k,v in os.environ.items() if k in ('PATH','HOME','TMPDIR','LANG','DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   FACTS_INGEST_URL=base+'/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'), NO_PROXY='127.0.0.1,localhost')
        def call(path='/healthz', body=None, key='retry', rid='fixture'):
            c = http.client.HTTPConnection('127.0.0.1',port,timeout=15)
            try:
                c.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body),
                          {'Content-Type':'application/json','Authorization':'Bearer '+key,'X-Request-ID':rid})
                r=c.getresponse(); return r.status,r.read()
            finally: c.close()
        count, expected_facts = 0, []
        with (root/'log').open('w') as log:
            proc=subprocess.Popen([str(binary)],env=env,cwd=root,stdout=log,stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[0]==200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('gateway fixture not ready')
                def check(body=BODY, key='retry', want=200, names=('first','fallback'), path='/v1/alpha/search'):
                    nonlocal count
                    rid='search-'+str(count)
                    upstream.hits=[]
                    status,raw=call(path,body,key,rid)
                    assert status==want,(count,key,path,status,raw)
                    assert [n for n,_,_ in upstream.hits]==list(names),(count,upstream.hits)
                    for _,p,wire in upstream.hits:
                        if path=='/v1/alpha/search':
                            assert p.endswith('/v1/alpha/search') and json.loads(wire)==body,(count,p)
                    if len(upstream.hits)>1:
                        assert all(w==upstream.hits[0][2] for _,_,w in upstream.hits), 'wire bytes changed on retry'
                    if want==200: assert json.loads(raw)==SUCCESS
                    if want==502: assert json.loads(raw)==ERROR
                    count+=1
                    return rid
                # Repeat after failure to prove no provider/key-wide cooldown is introduced.
                for key,want,names in [('retry',200,('first','fallback')), ('no-retry',502,('first',)),
                                       ('exhausted',502,('first','also-reject')), ('retry',200,('first','fallback')),
                                       ('codex-key',200,('codex','fallback'))]:
                    rid=check(key=key,want=want,names=names)
                    expected_facts.append((rid,want))
                for param in [None]:
                    upstream.error={'error':{**ERROR['error'],'param':param}}
                    check()
                upstream.error=ERROR
                for input_value in ['fixture query', [{'role':'user','content':'fixture query'}],
                                    [{'role':'assistant','content':[{'type':'output_text','text':'fixture query'}]}]]:
                    b=copy.deepcopy(BODY); b['input']=input_value
                    check(b)
                for patch in [{'type':'upstream_error'},{'code':'invalid_type'},{'code':400},{'code':None},
                              {'param':'input'},{'param':''},{'message':MESSAGE+'.'},
                              {'message':'Invalid input: '+MESSAGE},{'message':MESSAGE.lower()}]:
                    upstream.error={'error':{**ERROR['error'],**patch}}
                    check(want=400,names=('first',))
                for error in [{'detail':ERROR['error']},{'error':{'message':json.dumps(ERROR)}},
                              {'error':{'message':'invalid input'},'debug':ERROR}]:
                    upstream.error=error
                    check(want=400,names=('first',))
                upstream.error=ERROR
                for input_value in [None,'',' ',[],{},
                                    [{'type':'reasoning','encrypted_content':'opaque'}],
                                    [{'type':'function_call','name':'lookup','arguments':'{}'}],
                                    [{'type':'function_call_output','call_id':'x','output':'fixture'}],
                                    [{'role':'tool','content':'fixture'}],
                                    [{'role':'user','content':[]}],
                                    [{'role':'user','content':[{'type':'input_image','image_url':'fixture'}]}],
                                    [{'role':'user','content':[{'type':'input_text','text':1}]}],
                                    [{'role':'user','content':'fixture','encrypted_content':'opaque'}]]:
                    b=copy.deepcopy(BODY); b['input']=input_value
                    # The existing image resolver rejects malformed images before dispatch.
                    names=() if input_value==[{'role':'user','content':[{'type':'input_image','image_url':'fixture'}]}] else ('first',)
                    check(b,want=400,names=names)
                check({'model':'search-model','messages':[{'role':'user','content':'fixture'}]},
                      path='/v1/chat/completions',want=400,names=('first',))
                # A successful response is never reinterpreted because it contains this text.
                upstream.status=200
                upstream.hits=[]
                status,raw=call('/v1/alpha/search',BODY,rid='search-success-envelope')
                assert status==200 and json.loads(raw)==ERROR and len(upstream.hits)==1
                count+=1
                # All billed attempts retain raw400 while routing is502 and the final result is accurate.
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='request' and f.get('request_id')==rid for f in upstream.facts) for rid,_ in expected_facts): break
                    time.sleep(.1)
                for rid,want in expected_facts:
                    facts=[f for f in upstream.facts if f.get('request_id')==rid]
                    assert any(f.get('kind')=='billing' and f.get('status')==400 for f in facts),(rid,facts)
                    assert any(f.get('stage')=='upstream_attempt' and f.get('trace_detail',{}).get('attempt_status_code')==400 for f in facts),(rid,facts)
                    assert any(f.get('stage')=='alpha_search_text_fallback' and f.get('trace_detail',{}).get('routing_status_code')==502 for f in facts),(rid,facts)
                    assert any(f.get('kind')=='request' and f.get('status')==want for f in facts),(rid,facts)
                    if want==200: assert any(f.get('kind')=='billing' and f.get('status')==200 for f in facts)
                assert 'fixture-upstream' not in json.dumps(upstream.facts)
                print(f'PASS alpha/search fallback: {count} HTTP cases; raw400/routing502, unchanged wire, no global cooldown')
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                upstream.shutdown(); upstream.server_close()


if __name__=='__main__': verify(Path(sys.argv[1]).resolve())
