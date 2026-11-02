"""Exact standard-only relay rejection: retain priority and retry on loopback."""
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
from verify_encrypted_content_fallback import RESPONSE, payload

MESSAGE = 'oai-basispoints supports only the standard service tier; omit service_tier or use auto/default; Fast/priority is not supported'
ERROR = {'error': {'code': 'unsupported_service_tier', 'message': MESSAGE, 'type': 'upstream_error'}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        name = self.path.split('/')[1]
        self.server.hits.append((name, raw))
        reject = name != 'fallback' or self.server.mode == 'exhausted'
        status, content_type = (400 if reject else 200), 'application/json'
        if self.path.endswith('/chat/completions'):
            response = {'id':'chat_fixture', 'object':'chat.completion', 'model':'test-model',
                        'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}]}
            events = [{'choices':[{'index':0,'delta':{'content':'OK'},'finish_reason':None}]},
                      {'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}]
            success = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+b'data: [DONE]\n\n'
        else:
            response = RESPONSE
            events = [{'type':'response.output_text.delta','delta':'OK'},
                      {'type':'response.completed','response':response}]
            success = b''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)
        if self.server.mode == 'postcommit':
            status, content_type = 200, 'text/event-stream'
            event={'type':'error',**self.server.error}
            data = success.split(b'\n\n')[0]+b'\n\n'+f'event: error\ndata: {json.dumps(event)}\n\n'.encode()
        elif reject:
            data = json.dumps(self.server.error).encode()
        elif body.get('stream'):
            content_type, data = 'text/event-stream', success
        else:
            data = json.dumps(response).encode()
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        try: self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError): pass


def verify(binary, engine, chat=False, collect=False, hedging=False):
    server = ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    server.hits,server.facts,server.error,server.mode = [],[],ERROR,'reject'
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-tier-') as td:
        root,port = Path(td),free_port()
        base = f'http://127.0.0.1:{server.server_port}'
        path = '/v1/chat/completions' if chat else '/v1/responses'
        preferences = {'cooldown_period':300}
        if collect: preferences['upstream_stream_mode']='stream_only'
        config = {'providers':[{'provider':name,'engine':engine,'api':'fixture-upstream-secret',
                   'base_url':base+'/'+name+path,'model':['test-model'],'preferences':preferences}
                   for name in ['first','fallback','third','fourth','fifth']],
                  'api_keys':[{'api':key,'model':[name+'/*' for name in names],'preferences':{'AUTO_RETRY':limit}}
                    for key,names,limit in [('retry',['first','fallback'],1),('no-retry',['first','fallback'],False),
                      ('bounded',['first','fallback','third','fourth','fifth'],3)]],
                  'preferences':{'hedging':{'enabled':hedging,'max_inflight_attempts':2},'unknown_upstream_400_retries':3}}
        original_config=json.dumps(config).encode();(root/'config').write_bytes(original_config)
        env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','TMPDIR','LANG','DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'config'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   FACTS_INGEST_URL=base+'/facts',FACTS_INGEST_TOKEN='fixture',FACTS_SPOOL_DIR=str(root/'facts'),NO_PROXY='127.0.0.1,localhost')
        count,expected_facts=0,[]
        def body(stream=False):
            if chat:
                return {'model':'test-model','messages':[{'role':'user','content':'fixture'}],
                        'service_tier':'priority','stream':stream}
            return {**payload(stream),'service_tier':'priority'}
        def call(b=None,key='retry',target=False):
            rid=f'tier-{engine}-{chat}-{collect}-{hedging}-{count}'
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Request-ID':rid}
            if target:headers['X-Uni-API-Provider']='first'
            c=http.client.HTTPConnection('127.0.0.1',port,timeout=15)
            try:
                c.request('GET' if b is None else 'POST','/healthz' if b is None else path,None if b is None else json.dumps(b),headers)
                r=c.getresponse();return rid,r.status,r.read()
            finally:c.close()
        with (root/'log').open('w') as log:
            process=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[1]==200:break
                    except OSError:pass
                    time.sleep(.05)
                else:raise AssertionError('fixture not ready')
                def check(b=None,key='retry',target=False,want=200,names=('first','fallback'),facts=False):
                    nonlocal count
                    b=body() if b is None else b
                    server.hits=[]
                    rid,status,raw=call(b,key,target)
                    assert status==want,(engine,chat,collect,hedging,count,status,raw[:300])
                    assert [n for n,_ in server.hits]==list(names),(count,[n for n,_ in server.hits])
                    # All attempts preserve the requested tier and complete signed/tool history.
                    for _,wire in server.hits:
                        expected={**b,'stream':True} if collect else b
                        if engine=='codex':expected.setdefault('instructions','')
                        actual=json.loads(wire)
                        assert actual==expected,('wire values changed',[k for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k)])
                    if len(server.hits)>1:assert len({wire for _,wire in server.hits})==1,'wire bytes changed on fallback'
                    if facts:expected_facts.append((rid,want))
                    count+=1
                    return raw
                for stream in [False,True]:
                    if collect and stream:continue
                    for key,target,want,names in [('retry',False,200,('first','fallback')),('no-retry',False,502,('first',)),
                      ('retry',True,502,('first',)),('retry',False,200,('first','fallback'))]:
                        if chat and target:continue  # Chat's existing API rejects targeting before dispatch.
                        check(body(stream),key,target,want,names,True)
                server.mode='exhausted'
                check(key='bounded',want=502,names=('first','fallback','third','fourth'),facts=True)
                server.mode='reject'
                # Invalid/standard tiers and unrelated errors must stay 400 even with unknown400=3.
                for tier in [None,'default','auto','fast','PRIORITY',1]:
                    check({**body(),'service_tier':tier},want=400,names=('first',))
                b=body();del b['service_tier'];check(b,want=400,names=('first',))
                for patch in [{'code':'invalid_value'},{'type':'invalid_request_error'},{'param':'service_tier'},
                              {'param':''},{'message':MESSAGE+'.'},{'message':'Invalid input: '+MESSAGE}]:
                    server.error={'error':{**ERROR['error'],**patch}}
                    check(want=400,names=('first',))
                server.error={'error':{**ERROR['error'],'param':None}};check()
                for error in [{'detail':ERROR['error']},{'error':{'message':json.dumps(ERROR)}},
                              {'error':{'code':'invalid_type','message':'Invalid input'},'input':ERROR}]:
                    server.error=error;check(want=400,names=('first',))
                server.error=ERROR;server.mode='postcommit'
                check(body(True),want=200,names=('first',))
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='request' and f.get('request_id')==rid for f in server.facts) for rid,_ in expected_facts):break
                    time.sleep(.1)
                for rid,want in expected_facts:
                    selected=[f for f in server.facts if f.get('request_id')==rid]
                    assert any(f.get('kind')=='billing' and f.get('status')==400 for f in selected),rid
                    assert any(f.get('stage')=='service_tier_fallback' and f.get('trace_detail',{}).get('routing_status_code')==502 for f in selected),rid
                    assert any(f.get('kind')=='request' and f.get('status')==want for f in selected),rid
                for secret in ['fixture-upstream-secret','gAAAAopaque-do-not-log','c2c:opaque-do-not-log','private-history','private-result']:
                    assert secret not in json.dumps(server.facts)
                assert (root/'config').read_bytes()==original_config
                print(f'PASS service tier engine={engine} chat={chat} collect={collect} hedging={hedging}: {count} HTTP cases, unchanged priority/history, raw400/routing502, bounded retries, no cooldown')
            finally:
                process.terminate();process.wait(timeout=10)
                server.shutdown();server.server_close()


if __name__=='__main__':
    binary=Path(sys.argv[1]).resolve()
    for engine,chat,collect,hedging in [('gpt',False,False,False),('codex',False,False,False),
        ('gpt',True,False,False),('gpt',False,True,False),('codex',False,False,True)]:
        verify(binary,engine,chat,collect,hedging)
