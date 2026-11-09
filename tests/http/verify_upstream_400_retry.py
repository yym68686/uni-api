"""Every upstream 400: one budget, unchanged requests, loopback only."""
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
RESPONSE = {'id':'resp_fixture','object':'response','status':'completed','model':'test-model',
 'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'OK'}]}],
 'usage':{'input_tokens':10,'output_tokens':2,'total_tokens':12}}

ERROR = {'error': {'type': 'invalid_request_error', 'code': 'invalid_request_error',
                   'message': 'Relay backend hiccup', 'status_code': 400}}
SCOPE = '__uni_console_global_preferences__'
FIELD = '/preferences/unknown_upstream_400_retries'
CHAT = {'id':'chat_fixture','object':'chat.completion','model':'test-model',
        'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}]}


def frame(kind, **fields):
    return f'event: {kind}\ndata: {json.dumps(dict(type=kind, **fields))}\n\n'.encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        shared = self.server.shared
        if self.path == '/facts': return accept_facts(self, shared['facts'])
        raw = self.rfile.read(int(self.headers['Content-Length']))
        name = self.path.split('/')[1]
        shared['hits'].append((name, raw))
        if name=='first' and shared['mode']=='hold':
            shared['entered'].set()
            assert shared['release'].wait(10), 'in-flight fixture not released'
        if name=='second' and shared['mode']=='hedge_probe': time.sleep(.8)
        body = json.loads(raw)
        reject = (name=='first' or shared['mode'] in ['exhausted','postcommit','mixed','toolcommit']
                  or (name=='second' and shared['mode']=='hedge_probe'))
        status, kind = (400 if reject else 200), 'application/json'
        if shared['mode'] == 'mixed' and name == 'second': status = 500
        error = copy.deepcopy(shared['error'])
        if shared['mode'] == 'exact':
            error = {'error':{'message':'上游错误','type':'upstream_error','code':'upstream_error','param':'','status_code':400}}
        value = error if reject else (RESPONSE if 'input' in body else CHAT)
        if shared['mode'] != 'http' and (shared['semantic'] or shared['mode'] in ['postcommit','toolcommit']):
            status, kind = 200, 'text/event-stream'
            if reject:
                if shared['event']=='error':
                    terminal = (frame('error',code=error['error'].get('code'),message=error['error']['message'],
                                      param=error['error'].get('param'),sequence_number=1)
                                # Preserve relay-specific status/type metadata in
                                # nested fixtures; the standard code alone is 400.
                                if shared['flat'] and error['error'].get('code')=='invalid_request_error'
                                else frame('error',error=error['error']))
                else:
                    terminal = frame('response.failed',response={'status':'failed','error':error['error']})
                data = terminal
                if shared['mode'] == 'postcommit':
                    data = frame('response.output_text.delta',delta='started') + data
                elif shared['mode'] == 'toolcommit':
                    data = frame('response.output_item.done',item={'type':'function_call','name':'lookup','call_id':'c1','arguments':'{}'}) + data
                elif shared['mode'] != 'hedge_probe':
                    data = frame('response.created',response={'id':'failed-channel','status':'in_progress','output':[]}) + frame('response.output_text.delta',delta=' \n\t') + data
            else:
                data = frame('response.output_text.delta',delta='OK') + frame('response.completed',response=RESPONSE)
        elif not reject and body.get('stream'):
            kind = 'text/event-stream'
            data = frame('response.output_text.delta',delta='OK') + frame('response.completed',response=RESPONSE)
        else: data = b'<html>arbitrary non-JSON 400</html>' if reject and shared['mode']=='plain' else json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data))); self.end_headers()
        try: self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError): pass


def verify(binary, native=True, semantic=False, hedging=False, collect=False, error=ERROR, normalize=False, flat=False):
    selected_error = error
    shared = dict(hits=[],facts=[],error=error,mode='reject',semantic=semantic,event='error',flat=flat)
    servers = [ThreadingHTTPServer(('127.0.0.1',0),Upstream) for _ in range(4)]
    for server in servers:
        server.shared = shared
        threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='unknown400-') as td:
        root,port = Path(td),free_port()
        preferences = {'cooldown_period':0,'api_key_cooldown_period':0,'keepalive_interval':0}
        if normalize:
            preferences['normalize_responses_custom_tool_call_ids'] = True
        if collect: preferences['upstream_stream_mode'] = 'stream_only'
        # Leave enough scheduling margin for ordinary loopback responses; the
        # explicit slow probe above still crosses this hedge boundary.
        if hedging: preferences['timeout_policy']={'default':{'first_byte':.3,'total':5}}
        providers = [{'provider':name,'engine':'gpt','api':'fixture-upstream-secret',
            'base_url':f'http://127.0.0.1:{servers[index].server_port}/{name}/v1/'+('responses' if native or semantic else 'chat/completions'),
            'model':['test-model'],'preferences':preferences}
            for name,index in [('first',0),('second',0),('third',0),('fourth',0),('fifth',0)]]
        chain = [p['provider']+'/*' for p in providers]
        config = {'providers':providers,'api_keys':[
            {'api':key,'role':'admin' if key=='admin' else 'user','model':chain,
             'preferences':{'AUTO_RETRY':retry,'SCHEDULING_ALGORITHM':'fixed_priority'}}
            for key,retry in [('retry',6),('disabled',False),('one',1),('admin',True)]],
            'preferences':{'hedging':{'enabled':hedging,'max_inflight_attempts':2}}}
        original = json.dumps(config).encode(); (root/'config').write_bytes(original)
        env = {k:v for k,v in os.environ.items() if k in ['PATH','HOME','TMPDIR','LANG','DYLD_LIBRARY_PATH']}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config'),
            RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
            RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
            FACTS_INGEST_URL=f'http://127.0.0.1:{servers[0].server_port}/facts',FACTS_INGEST_TOKEN='fixture',
            FACTS_SPOOL_DIR=str(root/'facts'), NO_PROXY='127.0.0.1,localhost')
        count,expected = 0,[]
        path = '/v1/responses' if native else '/v1/chat/completions'
        body = {'model':'test-model','store':False,'stream':not collect if native else False}
        body['input' if native else 'messages'] = [{'role':'user','content':'fixture private input'}]
        def call(method='GET',path='/healthz',body=None,key='admin',extra=None):
            c = http.client.HTTPConnection('127.0.0.1',port,timeout=20)
            try:
                c.request(method,path,json.dumps(body) if body is not None else None,
                    {'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Request-ID':f'unknown400-{count}',**(extra or {})})
                r=c.getresponse(); return r.status,r.read()
            finally: c.close()
        def limit(value):
            code,raw=call(path='/v1/channel-settings?provider='+SCOPE)
            assert code==200
            view=json.loads(raw)
            mutation={'revision':view['revision'],'operation_id':f'set-{value}-{count}',
                'changes':[{'provider':SCOPE,'set':{FIELD:value}}]}
            code,raw=call('POST','/v1/channel-settings/validate',mutation)
            assert code==200,(code,raw)
            code,raw=call('PATCH','/v1/channel-settings',mutation)
            assert code==200,(code,raw)
        def check(want=200,names=('first','second'),key='retry',payload=None,target=False,flip=False,rescues=None):
            nonlocal count
            shared['hits']=[]
            rid=f'unknown400-{count}'
            def invoke():
                return call('POST',path,body if payload is None else payload,key,
                    {'X-Uni-API-Provider':'first'} if target else {})
            if flip:
                shared['entered'],shared['release']=threading.Event(),threading.Event()
                result=[]
                worker=threading.Thread(target=lambda:result.append(invoke()),daemon=True)
                worker.start()
                assert shared['entered'].wait(5)
                limit(0)
                shared['release'].set(); worker.join(10)
                assert not worker.is_alive() and len(result)==1
                code,raw=result[0]
            else: code,raw=invoke()
            # A precommit semantic failure retains its HTTP error status even
            # when the 400 rescue budget is zero; committed output stays 200.
            assert code==want,(native,semantic,hedging,collect,count,code,raw[:200])
            assert [n for n,_ in shared['hits']]==list(names),(count,[n for n,_ in shared['hits']],names)
            if len(names)>1:
                assert all(w==shared['hits'][0][1] for _,w in shared['hits']),'retry changed wire bytes'
            if want==200 and shared['mode'] not in ['postcommit','toolcommit']: assert b'OK' in raw
            if shared['mode']=='toolcommit' and not collect: assert b'c1' in raw
            if rescues is None: rescues = max(0,len(names)-1)
            if names: expected.append((rid,400 if shared['mode'] in ['postcommit','toolcommit'] else want,len(names),rescues))
            count+=1
        with (root/'log').open('w') as log:
            proc=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            try:
                for _ in range(150):
                    try:
                        if call()[0]==200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('mock gateway not ready')
                check(400,('first',)) # missing config defaults to zero
                limit(1)
                check(); check() # no provider-wide cooldown
                check(key='disabled')
                check(400,('first',) if native else (),key='admin',target=True)
                # Global 400 allowance does not depend on the key's other-error budget.
                check(key='one')
                shared['mode']='hold'; check(flip=True) # policy snapshot captured before request starts
                shared['mode']='reject'; check(400,('first',)) # following request sees zero
                for credits in [1,2,3]:
                    limit(credits); shared['mode']='exhausted'
                    check(400,tuple(['first','second','third','fourth'][:credits+1]))
                if hedging:
                    limit(2); shared['mode']='hedge_probe'
                    check(200,('first','second','third')) # a rescue's hedge trigger cannot burn the route cursor
                limit(1); shared['mode']='mixed'
                if not semantic: check(400,('first','second','third'),rescues=1) # 500 uses the ordinary budget
                shared['mode']='reject'
                for error in [
                    {'error':{**ERROR['error'],'param':'input'}},
                    {'error':{**ERROR['error'],'code':'invalid_value'}},
                    {'error':{**ERROR['error'],'message':'Missing input'}},
                    {'error':{**ERROR['error'],'message':'Content policy rejected'}},
                    {'error':{**ERROR['error'],'type':'safety_error'}},
                    {'error':{'type':'invalid_request_error','code':'invalid_request_error','message':'Invalid request parameter','param':'input'}},
                    {'error':{'type':'invalid_request_error','code':'invalid_value','message':'Invalid request parameter'}},
                    {'error':{'type':'invalid_request_error','code':'invalid_request_error','message':'Invalid request parameter','details':[{'field':'input'}]}},
                    {'error':{'type':'invalid_request_error','code':'invalid_request_error','message':'Invalid request parameter: input'}},
                ]:
                    shared['error']=error; check()
                shared['error']=selected_error
                for patch in [{'store':True},{'background':True},{'tools':[{'type':'web_search'}]}]:
                    b=copy.deepcopy(body); b.update(patch); check(payload=b)
                if native and semantic:
                    shared['event']='response.failed'; check()
                    for mode in ['postcommit','toolcommit']:
                        shared['mode']=mode; check(400 if collect else 200,('first',))
                if not semantic:
                    shared['mode']='plain'; check()
                shared['event']='error'; shared['mode']='reject'
                limit(0); check(400,('first',))
                # The same global zero disables former special-case errors too.
                shared['mode']='exact'; check(400,('first',))
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='request' and f.get('request_id')==rid for f in shared['facts']) for rid,_,_,_ in expected): break
                    time.sleep(.1)
                for rid,want,n,rescues in expected:
                    facts=[f for f in shared['facts'] if f.get('request_id')==rid]
                    assert any(f.get('kind')=='request' and f.get('status')==want for f in facts),(rid,want)
                    bills=[f for f in facts if f.get('kind')=='billing']
                    assert len(bills)==n,(rid,n,[f.get('status') for f in bills])
                    if semantic: assert all(f.get('status')==200 for f in bills),rid
                    events=[f for f in facts if f.get('stage')=='upstream_400_retry']
                    assert len(events)==rescues,(rid,rescues,len(events))
                    assert [f['trace_detail']['upstream_400_retry_used'] for f in events]==list(range(1,rescues+1)),rid
                    assert all(f['trace_detail']['attempt_status_code']==(200 if semantic else 400) for f in events),rid
                encoded=json.dumps(shared['facts'])+(root/'log').read_text()
                assert 'fixture-upstream-secret' not in encoded and 'fixture private input' not in encoded
                assert (root/'config').read_bytes()==original
                print(f'PASS all-upstream-400 native={native} semantic={semantic} hedging={hedging} collect={collect} flat={flat} generic_parameter={selected_error["error"]["message"]=="Invalid request parameter"}: {count} HTTP cases')
            finally:
                proc.terminate(); proc.wait(timeout=15)
                for server in servers: server.shutdown(); server.server_close()


if __name__=='__main__':
    binary=Path(sys.argv[1]).resolve()
    for error in [ERROR, {'error':{'type':'invalid_request_error','code':'unsupported_service_tier','param':'service_tier','message':'Invalid request parameter'}}]:
        verify(binary,native=False,error=error)
        verify(binary,error=error)
        verify(binary,semantic=True,error=error)
        verify(binary,semantic=True,collect=True,error=error)
        verify(binary,semantic=True,hedging=True,error=error)
        verify(binary,semantic=True,error=error,normalize=True)
        verify(binary,native=False,semantic=True,collect=True,error=error)
        verify(binary,native=False,hedging=True,error=error)
    verify(binary,semantic=True,flat=True)
    verify(binary,native=False,semantic=True,collect=True,flat=True)
