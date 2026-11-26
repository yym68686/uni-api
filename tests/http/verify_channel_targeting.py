"""Offline regression: diagnostics cannot retry or fall back to another channel."""
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
from verify_dispatch_timing import free_port

class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append((self.path, body.get('model')))
        self.server.requests.append((self.path, body, {k.lower():v for k,v in self.headers.items()}))
        bad = self.path.startswith('/bad/')
        response = {'error': {'message': 'fixture failure'}} if bad else {
            'id':'resp_check', 'object':'response', 'status':'completed', 'model':'gpt-6-astra',
            'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':'未知'}]}],
            'usage':{'input_tokens':25,'output_tokens':1,'total_tokens':26}}
        if not bad and self.path.endswith('/chat/completions'):
            response = {'id':'chat_check','object':'chat.completion','model':'gpt-6-astra','choices':[{'index':0,'message':{'role':'assistant','content':'未知'},'finish_reason':'stop'}],'usage':{'prompt_tokens':25,'completion_tokens':1,'total_tokens':26}}
        if not bad and (':streamGenerateContent' in self.path or body.get('stream')):
            frame = ({'candidates':[{'index':0,'content':{'role':'model','parts':[{'text':'native sample'}]},'finishReason':'STOP'}]}
                     if ':streamGenerateContent' in self.path else
                     {'id':'chat_check','model':body['model'],'choices':[{'index':0,'delta':{'content':'native sample'},'finish_reason':'stop'}]})
            raw = ('data: '+json.dumps(frame)+'\n\n'+('' if 'candidates' in frame else 'data: [DONE]\n\n')).encode()
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            return
        raw=json.dumps(response).encode()
        self.send_response(503 if bad else 200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

def verify(binary, endpoint):
    server=ThreadingHTTPServer(('127.0.0.1',0),Upstream);server.hits=[];server.requests=[]
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-channel-check-') as directory:
        root=Path(directory);port=free_port()
        config={'providers':[{'provider':p,'engine':'gpt','base_url':f'http://127.0.0.1:{server.server_port}/{p}{endpoint}','api':'upstream-test','model':['gpt-6-astra'],'preferences':{'cooldown_period':0}} for p in ['bad','good']], 'api_keys':[{'api':'admin-test','model':['bad/*']},{'api':'ordinary-test','model':['all']}], 'preferences':{'hedging':{'enabled':True,'max_inflight_attempts':2}}}
        for provider in config['providers']:
            provider['model'].append({'gemini-4-argon-upstream':'gemini-4-argon'})
        good=config['providers'][1]
        config['providers'].extend([
            dict(good,provider='custom',base_url=good['base_url']+'/deployment'),
            dict(good,provider='overridden',preferences={'post_body_parameter_overrides':{'reasoning_effort':'high'}}),
            dict(good,provider='excluded',exclude_request_rules=[{'match':{'reasoning_effort':'low'}}]),
        ])
        relay={**good,'provider':'relay','engine':'gemini','base_url':f'http://127.0.0.1:{server.server_port}/relay/v1beta',
               'model':[{'gemini-4-neon-upstream':'gemini-4-neon'}]}
        config['providers'].extend([
            relay,
            dict(relay,provider='relay-custom',base_url=relay['base_url']+'/deployment'),
            dict(relay,provider='relay-composite',api='account,secret'),
            dict(relay,provider='relay-overridden',preferences={'post_body_parameter_overrides':{'reasoning_effort':'high'}}),
        ])
        config['providers'][1]['preferences']['passthrough_request_headers'] = ['x-uni-api-provider','x-uni-api-fingerprint-protocol','x-api-key']
        (root/'api.json').write_text(json.dumps(config))
        env=dict(os.environ,PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'api.json'),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),NO_PROXY='127.0.0.1,localhost',RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0')
        # Do not inherit production export settings into a fixture.
        env={k:v for k,v in env.items() if not k.startswith('FACTS_')}
        with (root/'log').open('w+') as log:
            process=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            def request(method,path,key='admin-test',target=None,protocol=None,payload=None,native_auth=False):
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
                if target is not None:headers['X-Uni-API-Provider']=target
                if protocol:
                    headers['X-Uni-API-Fingerprint-Protocol']=protocol
                    headers['X-Uni-API-Probe-Unconfigured-Model']='true'
                if native_auth:
                    del headers['Authorization']
                    headers['x-goog-api-key']=key
                body=json.dumps(payload or {'model':'gpt-6-astra','input':[{'role':'user','content':'test'}]}) if method=='POST' else None
                c.request(method,path,body,headers);r=c.getresponse();status=r.status;raw=r.read();c.close();return status,raw
            try:
                for _ in range(100):
                    try:
                        if request('GET','/healthz')[0]==200:break
                    except OSError:pass
                    time.sleep(.05)
                status,raw=request('GET','/v1/observability/runtime');assert json.loads(raw)['capabilities']['targeted_responses']
                for key,target,expected in [('ordinary-test','good',403),('admin-test','missing',404)]:
                    assert request('POST','/v1/responses',key,target)[0]==expected
                assert not server.hits,server.hits
                status,raw=request('POST','/v1/responses',target='good')
                assert status==200,(status,raw)
                assert server.hits==[(f'/good{endpoint}','gpt-6-astra')],server.hits
                server.hits.clear()
                status,raw=request('POST','/v1/responses',target='bad')
                assert status>=400,(status,raw)
                assert server.hits==[(f'/bad{endpoint}','gpt-6-astra')],server.hits
                caps=json.loads(request('GET','/v1/observability/runtime')[1])['capabilities']
                assert set(caps['targeted_fingerprint_protocols'])=={'gemini','chat'}
                native='/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse'
                gbody={'contents':[{'role':'user','parts':[{'text':'fixture'}]}],
                       'generationConfig':{'maxOutputTokens':8192,'thinkingConfig':{'thinkingLevel':'low'}}}
                chat='/v1/chat/completions'
                cbody={'model':'deepseek-v4.1-flash-1017','messages':[{'role':'user','content':'fixture'}],
                       'reasoning_effort':'low','stream':True}
                server.hits.clear()
                for path,key,target,protocol,payload,expected in [
                    (native,'ordinary-test','good','gemini',gbody,403),
                    (native,'admin-test',None,'gemini',gbody,400),
                    (native,'admin-test','good','chat',gbody,400),
                    (native.replace('streamGenerateContent','countTokens'),'admin-test','good','gemini',gbody,400),
                    (chat,'admin-test','good','unknown',cbody,400),
                    (native,'admin-test','custom','gemini',gbody,400),
                    (chat,'admin-test','overridden','chat',cbody,400),
                    (native,'admin-test','excluded','gemini',gbody,404),
                ]:
                    status,raw=request('POST',path,key,target,protocol,payload)
                    assert status==expected,(path,status,raw)
                assert not server.hits,server.hits
                for path,protocol,payload,upath in [
                    (native,'gemini',gbody,'/v1beta/models/gemini-4-argon-upstream:streamGenerateContent?alt=sse'),
                    (chat,'chat',cbody,chat),
                ]:
                    for target in ['good','bad']:
                        before=len(server.requests)
                        status,raw=request('POST',path,target=target,protocol=protocol,payload=payload,native_auth=protocol=='gemini')
                        assert (status==200 and b'native sample' in raw) if target=='good' else status>=400,(status,raw)
                        sent=server.requests[before:]
                        assert len(sent)==1 and sent[0][:2]==('/'+target+upath,payload),sent
                        headers=sent[0][2]
                        assert headers.get('x-goog-api-key' if protocol=='gemini' else 'authorization')==('upstream-test' if protocol=='gemini' else 'Bearer upstream-test'),headers
                        assert 'admin-test' not in headers.values() and not any(k.startswith('x-uni-api-') for k in headers),headers
                # Same contract as a catalog-driven Neon fingerprint on a /v1beta relay.
                neon=dict(cbody,model='gemini-4-neon',max_tokens=8192)
                before=len(server.requests)
                for key,target,expected in [('ordinary-test','relay',403),('admin-test','relay-custom',400),
                                            ('admin-test','relay-composite',400),('admin-test','relay-overridden',400)]:
                    status,raw=request('POST',chat,key,target,'chat',neon)
                    assert status==expected,(target,status,raw)
                assert len(server.requests)==before,server.requests[before:]
                status,raw=request('POST',chat,target='relay',protocol='chat',payload=neon)
                assert status==200 and b'native sample' in raw and b'[DONE]' in raw,(status,raw)
                [(path,body,headers)]=server.requests[before:]
                assert path=='/relay/v1/chat/completions' and body==dict(neon,model='gemini-4-neon-upstream'),(path,body)
                assert headers['authorization']=='Bearer upstream-test' and 'admin-test' not in headers.values(),headers
                assert not any(k.startswith('x-uni-api-') for k in headers),headers
                neon_native='/v1beta/models/gemini-4-neon:streamGenerateContent?alt=sse'
                status,raw=request('POST',neon_native,target='relay',protocol='gemini',payload=gbody,native_auth=True)
                assert status==200 and b'native sample' in raw,(status,raw)
                assert server.requests[-1][:2]==('/relay/v1beta/models/gemini-4-neon-upstream:streamGenerateContent?alt=sse',gbody)
                # Diagnostic wire selection must not mutate the serving provider.
                assert request('POST','/v1/responses',target='good')[0]==200
                assert server.hits[-1]==(f'/good{endpoint}','gpt-6-astra'),server.hits
                print('PASS targeted diagnostics:',endpoint,'auth, exact target, no retry or fallback')
            except Exception:
                print((root/'log').read_text()[-8000:]);raise
            finally:
                process.terminate()
                try:process.wait(timeout=4)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                server.shutdown();server.server_close()

if __name__=='__main__':
    binary=Path(sys.argv[1]).resolve()
    for endpoint in ['/v1/responses','/v1/chat/completions']:verify(binary,endpoint)
