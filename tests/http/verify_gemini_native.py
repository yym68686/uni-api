"""Offline HTTP protocol matrix. No provider credentials or paid requests."""
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

USAGE = {"promptTokenCount":100,"cachedContentTokenCount":60,"candidatesTokenCount":12,"thoughtsTokenCount":8,"totalTokenCount":120}
CHAT_USAGE = {"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":60},"completion_tokens_details":{"reasoning_tokens":8}}

class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append((self.path,body,dict(self.headers)))
        stream = body.get('stream') or ':streamGenerateContent' in self.path
        model = body.get('model','')
        if ':countTokens' in self.path:
            value = {'totalTokens':17}
        elif '/models/' in self.path:
            assert 'contents' in body and 'model' not in body and 'stream' not in body, body
            value = {'candidates':[{'index':0,'content':{'role':'model','parts':[{'text':'hello','thoughtSignature':'preserved'}]},'finishReason':'STOP'}],'usageMetadata':USAGE,'modelVersion':'native-upstream','extraNativeField':{'untouched':True}}
            events = [value]
            mode = body.get('contents',[{}])[0].get('parts',[{}])[0].get('text')
            if mode == 'late-usage':
                terminal = dict(value); terminal.pop('usageMetadata')
                events = [terminal, {'usageMetadata': USAGE}]
            if mode == 'late-error': events = [value, {'error': {'code':503, 'message':'late failure'}}]
            if mode == 'truncated':
                value['candidates'][0]['finishReason'] = 'MAX_TOKENS'
            if mode == 'json-error': value = {'error': {'code':503,'message':'semantic failure'}}
        elif self.path.endswith('/responses'):
            assert 'input' in body and 'messages' not in body, body
            value = {'id':'resp_fixture','object':'response','status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'hello'}]}],'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120,'input_tokens_details':{'cached_tokens':60},'output_tokens_details':{'reasoning_tokens':8}}}
            events = [{'type':'response.created','response':{'id':'resp_fixture','status':'in_progress'}},{'type':'response.output_text.delta','delta':'hello'},{'type':'response.completed','response':value}]
        elif self.path.endswith('/messages'):
            assert 'messages' in body and body['max_tokens']>0, body
            value = {'id':'msg_fixture','type':'message','role':'assistant','content':[{'type':'text','text':'hello'}],'stop_reason':'end_turn','usage':{'input_tokens':40,'cache_read_input_tokens':60,'output_tokens':20}}
            events = [{'type':'message_start','message':{'id':'msg_fixture','usage':value['usage']}},{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'hello'}},{'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':20}},{'type':'message_stop'}]
        else:
            assert 'messages' in body, body
            value = {'id':'chat_fixture','choices':[{'index':0,'message':{'role':'assistant','content':'hello'},'finish_reason':'stop'}],'usage':CHAT_USAGE}
            events = [{'choices':[{'index':0,'delta':{'content':'hello'},'finish_reason':None}]},{'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},{'choices':[],'usage':CHAT_USAGE}]
        if model == 'tool':
            events = [{'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'call_fixture','function':{'name':'weather','arguments':'{"city":'}}]},'finish_reason':None}]},{'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"Paris"}'}}]},'finish_reason':None}]},{'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]},{'choices':[],'usage':CHAT_USAGE}]
        if model == 'broken': events = [{'choices':[{'index':0,'delta':{'content':'partial'}}]}]
        if model == 'error': events = [{'error':{'message':'fixture failure','code':429}}]
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream' if stream else 'application/json')
        self.end_headers()
        raw = ''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode() if stream else json.dumps(value).encode()
        try:
            for offset in range(0,len(raw),7): self.wfile.write(raw[offset:offset+7]); self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError): pass

def verify(binary):
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream); upstream.hits=[]
    threading.Thread(target=upstream.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-gemini-') as directory:
        root=Path(directory); port=free_port(); base=f'http://127.0.0.1:{upstream.server_port}'
        providers=[]
        providers.append({'provider':'vertex','engine':'vertex','base_url':base,'api':['ab.fixture'],'model':[{'vertex':'vertex'}]})
        for name,engine,path in [('gemini','gemini','/v1beta'),('responses','gpt','/v1/responses'),('codex','codex','/v1/responses'),('claude','claude','/v1/messages'),('chat','gpt','/v1/chat/completions')]:
            providers.append({'provider':name,'engine':engine,'base_url':base+path,'api':'fixture-provider','model':[{name:name}]+([{'tool':'tool'},{'broken':'broken'},{'error':'error'}] if name=='chat' else [])})
        (root/'config.json').write_text(json.dumps({'providers':providers,'api_keys':[{'api':'fixture-client','model':['all'],'preferences':{'AUTO_RETRY':False}}]}))
        env={k:v for k,v in os.environ.items() if not k.startswith(('FACTS_S3_','UNI_API_CONTROL_RESTORE_','OTEL_','FUGUE_OTEL_')) and k!='CONFIG_URL'}
        env.update(PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'config.json'),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',NO_PROXY='127.0.0.1,localhost')
        def call(path,body=None,auth='header',extra=None):
            headers={'Content-Type':'application/json'}
            if auth=='header': headers['x-goog-api-key']='fixture-client'
            elif auth=='bearer': headers['Authorization']='Bearer fixture-client'
            elif auth=='query': path+=('&' if '?' in path else '?')+'key=fixture-client'
            headers.update(extra or {})
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=15)
            try:
                conn.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body),headers)
                res=conn.getresponse(); raw=res.read(); return res.status,dict(res.getheaders()),raw
            finally: conn.close()
        def decode(raw): return [json.loads(line[5:]) for line in raw.decode().splitlines() if line.startswith('data:')]
        with (root/'gateway.log').open('w+') as log:
            process=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            try:
                for _ in range(160):
                    try:
                        if call('/healthz')[0]==200: break
                    except OSError: pass
                    time.sleep(.05)
                body={'contents':[{'role':'user','parts':[{'text':'hello'}]}]}
                for model in ['gemini','responses','codex','claude','chat']:
                    for version in ['v1beta','v1']:
                        for action,suffix in [('generateContent',''),('streamGenerateContent','?alt=sse'),('streamGenerateContent','')]:
                            status,headers,raw=call(f'/{version}/models/{model}:{action}{suffix}',body)
                            assert status==200,(model,action,status,raw)
                            frames=decode(raw) if suffix else (json.loads(raw) if action.startswith('stream') else [json.loads(raw)])
                            assert any(f.get('candidates') for f in frames),(model,frames)
                            assert not any(f.get('error') for f in frames),(model,frames)
                            usage=[f['usageMetadata'] for f in frames if 'usageMetadata' in f][-1]
                            assert usage['promptTokenCount']==100 and usage['candidatesTokenCount']+usage.get('thoughtsTokenCount',0)==20,(model,usage)
                            assert usage['cachedContentTokenCount']==60,(model,usage)
                            if model=='gemini': assert frames[-1]['extraNativeField']=={'untouched':True}
                for auth in ['query','bearer']:
                    assert call('/v1beta/models/chat:generateContent',body,auth)[0]==200
                assert call('/v1beta/models/chat:generateContent',body,'none')[0] in [401,403]
                assert call('/v1beta/models/gemini:countTokens',body)[0]==200
                assert call('/v1beta/models/vertex:countTokens',body)[0]==200
                assert call('/v1beta/models/chat:countTokens',body)[0]>=400
                status,_,raw=call('/v1beta/models',auth='query'); assert status==200 and any(m['name']=='models/responses' for m in json.loads(raw)['models'])
                status,_,raw=call('/v1beta/models/chat:streamGenerateContent?alt=sse',body,extra={'Idempotency-Key':'native-fixture'}); assert status==200 and decode(raw)[-1]['usageMetadata']['totalTokenCount']==120
                status,_,raw=call('/v1beta/models/tool:streamGenerateContent?alt=sse',body)
                calls=[p['functionCall'] for f in decode(raw) for c in f.get('candidates',[]) for p in c['content']['parts'] if 'functionCall' in p]
                assert calls==[{'id':'call_fixture','name':'weather','args':{'city':'Paris'}}],calls
                status,_,raw=call('/v1beta/models/broken:streamGenerateContent?alt=sse',body)
                assert status==200 and decode(raw)[-1].get('error'),raw
                status,_,raw=call('/v1beta/models/error:streamGenerateContent?alt=sse',body)
                assert status>=400,(status,raw)
                for streaming in [False,True]:
                    status,_,raw=call('/v1/responses',{'model':'gemini','input':'hello','stream':streaming},'bearer')
                    assert status==200,(status,raw)
                    response=decode(raw)[-1]['response'] if streaming else json.loads(raw)
                    assert response['usage']['input_tokens']==100 and response['usage']['output_tokens']==20,response
                    assert response['usage']['input_tokens_details']['cached_tokens']==60,response
                for streaming in [False, True]:
                    status,_,raw=call('/v1/responses',{'model':'gemini','input':'truncated','stream':streaming},'bearer')
                    response=decode(raw)[-1]['response'] if streaming else json.loads(raw)
                    assert status==200 and response['status']=='incomplete' and response['incomplete_details']['reason']=='max_output_tokens',response
                status,_,raw=call('/v1/responses',{'model':'gemini','input':'late-usage','stream':True},'bearer')
                assert decode(raw)[-1]['response']['usage']['input_tokens_details']['cached_tokens']==60,raw
                for marker in ['json-error','late-error']:
                    payload={'contents':[{'parts':[{'text':marker}]}]}
                    endpoint='streamGenerateContent?alt=sse' if marker=='late-error' else 'generateContent'
                    status,_,raw=call('/v1beta/models/gemini:'+endpoint,payload)
                    assert (status>=400 if marker=='json-error' else decode(raw)[-1].get('error')),raw
                assert all('fixture-client' not in path and headers.get('X-Goog-Api-Key') is None for path,_,headers in upstream.hits)
                print('Gemini HTTP matrix passed: 30 generation combinations, auth, catalog, countTokens, idempotency, tools, failures, Responses reverse conversion')
            except Exception:
                log.flush(); log.seek(0); print(log.read()[-7000:]); raise
            finally:
                process.terminate(); process.wait(timeout=10); upstream.shutdown(); upstream.server_close()

if __name__=='__main__': verify(Path(sys.argv[1]).resolve())
