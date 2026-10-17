"""Offline stream-only transport, JSON response, failure, and settings contracts."""
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


def event(kind, **values):
    return ('event: ' + kind + '\r\ndata: ' + json.dumps({'type': kind, **values}, ensure_ascii=False) + '\r\n\r\n').encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        assert self.headers.get('X-Uni-Upstream-Stream-Mode') == '1'
        raw=json.dumps({'enabled':True,'snapshot_id':'a'*64,'snapshot':self.server.snapshot}).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

    def do_POST(self):
        p = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        name, model = self.path.split('/')[1], p['model']
        self.server.hits.append((name, model, p.get('stream', False)))
        result = {'id': 'resp_fixture', 'object': 'response', 'status': 'completed', 'model': model,
                  'output': [{'id':'msg_1','type':'message','role':'assistant','status':'completed',
                              'content':[{'type':'output_text','text':'test 中文','annotations':[]}]},
                             {'id':'rs_1','type':'reasoning','encrypted_content':'opaque','summary':[]},
                             {'id':'fc_1','type':'function_call','call_id':'call_1','name':'lookup','arguments':'{"x":1}'}],
                  'usage': {'input_tokens':12,'output_tokens':5,'total_tokens':17,
                            'input_tokens_details':{'cached_tokens':4},'output_tokens_details':{'reasoning_tokens':2}},
                  'vendor_extension': {'preserve': True}}
        if self.path.endswith('/compact') or model == 'follow':
            raw = json.dumps(result).encode(); content_type = 'application/json'; status = 200
        elif not p.get('stream'):
            raw = b'{"error":{"code":"stream_only_mode","type":"stream_required","message":"Set stream true"}}'
            content_type = 'application/json'; status = 400
        else:
            raw = b': heartbeat\r\n\r\n' + event('response.created', response={'id':'resp_fixture','status':'in_progress'})
            if model != 'early-fail':
                raw += event('response.output_text.delta', delta='test 中文')
            if model in ('failed', 'early-fail') and name != 'fallback':
                raw += event('response.failed', response={'status':'failed','error':{'code':'server_error','message':'fixture failed'}})
            elif model == 'truncated':
                pass
            elif model == 'malformed':
                raw += b'data: {bad json}\n\n'
            elif model == 'done-only':
                raw += b'data: [DONE]\n\n'
            elif model == 'oversize':
                raw += b':' + b'x' * 20000 + b'\n\n'
            else:
                kind = 'incomplete' if model == 'incomplete' else 'completed'
                result['status'] = kind
                if kind == 'incomplete': result['incomplete_details'] = {'reason':'max_output_tokens'}
                raw += event('response.' + kind, response=result)
            content_type = 'text/event-stream'; status = 200
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('X-OAIX-Token-ID', 'fixture')
        self.send_header('X-Client-Request-ID', 'fixture-correlation')
        self.end_headers()
        try:
            if model == 'idle': time.sleep(.25)
            if model == 'total':
                for _ in range(10):
                    self.wfile.write(b':\n\n'); self.wfile.flush(); time.sleep(.04)
                return
            # Split UTF-8 and SSE boundaries across separate writes.
            for i in range(0, len(raw), 7):
                self.wfile.write(raw[i:i+7]); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass


def verify(binary):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream); upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    models = ['normal','follow','failed','early-fail','truncated','malformed','done-only','incomplete','oversize','idle','total']
    with tempfile.TemporaryDirectory(prefix='uni-stream-only-') as directory:
        root = Path(directory); port = free_port()
        mode = {'default':'stream_only', 'models':{'follow':'follow_client'}}
        config = {'providers':[{'provider':n,'engine':'gpt', 'base_url':f'http://127.0.0.1:{upstream.server_port}/{n}/v1/'+('chat/completions' if n=='chat' else 'responses'),
                               'api':'fixture','model':models,'preferences':{'upstream_stream_mode':mode, 'cooldown_period':0,
                               'timeout_policy':{'default':{'total':2,'idle':1},'rules':[
                                   {'match':{'model':'idle','stream':False},'timeout':{'idle':.05}},
                                   {'match':{'model':'total','stream':False},'timeout':{'total':.15}}]}}}
                              for n in ['native','fallback','chat']],
                  'api_keys':[{'api':'admin-fixture','model':['all']},
                              {'api':'native-fixture','model':['native/*'],'preferences':{'AUTO_RETRY':False}},
                              {'api':'mixed-fixture','model':['native/*','chat/*'],'preferences':{'AUTO_RETRY':False}},
                              {'api':'retry-fixture','model':['native/*','fallback/*'],'preferences':{'AUTO_RETRY':1}}],
                  'preferences':{'hedging':{'enabled':False}}}
        config_path = root/'api.json'; config_path.write_text(json.dumps(config)); original = config_path.read_bytes()
        env = {k:v for k,v in os.environ.items() if not k.startswith(('FACTS_S3_','UNI_API_CONTROL_RESTORE_'))}
        env.update(PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(config_path),NO_PROXY='127.0.0.1,localhost',
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   RUST_GENERIC_UPSTREAM_RESPONSE_MAX_BYTES='8192')
        def call(method, path, body=None, key='admin-fixture'):
            c=http.client.HTTPConnection('127.0.0.1',port,timeout=8)
            c.request(method,path,json.dumps(body) if body is not None else None,{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            r=c.getresponse(); status=r.status; headers=dict(r.getheaders()); raw=r.read(); c.close()
            return status, headers, raw
        def read(path):
            status, _, raw=call('GET',path); assert status==200,(status,raw); return json.loads(raw)
        with (root/'log').open('w+') as log:
            process=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            try:
                for _ in range(100):
                    try:
                        if call('GET','/healthz')[0]==200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('gateway not ready')
                for key in ['native-fixture','mixed-fixture']:
                    for endpoint in ['/v1/responses','/v1/chat/completions']:
                        for stream in [None,False,True]:
                            p={'model':'normal', 'input':'say test'} if endpoint.endswith('responses') else {'model':'normal','messages':[{'role':'user','content':'say test'}]}
                            if stream is not None: p['stream']=stream
                            status,headers,raw=call('POST',endpoint,p,key)
                            assert status==200,(key,endpoint,status,raw)
                            if stream:
                                assert 'text/event-stream' in headers.get('content-type',''),headers
                            else:
                                assert 'application/json' in headers.get('content-type',''),headers
                                v=json.loads(raw)
                                if endpoint.endswith('responses'):
                                    assert headers.get('x-client-request-id')=='fixture-correlation',headers
                                    assert v['output'][1]['encrypted_content']=='opaque' and v['vendor_extension']=={'preserve':True},v
                                    assert v['usage']['input_tokens_details']['cached_tokens']==4,v
                                else:
                                    assert v['choices'][0]['message']['content']=='test 中文',v
                                    assert v['choices'][0]['message']['tool_calls'][0]['function']['arguments']=='{"x":1}',v
                for key in ['native-fixture','mixed-fixture']:
                    for model in ['failed','truncated','malformed','done-only','oversize','idle','total']:
                        status,_,raw=call('POST','/v1/responses',{'model':model,'input':'test'},key)
                        assert status==(500 if model=="failed" else 502),(key,model,status,raw)
                    status,_,raw=call('POST','/v1/responses',{'model':'incomplete','input':'test'},key)
                    assert status==200 and json.loads(raw)['status']=='incomplete',raw
                    for path,model in [('/v1/responses','follow'),('/v1/responses/compact','normal')]:
                        before=len(upstream.hits);status,_,raw=call('POST',path,{'model':model,'input':'test'},key)
                        assert status==200 and upstream.hits[before][2] is False,(status,raw)
                for model,count,status in [('early-fail',2,200),('failed',1,500),('truncated',1,502)]:
                    before=len(upstream.hits)
                    code,_,raw=call('POST','/v1/responses',{'model':model,'input':'test'},'retry-fixture')
                    assert code==status and len(upstream.hits)-before==count,(model,code,upstream.hits[before:],raw)
                schema=read('/v1/channel-settings/schema');assert schema['upstream_stream_mode']==1
                view=read('/v1/channel-settings?provider=native')
                patch={'revision':view['revision'],'operation_id':'stream-mode-change','changes':[{'provider':'native','set':{'/preferences/upstream_stream_mode':'invalid'}}]}
                assert call('POST','/v1/channel-settings/validate',patch)[0]==400
                patch['changes'][0]['set']['/preferences/upstream_stream_mode']='follow_client'
                assert call('PATCH','/v1/channel-settings',patch)[0]==200
                assert call('POST','/v1/responses',{'model':'normal','input':'test'},'native-fixture')[0]==400
                patch['revision']=read('/v1/channel-settings?provider=native')['revision']
                patch['operation_id']='enable-stream-mode'
                patch['sample']={'model':'normal','endpoint':'/v1/responses','stream':False,'body':{'model':'normal','input':'test','stream':False}}
                patch['changes'][0]['set']['/preferences/upstream_stream_mode']={'default':'follow_client','models':{'normal':'stream_only'}}
                code,_,raw=call('POST','/v1/channel-settings/validate',patch)
                assert code==200,raw
                preview=json.loads(raw)['previews'][0]['sample']
                assert preview['wire_request']['upstream_stream'] is True and preview['wire_request']['downstream_stream'] is False,preview
                assert call('PATCH','/v1/channel-settings',patch)[0]==200
                exported=read('/v1/channel-settings/export')
                upstream.snapshot={'version':2,'rules':[],'temporary_channels':[],'channel_settings':exported['channel_settings']}
                process.terminate();process.wait(timeout=5)
                env.update(UNI_API_CONTROL_RESTORE_URL=f'http://127.0.0.1:{upstream.server_port}/restore',UNI_API_CONTROL_RESTORE_TOKEN='admin-fixture')
                process=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
                for _ in range(100):
                    try:
                        if call('GET','/healthz')[0]==200:break
                    except OSError:pass
                    time.sleep(.05)
                else:raise AssertionError('restored gateway not ready')
                assert read('/v1/channel-settings?provider=native')['effective']['preferences']['upstream_stream_mode']==patch['changes'][0]['set']['/preferences/upstream_stream_mode']
                assert call('POST','/v1/responses',{'model':'normal','input':'test'},'native-fixture')[0]==200
                assert config_path.read_bytes()==original
                print('PASS stream-only: native/mixed/chat, default/false/true, full output and usage, failure/EOF/size/timeouts, incomplete, scope, bounded retries, settings')
            except Exception:
                print((root/'log').read_text()[-5000:]); raise
            finally:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                upstream.shutdown();upstream.server_close()

if __name__=='__main__': verify(Path(sys.argv[1]).resolve())
