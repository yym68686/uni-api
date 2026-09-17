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
        self.server.hits.append((self.path, body['model']))
        bad = self.path.startswith('/bad/')
        response = {'error': {'message': 'fixture failure'}} if bad else {
            'id':'resp_check', 'object':'response', 'status':'completed', 'model':'gpt-6-astra',
            'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':'未知'}]}],
            'usage':{'input_tokens':25,'output_tokens':1,'total_tokens':26}}
        if not bad and self.path.endswith('/chat/completions'):
            response = {'id':'chat_check','object':'chat.completion','model':'gpt-6-astra','choices':[{'index':0,'message':{'role':'assistant','content':'未知'},'finish_reason':'stop'}],'usage':{'prompt_tokens':25,'completion_tokens':1,'total_tokens':26}}
        raw=json.dumps(response).encode()
        self.send_response(503 if bad else 200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

def verify(binary, endpoint):
    server=ThreadingHTTPServer(('127.0.0.1',0),Upstream);server.hits=[]
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-channel-check-') as directory:
        root=Path(directory);port=free_port()
        config={'providers':[{'provider':p,'engine':'gpt','base_url':f'http://127.0.0.1:{server.server_port}/{p}{endpoint}','api':'upstream-test','model':['gpt-6-astra'],'preferences':{'cooldown_period':0}} for p in ['bad','good']], 'api_keys':[{'api':'admin-test','model':['bad/*']},{'api':'ordinary-test','model':['all']}], 'preferences':{'hedging':{'enabled':True,'max_inflight_attempts':2}}}
        (root/'api.json').write_text(json.dumps(config))
        env=dict(os.environ,PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'api.json'),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),NO_PROXY='127.0.0.1,localhost',RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0')
        # Do not inherit production export settings into a fixture.
        env={k:v for k,v in env.items() if not k.startswith('FACTS_S3_')}
        with (root/'log').open('w+') as log:
            process=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            def request(method,path,key='admin-test',target=None):
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
                if target is not None:headers['X-Uni-API-Provider']=target
                body=json.dumps({'model':'gpt-6-astra','input':[{'role':'user','content':'test'}]}) if method=='POST' else None
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
