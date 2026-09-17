"""Isolated gateway regression: controls affect routing, never the config file."""
import hashlib
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
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append(self.path.split('/')[1])
        response={'id':'resp_fixture','object':'response','status':'completed','model':payload['model'],'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'OK'}]}],'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}}
        if self.path.endswith('/chat/completions'):
            response={'id':'chat_fixture','object':'chat.completion','model':payload['model'],'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        body=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

def verify(binary):
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream);upstream.hits=[]
    threading.Thread(target=upstream.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='uni-controls-') as directory:
        root=Path(directory);port=free_port()
        config={'providers':[{'provider':name,'base_url':f'http://127.0.0.1:{upstream.server_port}/{name}/v1/{endpoint}','engine':'gpt','api':'test-upstream','model':['model-a','model-b']}for name,endpoint in [('native','responses'),('compat','chat/completions')]],'api_keys':[{'api':'first','model':['all']},{'api':'ordinary','model':['native/*','compat/*']}]}
        config_path=root/'api.json';config_path.write_text(json.dumps(config));original=config_path.read_bytes()
        env=dict(os.environ,PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(config_path),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),NO_PROXY='127.0.0.1,localhost',RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0')
        env={k:v for k,v in env.items()if not k.startswith('FACTS_S3_')}
        def call(method,path,body=None,key='first'):
            c=http.client.HTTPConnection('127.0.0.1',port,timeout=10);c.request(method,path,json.dumps(body)if body is not None else None,{'Authorization':'Bearer '+key,'Content-Type':'application/json'});r=c.getresponse();code=r.status;raw=r.read();c.close();return code,json.loads(raw)
        def start(log):
            p=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            for _ in range(100):
                try:
                    if call('GET','/healthz')[0]==200:return p
                except OSError:pass
                time.sleep(.05)
            raise AssertionError('gateway not ready')
        def stop(p):
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        with (root/'log').open('w+')as log:
            process=start(log)
            try:
                def state():
                    code,data=call('GET','/v1/channel-controls');assert code==200;return data
                def change(action='set',order=None,disabled=None,key='',model='',revision=None):
                    return call('POST','/v1/channel-controls',{'revision':revision or state()['revision'],'action':action,'api_key_id':key,'model':model,'order':order or [],'disabled':disabled or []})
                def route(expected,model='model-a'):
                    for endpoint in ['/v1/responses','/v1/chat/completions']:
                        payload={'model':model,'input':'hi'}if endpoint.endswith('responses')else{'model':model,'messages':[{'role':'user','content':'hi'}]}
                        upstream.hits.clear();code,_=call('POST',endpoint,payload,'ordinary');assert code==200,code;assert upstream.hits==[expected],upstream.hits
                assert call('GET','/v1/channel-controls',key='ordinary')[0]==403
                assert call('POST','/v1/channel-controls',{},key='ordinary')[0]==403
                route('native')
                old=state()['revision'];assert change(order=['compat','native'])[0]==200;route('compat')
                assert change(disabled=['native'],revision=old)[0]==409
                key='key-'+hashlib.sha256(b'ordinary').hexdigest()
                assert change(disabled=['compat'],key=key,model='model-a')[0]==200;route('native');route('compat','model-b')
                assert change(action='reset',key=key,model='model-a')[0]==200;route('compat')
                assert change(disabled=['native','compat'])[0]==200
                upstream.hits.clear();assert call('POST','/v1/responses',{'model':'model-a','input':'hi'},'ordinary')[0]==503;assert not upstream.hits
                assert config_path.read_bytes()==original
                old=state();stop(process);process=start(log)
                new=state();assert new['rules']==[] and new['instance_id']!=old['instance_id'];assert new['expires_at'] is None
                route('native');assert config_path.read_bytes()==original
                print('PASS controls: auth, ordering, scoped disable, conflict, reset, native/compat routing, restart clears, config unchanged')
            except Exception:
                print((root/'log').read_text()[-8000:]);raise
            finally:stop(process);upstream.shutdown();upstream.server_close()
if __name__=='__main__':verify(Path(sys.argv[1]).resolve())
