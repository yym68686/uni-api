"""Isolated regression; uses only local stub upstream and storage endpoints."""
import json,subprocess,tempfile,pathlib,os,threading,time,http.client,sys,hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from verify_dispatch_timing import free_port
key='fixture-admin';kid='key-'+hashlib.sha256(key.encode()).hexdigest()
snapshot={'version':1,'temporary_channels':[{'provider':'sub2api-saved','api_key_id':kid,'base_url':'https://fixture.example/v1/responses','api_key':'fixture-upstream','models':['m']}],'rules':[{'api_key_id':kid,'model':'m','order':['sub2api-saved','configured'],'disabled':['configured']}]}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  assert self.headers['Authorization']=='Bearer '+key
  body=json.dumps({'enabled':True,'snapshot':snapshot}).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory(prefix='uni-restore-')as path:
 root=pathlib.Path(path);port=free_port();(root/'api.json').write_text(json.dumps({'providers':[{'provider':'configured','engine':'gpt','base_url':'https://fixture.example/v1/responses','api':'fixture','model':['m']}],'api_keys':[{'api':key,'model':['all']}]}))
 env={k:v for k,v in os.environ.items()if not k.startswith(('FACTS_S3_','UNI_API_CONTROL_'))};env.update(PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'api.json'),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'),UNI_API_CONTROL_RESTORE_URL=f'http://127.0.0.1:{server.server_port}/bootstrap',UNI_API_CONTROL_RESTORE_TOKEN=key,NO_PROXY='127.0.0.1,localhost')
 observed=[]
 with (root/'log').open('w')as log:
  for _ in range(2):
   p=subprocess.Popen([str(pathlib.Path(sys.argv[1]).resolve())],env=env,cwd=root,stdout=log,stderr=log)
   try:
    for attempt in range(100):
     try:
      c=http.client.HTTPConnection('127.0.0.1',port,timeout=3);c.request('GET','/v1/channel-controls',headers={'Authorization':'Bearer '+key});r=c.getresponse();d=json.loads(r.read());c.close();assert r.status==200;break
     except OSError:time.sleep(.05)
    assert d['rules']==snapshot['rules'];assert d['temporary_channels']==[{k:v for k,v in snapshot['temporary_channels'][0].items()if k not in ['base_url','api_key']}];observed.append(d['instance_id'])
   finally:p.terminate();p.wait(timeout=5)
 assert observed[0]!=observed[1]
 print('PASS: two real process boots restore channels/order/disabled rules before first accepted HTTP request; different instance ids; raw credentials absent from public controls')
server.shutdown();server.server_close()
