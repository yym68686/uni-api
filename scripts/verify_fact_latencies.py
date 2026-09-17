"""Isolated regression; uses only local stub upstream and storage endpoints."""
import json,os,pathlib,subprocess,tempfile,threading,time,http.client,sys,hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from verify_dispatch_timing import free_port
facts=[];measurements=[]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_PUT(self):
  body=self.rfile.read(int(self.headers['Content-Length']));facts.extend(json.loads(l)for l in body.splitlines() if l);self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']));name=self.path.split('/')[1];started=time.monotonic();time.sleep(.4)
  delta={'type':'response.output_text.delta','delta':'test','item_id':'msg_fixture','output_index':0,'content_index':0}
  done={'type':'response.completed','response':{'id':'resp_fixture','status':'completed','model':'m','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'test'}]}],'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}}}
  body=b''.join(('event: '+x['type']+'\ndata: '+json.dumps(x)+'\n\n').encode() for x in [delta,done]);self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(body)));self.end_headers()
  if name!='codex-buffered':time.sleep(.25)
  measurements.append({'provider':name,'send_to_first_text_ms':(time.monotonic()-started)*1000});self.wfile.write(body);self.wfile.flush()
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
with tempfile.TemporaryDirectory(prefix='uni-latency-repro-') as path:
 root=pathlib.Path(path);port=free_port();providers=[('gpt-imported','gpt'),('codex-stream','codex'),('codex-buffered','codex')]
 cfg={'providers':[{'provider':name,'engine':engine,'base_url':f'http://127.0.0.1:{server.server_port}/{name}/v1/responses','api':'fixture-upstream','model':['m']}for name,engine in providers],'api_keys':[{'api':'fixture-admin','model':['all']}]};(root/'api.json').write_text(json.dumps(cfg))
 env={k:v for k,v in os.environ.items()if not k.startswith(('FACTS_S3_','AWS_'))};env.update(PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'api.json'),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),RUST_REQUEST_SPOOL_DIRECTORY=str(root/'body-spool'),RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',NO_PROXY='127.0.0.1,localhost',FACTS_S3_ENDPOINT=f'http://127.0.0.1:{server.server_port}',FACTS_S3_BUCKET='fixture',FACTS_S3_ACCESS_KEY_ID='fixture',FACTS_S3_SECRET_ACCESS_KEY='fixture',FACTS_S3_SPOOL_DIR=str(root/'facts-spool'))
 def call(method,path,payload=None,provider=None):
  c=http.client.HTTPConnection('127.0.0.1',port,timeout=5);headers={'Authorization':'Bearer fixture-admin','Content-Type':'application/json'}
  if provider:headers['x-uni-api-provider']=provider
  c.request(method,path,json.dumps(payload)if payload else None,headers);r=c.getresponse();status=r.status;body=r.read();c.close();return status,body
 with (root/'runtime.log').open('w')as log:
  p=subprocess.Popen([str(pathlib.Path(sys.argv[1]).resolve())],cwd=root,env=env,stdout=log,stderr=log)
  try:
   for _ in range(100):
    try:
     if call('GET','/healthz')[0]==200:break
    except OSError:pass
    time.sleep(.05)
   for name,_ in providers:
    status,body=call('POST','/v1/responses',{'model':'m','input':'fixture','stream':True},name);assert status==200;assert b'response.output_text.delta' in body
   deadline=time.monotonic()+8
   while time.monotonic()<deadline and len(facts)<9:time.sleep(.1)
   for measurement in measurements:
    name=measurement['provider'];attempt=next(f for f in facts if f['kind']=='attempt'and f['provider']==name);dispatch=next(f for f in facts if f['kind']=='dispatch'and f['provider']==name)
    assert attempt.get('first_output_ms') is not None and attempt['first_output_ms'] >= measurement['send_to_first_text_ms'] - 25
    assert attempt.get('key_id') == dispatch.get('key_id') and dispatch.get('key_id')
    print(json.dumps(measurement|{'success':attempt['outcome'],'recorded_first_output_ms':attempt.get('first_output_ms'),'attempt_has_key_id':bool(attempt.get('key_id')),'dispatch_has_key_id':bool(dispatch.get('key_id')),'dispatch_ms':dispatch.get('dispatch_ms')}))
  finally:p.terminate();p.wait(timeout=5)
server.shutdown();server.server_close()
