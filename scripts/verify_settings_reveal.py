"""Isolated administrator-only API-key reveal regression. No upstream calls."""
import http.client,json,os,pathlib,socket,subprocess,sys,tempfile,time,urllib.parse
binary=pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='reveal-qa-')as tmp:
 root=pathlib.Path(tmp);s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
 config={'providers':[{'provider':'one','base_url':'http://127.0.0.1:1/v1/responses','api':['fixture-a','fixture-b'],'model':['model-a']}],'api_keys':[{'api':'catalog','model':['all']},{'api':'admin-token','role':'admin','model':['all']},{'api':'ordinary','model':['all']}]}
 (root/'api.json').write_text(json.dumps(config))
 env={k:v for k,v in os.environ.items()if not k.startswith(('FACTS_S3_','UNI_API_CONTROL_RESTORE'))}
 env.update(PORT=str(port),DISABLE_DATABASE='true',UNI_API_CONFIG_PATH=str(root/'api.json'),RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'),UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),NO_PROXY='127.0.0.1,localhost')
 with (root/'log').open('w')as log:
  p=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
  def get(path,key='admin-token'):
   c=http.client.HTTPConnection('127.0.0.1',port,timeout=5);c.request('GET',path,headers={'Authorization':'Bearer '+key});r=c.getresponse();body=r.read();headers=dict(r.getheaders());c.close();return r.status,headers,json.loads(body)
  try:
   for _ in range(100):
    try:
     if get('/healthz')[0]==200:break
    except OSError:time.sleep(.05)
   code,_,view=get('/v1/channel-settings?provider=one');assert code==200
   path='/v1/channel-settings/secrets?'+urllib.parse.urlencode({'provider':'one','revision':view['revision']})
   for caller in ['ordinary','catalog','']:
    assert get(path,caller)[0]==403,'non-admin reveal allowed'
   code,headers,revealed=get(path);assert code==200 and headers['cache-control']=='no-store'
   assert revealed['keys'][view['effective']['api'][1]['$secret']]=='fixture-b'
   assert 'fixture-a'not in json.dumps(view)
   assert get('/v1/channel-settings/secrets?provider=one&revision=old')[0]==409
   assert get('/v1/channel-settings?provider=one')[2]==view
   print('PASS: explicit administrator, exact key refs, no-store, revision conflict, read-only')
  finally:
   p.terminate();p.wait(timeout=10)
