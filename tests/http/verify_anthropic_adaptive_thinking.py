"""Offline wire-contract checks for Claude adaptive thinking and effort."""
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
from verify_dispatch_timing import free_port


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.hits.append(payload)
        config = payload.get('output_config')
        thinking = payload.get('thinking', {})
        invalid = 'output_config' in payload and (not isinstance(config, dict) or
                  ('effort' in config and config['effort'] not in ['low', 'medium', 'high']))
        if isinstance(thinking, dict) and thinking.get('type') == 'enabled' and payload['model'] == 'claude-sonnet-5':
            invalid = True
        if invalid:
            raw = b'{"error":{"type":"invalid_request_error","message":"Invalid thinking configuration"}}'
            status, content_type = 400, 'application/json'
        else:
            status = 200
            response = {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'model': payload['model'],
                        'content': [{'type': 'text', 'text': 'test'}], 'stop_reason': 'end_turn',
                        'usage': {'input_tokens': 10, 'output_tokens': 1}}
            events = [
                {'type': 'message_start', 'message': {**response, 'content': [], 'stop_reason': None}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'test'}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
                {'type': 'message_stop'},
            ]
            content_type = 'text/event-stream' if payload.get('stream') else 'application/json'
            raw = (''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                   if payload.get('stream') else json.dumps(response).encode())
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(binary):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='claude-adaptive-') as temporary:
        root, port = Path(temporary), free_port()
        providers = []
        for name, target, override in [
            ('adaptive', 'claude-sonnet-5', None),
            ('legacy', 'claude-sonnet-4-6', None),
            ('unknown', 'claude-sonnet-5-custom', None),
            ('override-adaptive', 'claude-sonnet-4-6', 'claude-sonnet-5'),
            ('override-legacy', 'claude-sonnet-5', 'claude-sonnet-4-6'),
        ]:
            prefs = {'cooldown_period': 0}
            if override:
                prefs['post_body_parameter_overrides'] = {'model': override}
            providers.append({'provider': name, 'engine': 'claude', 'base_url': f'http://127.0.0.1:{upstream.server_port}/v1/messages',
                              'api': 'fixture', 'model': [{target: 'client-alias'}], 'preferences': prefs})
        config = {'providers': providers, 'api_keys': [
            {'api': 'key-'+p['provider'], 'model': [p['provider']+'/*'], 'preferences': {'AUTO_RETRY': False}}
            for p in providers]}
        config_path = root/'config.json'
        config_path.write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('FACTS_S3_', 'OTEL_', 'FUGUE_OTEL_', 'UNI_API_CONTROL_RESTORE_')) and k != 'CONFIG_URL'}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0',
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', PROVIDER_MODEL_CIRCUIT_FAILURE_THRESHOLD='1000', NO_PROXY='127.0.0.1,localhost')
        def request(path, body=None, key='adaptive'):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
            try:
                connection.request('GET' if body is None else 'POST', path, None if body is None else json.dumps(body),
                                   {'Authorization':'Bearer key-'+key, 'Content-Type':'application/json'})
                r = connection.getresponse()
                return r.status, r.read()
            finally:
                connection.close()
        with (root/'log').open('w') as log:
            proc = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        if request('/healthz')[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError('gateway did not start: '+(root/'log').read_text()[-4000:])
                count = 0
                cases = [
                    ('adaptive', {'reasoning_effort':'high'}, 'adaptive', 'high', 256),
                    ('adaptive', {'reasoning':{'effort':'medium'}}, 'adaptive', 'medium', 256),
                    ('adaptive', {'reasoning_effort':'high', 'output_config':{'effort':'low','format':{'type':'json_schema','schema':{'type':'object'}}}}, 'adaptive', 'low', 256),
                    ('adaptive', {'reasoning_effort':'high', 'output_config':{'format':{'type':'json_schema','schema':{'type':'object'}}}}, 'adaptive', 'high', 256),
                    ('adaptive', {'output_config':{'effort':'high'}}, 'adaptive', 'high', 256),
                    ('adaptive', {'thinking':{'type':'disabled'}, 'reasoning_effort':'high','temperature':.4,'top_p':.8}, 'disabled', None, 256),
                    ('adaptive', {'reasoning_effort':'none'}, 'disabled', None, 256),
                    ('adaptive', {}, None, None, 256),
                    ('legacy', {'reasoning_effort':'high'}, 'enabled', None, 9216),
                    ('unknown', {'reasoning_effort':'high'}, 'enabled', None, 9216),
                    ('legacy', {'thinking':{'type':'adaptive'},'reasoning_effort':'high'}, 'adaptive', 'high', 256),
                    ('legacy', {'thinking':{'type':'enabled','budget_tokens':1024},'reasoning_effort':'high'}, 'enabled', None, 2048),
                    ('override-adaptive', {'reasoning_effort':'high'}, 'adaptive', 'high', 256),
                    ('override-legacy', {'reasoning_effort':'high'}, 'enabled', None, 9216),
                ]
                for endpoint in ['/v1/chat/completions','/v1/responses']:
                    for stream in [False,True]:
                        for key,fields,mode,effort,limit in cases:
                            body={'model':'client-alias','stream':stream,**copy.deepcopy(fields)}
                            if endpoint.endswith('/responses'):
                                body.update(input=[{'role':'user','content':'fixture'}], max_output_tokens=256,
                                            tools=[{'type':'function','name':'lookup','parameters':{'type':'object'}}],tool_choice={'type':'function','name':'lookup'})
                            else:
                                body.update(messages=[{'role':'user','content':'fixture'}],max_tokens=256,
                                            tools=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}],tool_choice={'type':'function','function':{'name':'lookup'}})
                            before=copy.deepcopy(body);upstream.hits=[]
                            status,raw=request(endpoint,body,key)
                            assert status==200 and b'test' in raw,(endpoint,stream,key,fields,status,raw)
                            assert len(upstream.hits)==1
                            wire=upstream.hits[0]
                            assert wire.get('thinking',{}).get('type')==mode,(fields,wire)
                            assert wire.get('output_config',{}).get('effort')==effort,(fields,wire)
                            assert wire['max_tokens']==limit,(fields,wire)
                            assert wire['tool_choice']=={'type':'tool','name':'lookup'}
                            assert wire['tools'][0]['name']=='lookup'
                            if 'format' in fields.get('output_config',{}):assert wire['output_config']['format']==fields['output_config']['format']
                            if mode=='adaptive':assert 'budget_tokens' not in wire['thinking']
                            if mode=='disabled' and 'temperature' in fields:assert wire['temperature']==.4 and wire['top_p']==.8
                            assert body==before
                            count+=1
                # Explicit invalid parameters are preserved and still return a single 400.
                for fields in [{'thinking':{'type':'enabled','budget_tokens':1024}},
                               {'reasoning_effort':'high','output_config':{'effort':None}},
                               {'reasoning_effort':'xhigh'},{'output_config':'invalid','thinking':{'type':'adaptive'}}]:
                    upstream.hits=[]
                    status,_=request('/v1/chat/completions',{'model':'client-alias','messages':[{'role':'user','content':'fixture'}],**fields})
                    assert status==400 and len(upstream.hits)==1,(fields,status)
                    count+=1
                # Native Messages intent and signed tool history pass through unchanged.
                body={'model':'client-alias','max_tokens':256,'thinking':{'type':'adaptive'},'output_config':{'effort':'high'},
                      'messages':[{'role':'assistant','content':[{'type':'thinking','thinking':'fixture','signature':'exact-signature'},
                       {'type':'tool_use','id':'call','name':'lookup','input':{'value':' preserve '}}]},
                       {'role':'user','content':[{'type':'tool_result','tool_use_id':'call','content':'result'}]}]}
                upstream.hits=[];status,_=request('/v1/messages',body)
                assert status==200 and upstream.hits[0]['messages']==body['messages']
                assert upstream.hits[0]['output_config']==body['output_config'];count+=1
                print(f'PASS Claude adaptive thinking: {count} HTTP wire contracts including mapping, overrides, explicit fields and 400 preservation')
            finally:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                upstream.shutdown();upstream.server_close()


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
