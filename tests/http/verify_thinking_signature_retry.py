"""Signed Messages history: fallback without mutation/cooldown; no paid calls."""
import copy
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
from fact_sink import accept_facts
from verify_dispatch_timing import free_port

ERROR = {"error": {"type": "invalid_request_error", "message":
         "messages.1.content.0: Invalid `signature` in `thinking` block"}}
RELAY_ERROR = {"error": {"type": "<nil>", "message":
    "messages.3.content.79: Invalid `signature` in `thinking` block "
    "(luban request id: req_fixture_1) (request id: fixture-request-2)"}}
MASKED_ERROR = {"error": {"type": "<nil>", "message":
    "***.***.content.0: Invalid `signature` in `thinking` block "
    "(request id: fixture-request-3)"}}
BODY = {"model": "claude-opus-5-5", "max_tokens": 32000, "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"}, "tools": [{"name": "read", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "fixture"},
          {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "fixture reasoning", "signature": "opaque-signed-history"},
            {"type": "tool_use", "id": "tool_1", "name": "read", "input": {"path": "fixture"}}]},
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}]}]}
RESPONSE = {"id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}}
EVENTS = [
    {"type": "message_start", "message": {**RESPONSE, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
    {"type": "message_stop"}]


def sse(events):
    return ''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        if self.path == '/facts': return accept_facts(self, self.server.facts)
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        channel = self.path.split('/')[1]
        self.server.hits.append((channel, body))
        if self.server.mode == 'postcommit' and channel == 'first':
            status, content_type = 200, 'text/event-stream'
            raw = sse(EVENTS[:3] + [{"type": "error", **self.server.error}])
        elif channel != 'fallback':
            status, content_type, raw = 400, 'application/json', json.dumps(self.server.error).encode()
        else:
            status = 200
            content_type = 'text/event-stream' if body.get('stream') else 'application/json'
            raw = sse(EVENTS) if body.get('stream') else json.dumps(RESPONSE).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(binary):
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    upstream.hits, upstream.facts, upstream.mode, upstream.error = [], [], 'http', ERROR
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix='thinking-signature-') as directory:
        root, port = Path(directory), free_port()
        providers = [{"provider": n, "engine": "claude",
                      "base_url": f"http://127.0.0.1:{upstream.server_port}/{n}/v1/messages",
                      "api": "fixture-upstream", "model": ["claude-opus-5-5"],
                      "preferences": {"cooldown_period": 300}} for n in ['first', 'fallback', 'also-reject']]
        # Non-Messages wire must not treat messages[] text in an unrelated body as signed history.
        providers.append({**providers[0], "provider": "gpt", "engine": "gpt",
                          "base_url": f"http://127.0.0.1:{upstream.server_port}/gpt/v1/chat/completions"})
        config = {"providers": providers, "preferences": {"hedging": {"enabled": False}}, "api_keys": [
            {"api": "retry", "model": ["first/*", "fallback/*"]},
            {"api": "no-retry", "model": ["first/*", "fallback/*"], "preferences": {"AUTO_RETRY": False}},
            {"api": "exhausted", "model": ["first/*", "also-reject/*"]},
            {"api": "gpt-only", "model": ["gpt/*"]}]}
        (root/'config.json').write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k in ('PATH','HOME','TMPDIR','LANG','DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'config.json'),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root/'spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0',
                   FACTS_INGEST_URL=f'http://127.0.0.1:{upstream.server_port}/facts', FACTS_INGEST_TOKEN='fixture', FACTS_SPOOL_DIR=str(root/'facts'),
                   NO_PROXY='127.0.0.1,localhost')
        def call(body=None, key='retry', rid='test', path='/v1/messages'):
            conn = http.client.HTTPConnection('127.0.0.1',port,timeout=10)
            try:
                conn.request('POST' if body is not None else 'GET', path if body is not None else '/healthz',
                             json.dumps(body) if body is not None else None,
                             {'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Request-ID':rid})
                r=conn.getresponse(); return r.status,r.read()
            finally: conn.close()
        with (root/'log').open('w') as log:
            proc=subprocess.Popen([str(binary)],cwd=root,env=env,stdout=log,stderr=log)
            try:
                for _ in range(100):
                    try:
                        if call()[0]==200: break
                    except OSError: pass
                    time.sleep(.05)
                else: raise AssertionError('not ready')
                count=0
                fact_rids = []
                for error in [ERROR, RELAY_ERROR, MASKED_ERROR, {"error": {"message": json.dumps(RELAY_ERROR)}},
                              {"error": {"message": json.dumps(MASKED_ERROR)}}]:
                    upstream.error = error
                    fact_rids.append(f'signature-{count}')
                    for stream in [False,True]:
                        for key,expected,hits in [('retry',200,['first','fallback']),('no-retry',502,['first']),
                                                  ('exhausted',502,['first','also-reject']*3),('retry',200,['first','fallback'])]:
                            upstream.hits=[];body={**copy.deepcopy(BODY),'stream':stream};rid=f'signature-{count}'
                            status,raw=call(body,key,rid)
                            assert status==expected,(key,status,raw)
                            assert [c for c,_ in upstream.hits]==hits,upstream.hits
                            assert all(b==body for _,b in upstream.hits),'request history mutated'
                            if status==502: assert json.loads(raw)==error
                            if status==200 and stream: assert b'message_stop' in raw
                            count+=1
                # Same first provider is still selected immediately: no long global cooldown.
                negatives=[('missing-signature',None),('empty-signature',''),('blank-signature',' \n'),('non-string',42)]
                for error in [ERROR, RELAY_ERROR, MASKED_ERROR]:
                    upstream.error = error
                    for name,sig in negatives:
                        upstream.hits=[];body=copy.deepcopy(BODY);body['messages'][1]['content'][0]['signature']=sig
                        status,_=call(body,rid=name);assert status==400 and len(upstream.hits)==1
                        count+=1
                for history_error in (RELAY_ERROR, MASKED_ERROR):
                    upstream.error = history_error
                    for variant in ('no-thinking', 'mixed-invalid', 'user-thinking'):
                        body = copy.deepcopy(BODY)
                        if variant == 'no-thinking': body['messages'][1]['content'].pop(0)
                        elif variant == 'mixed-invalid': body['messages'][1]['content'].append({'type':'thinking','thinking':'','signature':''})
                        else: body['messages'][1]['role'] = 'user'
                        upstream.hits = []
                        status,_ = call(body, rid=variant)
                        assert status == 400 and len(upstream.hits) == 1
                        count += 1
                for error in [
                    {"error":{"type":"invalid_request_error","message":"Missing required parameter: max_tokens"}},
                    {"error":{**ERROR['error'],'code':'invalid_type'}},
                    {"error":{**ERROR['error'],'message':ERROR['error']['message']+'. The block is bound to a different conversation.'}},
                    {"error":{**ERROR['error'],'message':ERROR['error']['message'].replace('1.content.0','9.content.0')}}]:
                    upstream.hits=[];upstream.error=error
                    status,_=call(BODY,rid=f'negative-{count}');assert status==400 and len(upstream.hits)==1
                    count+=1
                for patch in [
                    {'code':'invalid_type'}, {'code':'invalid_request_error'}, {'param':'messages'},
                    {'type':'invalid_request_error'},
                    {'message':MASKED_ERROR['error']['message']+'. The block is bound to a different conversation.'},
                    {'message':MASKED_ERROR['error']['message'].replace('***.***','messages.1')},
                    {'message':MASKED_ERROR['error']['message'].replace('content.0','content.1')}]:
                    upstream.error={'error':{**MASKED_ERROR['error'],**patch}};upstream.hits=[]
                    status,_=call(BODY,rid=f'masked-negative-{count}')
                    assert status==400 and len(upstream.hits)==1;count+=1
                for error in [ERROR, RELAY_ERROR, MASKED_ERROR]:
                    upstream.error=error;upstream.hits=[];upstream.mode='http'
                    status,_=call(BODY,key='gpt-only',path='/v1/chat/completions',rid=f'not-messages-{count}')
                    assert status==400 and len(upstream.hits)==1;count+=1
                    upstream.hits=[];upstream.mode='postcommit'
                    status,raw=call({**BODY,'stream':True},rid=f'after-output-{count}')
                    assert status==200 and b'done' in raw and len(upstream.hits)==1;count+=1
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    if all(any(f.get('kind')=='request' and f.get('request_id')==rid for f in upstream.facts) for rid in fact_rids):break
                    time.sleep(.1)
                for rid in fact_rids:
                    selected=[f for f in upstream.facts if f.get('request_id')==rid]
                    assert any(f.get('kind')=='billing' and f.get('status')==400 for f in selected),selected
                    assert any(f.get('stage')=='upstream_attempt' and f.get('trace_detail',{}).get('attempt_status_code')==400 for f in selected),selected
                    assert any(f.get('stage')=='thinking_signature_fallback' and f.get('trace_detail',{}).get('routing_status_code')==502 for f in selected),selected
                    assert any(f.get('kind')=='request' and f.get('status')==200 for f in selected),selected
                encoded=json.dumps(upstream.facts)
                for secret in ('opaque-signed-history','fixture reasoning','fixture-upstream'):assert secret not in encoded
                print(f'PASS thinking signature fallback: {count} HTTP cases, intact history, raw400/routing502/final200, no cooldown or post-output retry')
            finally:
                proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                upstream.shutdown();upstream.server_close()

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
