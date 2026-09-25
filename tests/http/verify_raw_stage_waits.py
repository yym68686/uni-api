"""Only loopback fixtures: heartbeat, split event, and delayed semantic event."""
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

facts = []
heartbeat = b': keepalive\n\n'
created = b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_fixture","output":[]}}\n\n'
delta = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"PRIVATE_FIXTURE_TEXT"}\n\n'
completed = b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
wire = heartbeat + created + delta + completed

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_PUT(self):
        facts.extend(json.loads(line) for line in self.rfile.read(int(self.headers['Content-Length'])).splitlines())
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(wire)))
        self.end_headers()
        self.wfile.write(heartbeat)
        self.wfile.flush()
        time.sleep(.25)
        self.wfile.write(created[:30])
        self.wfile.flush()
        time.sleep(.25)
        self.wfile.write(created[30:] + delta)
        self.wfile.flush()
        time.sleep(.05)
        self.wfile.write(completed)
        self.wfile.flush()

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    with tempfile.TemporaryDirectory(prefix='uni-raw-stages-') as directory:
        root = Path(directory)
        port = free_port()
        config = {'providers': [{'provider': 'raw', 'engine': 'gpt', 'api': 'fixture', 'base_url': f'http://127.0.0.1:{server.server_port}/v1/responses', 'model': ['m']}], 'api_keys': [{'api': 'fixture-admin', 'model': ['all']}]}
        (root/'api.json').write_text(json.dumps(config))
        env = {k:v for k,v in os.environ.items() if k in ('PATH','HOME','TMPDIR','LANG','DYLD_LIBRARY_PATH')}
        env.update(PORT=str(port), DISABLE_DATABASE='true', UNI_API_CONFIG_PATH=str(root/'api.json'), RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root/'snapshot.json'), UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root/'ledger'), RUST_REQUEST_SPOOL_DIRECTORY=str(root/'body-spool'), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS='0', RUST_REQUEST_SPOOL_INODE_RESERVE_BPS='0', NO_PROXY='127.0.0.1,localhost', FACTS_S3_ENDPOINT=f'http://127.0.0.1:{server.server_port}', FACTS_S3_BUCKET='fixture', FACTS_S3_ACCESS_KEY_ID='fixture', FACTS_S3_SECRET_ACCESS_KEY='fixture', FACTS_S3_SPOOL_DIR=str(root/'facts-spool'))
        with (root/'runtime.log').open('w') as log:
            p = subprocess.Popen([str(Path(sys.argv[1]).resolve())], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        c = http.client.HTTPConnection('127.0.0.1',port,timeout=1)
                        c.request('GET','/healthz')
                        r=c.getresponse();r.read();c.close()
                        if r.status==200:break
                    except OSError:pass
                    time.sleep(.05)
                c=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
                start=time.monotonic()
                c.request('POST','/v1/responses',json.dumps({'model':'m','input':'fixture','stream':True}),{'Authorization':'Bearer fixture-admin','Content-Type':'application/json'})
                r=c.getresponse();first=r.read(1);first_ms=(time.monotonic()-start)*1000;body=first+r.read();c.close()
                assert r.status==200 and body==wire, (r.status,body)
                assert first==b':' and first_ms<400, first_ms
                deadline=time.monotonic()+8
                while time.monotonic()<deadline and not any(f['kind']=='attempt' for f in facts):time.sleep(.1)
                attempt=next(f for f in facts if f['kind']=='attempt')
                t=attempt['transport_timing'];raw=t['raw_stream'];snapshot=raw['at_response_created']
                assert attempt['response_created_ms']>=450, attempt
                assert snapshot['upstream_read_wait_ms']>=450, snapshot
                assert snapshot['output_send_ms']<snapshot['upstream_read_wait_ms'], snapshot
                assert snapshot['process_ms']<snapshot['upstream_read_wait_ms'], snapshot
                assert snapshot['comment_frames']==1 and snapshot['frames']>=2, snapshot
                assert snapshot['max_pending_frame_bytes']>=30, snapshot
                assert snapshot['upstream_read_calls']>=3, snapshot
                assert raw['at_first_text'] is not None
                assert 'PRIVATE_FIXTURE_TEXT' not in json.dumps(attempt)
                print('PASS raw wire unchanged, heartbeat forwarded early, split-event wait separated:',json.dumps({'client_first_ms':first_ms,'response_created_ms':attempt['response_created_ms'],'raw_stream':raw}))
            finally:
                p.terminate();p.wait(timeout=5)
finally:
    server.shutdown();server.server_close()
