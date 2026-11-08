"""Compaction policies survive native, generic, and native-fallback routing."""

import http.client
import itertools
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
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append(self.path.split("/")[1])
        if payload.get("metadata", {}).get("fixture_delay"):
            time.sleep(.6)
        response = {
            "id": "resp_fixture", "object": "response", "status": "completed",
            "model": payload["model"],
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "OK"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        compact = self.path.rstrip("/").endswith("/compact") or any(
            item.get("type") == "compaction_trigger" for item in payload.get("input", [])
        )
        if compact:
            response["output"] = [{"type": "compaction", "encrypted_content": "fixture-state"}]
        if payload.get("stream"):
            events = [
                {"type": "response.created", "response": {"status": "in_progress"}},
                {"type": "response.output_text.delta", "delta": "OK"},
                {"type": "response.completed", "response": response},
            ]
            if compact:
                events.insert(-1, {"type": "response.output_item.done", "item": response["output"][0], "output_index": 0})
            raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        else:
            raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if payload.get("stream") else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def verify(binary, mode):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="uni-compaction-") as directory:
            root = Path(directory)
            port = free_port()
            providers = [
                {"provider": name, "engine": "gpt", "api": "fixture-upstream",
                 "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}/v1/responses",
                 "model": ["model-a"], **policy}
                for name, policy in [
                    ("regular", {}), ("compact", {"only_request_types": ["compaction"]})]
            ]
            providers[1]["preferences"] = {"timeout_policy": {
                "default": {"first_byte": .2},
                "rules": [{"match": {"request_type": "compaction"}, "timeout": {"first_byte": 3}}],
            }}
            if mode == "fallback":
                # A compatible fallback candidate makes native preparation hand
                # the request back to generic routing after its first filtering.
                providers.append({**providers[0], "provider": "compat",
                                  "base_url": f"http://127.0.0.1:{upstream.server_port}/compat/v1/chat/completions"})
            config = {
                "providers": providers,
                "api_keys": [
                    {"api": "fixture-admin", "role": "admin", "model": ["all"],
                     "preferences": {"AUTO_RETRY": False}},
                    {"api": "fixture-regular", "model": ["regular/*"]},
                    {"api": "fixture-compact", "model": ["compact/*"]},
                ],
                "preferences": {"AUTO_RETRY": False, "SCHEDULING_ALGORITHM": "fixed_priority"},
            }
            (root / "api.json").write_text(json.dumps(config))
            env = {k: v for k, v in os.environ.items()
                   if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
            env.update(
                PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "api.json"),
                RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                NO_PROXY="127.0.0.1,localhost",
                UNI_API_RUST_RESPONSES_DATA_PLANE="false" if mode == "generic" else "true",
            )

            def call(method, path, body=None, key="fixture-admin", request_id=None):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
                if request_id:
                    headers["X-Request-ID"] = request_id
                conn.request(method, path, json.dumps(body) if body is not None else None, headers)
                response = conn.getresponse()
                raw = response.read()
                conn.close()
                return response.status, raw

            successes = []
            sequence = itertools.count()

            def route(endpoint, triggered, expected, key="fixture-admin", stream=False, delayed=False):
                payload = {"model": "model-a", "input": [{"type": "message", "role": "user",
                           "content": [{"type": "input_text", "text": "fixture"}]}], "stream": stream}
                if triggered:
                    payload["input"].append({"type": "compaction_trigger"})
                if delayed:
                    payload["metadata"] = {"fixture_delay": True}
                request_id = f"compaction-{mode}-{next(sequence)}"
                upstream.hits.clear()
                status, raw = call("POST", endpoint, payload, key, request_id)
                if expected is None:
                    assert status == 404 and not upstream.hits, (mode, endpoint, key, status, upstream.hits, raw)
                else:
                    assert status == 200 and upstream.hits == [expected], (mode, endpoint, key, status, upstream.hits, raw)
                    generic = mode == "generic" or (mode == "fallback" and key == "fixture-admin")
                    successes.append((request_id, generic))

            with (root / "runtime.log").open("w") as log:
                process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
                try:
                    for _ in range(100):
                        try:
                            if call("GET", "/healthz")[0] == 200:
                                break
                        except OSError:
                            pass
                        time.sleep(.05)
                    else:
                        raise AssertionError("gateway not ready")
                    route("/v1/responses", True, "regular")
                    status, raw = call("GET", "/v1/channel-settings?provider=regular")
                    assert status == 200, raw
                    status, raw = call("PATCH", "/v1/channel-settings", {
                        "revision": json.loads(raw)["revision"], "operation_id": "exclude-compaction",
                        "changes": [{"provider": "regular", "set": {"/exclude_request_types": ["compaction"]}}],
                    })
                    assert status == 200, raw
                    for endpoint in ("/v1/responses", "/v1/responses/"):
                        for stream in (False, True):
                            route(endpoint, True, "compact", stream=stream)
                            route(endpoint, False, "regular", stream=stream)
                            route(endpoint, True, None, "fixture-regular", stream)
                            route(endpoint, False, None, "fixture-compact", stream)
                            route(endpoint, True, "compact", "fixture-compact", stream)
                    for endpoint in ("/v1/responses/compact", "/v1/responses/compact/"):
                        route(endpoint, False, "compact")
                        route(endpoint, True, "compact")
                        route(endpoint, False, None, "fixture-regular")
                        route(endpoint, False, "compact", "fixture-compact")
                    route("/v1/responses", True, "compact", stream=True, delayed=True)
                    route("/v1/responses/compact", False, "compact", delayed=True)
                except Exception:
                    log.flush()
                    print((root / "runtime.log").read_text()[-8000:])
                    raise
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            events = []
            for line in (root / "runtime.log").read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            for request_id, generic in successes:
                marker = "rust_generic_data_plane" if generic else "rust_responses_data_plane"
                assert any(e.get("request_id") == request_id and e.get(marker) is True for e in events), (mode, request_id, "wrong data plane")
            print(f"PASS compaction {mode}: live settings, trigger/compact endpoints, scoped rejection, ordinary requests, timeout policy")
    finally:
        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    for mode in sys.argv[2:] or ("native", "generic", "fallback"):
        verify(Path(sys.argv[1]).resolve(), mode)
