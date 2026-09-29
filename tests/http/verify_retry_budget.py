"""Physical retry caps, slow hedges and same-channel repair on loopback only."""
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
from verify_empty_name_retry import ERROR as REPAIR_ERROR, payload as repair_payload


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.hits.append((self.path, body))
            mode = self.server.mode
            attempt = len(self.server.hits)
            self.server.active += 1
        try:
            if mode == "slow":
                time.sleep(0.18)
            repaired = any(item.get("type") == "message" and item.get("role") == "assistant"
                           for item in body.get("input", []) if isinstance(item, dict))
            status, error = 503, {"error": {"message": "fixture upstream unavailable"}}
            if ((mode == "repair" and not repaired)
                    or (mode == "late-repair" and attempt == 4)):
                status, error = 400, REPAIR_ERROR
            raw = json.dumps(error).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            with self.server.lock:
                self.server.active -= 1


def verify(binary, hedging):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits, server.lock, server.active, server.mode = [], threading.Lock(), 0, "fast"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="uni-retry-budget-") as directory:
            root, port = Path(directory), free_port()
            def provider(name, model, keys=1):
                return {"provider": name, "engine": "gpt",
                        "base_url": f"http://127.0.0.1:{server.server_port}/{name}/v1/responses",
                        "api": [f"fixture-upstream-{i}" for i in range(keys)], "model": [model],
                        "preferences": {"cooldown_period": 0, "api_key_cooldown_period": 0}}
            values = {"three": 3, "string-three": " 3 ", "one": 1, "false": False,
                      "zero": 0, "string-false": "false", "off": " OFF ",
                      "invalid": "invalid", "negative": -1, "fractional": 1.5,
                      "true": True, "string-true": "true", "large": 2**64-1}
            api_keys = [{"api": name, "model": ["all"],
                         "preferences": {"AUTO_RETRY": value}} for name, value in values.items()]
            api_keys.append({"api": "default", "model": ["all"]})
            config = {
                "providers": [provider(f"channel-{i:02}", "many-channels") for i in range(50)]
                             + [provider("pool", "many-keys", 60)]
                             + [provider(f"small-{i}", "small") for i in range(3)],
                "api_keys": api_keys,
                "preferences": {"hedging": {"enabled": hedging, "max_inflight_attempts": 2},
                                "timeout_policy": {"default": {"first_byte": 0.06, "total": 2}}},
            }
            config_path = root / "api.json"
            original = json.dumps(config).encode()
            config_path.write_bytes(original)
            env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG")}
            env.update(PORT=str(port), DISABLE_DATABASE="true", NO_PROXY="127.0.0.1,localhost",
                       UNI_API_CONFIG_PATH=str(config_path),
                       RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                       UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                       RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                       RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0")

            def request(endpoint="/healthz", body=None, key="three", target=None):
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                if target:
                    headers["X-Uni-API-Provider"] = target
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                try:
                    conn.request("GET" if body is None else "POST", endpoint,
                                 None if body is None else json.dumps(body), headers)
                    response = conn.getresponse()
                    return response.status, response.read()
                finally:
                    conn.close()

            def check(endpoint, stream, key, model="small", expected=4, mode="fast", target=None):
                with server.lock:
                    assert server.active == 0
                    server.mode, server.hits = mode, []
                body = {"model": model, "stream": stream}
                if mode in ("repair", "late-repair"):
                    body = dict(repair_payload(stream), model=model)
                elif endpoint == "/v1/chat/completions":
                    body["messages"] = [{"role": "user", "content": "fixture"}]
                else:
                    body["input"] = "fixture"
                status, raw = request(endpoint, body, key, target)
                # Aborted hedge sockets may leave a sleeping mock handler behind.
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    with server.lock:
                        if server.active == 0:
                            break
                    time.sleep(0.01)
                with server.lock:
                    hits = list(server.hits)
                    assert server.active == 0
                context = (hedging, endpoint, stream, key, model, mode, status, len(hits), raw[:200])
                assert len(hits) == expected, context
                assert status in ((502, 503, 504) if mode == "slow" else (400, 503)), context
                if mode == "repair" and expected > 1:
                    assert hits[0][0] == hits[1][0], context
                    assert hits[0][1]["input"] != hits[1][1]["input"], context
                return 1

            with (root / "log").open("w+") as log:
                process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
                try:
                    for _ in range(150):
                        try:
                            if request()[0] == 200:
                                break
                        except OSError:
                            pass
                        time.sleep(0.05)
                    else:
                        raise AssertionError("fixture startup failed")
                    count = 0
                    for endpoint in ("/v1/chat/completions", "/v1/responses"):
                        for stream in (False, True):
                            for model in ("many-channels", "many-keys"):
                                count += check(endpoint, stream, "three", model)
                            for key, expected in (("string-three", 4), ("one", 2), ("false", 1),
                                                  ("zero", 1), ("string-false", 1), ("off", 1),
                                                  ("invalid", 1), ("negative", 1), ("fractional", 1),
                                                  ("true", 9), ("string-true", 9), ("default", 9),
                                                  ("large", 100)):
                                count += check(endpoint, stream, key, expected=expected)
                            if endpoint == "/v1/responses":
                                count += check(endpoint, stream, "three", target="small-0", expected=1)
                            for key, expected in (("three", 4), ("false", 1), ("zero", 1), ("string-false", 1)):
                                count += check(endpoint, stream, key, expected=expected, mode="slow")
                    for stream in (False, True):
                        for key, expected in (("three", 4), ("one", 2), ("false", 1), ("string-false", 1)):
                            count += check("/v1/responses", stream, key, expected=expected, mode="repair")
                    for stream in (False, True):
                        count += check("/v1/responses", stream, "three", mode="late-repair")
                    count += check("/v1/responses/compact", False, "three")
                    assert config_path.read_bytes() == original
                    print(f"PASS retry budget: {count} cases, hedging={hedging}")
                except Exception:
                    log.flush()
                    print((root / "log").read_text()[-6000:])
                    raise
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    for enabled in (False, True):
        verify(Path(sys.argv[1]).resolve(), enabled)
