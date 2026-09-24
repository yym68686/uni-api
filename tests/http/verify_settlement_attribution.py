"""Loopback-only HTTP contract: preserve final OAIX receipts through adapters."""
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
from verify_dispatch_timing import free_port, get_json

RECEIPT = {"payload": "opaque-signed-final-owner", "signatures": ["fixture"]}
NONCE = "a" * 64


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append((body["model"], self.headers.get("X-OAIX-Settlement-Nonce")))
        response = self.path.endswith("/responses")
        usage = {("input_tokens" if response else "prompt_tokens"): 12,
                 ("output_tokens" if response else "completion_tokens"): 3,
                 "total_tokens": 15, "oaix_settlement_receipt": RECEIPT}
        if response:
            value = {"id": "resp_fixture", "object": "response", "status": "completed",
                     "model": body["model"], "output": [{"id": "msg_fixture", "type": "message",
                     "role": "assistant", "status": "completed", "content": [
                         {"type": "output_text", "text": "OK"}]}], "usage": usage}
            events = [{"type": "keepalive"}, {"type": "response.output_text.delta", "delta": "OK"},
                      {"type": "response.completed", "response": value}]
        else:
            value = {"id": "chatcmpl_fixture", "object": "chat.completion", "model": body["model"],
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                  "finish_reason": "stop"}], "usage": usage}
            events = [{"id": "chatcmpl_fixture", "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]},
                {"id": "chatcmpl_fixture", "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"id": "chatcmpl_fixture", "object": "chat.completion.chunk", "choices": [], "usage": usage}]
        streaming = body.get("stream")
        raw = json.dumps(value).encode()
        if streaming:
            raw = "".join((f"event: {v['type']}\n" if response else "") +
                          f"data: {json.dumps(v)}\n\n" for v in events).encode() + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("X-OAIX-Token-ID", "100")  # Deliberately stale; receipt is authoritative.
        self.send_header("X-OAIX-Attribution-Contract", "receipt-v1")
        self.end_headers()
        self.wfile.write(raw)


def verify(binary, hedging):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cases = [(up, down, stream, force) for up in ["responses", "chat"]
             for down in ["responses", "chat"] for stream in [False, True]
             for force in ([False, True] if stream and up != down else [False])]
    with tempfile.TemporaryDirectory(prefix="settlement-transport-") as directory:
        root, port = Path(directory), free_port()
        models = {up: [f"{up}-{down}-{stream}-{force}-" + ("forced-json" if force else "normal")
                       for u, down, stream, force in cases if u == up] for up in ["responses", "chat"]}
        providers = []
        for up in models:
            for force in [False, True]:
                names = [m for m in models[up] if ("forced-json" in m) == force]
                if not names:
                    continue
                providers.append({"provider": up + str(force), "engine": "gpt", "api": "fixture",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1/" +
                    ("responses" if up == "responses" else "chat/completions"), "model": names,
                    "preferences": {"post_body_parameter_overrides": {"stream": False}} if force else {}})
        config = {"providers": providers, "api_keys": [{"api": "fixture-key", "model": ["all"]}],
                  "preferences": {"hedging": {"enabled": hedging}}}
        path = root / "api.json"
        path.write_text(json.dumps(config))
        env = dict(os.environ, PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(path),
                   NO_PROXY="127.0.0.1,localhost", RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "memory"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0")
        for key in list(env):
            if key.startswith(("FACTS_S3_", "UNI_API_CONTROL_RESTORE_")):
                env.pop(key)
        with (root / "log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        get_json(port, "/healthz")
                        break
                    except OSError:
                        time.sleep(0.05)
                for up, down, stream, force in cases:
                    model = f"{up}-{down}-{stream}-{force}-" + ("forced-json" if force else "normal")
                    body = {"model": model, "stream": stream}
                    body.update({"input": "fixture"} if down == "responses" else
                                {"messages": [{"role": "user", "content": "fixture"}]})
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    conn.request("POST", "/v1/" + ("responses" if down == "responses" else "chat/completions"),
                                 json.dumps(body), {"Authorization": "Bearer fixture-key", "Content-Type": "application/json",
                                                    "X-OAIX-Settlement-Nonce": NONCE})
                    response = conn.getresponse()
                    raw = response.read()
                    context = (hedging, model, response.status, raw[:1000])
                    assert response.status == 200, context
                    values = [json.loads(line[5:]) for line in raw.splitlines()
                              if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"] if stream else [json.loads(raw)]
                    usages = [v.get("response", v).get("usage") for v in values]
                    receipts = [u["oaix_settlement_receipt"] for u in usages if isinstance(u, dict) and "oaix_settlement_receipt" in u]
                    assert receipts and all(r == RECEIPT for r in receipts), context
                    assert response.getheader("X-OAIX-Attribution-Contract") == "receipt-v1", context
                    assert response.getheader("X-OAIX-Token-ID") == "100", context
                    assert server.hits[-1] == (model, NONCE), (context, server.hits)
                    conn.close()
                    print(f"PASS hedging={hedging} {model}")
            except Exception:
                log.flush()
                print((root / "log").read_text()[-6000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)
    server.shutdown()
    server.server_close()


if __name__ == "__main__":
    for enabled in [False, True]:
        verify(Path(sys.argv[1]).resolve(), enabled)
