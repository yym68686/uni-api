"""Offline HTTP contract checks for the native TypeSafe/Jev data plane."""
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


QUESTIONS = {
    "yes": {"type": "noul", "instructions": {"question": "Is the report urgent?"},
            "criteria": {"true": ["Current outage"], "false": "No current outage"}},
    "route": {"type": "choice", "instructions": ["Classify the report"],
              "criteria": {"outage": {"description": "Service down"}, "other": None}},
    "impact": {"type": "score", "instructions": "Rate the report's impact",
               "criteria": ["None", {"description": "Partial"}, ["Complete"]]},
}
ANSWER = {"model": "jev-1.13.0", "answers": {
    "yes": {"type": "noul", "noul": .99},
    "route": {"type": "choice", "choice": "outage", "confidence": .98,
              "probabilities": {"outage": .99, "other": .01}},
    "impact": {"type": "score", "score": 1.99, "confidence": .98,
               "probabilities": {"0": 0, "1": .01, "2": .99},
               "legend": {"0": "None", "1": {"description": "Partial"}, "2": ["Complete"]}},
}, "usage": {"input_tokens": 2000, "output_tokens": 137}}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if status in (429, 529):
            self.send_header("Retry-After", "17")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        assert self.path == "/v1/models", self.path
        assert self.headers["Authorization"] == "Bearer fixture-upstream"
        self.reply(200, {"models": [{"name": "jev-latest", "description": "Fixture", "release_date": "2026-09-10"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append((self.path, dict(self.headers), payload))
        assert self.path == "/v1/systemone", self.path
        assert self.headers["Authorization"] == "Bearer fixture-upstream"
        if isinstance(payload.get("state"), dict) and "error" in payload["state"]:
            return self.reply(payload["state"]["error"], {"detail": "Fixture upstream failure"})
        self.reply(200, ANSWER)


def verify(binary):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="uni-typesafe-") as directory:
        root = Path(directory)
        port = free_port()
        config = {
            "providers": [
                {"provider": "jev", "base_url": f"http://127.0.0.1:{upstream.server_port}",
                 "engine": "typesafe", "api": "fixture-upstream",
                 "model": ["jev-1.13.0", {"jev-1.13.0": "decision-alias"}],
                 "preferences": {"AUTO_RETRY": False, "cooldown_period": 0}},
                {"provider": "jev-discovery", "base_url": f"http://127.0.0.1:{upstream.server_port}/v1/systemone",
                 "engine": "typesafe", "api": "fixture-upstream"},
                {"provider": "chat", "base_url": f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions",
                 "engine": "gpt", "api": "fixture-chat", "model": ["chat-model"]},
            ],
            "api_keys": [
                {"api": "admin", "model": ["all"]},
                {"api": "business", "model": ["jev/*", "jev-discovery/*"], "preferences": {"AUTO_RETRY": False}},
                {"api": "other", "model": ["chat/*"]},
            ],
        }
        config_path = root / "api.json"
        config_path.write_text(json.dumps(config))
        env = dict(os.environ, PORT=str(port), DISABLE_DATABASE="true",
                   UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   NO_PROXY="127.0.0.1,localhost", RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0")
        env = {k: v for k, v in env.items() if not k.startswith(("FACTS_S3_", "UNI_API_CONTROL_RESTORE_"))}

        def call(method, path, body=None, key="business"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            connection.request(method, path, json.dumps(body) if body is not None else None,
                               {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), json.loads(response.read())
            connection.close()
            return result

        with (root / "gateway.log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(120):
                    try:
                        if call("GET", "/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    log.seek(0)
                    raise AssertionError("Gateway not ready: " + log.read())
                status, _, models = call("GET", "/v1/models")
                assert status == 200
                ids = {m["id"] for m in models["data"]}
                assert ids == {"jev-latest", "jev-1.13.0", "decision-alias"}, ids
                assert {m["name"] for m in models["models"]} == ids
                assert all(isinstance(m["release_date"], str) for m in models["models"])
                for state in ["服务全部中断", {"report": "All requests fail"}, ["Outage", {"all_users": True}]]:
                    for model in ["jev-latest", "jev-1.13.0", "decision-alias"]:
                        body = {"model": model, "state": state, "questions": QUESTIONS}
                        status, _, result = call("POST", "/v1/systemone", body)
                        assert status == 200 and result == ANSWER, (status, result)
                        sent = upstream.hits[-1][2]
                        expected = {**body, "model": "jev-1.13.0" if model == "decision-alias" else model}
                        assert sent == expected, sent
                        assert "stream" not in sent and "messages" not in sent
                for error in (422, 429, 529):
                    status, headers, result = call("POST", "/v1/systemone", {"model": "jev-1.13.0", "state": {"error": error}, "questions": QUESTIONS})
                    assert status == error and result == {"detail": "Fixture upstream failure"}, (status, result)
                    if error in (429, 529):
                        assert next(v for k, v in headers.items() if k.lower() == "retry-after") == "17"
                hits = len(upstream.hits)
                body = {"model": "jev-1.13.0", "state": "report", "questions": QUESTIONS}
                assert call("POST", "/v1/systemone", body, "invalid")[0] in (401, 403)
                assert call("POST", "/v1/systemone", body, "other")[0] in (403, 404)
                assert call("POST", "/v1/chat/completions", {"model": "jev-1.13.0", "messages": [{"role": "user", "content": "hi"}]})[0] in (404, 503)
                assert call("POST", "/v1/systemone", {**body, "model": "chat-model"}, "admin")[0] in (404, 503)
                for field in ("model", "state", "questions"):
                    invalid = {k: v for k, v in body.items() if k != field}
                    assert call("POST", "/v1/systemone", invalid)[0] == 422
                assert len(upstream.hits) == hits
                status, _, catalog = call("GET", "/v1/model-channels?endpoint=/v1/systemone&stream=false", key="admin")
                for path in ["model-channels", "channel-metrics"]:
                    status, _, all_catalog = call("GET", f"/v1/{path}?endpoint=all&stream=all", key="admin")
                    assert status == 200 and any(r["engine"] == "typesafe" for r in all_catalog["data"]), all_catalog
                    status, _, streaming = call("GET", f"/v1/{path}?endpoint=all&stream=true", key="admin")
                    assert status == 200 and all(r["engine"] != "typesafe" for r in streaming["data"]), streaming
                assert status == 200 and all(row["engine"] == "typesafe" for row in catalog["data"])
                print("PASS: TypeSafe model discovery, SDK catalog, all state/question shapes, aliases, usage, errors, permissions and endpoint isolation")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    upstream.shutdown()


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
