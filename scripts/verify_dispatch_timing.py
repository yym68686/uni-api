"""Offline regression: arrival-to-send timing across body upload, retries and hedges."""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlencode


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.hits.append(self.path)
        if self.path.startswith("/first/"):
            time.sleep(0.45)
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(b'{"error":{"message":"fixture temporarily unavailable"}}')
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.server.second_started.set()
        self.server.release_second.wait(8)
        if self.path.endswith("/messages"):
            response = {"id": "msg_fixture", "type": "message", "role": "assistant", "model": "m",
                        "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1}}
            frames = [
                {"type": "message_start", "message": {**response, "content": [], "stop_reason": None}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            ]
            stream_body = "".join(f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n" for frame in frames).encode()
        elif self.path.endswith("/responses"):
            response = {"id": "resp_fixture", "object": "response", "status": "completed",
                        "model": "m", "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": "OK", "annotations": []}]}],
                        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
            frames = [{"type": "response.output_text.delta", "delta": "OK", "item_id": "msg_fixture", "output_index": 0, "content_index": 0},
                      {"type": "response.completed", "response": response}]
            stream_body = "".join(f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n" for frame in frames).encode()
        else:
            response = {"id": "chat_fixture", "object": "chat.completion", "model": "m",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            stream_body = b'data: {"id":"chat_fixture","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}\n\ndata: {"id":"chat_fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        body = stream_body if payload.get("stream") else json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if payload.get("stream") else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def get_json(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
    try:
        conn.request("GET", path, headers={"Authorization": "Bearer fixture-key"})
        response = conn.getresponse()
        assert response.status == 200, response.status
        return json.loads(response.read())
    finally:
        conn.close()


def verify(binary, endpoint, streaming, hedging, idempotent=False):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    upstream.second_started = threading.Event()
    upstream.release_second = threading.Event()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    port = free_port()
    request_id = f"fixture-{endpoint.split('/')[-1]}-{streaming}-{hedging}-{idempotent}"
    with tempfile.TemporaryDirectory(prefix="uni-dispatch-") as directory:
        root = Path(directory)
        config = {"providers": [
            {"provider": name, "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}{endpoint}",
             "api": "fixture-upstream-key", "model": ["m"], "engine": "claude" if endpoint == "/v1/messages" else "gpt",
             "preferences": {"cooldown_period": 0, "timeout_policy": {"default": {"first_byte": 0.18 if hedging else 3, "total": 5}}}}
            for name in ["first", "second"]],
            "api_keys": [{"api": "fixture-key", "model": ["first/*", "second/*"], "preferences": {"AUTO_RETRY": True}}],
            "preferences": {"hedging": {"enabled": hedging, "max_inflight_attempts": 2}}}
        config_file = root / "api.json"
        config_file.write_text(json.dumps(config))
        env = dict(os.environ, PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_file),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"), NO_PROXY="127.0.0.1,localhost",
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "memory-ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0")
        with (root / "log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        get_json(port, "/healthz")
                        break
                    except OSError:
                        time.sleep(0.05)
                payload = {"model": "m", "stream": streaming}
                payload.update({"input": "fixture"} if endpoint == "/v1/responses" else {"messages": [{"role": "user", "content": "fixture"}]})
                body = json.dumps(payload).encode()
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
                conn.putrequest("POST", endpoint)
                conn.putheader("Authorization", "Bearer fixture-key")
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(len(body)))
                conn.putheader("X-Request-ID", request_id)
                conn.putheader("X-Request-Start", "0")
                if idempotent:
                    conn.putheader("Idempotency-Key", request_id)
                conn.endheaders(body[:1])
                time.sleep(0.25)
                conn.send(body[1:])
                assert upstream.second_started.wait(6), "second channel was not dispatched"
                query = "/v1/channel-metrics?" + urlencode({"endpoint": endpoint, "stream": str(streaming).lower(), "window": "15m", "model": "m"})
                data = get_json(port, query)
                timings = {row["provider"]: row["stats"]["request_to_dispatch"] for row in data["data"]}
                assert set(timings) == {"first", "second"}
                first, second = timings["first"], timings["second"]
                assert first["sample_count"] == second["sample_count"] == 1, timings
                assert first["last_ms"] >= 200, timings
                assert second["last_ms"] - first["last_ms"] >= (130 if hedging else 400), timings
                assert second["last_ms"] < 5000, timings
                upstream.release_second.set()
                response = conn.getresponse()
                output = response.read()
                assert response.status == 200 and b"OK" in output, (response.status, output)
                conn.close()
                after = get_json(port, query)
                assert all(row["stats"]["request_to_dispatch"]["sample_count"] == 1 for row in after["data"])
                # Completion is recorded by a detached observer after stream EOF.
                for _ in range(40):
                    after = get_json(port, query)
                    if sum(row["stats"]["success"] for row in after["data"]) == 1:
                        break
                    time.sleep(0.025)
                stats = {row["provider"]: row["stats"] for row in after["data"]}
                assert stats["first"]["started"] == stats["second"]["started"] == 1, stats
                assert stats["second"]["success"] == 1, stats
                if not hedging or endpoint == "/v1/chat/completions":
                    assert stats["first"]["failed"] + stats["first"]["hedge_cancelled"] == 1, stats
                other = get_json(port, "/v1/channel-metrics?" + urlencode({"endpoint": endpoint, "stream": str(not streaming).lower(), "window": "15m", "model": "m"}))
                assert all(row["stats"]["started"] == row["stats"]["success"] == row["stats"]["failed"] == 0 for row in other["data"]), other
                combined = get_json(port, "/v1/channel-metrics?" + urlencode({"endpoint": "all", "stream": "all", "window": "15m", "model": "m"}))
                assert len(combined["data"]) == 2
                assert all(row["endpoint"] == "all" and row["stream"] is None for row in combined["data"])
                assert {row["provider"]: row["stats"] for row in combined["data"]} == stats, combined
                points = get_json(port, query.replace("/channel-metrics?", "/channel-metrics/timeseries?"))
                for row in points["data"]:
                    assert sum(point["request_to_dispatch"]["sample_count"] for point in row["points"]) == 1
                    assert sum(point["started"] for point in row["points"]) == 1
                    assert sum(point["success"] for point in row["points"]) == stats[row["provider"]]["success"]
                log.flush()
                events = [json.loads(line) for line in (root / "log").read_text().splitlines() if line.startswith('{"')]
                dispatched = [event for event in events if event.get("event") == "channel_dispatch"]
                assert len(dispatched) == 2
                assert all(event["request_id"] == request_id for event in dispatched)
                assert len({event["attempt_id"] for event in dispatched}) == 2
                print(f"PASS {endpoint} stream={streaming} hedge={hedging} idempotent={idempotent}: {first['last_ms']:.1f} -> {second['last_ms']:.1f} ms")
            except Exception:
                print("fixture upstream paths:", upstream.hits, flush=True)
                print(get_json(port, "/v1/observability/runtime"), flush=True)
                print((root / "log").read_text()[-16000:], flush=True)
                raise
            finally:
                if "conn" in locals():
                    conn.close()
                upstream.release_second.set()
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                upstream.shutdown()
                upstream.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    binary = parser.parse_args().binary.resolve()
    for endpoint, streaming, hedging, idempotent in [
        ("/v1/messages", True, False, False),
        ("/v1/messages", False, False, False),
        ("/v1/responses", True, False, False),
        ("/v1/responses", False, False, False),
        ("/v1/chat/completions", True, False, False),
        ("/v1/chat/completions", False, False, True),
        ("/v1/responses", True, True, False),
        ("/v1/chat/completions", False, True, False),
    ]:
        verify(binary, endpoint, streaming, hedging, idempotent)
