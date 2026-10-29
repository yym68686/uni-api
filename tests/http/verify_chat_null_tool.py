"""Null Chat tool text is omitted only by Chat->Responses; local mocks only."""
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

from verify_codex_defaults import Upstream, ThreadingHTTPServer, free_port


def verify(binary):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    valid = {"id": "call-null", "type": "function", "function": {
        "name": "lookup", "arguments": '{"content":null,"text":""}'}}
    forced_input = [{"role": "assistant", "content": None}, {
        "type": "function_call", "call_id": "override-call", "name": "lookup", "arguments": "{}"}]
    with tempfile.TemporaryDirectory(prefix="uni-null-tool-") as directory:
        root = Path(directory)
        port = free_port()
        providers = []
        for name in ("auto", "gpt", "codex", "override"):
            provider = {"provider": name, "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}/v1/responses",
                        "api": "fixture", "model": [{"upstream-model": "public-model"}]}
            if name in ("gpt", "codex"):
                provider["engine"] = name
            if name == "override":
                provider["preferences"] = {"post_body_parameter_overrides": {"input": forced_input}}
            providers.append(provider)
        config = {"providers": providers, "preferences": {"AUTO_RETRY": False},
                  "api_keys": [{"api": f"{name}-fixture", "model": [f"{name}/*"]}
                               for name in ("auto", "gpt", "codex", "override")]}
        config_path = root / "config.json"
        original_config = json.dumps(config).encode()
        config_path.write_bytes(original_config)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0",
                   RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0", NO_PROXY="127.0.0.1,localhost")

        def request(path, payload=None, provider="auto"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                conn.request("GET" if payload is None else "POST", path,
                             None if payload is None else json.dumps(payload),
                             {"Authorization": f"Bearer {provider}-fixture", "Content-Type": "application/json"})
                response = conn.getresponse()
                return response.status, response.read()
            finally:
                conn.close()

        def outbound(payload, provider, path="/v1/chat/completions"):
            before = copy.deepcopy(payload)
            upstream.hits.clear()
            status, raw = request(path, payload, provider)
            assert status == 200 and b"OK" in raw, (status, raw)
            assert len(upstream.hits) == 1, upstream.hits
            assert payload == before
            return upstream.hits[0][1]

        with (root / "gateway.log").open("w") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(120):
                    try:
                        if request("/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError("local gateway did not start")
                checked = 0
                for provider in ("auto", "gpt", "codex"):
                    for stream in (False, True):
                        for placeholder in ("", None, "absent"):
                            turn = {"role": "assistant", "tool_calls": [valid]}
                            if placeholder != "absent":
                                turn["content"] = placeholder
                            payload = {"model": "public-model", "stream": stream,
                                       "messages": [{"role": "user", "content": "start"}, turn,
                                                    {"role": "tool", "tool_call_id": "call-null", "content": ""},
                                                    {"role": "assistant", "content": " "}],
                                       "max_tokens": 65535, "reasoning_effort": "medium"}
                            body = outbound(payload, provider)
                            assert body["input"] == [
                                {"role": "user", "content": [{"type": "input_text", "text": "start"}]},
                                {"type": "function_call", "call_id": "call-null", "name": "lookup",
                                 "arguments": valid["function"]["arguments"]},
                                {"type": "function_call_output", "call_id": "call-null", "output": ""},
                                {"role": "assistant", "content": [{"type": "output_text", "text": " "}]},
                            ], body
                            assert body["max_output_tokens"] == 65535 and body["reasoning"]["effort"] == "medium"
                            checked += 1

                        negatives = [
                            {"role": "assistant", "content": None},
                            {"role": "assistant", "content": None, "tool_calls": []},
                            {"role": "user", "content": None, "tool_calls": [valid]},
                            {"role": "assistant", "content": [], "tool_calls": [valid]},
                            {"role": "assistant", "content": False, "tool_calls": [valid]},
                            {"role": "assistant", "content": 42, "tool_calls": [valid]},
                        ]
                        for field, value in (("id", " "), ("name", " "), ("arguments", "bad JSON"),
                                             ("arguments", "[]"), ("arguments", {})):
                            invalid = copy.deepcopy(valid)
                            (invalid if field == "id" else invalid["function"])[field] = value
                            negatives.append({"role": "assistant", "content": None, "tool_calls": [invalid]})
                        negatives.append({"role": "assistant", "content": None, "tool_calls": [valid, {"type": "custom"}]})
                        for message in negatives:
                            body = outbound({"model": "public-model", "stream": stream, "messages": [message]}, provider)
                            assert body["input"][0] == {"role": message["role"], "content": []}, body
                            checked += 1

                        native = [{"role": "assistant", "content": None}, {
                            "type": "function_call", "call_id": "native-call", "name": "lookup", "arguments": "{}"}]
                        body = outbound({"model": "public-model", "stream": stream, "input": native}, provider, "/v1/responses")
                        assert body["input"] == native
                        checked += 1
                        body = outbound({"model": "public-model", "stream": stream, "messages": [
                            {"role": "assistant", "content": None, "tool_calls": [valid]}]}, "override")
                        assert body["input"] == forced_input
                        checked += 1
                    body = outbound({"model": "public-model", "input": forced_input}, provider, "/v1/responses/compact")
                    assert body["input"] == forced_input
                    checked += 1
                assert config_path.read_bytes() == original_config
                print(f"PASS null tool text: {checked} HTTP cases; auto/GPT/Codex, stream/nonstream, "
                      "calls/results/order/limits preserved, invalid inputs/native/override unchanged")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                upstream.shutdown()
                upstream.server_close()


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
