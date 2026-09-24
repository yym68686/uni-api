"""Offline HTTP checks for key-scoped, version-independent Codex model cards."""
import hashlib
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


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.server.hits.append(self.path)
        self.send_response(500)
        self.end_headers()

    do_POST = do_GET


def verify(binary):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    templates = json.loads(
        (Path(__file__).resolve().parents[2] / "assets/codex/codex_models_pro_0_153_2.json").read_text()
    )["models"]
    known = {model["slug"]: model for model in templates}
    expected = {
        "gpt-6-astra", "gpt-6-sol", "gpt-5.6-sol", "codex-auto-review",
        "claude-opus-5", "gemini-3.8-flash", "grok-4.6", "deepseek-v4-pro", "custom-chat",
    }
    excluded = {
        "gpt-image-2", "gpt-image-2.5", "gemini-embedding-001", "text-embedding-004",
        "gemini-3.1-flash-image", "gemini-2.5-flash-tts", "sora-2", "seedance-2-0",
    }
    with tempfile.TemporaryDirectory(prefix="uni-codex-models-") as directory:
        root = Path(directory)
        port = free_port()
        base = f"http://127.0.0.1:{upstream.server_port}"

        def provider(name, models, **extra):
            return {"provider": name, "base_url": base + "/v1/responses",
                    "engine": "gpt", "api": "fixture-upstream-key", "model": models, **extra}

        config = {
            "providers": [
                provider("primary", sorted(expected | excluded) + [{"gpt-image-2.5": "masked-media"}]),
                provider("fallback", ["gpt-6-sol"]),
                provider("private", ["private-model"]),
                provider("blocked", ["blocked-model"], exclude_endpoints=["/v1/responses"]),
                provider("systemone", ["jev-latest"], engine="typesafe"),
            ],
            "api_keys": [
                {"api": "admin-fixture", "model": ["all"]},
                {"api": "business-fixture", "model": ["primary/*", "fallback/*", "blocked/*", "systemone/*"]},
                {"api": "restricted-fixture", "model": ["primary/gpt-6-sol"]},
                {"api": "nested-fixture", "model": ["restricted-fixture/*"]},
                {"api": "media-fixture", "model": ["primary/gpt-image-2.5", "primary/gemini-embedding-001"]},
                {"api": "other-fixture", "model": ["private/*"]},
            ],
        }
        config_path = root / "api.json"
        config_path.write_text(json.dumps(config))
        original = config_path.read_bytes()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("FACTS_S3_", "UNI_API_CONTROL_RESTORE_", "OTEL_", "FUGUE_OTEL_"))
               and k != "CONFIG_URL"}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(config_path),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"),
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0",
                   NO_PROXY="127.0.0.1,localhost")

        def call(path, key="business-fixture", body=None, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            request_headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
            request_headers.update(headers or {})
            try:
                connection.request("GET" if body is None else "POST", path,
                                   None if body is None else json.dumps(body), request_headers)
                response = connection.getresponse()
                raw = response.read()
                return response.status, {k.lower(): v for k, v in response.getheaders()}, raw
            finally:
                connection.close()

        def slugs(key="business-fixture"):
            status, _, body = call("/v1/models?client_version=arbitrary", key)
            assert status == 200, (status, body)
            return {model["slug"] for model in json.loads(body)["models"]}

        with (root / "gateway.log").open("w+") as log:
            process = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(160):
                    try:
                        if call("/healthz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError("fixture did not become ready")
                baseline = None
                baseline_etag = None
                for version in ["", "0.1.0", "0.153.2", "0.155.1", "999.999.999", "future-version"]:
                    status, headers, raw = call("/v1/models?client_version=" + version)
                    assert status == 200 and headers["x-uni-api-models-source"] == "key-scoped-catalog"
                    assert int(headers["content-length"]) == len(raw)
                    assert "private" in headers["cache-control"]
                    assert "x-uni-api-models-snapshot-client-version" not in headers
                    assert "x-uni-api-models-upstream-etag" not in headers
                    if baseline is None:
                        baseline, baseline_etag = raw, headers["etag"]
                    assert raw == baseline and headers["etag"] == baseline_etag, version
                cards = {model["slug"]: model for model in json.loads(baseline)["models"]}
                assert set(cards) == expected, (set(cards), expected)
                for slug in expected & known.keys():
                    assert cards[slug] == known[slug], slug
                assert cards["gpt-6-astra"]["context_window"] == 600000
                for slug in expected - known.keys():
                    assert cards[slug]["display_name"] == slug
                    for field, value in known["gpt-5.6-sol"].items():
                        if field not in {"slug", "display_name", "description", "priority"}:
                            assert cards[slug][field] == value, (slug, field)
                status, _, raw = call("/v1/models?client_version=another",
                                      headers={"If-None-Match": "W/" + baseline_etag})
                assert status == 304 and not raw
                assert slugs("restricted-fixture") == {"gpt-6-sol"}
                assert slugs("nested-fixture") == {"gpt-6-sol"}
                assert slugs("media-fixture") == set()
                assert slugs("other-fixture") == {"private-model"}
                assert slugs("admin-fixture") == expected | {"private-model"}
                assert call("/v1/models?client_version=1", "invalid")[0] == 403
                assert call("/v1/models?client_version=1", "")[0] == 403
                # A key with fewer grants cannot reuse another key's ETag.
                assert call("/v1/models?client_version=1", "restricted-fixture",
                            headers={"If-None-Match": baseline_etag})[0] == 200
                status, _, ordinary = call("/v1/models")
                ids = {model["id"] for model in json.loads(ordinary)["data"]}
                assert status == 200 and excluded <= ids
                assert "jev-latest" in ids and "blocked-model" in ids

                def controls():
                    status, _, raw = call("/v1/channel-controls", "admin-fixture")
                    assert status == 200
                    return json.loads(raw)

                key_id = "key-" + hashlib.sha256(b"business-fixture").hexdigest()

                def disable(providers):
                    status, _, body = call("/v1/channel-controls", "admin-fixture", {
                        "revision": controls()["revision"], "action": "set", "api_key_id": key_id,
                        "model": "gpt-6-sol", "order": [], "disabled": providers,
                    })
                    assert status == 200, body

                disable(["primary"])
                assert "gpt-6-sol" in slugs()  # Another permitted route remains.
                disable(["primary", "fallback"])
                assert "gpt-6-sol" not in slugs()
                assert "gpt-6-sol" in slugs("restricted-fixture")
                disable([])
                status, _, body = call("/v1/temporary-channels", "admin-fixture", {
                    "revision": controls()["revision"], "api_key_id": key_id,
                    "provider": "sub2api-temporary-fixture", "base_url": base + "/v1/responses",
                    "api_key": "temporary-upstream", "models": ["new-temporary-model"], "position": 1,
                })
                assert status == 200, body
                assert slugs() == expected | {"new-temporary-model"}
                assert "new-temporary-model" not in slugs("admin-fixture")
                assert not upstream.hits, upstream.hits
                assert config_path.read_bytes() == original
                print("PASS Codex catalog: per-key/nested grants, aliases, modality exclusions, "
                      "all client versions identical, 600k Astra, template metadata, ETag, "
                      "local disable/fallback/temporary scope, ordinary catalog unchanged, no upstream calls")
            except Exception:
                log.flush()
                print((root / "gateway.log").read_text()[-6000:])
                raise
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
