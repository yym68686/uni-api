"""Service tier evidence and routing; every upstream is loopback."""
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

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tier = body.get("service_tier")
        self.server.hits.append((self.path, self.headers["Authorization"], body))
        name = self.path.split("/")[1]
        fault = self.server.faults.get((name, tier))
        if fault and body.get("input") != "Reply OK.":
            status, error = fault
            raw = json.dumps({"error": error}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        response = {"id": "resp_test", "status": "completed", "object": "response",
                    "model": body["model"], "output": [{"id": "msg", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "text": "OK"}]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "service_tier": {"fast": "priority", "ultrafast": (
                        "ultrafast" if self.path.startswith("/verified/") else "default"
                    )}.get(tier, tier)}
        items = body.get("input")
        if self.path.endswith("/compact") or (isinstance(items, list) and any(
            item.get("type") == "compaction_trigger" for item in items
        )):
            response["output"] = [{"type": "compaction", "encrypted_content": "fixture"}]
        if name == "compat":
            response = {"id": "chatcmpl_1", "object": "chat.completion", "model": body["model"],
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        kind = "application/json"
        raw = json.dumps(response).encode()
        if body.get("stream"):
            kind = "text/event-stream"
            events = [{"type": "response.created", "response": dict(response, status="in_progress", service_tier=tier)}]
            if name != "early":
                events.append({"type": "response.output_text.delta", "delta": "OK", "item_id": "msg",
                               "output_index": 0, "content_index": 0})
                if (name, tier) in self.server.fail_after_output:
                    events.append({"type": "response.failed", "response": dict(
                        response, status="failed", error={"code": "server_error", "message": "Upstream unavailable"})})
                else:
                    events.append({"type": "response.completed", "response": response})
            raw = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main():
    binary = Path(sys.argv[1]).resolve()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.hits = []
    upstream.faults = {}
    upstream.fail_after_output = set()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    probe_path = "/v1/channel-settings/service-tier-probe"
    global_scope = "__uni_console_global_preferences__"
    with tempfile.TemporaryDirectory(prefix="service-tier-test-") as directory:
        port, root = free_port(), Path(directory)
        providers = [
            {"provider": name, **({} if name == "portable" else {"engine": "gpt"}),
             "base_url": f"http://127.0.0.1:{upstream.server_port}/{name}/v1/responses",
             "api": ["first-" + name, "second-" + name], "model": [{"wire": "public"}],
             "preferences": {"compaction_mode": "native", "api_key_schedule_algorithm": "fixed_priority", **(
                 {"upstream_stream_mode": "stream_only"} if name in ("early", "full") else {}
             )}}
            for name in ("peer", "verified", "early", "full", "portable")
        ]
        providers.append({
            "provider": "compat", "engine": "gpt",
            "base_url": f"http://127.0.0.1:{upstream.server_port}/compat/v1/chat/completions",
            "api": "compat-key", "model": [{"wire": "public"}],
        })
        config = {"providers": providers, "api_keys": [
            {"api": "admin", "role": "admin", "model": ["all"]},
            {"api": "caller", "role": "user", "model": ["peer/*", "verified/*"],
             "preferences": {"SCHEDULING_ALGORITHM": "fixed_priority"}},
            {"api": "peer-only", "role": "user", "model": ["peer/*"]},
            {"api": "verified-only", "role": "user", "model": ["verified/*"]},
            {"api": "mixed", "role": "user", "model": ["compat/*", "verified/*"]},
            {"api": "no-retry", "role": "user", "model": ["peer/*", "verified/*"],
             "preferences": {"AUTO_RETRY": False}},
        ]}
        (root / "config.json").write_text(json.dumps(config))
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "TMPDIR", "LANG", "DYLD_LIBRARY_PATH")}
        env.update(PORT=str(port), DISABLE_DATABASE="true", UNI_API_CONFIG_PATH=str(root / "config.json"),
                   RUST_RESPONSES_CONFIG_SNAPSHOT_PATH=str(root / "snapshot.json"),
                   UNI_API_SHARED_MEMORY_RESERVATION_PATH=str(root / "ledger"),
                   RUST_REQUEST_SPOOL_DIRECTORY=str(root / "spool"), NO_PROXY="127.0.0.1,localhost",
                   RUST_REQUEST_SPOOL_DISK_RESERVE_BPS="0", RUST_REQUEST_SPOOL_INODE_RESERVE_BPS="0")

        def call(method, path, body=None, key="admin"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=6)
            try:
                conn.request(method, path, json.dumps(body) if body is not None else None,
                             {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                response = conn.getresponse()
                raw = response.read().decode()
                try:
                    return response.status, json.loads(raw)
                except ValueError:
                    return response.status, raw
            finally:
                conn.close()

        def setting(provider, changes, operation, remove=None):
            code, view = call("GET", "/v1/channel-settings?provider=" + provider)
            assert code == 200, (code, view)
            code, applied = call("PATCH", "/v1/channel-settings", {
                "revision": view["revision"], "operation_id": operation,
                "changes": [{"provider": provider, "set": changes, "remove": remove or []}],
            })
            assert code == 200, (code, applied)

        def request(tier=None, key="caller", endpoint="/v1/responses", **options):
            return call("POST", endpoint, {"model": "public", "input": "test",
                                                 **({"service_tier": tier} if tier else {}), **options}, key)

        def check_route(tier, expected, key="caller", status=200, **options):
            before = len(upstream.hits)
            code, body = request(tier, key, **options)
            assert code == status, (code, body, expected)
            attempts = [(path.split("/")[1], sent.get("service_tier"))
                        for path, _, sent in upstream.hits[before:]]
            assert attempts == expected, (attempts, expected, body)
            return body

        def assert_catalog(tiers, key="caller"):
            before = len(upstream.hits)
            for path in ("/v1/codex/models", "/v1/models?client_version="):
                status, catalog = call("GET", path, key=key)
                assert status == 200, (path, status, catalog)
                card, = catalog["models"]
                assert card["slug"] == "public"
                assert [tier["id"] for tier in card.get("service_tiers", [])] == tiers, card
                assert all(tier["name"] and tier["description"] for tier in card.get("service_tiers", []))
                assert card.get("additional_speed_tiers", []) == (["fast"] if "priority" in tiers else [])
                assert "default_service_tier" not in card
            assert len(upstream.hits) == before

        with (root / "process.log").open("w+") as log:
            proc = subprocess.Popen([str(binary)], cwd=root, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError(f"Gateway exited unexpectedly: {proc.returncode}")
                    try:
                        if call("GET", "/readyz")[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(.05)
                assert call("GET", "/v1/observability/runtime")[1]["service_tier_probe_version"] == 1
                assert_catalog([])
                req = {"provider": "verified", "model": "public", "tier": "fast"}
                assert call("POST", probe_path, req, "caller")[0] == 403
                code, pre = call("POST", probe_path, dict(req, verify_only=True))
                assert code == 200 and pre["status"] == "ready" and not upstream.hits, pre
                code, portable = call("POST", probe_path, dict(req, provider="portable", verify_only=True))
                assert code == 200 and portable["status"] == "ready" and portable["compatibility"]
                candidate = {"base_url": next(p["base_url"] for p in providers if p["provider"] == "portable"),
                             "api_key": "first-portable",
                             "models": {"new-alias": "wire"}}
                preview = dict(req, provider="candidate", verify_only=True, candidate=candidate)
                assert call("POST", probe_path, dict(preview, verify_only=False))[0] == 400
                code, options = call("POST", probe_path, preview)
                assert code == 200, options
                option = options["models"]["new-alias"]
                assert option["binding"] != portable["binding"]
                assert option["compatibility"] == portable["compatibility"]
                code, options = call("POST", probe_path, dict(preview, candidate=dict(candidate, api_key="rotated")))
                assert code == 200 and options["models"]["new-alias"]["compatibility"] != option["compatibility"]
                assert call("POST", probe_path, dict(req, expected_scope="0" * 64))[0] == 409
                assert not upstream.hits
                code, result = call("POST", probe_path, dict(req, expected_scope=pre["scope"]))
                assert code == 200 and result["status"] == "supported" and result["observed"] == "priority", result
                assert result["binding"] == pre["binding"]
                assert len(upstream.hits) == 1 and upstream.hits[-1] == (
                    "/verified/v1/responses", "Bearer first-verified",
                    {"model": "wire", "input": "Reply OK.", "service_tier": "fast", "store": False, "stream": False})
                code, slow = call("POST", probe_path, dict(req, provider="peer", tier="ultrafast"))
                assert code == 200 and slow["status"] == "not_honored" and slow["observed"] == "default", slow
                for provider, status in (("early", "inconclusive"), ("full", "supported")):
                    code, streamed = call("POST", probe_path, dict(req, provider=provider))
                    assert code == 200 and streamed["status"] == status, (provider, streamed)
                assert len(upstream.hits) == 4
                check_route("ultrafast", [("peer", None)])
                check_route("fast", [("peer", None)])
                path = "/preferences/service_tier_evidence/" + result["binding"] + "/fast"
                setting("verified", {path: result["checked_at"] - 7 * 86400}, "expired")
                check_route("fast", [("peer", None)])
                assert_catalog([])
                setting("verified", {path: result["checked_at"]}, "publish")
                assert_catalog(["priority"])
                assert_catalog([], "peer-only")
                for tier in ("fast", "priority"):
                    code, served = request(tier)
                    assert code == 200 and served["service_tier"] == "priority", served
                    assert upstream.hits[-1][:2] == ("/verified/v1/responses", "Bearer first-verified")
                    assert upstream.hits[-1][2]["service_tier"] == "fast"
                check_route("fast", [("peer", None)], "peer-only")
                check_route("ultrafast", [("peer", None)])
                check_route("fast", [("verified", "fast")], "mixed")
                rejection = (400, {"code": "unsupported_service_tier", "param": "service_tier"})
                upstream.faults["verified", "fast"] = rejection
                check_route("fast", [("verified", "fast"), ("peer", None)])
                check_route("fast", [("verified", "fast"), ("compat", None)], "mixed")
                streamed = check_route("fast", [("verified", "fast"), ("peer", None)], stream=True)
                assert "response.completed" in streamed, streamed
                upstream.faults.clear()
                upstream.fail_after_output.add(("verified", "fast"))
                streamed = check_route("fast", [("verified", "fast")], stream=True)
                assert "response.output_text.delta" in streamed and '"delta": "OK"' in streamed, streamed
                assert "response.completed" not in streamed, streamed
                upstream.fail_after_output.clear()
                upstream.faults["verified", "fast"] = rejection
                check_route("fast", [("verified", "fast"), ("verified", None)], "verified-only")
                upstream.faults["verified", "fast"] = (400, {"code": "bad_input", "param": "input"})
                check_route("fast", [("verified", "fast")], status=400)
                check_route("fast", [("verified", "fast")], "mixed", status=400)
                upstream.faults.clear()
                code, ultra = call("POST", probe_path, dict(req, tier="ultrafast"))
                assert code == 200 and ultra["status"] == "supported", ultra
                ultra_path = "/preferences/service_tier_evidence/" + ultra["binding"] + "/ultrafast"
                disabled_path = "/preferences/service_tier_disabled/fast"
                setting("verified", {ultra_path: ultra["checked_at"], disabled_path: True}, "enable-ultra-disable-fast")
                assert_catalog(["ultrafast"])
                assert_catalog(["ultrafast"], "verified-only")
                code, peer_fast = call("POST", probe_path, dict(req, provider="peer"))
                assert code == 200 and peer_fast["status"] == "supported", peer_fast
                peer_path = "/preferences/service_tier_evidence/" + peer_fast["binding"] + "/fast"
                setting("peer", {peer_path: peer_fast["checked_at"]}, "enable-fast-on-peer")
                assert_catalog(["priority", "ultrafast"])
                assert_catalog(["priority"], "peer-only")
                assert_catalog(["ultrafast"], "verified-only")
                check_route("fast", [("verified", None)], "verified-only")
                assert request("ultrafast", "verified-only")[0] == 200
                assert upstream.hits[-1][0] == "/verified/v1/responses"
                assert request(key="verified-only")[0] == 200
                assert upstream.hits[-1][0] == "/verified/v1/responses"
                # Renewing detection evidence must not undo an explicit routing preference.
                setting("verified", {path: result["checked_at"]}, "renew-fast-evidence")
                check_route("fast", [("verified", None)], "verified-only")
                setting(global_scope, {"/preferences/service_tier_routing": False}, "disable-global-filter")
                assert_catalog(["priority", "ultrafast"])
                assert request("fast", "verified-only")[0] == 503
                assert request("ultrafast", "verified-only")[0] == 200
                check_route("ultrafast", [("peer", "ultrafast")])
                assert request(key="verified-only")[0] == 200
                setting(global_scope, {"/preferences/service_tier_routing": True}, "restore-global-filter")
                setting("verified", {disabled_path: False}, "reenable-fast")
                assert request("fast", "verified-only")[0] == 200
                assert request("ultrafast", "verified-only")[0] == 200
                upstream.faults["peer", "fast"] = rejection
                upstream.faults["verified", "fast"] = rejection
                check_route("fast", [("peer", "fast"), ("verified", "fast"), ("peer", None)])
                upstream.faults.clear()
                code, response = request("default")
                assert code == 200 and upstream.hits[-1][0] == "/peer/v1/responses", response
                setting("verified", {}, "withdraw", [path])
                setting("peer", {}, "withdraw-peer-fast", [peer_path])
                assert_catalog(["ultrafast"])
                check_route("fast", [("peer", None)])
                assert request()[0] == 200 and upstream.hits[-1][0] == "/peer/v1/responses"
                policy = "/preferences/compaction_ultrafast_only"
                setting("verified", {policy: True}, "ultrafast-compaction-only")
                assert_catalog(["ultrafast"])
                check_route(None, [("verified", None)], "verified-only")
                check_route("fast", [("verified", None)], "verified-only")
                check_route("ultrafast", [("verified", "ultrafast")], "verified-only")
                for endpoint in ("/v1/responses", "/v1/responses/compact"):
                    fixture = dict(endpoint=endpoint, input=[
                        {"type": "message", "role": "user", "content": "test"},
                        *([{"type": "compaction_trigger"}] if endpoint == "/v1/responses" else []),
                    ])
                    for tier in (None, "fast", "priority", "default"):
                        check_route(tier, [], "verified-only", status=404, **fixture)
                    check_route("ultrafast", [("verified", "ultrafast")], "verified-only", **fixture)
                fixture["endpoint"] = "/v1/responses"
                fixture["input"].append({"type": "compaction_trigger"})
                upstream.faults["verified", "ultrafast"] = rejection
                check_route("ultrafast", [("verified", "ultrafast"), ("peer", None)], **fixture)
                check_route("ultrafast", [("verified", "ultrafast")], "verified-only", status=400, **fixture)
                upstream.faults.clear()
                setting("verified", {}, "remove-ultrafast-evidence", [ultra_path])
                check_route("ultrafast", [], "verified-only", status=503, **fixture)
                setting("verified", {ultra_path: ultra["checked_at"]}, "restore-ultrafast-evidence", [policy])
                # An unavailable accelerated route must not take up a send
                # on the next request, even when AUTO_RETRY disables replays.
                setting("verified", {"/preferences/cooldown_period": 30}, "cool-after-failure")
                upstream.faults["verified", "ultrafast"] = (503, {"message": "Upstream unavailable"})
                check_route("ultrafast", [("verified", "ultrafast"), ("peer", None)])
                check_route("ultrafast", [("peer", None)], "no-retry")
                check_route("ultrafast", [("compat", None)], "mixed")
            except Exception:
                print((root / "process.log").read_text()[-8000:])
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                upstream.shutdown()
                upstream.server_close()
    print("PASS verified tier priority, cooldown/error fallback, native/compat streaming, safety policies and catalogs")


if __name__ == "__main__":
    main()
