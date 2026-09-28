#!/usr/bin/env python3
"""Roll out uni-api on DigitalOcean through a restored loopback candidate.

The script deliberately keeps the serving container on 8001 until a new
candidate on 8002 has restored the uni-api-web control snapshot and matched the
current public control intent. It never prints environment values or request
bodies.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


REMOTE_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

PRIMARY = "uni-api"
CANDIDATE = "uni-api-management-candidate"
PRIMARY_PORT = 8001
CANDIDATE_PORT = 8002
COMPOSE = Path("/root/docker-compose.yml")
CONFIG = Path("/root/api-copy.yaml")
CANDIDATE_ENV = Path("/root/uni-api-rollout-candidate.env")
CANDIDATE_DATA = Path("/root/uni-api-management-candidate-data")
ORIGIN_POLICY = Path("/etc/caddy/origin-switch-policy.json")
ORIGIN_HELPER = "/usr/local/sbin/fugue-caddy-origin-switch"


class RolloutError(RuntimeError):
    pass


def run(command, *, check=True, capture=True):
    result = subprocess.run(
        command,
        check=False,
        capture_output=capture,
        text=True,
        timeout=None,
    )
    if check and result.returncode != 0:
        raise RolloutError("command failed: " + str(command[:2]))
    return result


def docker(*args):
    return run(["docker", *args]).stdout.strip()


def inspect(name, *, missing_ok=False):
    result = subprocess.run(
        ["docker", "inspect", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if missing_ok:
            return None
        raise RolloutError("docker inspect failed")
    return json.loads(result.stdout)[0]


def env_map(container):
    return dict(entry.split("=", 1) for entry in container["Config"].get("Env", []) if "=" in entry)


def image_info(image):
    rows = json.loads(docker("image", "inspect", image))
    if not rows:
        raise RolloutError("image inspect returned no image")
    row = rows[0]
    env = dict(entry.split("=", 1) for entry in row.get("Config", {}).get("Env", []) if "=" in entry)
    source = env.get("SOURCE_COMMIT", "")
    if not source:
        raise RolloutError("release image has no SOURCE_COMMIT")
    return {"id": row["Id"], "source_commit": source, "repo_digests": row.get("RepoDigests", [])}


def admin_token():
    import yaml

    try:
        return yaml.safe_load(CONFIG.read_text())["api_keys"][0]["api"]
    except Exception as exc:
        raise RolloutError("cannot read administrator key from api-copy.yaml") from exc


def get_json(port, path):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        headers={"Authorization": "Bearer " + admin_token()},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise RolloutError("local endpoint returned a non-200 status")
        return json.load(response)


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=8) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def intent(state):
    rules = sorted(state.get("rules") or [], key=lambda item: json.dumps(item, sort_keys=True))
    channels = sorted(state.get("temporary_channels") or [], key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "config_revision": state.get("config_revision"),
        "channel_definitions_digest": state.get("channel_definitions_digest"),
        "channel_settings_digest": state.get("channel_settings_digest"),
        "rules": rules,
        "temporary_channels": channels,
    }


def control_summary(port, *, require_bootstrap=False):
    state = get_json(port, "/v1/channel-controls")
    if not state.get("temporary_channel_restore"):
        raise RolloutError(f"port {port} does not expose temporary control restore")
    if require_bootstrap and not state.get("bootstrap_restore"):
        raise RolloutError(f"port {port} has no bootstrap_restore receipt")
    public_intent = intent(state)
    encoded = json.dumps(public_intent, sort_keys=True, separators=(",", ":")).encode()
    return {
        "intent": public_intent,
        "state_hash": hashlib.sha256(encoded).hexdigest(),
        "channels": len(public_intent["temporary_channels"]),
        "rules": len(public_intent["rules"]),
        "bootstrap_restore": bool(state.get("bootstrap_restore")),
    }


def require_equal(first, second, label):
    if first["intent"] != second["intent"]:
        raise RolloutError(f"{label} control intent differs; serving instance remains active")


def wait_ready(name, port, *, require_bootstrap, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        container = inspect(name, missing_ok=True)
        if not container or not container["State"].get("Running"):
            raise RolloutError(f"{name} stopped before readiness")
        if health(port):
            try:
                summary = control_summary(port, require_bootstrap=require_bootstrap)
                return summary
            except Exception as exc:
                last_error = type(exc).__name__
        time.sleep(1)
    raise RolloutError(f"{name} did not become ready ({last_error})")


def caddy_config():
    request = urllib.request.Request("http://127.0.0.1:2019/config/")
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def current_origin_port():
    config = caddy_config()
    found = []

    def walk(value, host_match=False):
        if isinstance(value, dict):
            matches = value.get("match") or []
            here = host_match or any(
                "api-origin.0-0.pro" in (matcher.get("host") or []) for matcher in matches if isinstance(matcher, dict)
            )
            for key, child in value.items():
                if here and key == "dial" and isinstance(child, str) and child.startswith("localhost:"):
                    found.append(child)
                else:
                    walk(child, here)
        elif isinstance(value, list):
            for child in value:
                walk(child, host_match)

    walk(config)
    ports = {int(value.rsplit(":", 1)[1]) for value in found}
    if len(ports) != 1 or ports not in ({PRIMARY_PORT}, {CANDIDATE_PORT}):
        raise RolloutError("Caddy does not expose exactly one allowed api-origin port")
    return next(iter(ports))


def adapted_header_count():
    policy = json.loads(ORIGIN_POLICY.read_text())
    result = run(
        ["/usr/bin/caddy", "adapt", "--config", policy["config_path"], "--adapter", "caddyfile"],
        check=False,
    )
    if result.returncode != 0:
        raise RolloutError("Caddy adaptation failed")
    config = json.loads(result.stdout)
    key = policy["credential_header"]
    count = 0

    def walk(value):
        nonlocal count
        if isinstance(value, dict):
            for name, child in value.items():
                if name == key:
                    count += 1
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(config)
    return count


def guard(policy, from_port=None, to_port=None):
    command = [ORIGIN_HELPER, "--policy", str(policy)]
    if from_port is not None:
        command += ["--from-port", str(from_port), "--to-port", str(to_port)]
    result = run(command, check=False)
    return result.returncode == 0


def choose_policy():
    if guard(ORIGIN_POLICY):
        print("caddy_policy=system", flush=True)
        return ORIGIN_POLICY, None
    base = json.loads(ORIGIN_POLICY.read_text())
    actual = adapted_header_count()
    expected = base.get("credential_header_count")
    if (expected, actual) != (2, 0):
        raise RolloutError("Caddy guard check failed; policy mismatch was not the header count")
    temporary = tempfile.NamedTemporaryFile(
        mode="w", prefix="uni-api-origin-policy-", suffix=".json", dir="/tmp", delete=False
    )
    temporary_path = Path(temporary.name)
    try:
        base["credential_header_count"] = actual
        json.dump(base, temporary, separators=(",", ":"))
        temporary.close()
        if not guard(temporary_path):
            raise RolloutError("temporary Caddy policy also failed check-only")
        print(f"caddy_policy=temporary observed_header_count={actual}", flush=True)
        return temporary_path, temporary_path
    except Exception:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise


def switch(policy, from_port, to_port):
    if current_origin_port() != from_port:
        raise RolloutError(f"Caddy is not serving from {from_port}")
    if not guard(policy, from_port, to_port):
        raise RolloutError(f"Caddy switch {from_port}->{to_port} failed")
    if current_origin_port() != to_port:
        raise RolloutError("Caddy switch returned without the requested serving port")
    print(f"route={to_port}", flush=True)


def public_probes():
    for host in ("api.0-0.pro", "i00.pro"):
        request = urllib.request.Request(f"https://{host}/v1/health")
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise RolloutError(f"public probe failed for {host}")
    print("public_probes=200,200", flush=True)


def stop_clean(name):
    ports = {PRIMARY: PRIMARY_PORT, CANDIDATE: CANDIDATE_PORT}
    if name not in ports or current_origin_port() == ports[name]:
        raise RolloutError("refusing to stop the serving port")
    print(f"draining={name} force_timeout=disabled", flush=True)
    run(["docker", "stop", "-t", "-1", name])
    state = inspect(name)["State"]
    if state.get("ExitCode") != 0 or state.get("OOMKilled"):
        raise RolloutError(f"{name} did not exit cleanly")
    print(f"drained={name}", flush=True)


def compose_metadata(image):
    result = run(
        ["docker", "compose", "--project-directory", "/root", "-f", str(COMPOSE), "config", "--format", "json"]
    )
    service = json.loads(result.stdout).get("services", {}).get("uni-api")
    if not service:
        raise RolloutError("Compose has no uni-api service")
    if service.get("image") != image:
        raise RolloutError("Compose image does not match the requested release image")
    env = service.get("environment") or {}
    for name in ("UNI_API_CONTROL_RESTORE_URL", "UNI_API_CONTROL_RESTORE_TOKEN"):
        if not env.get(name):
            raise RolloutError(f"Compose is missing {name}")
    mounts = service.get("volumes") or []
    if not any(item.get("target") == "/home/data" for item in mounts if isinstance(item, dict)):
        raise RolloutError("Compose is missing the primary /home/data mount")
    return service


def write_candidate_env(primary, source_commit):
    values = env_map(primary)
    for name in ("UNI_API_CONTROL_RESTORE_URL", "UNI_API_CONTROL_RESTORE_TOKEN"):
        if not values.get(name):
            raise RolloutError(f"primary environment is missing {name}")
    values["SOURCE_COMMIT"] = source_commit
    if any("\n" in value or "\r" in value for value in values.values()):
        raise RolloutError("environment contains a newline; refusing an ambiguous env-file")
    CANDIDATE_ENV.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    CANDIDATE_ENV.chmod(0o600)
    CANDIDATE_DATA.mkdir(mode=0o700, parents=True, exist_ok=True)


def candidate_volume_is_separate(primary):
    for mount in primary.get("Mounts", []):
        if mount.get("Destination") == "/home/data" and mount.get("Source") == str(CANDIDATE_DATA):
            raise RolloutError("candidate data directory is shared with the primary")


def start_candidate(primary, image_id, source_commit):
    existing = inspect(CANDIDATE, missing_ok=True)
    if existing:
        if existing["State"].get("Running"):
            raise RolloutError("candidate is already running; refusing to replace it")
        run(["docker", "rm", CANDIDATE])
    candidate_volume_is_separate(primary)
    network = primary.get("HostConfig", {}).get("NetworkMode") or "root_default"
    run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CANDIDATE,
            "--restart",
            "unless-stopped",
            "--network",
            network,
            "--env-file",
            str(CANDIDATE_ENV),
            "-p",
            "127.0.0.1:8002:8000",
            "-v",
            "/root/api-copy.yaml:/home/api.yaml",
            "-v",
            f"{CANDIDATE_DATA}:/home/data",
            "-v",
            "/etc/localtime:/etc/localtime:ro",
            image_id,
        ]
    )
    print(f"candidate_started source_commit={source_commit}", flush=True)


def final_report(image):
    primary = inspect(PRIMARY)
    candidate = inspect(CANDIDATE, missing_ok=True)
    env = env_map(primary)
    summary = control_summary(PRIMARY_PORT, require_bootstrap=True)
    if current_origin_port() != PRIMARY_PORT:
        raise RolloutError("final Caddy port is not 8001")
    if primary["Image"] != image["id"]:
        raise RolloutError("primary image does not match the release image")
    if env.get("SOURCE_COMMIT") != image["source_commit"]:
        raise RolloutError("primary SOURCE_COMMIT does not match the release image")
    if not primary["State"].get("Running") or primary.get("RestartCount") != 0:
        raise RolloutError("primary is not healthy after rollout")
    if not health(PRIMARY_PORT):
        raise RolloutError("primary health check failed")
    retained_equal(PRIMARY_PORT)
    public_probes()
    print(
        json.dumps(
            {
                "rollout_complete": True,
                "image_id": image["id"],
                "source_commit": env.get("SOURCE_COMMIT"),
                "primary_restart_count": primary.get("RestartCount"),
                "channels": summary["channels"],
                "rules": summary["rules"],
                "control_state_hash": summary["state_hash"],
                "candidate_status": candidate["State"]["Status"] if candidate else "removed",
                "candidate_exit_code": candidate["State"].get("ExitCode") if candidate else 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def check_only(image_ref, expected_commit=""):
    primary = inspect(PRIMARY)
    image = image_info(image_ref)
    if current_origin_port() != PRIMARY_PORT:
        raise RolloutError("check-only requires Caddy to serve 8001")
    summary = control_summary(PRIMARY_PORT, require_bootstrap=True)
    if not health(PRIMARY_PORT):
        raise RolloutError("primary health check failed")
    if expected_commit and env_map(primary).get("SOURCE_COMMIT") != expected_commit:
        raise RolloutError("serving SOURCE_COMMIT differs from the expected release")
    retained_equal(PRIMARY_PORT)
    print(
        json.dumps(
            {
                "check_only": True,
                "primary_image": primary["Image"],
                "local_tag_image": image["id"],
                "source_commit": env_map(primary).get("SOURCE_COMMIT"),
                "channels": summary["channels"],
                "rules": summary["rules"],
                "control_state_hash": summary["state_hash"],
                "bootstrap_restore": summary["bootstrap_restore"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def retained_equal(port):
    """Compare routing/settings with the console's durable source snapshot."""
    live = get_json(port, "/v1/channel-controls")
    serving = inspect(PRIMARY if port == PRIMARY_PORT else CANDIDATE)
    env = env_map(serving)
    request = urllib.request.Request(env["UNI_API_CONTROL_RESTORE_URL"], headers={
        "Authorization": "Bearer " + env["UNI_API_CONTROL_RESTORE_TOKEN"],
        "X-Uni-Channel-Settings-Version": "1",
    })
    with urllib.request.urlopen(request, timeout=25) as response:
        retained = json.load(response)
    if not retained.get("enabled") or not retained.get("snapshot_id"):
        raise RolloutError("console retention is disabled or has no verified snapshot")
    saved = retained["snapshot"]
    def routing(value):
        return {
            "rules": sorted(value.get("rules", []), key=lambda row: (row["api_key_id"], row["model"])),
            "channels": sorted([
                {"provider": row["provider"], "api_key_id": row["api_key_id"], "models": sorted(row["models"])}
                for row in value.get("temporary_channels", [])
            ], key=lambda row: row["provider"]),
        }
    def digest(value):
        # Match control/settings.rs, which canonicalizes integral JSON floats.
        def normalize(item):
            if isinstance(item, dict):
                return {key: normalize(child) for key, child in item.items()}
            if isinstance(item, list):
                return [normalize(child) for child in item]
            if isinstance(item, float) and item.is_integer() and abs(item) < 9007199254740992:
                return int(item)
            return item
        return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    definitions = {row["provider"]: row["definition"] for row in saved.get("temporary_channels", []) if row.get("definition")}
    if (routing(saved) != routing(live)
        or digest(saved.get("channel_settings", {})) != live.get("channel_settings_digest")
        or digest(definitions) != live.get("channel_definitions_digest")):
        raise RolloutError("console saved intent differs; preserve the serving instance")
    print(f"retained_equal_port={port}", flush=True)


def rollout(args):
    # This lock serializes invocations of this script, including long drains.
    primary = inspect(PRIMARY)
    service = compose_metadata(args.image)
    baseline = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (COMPOSE, CONFIG)}
    def unchanged():
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != value for p, value in baseline.items()):
            raise RolloutError("base config or Compose changed during rollout")
        if compose_metadata(args.image) != service:
            raise RolloutError("effective Compose configuration changed during rollout")
        if image_info(args.image)["id"] != image["id"]:
            raise RolloutError("image tag moved during rollout; preserve serving candidate")

    current_port = current_origin_port()
    candidate = inspect(CANDIDATE, missing_ok=True)
    if candidate and candidate["State"].get("Running"):
        # Resume the pinned candidate, never pull a newer tag over an active rollout.
        image = image_info(candidate["Image"])
        if image_info(args.image)["id"] != image["id"]:
            raise RolloutError("active candidate and image tag differ; manual reconciliation required")
    else:
        if current_port != PRIMARY_PORT:
            raise RolloutError("Caddy is on 8002 without a running candidate")
        run(["docker", "pull", args.image])
        image = image_info(args.image)
    if args.expected_commit and image["source_commit"] != args.expected_commit:
        raise RolloutError("image SOURCE_COMMIT differs from the expected release")
    policy, temporary_policy = choose_policy()
    try:
        retained_equal(current_port)
        if current_port == PRIMARY_PORT and primary["Image"] == image["id"]:
            if candidate and candidate["State"].get("Running"):
                stop_clean(CANDIDATE)
            final_report(image)
            print("no_update_needed=true", flush=True)
            return
        if not (candidate and candidate["State"].get("Running")):
            # Existing stopped spools are preserved and mounted again, never deleted.
            write_candidate_env(primary, image["source_commit"])
            start_candidate(primary, image["id"], image["source_commit"])
        wait_ready(CANDIDATE, CANDIDATE_PORT, require_bootstrap=True,
                   timeout_seconds=args.ready_timeout)
        retained_equal(CANDIDATE_PORT)
        unchanged()
        if current_port == PRIMARY_PORT:
            require_equal(control_summary(PRIMARY_PORT), control_summary(CANDIDATE_PORT, require_bootstrap=True), "pre-switch")
            switch(policy, PRIMARY_PORT, CANDIDATE_PORT)

        # Check the live serving candidate even on resume before stopping anything.
        if current_origin_port() != CANDIDATE_PORT or not health(CANDIDATE_PORT):
            raise RolloutError("candidate is not serving and healthy")
        primary = inspect(PRIMARY)
        if primary["Image"] != image["id"] or not primary["State"].get("Running"):
            if primary["State"].get("Running"):
                stop_clean(PRIMARY)
            unchanged()
            run(["docker", "compose", "--project-directory", "/root", "-f", str(COMPOSE),
                 "up", "-d", "--no-deps", "--pull", "never", "uni-api"])
        primary_summary = wait_ready(PRIMARY, PRIMARY_PORT, require_bootstrap=True,
                                     timeout_seconds=args.ready_timeout)
        require_equal(primary_summary, control_summary(CANDIDATE_PORT, require_bootstrap=True), "pre-switch-back")
        retained_equal(PRIMARY_PORT)
        unchanged()
        if inspect(PRIMARY)["Image"] != image["id"]:
            raise RolloutError("Compose started a different image")
        switch(policy, CANDIDATE_PORT, PRIMARY_PORT)
        stop_clean(CANDIDATE)
        final_report(image)
    except Exception:
        print("rollout_failed=true serving_instance_preserved=true", flush=True)
        raise
    finally:
        if temporary_policy:
            temporary_policy.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="yym68686/uni-api:latest")
    parser.add_argument("--ready-timeout", type=int, default=300)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--expected-commit", default="")
    args = parser.parse_args()
    try:
        if args.check_only:
            check_only(args.image, args.expected_commit)
        else:
            os.umask(0o077)
            with open("/root/uni-api-rollout.lock", "a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RolloutError("another DigitalOcean rollout is already running")
                rollout(args)
    except (OSError, RolloutError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="digitalocean")
    parser.add_argument("--image", default="yym68686/uni-api:latest")
    parser.add_argument("--ready-timeout", type=int, default=300)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--ssh-option", action="append", default=[],
                        help="SSH -o option; repeat for a connection-specific override")
    args = parser.parse_args()

    remote_command = [
        "python3",
        "-",
        "--image",
        args.image,
        "--ready-timeout",
        str(args.ready_timeout),
    ]
    remote_command += ["--expected-commit", args.expected_commit]
    if args.check_only:
        remote_command.append("--check-only")
    command = ["ssh", "-T"]
    for option in args.ssh_option + ["BatchMode=yes", "ConnectTimeout=15", "ServerAliveInterval=15", "ServerAliveCountMax=3"]:
        command += ["-o", option]
    command += [args.remote, shlex.join(remote_command)]
    result = subprocess.run(command, input=REMOTE_SCRIPT, text=True, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
