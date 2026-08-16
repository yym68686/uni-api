from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi.encoders import jsonable_encoder

from core.utils import get_model_dict
from uni_api.routing.request_rules import provider_exclude_request_rules
from uni_api.routing.request_types import provider_request_type_values


RUST_RESPONSES_SNAPSHOT_SCHEMA_VERSION = 1
RUST_RESPONSES_SNAPSHOT_PATH = os.getenv(
    "RUST_RESPONSES_CONFIG_SNAPSHOT_PATH",
    "/tmp/uni-api-rust-responses-config-v1.json",
)


def build_rust_responses_snapshot(
    config: dict[str, Any],
    api_list: list[str],
    *,
    database_disabled: bool,
) -> dict[str, Any]:
    """Build the immutable configuration Rust needs on the Responses hot path."""

    api_keys = list(config.get("api_keys", []) or [])
    snapshot_api_keys: list[dict[str, Any]] = []
    for index, token in enumerate(api_list):
        item = api_keys[index] if index < len(api_keys) else {}
        preferences = dict((item or {}).get("preferences") or {})
        snapshot_api_keys.append(
            {
                "token": str(token),
                "model_rules": list((item or {}).get("model") or []),
                "role": str((item or {}).get("role") or str(token)[:8] or "None"),
                "weights": dict((item or {}).get("weights") or {}),
                "preferences": preferences,
                # Paid-key state changes independently of config. Rust must
                # fall back unless database-backed accounting is disabled.
                "native_paid_state_safe": bool(database_disabled),
            }
        )

    snapshot_providers: list[dict[str, Any]] = []
    for provider in config.get("providers", []) or []:
        provider_name = str(provider.get("provider") or "").strip()
        if not provider_name:
            continue
        model_dict = provider.get("_model_dict_cache")
        if not isinstance(model_dict, dict):
            model_dict = get_model_dict(provider)
        provider_preferences = dict(provider.get("preferences") or {})
        if (
            "max_request_body_bytes" not in provider_preferences
            and provider.get("max_request_body_bytes") is not None
        ):
            provider_preferences["max_request_body_bytes"] = provider.get(
                "max_request_body_bytes"
            )
        excluded_endpoints: list[Any] = []
        for configured in (
            provider.get("exclude_endpoints"),
            provider_preferences.get("exclude_endpoints"),
        ):
            if isinstance(configured, (list, tuple, set)):
                excluded_endpoints.extend(configured)
            elif configured:
                excluded_endpoints.append(configured)
        snapshot_providers.append(
            {
                "name": provider_name,
                "base_url": str(provider.get("base_url") or ""),
                "engine": provider.get("engine"),
                "api": provider.get("api"),
                "models": {
                    str(request_model): str(upstream_model)
                    for request_model, upstream_model in model_dict.items()
                },
                "preferences": provider_preferences,
                "exclude_endpoints": excluded_endpoints,
                "only_request_types": provider_request_type_values(
                    provider, "only_request_types"
                ),
                "exclude_request_types": provider_request_type_values(
                    provider, "exclude_request_types"
                ),
                "exclude_request_rules": list(
                    provider_exclude_request_rules(provider)
                ),
            }
        )

    payload = {
        "schema_version": RUST_RESPONSES_SNAPSHOT_SCHEMA_VERSION,
        "generated_unix_ms": int(time.time() * 1000),
        "database_disabled": bool(database_disabled),
        "preferences": dict(config.get("preferences") or {}),
        "api_keys": snapshot_api_keys,
        "providers": snapshot_providers,
    }
    revision_payload = dict(payload)
    revision_payload.pop("generated_unix_ms", None)
    canonical = json.dumps(
        jsonable_encoder(revision_payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload["revision"] = hashlib.sha256(canonical).hexdigest()
    return jsonable_encoder(payload)


def publish_rust_responses_snapshot(
    config: dict[str, Any],
    api_list: list[str],
    *,
    database_disabled: bool,
    path: str | os.PathLike[str] = RUST_RESPONSES_SNAPSHOT_PATH,
) -> str:
    """Atomically publish a mode-0600 snapshot and return its revision."""

    snapshot = build_rust_responses_snapshot(
        config,
        api_list,
        database_disabled=database_disabled,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(
                snapshot,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target)
        os.chmod(target, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return str(snapshot["revision"])
