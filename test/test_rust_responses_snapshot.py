from __future__ import annotations

import json
import stat

from uni_api.rust_responses_snapshot import (
    build_rust_responses_snapshot,
    publish_rust_responses_snapshot,
)


def _config():
    return {
        "preferences": {"cooldown_period": 30},
        "providers": [
            {
                "provider": "primary",
                "base_url": "https://example.com/v1/responses",
                "engine": "codex",
                "api": ["provider-key-1", "provider-key-2"],
                "model": [{"gpt-upstream": "gpt-public"}],
                "preferences": {
                    "post_body_parameter_overrides": {"store": False},
                    "exclude_request_types": ["future-type"],
                },
                "only_request_types": ["compaction"],
                "exclude_request_rules": [
                    {
                        "match": {
                            "request_model": "gpt-public",
                            "upstream_model": "gpt-upstream",
                            "reasoning_effort": ["max"],
                        },
                        "reason": "unsupported_reasoning_effort",
                    }
                ],
            }
        ],
        "api_keys": [
            {
                "api": "client-key",
                "model": ["primary/*"],
                "preferences": {
                    "SCHEDULING_ALGORITHM": "fixed_priority",
                    "AUTO_RETRY": True,
                },
            }
        ],
    }


def test_snapshot_contains_compiled_models_and_hot_path_configuration():
    snapshot = build_rust_responses_snapshot(
        _config(),
        ["client-key"],
        database_disabled=True,
    )

    assert snapshot["schema_version"] == 1
    assert len(snapshot["revision"]) == 64
    assert snapshot["api_keys"][0]["token"] == "client-key"
    assert snapshot["providers"][0]["models"] == {
        "gpt-public": "gpt-upstream"
    }
    assert snapshot["providers"][0]["api"] == [
        "provider-key-1",
        "provider-key-2",
    ]
    assert snapshot["providers"][0]["only_request_types"] == ["compaction"]
    assert snapshot["providers"][0]["exclude_request_types"] == ["future-type"]
    assert snapshot["providers"][0]["exclude_request_rules"][0]["match"] == {
        "request_model": "gpt-public",
        "upstream_model": "gpt-upstream",
        "reasoning_effort": ["max"],
    }
    assert build_rust_responses_snapshot(
        _config(),
        ["client-key"],
        database_disabled=True,
    )["revision"] == snapshot["revision"]


def test_snapshot_publish_is_atomic_private_and_valid_json(tmp_path):
    path = tmp_path / "responses.json"
    revision = publish_rust_responses_snapshot(
        _config(),
        ["client-key"],
        database_disabled=True,
        path=path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["revision"] == revision
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))
