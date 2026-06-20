import pytest
from types import SimpleNamespace

import main
import utils
from uni_api.config.compiler import compile_runtime_config
from uni_api.config.env import expand_config_environment
from uni_api.config.schema import validate_config_data


def _config():
    return {
        "providers": [
            {
                "provider": "openai-a",
                "base_url": "https://api.example.com/v1/chat/completions",
                "api": "upstream-key",
                "model": [{"gpt-4.1": "gpt-alias"}, "text-embedding-3-small"],
                "exclude_endpoints": ["v1/responses"],
                "preferences": {
                    "exclude_endpoints": ["/v1/images/generations"],
                    "model_timeout": {"gpt-4.1": 30},
                    "keepalive_interval": {"gpt-4.1": 10},
                },
            }
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": [{"openai-a/gpt-alias": 2}, "openai-a/text-embedding-3-small"],
                "weights": {"openai-a/gpt-alias": 2},
            }
        ],
        "preferences": {
            "model_timeout": {"default": 100},
            "keepalive_interval": {"default": 99999},
        },
    }


def test_validate_config_data_rejects_invalid_provider_shape():
    with pytest.raises(Exception):
        validate_config_data({"providers": [{"base_url": "https://example.com"}], "api_keys": []})


def test_validate_config_data_error_message_identifies_missing_provider_field():
    with pytest.raises(Exception) as exc_info:
        validate_config_data({"providers": [{"base_url": "https://example.com"}], "api_keys": []})

    message = str(exc_info.value)
    assert "providers.0.provider" in message
    assert "Field required" in message


def test_compile_runtime_config_accepts_empty_config():
    config = validate_config_data({})
    runtime = compile_runtime_config(config, [], models_list={})

    assert runtime.api_key_by_token == {}
    assert runtime.provider_by_name == {}
    assert runtime.api_key_model_response_cache == {}


def test_compile_runtime_config_builds_hot_path_indexes():
    config = validate_config_data(_config())
    runtime = compile_runtime_config(
        config,
        ["sk-test"],
        models_list={"sk-test": ["gpt-alias", "text-embedding-3-small"]},
        default_timeout=100,
    )

    assert runtime.api_key_index_by_token == {"sk-test": 0}
    assert runtime.api_list == ("sk-test",)
    assert runtime.api_key_by_token["sk-test"]["api"] == "sk-test"
    assert runtime.api_key_model_rules_by_index == (
        ({"openai-a/gpt-alias": 2}, "openai-a/text-embedding-3-small"),
    )
    assert runtime.api_key_preferences_by_index == ({},)
    assert runtime.api_key_roles_by_index == ("sk-test",)
    assert runtime.api_key_weights_by_index == ({"openai-a/gpt-alias": 2},)
    assert runtime.provider_by_name["openai-a"]["provider"] == "openai-a"
    assert runtime.models_by_provider["openai-a"] == ("gpt-alias", "text-embedding-3-small")
    assert "openai-a" in [provider["provider"] for provider in runtime.providers_by_model["gpt-alias"]]
    assert runtime.api_key_allowed_models["sk-test"] == ["gpt-alias", "text-embedding-3-small"]
    assert [item["id"] for item in runtime.api_key_model_response_cache["sk-test"]] == [
        "gpt-alias",
        "text-embedding-3-small",
    ]
    assert runtime.endpoint_exclusions_by_provider["openai-a"] == frozenset(
        {"/v1/responses", "/v1/images/generations"}
    )
    assert runtime.weights_by_api_key_model["sk-test"] == {"openai-a/gpt-alias": 2}
    assert runtime.provider_preferences["openai-a"]["model_timeout"] == {"gpt-4.1": 30}
    assert runtime.model_timeout_index["global"]["default"] == 100
    assert runtime.model_timeout_index["openai-a"]["gpt-4.1"] == 30
    assert runtime.keepalive_interval_index["openai-a"]["gpt-4.1"] == 10


def test_timeout_policy_matches_most_specific_provider_rule():
    policy = main.init_timeout_policy(
        {
            "preferences": {
                "timeout_policy": {
                    "default": {"first_byte": 30},
                    "rules": [
                        {
                            "match": {"endpoint": "/v1/responses/compact"},
                            "timeout": {"first_byte": 90},
                        }
                    ],
                }
            },
            "providers": [
                {
                    "provider": "openai-a",
                    "preferences": {
                        "timeout_policy": {
                            "rules": [
                                {
                                    "match": {
                                        "endpoint": "/v1/responses/compact",
                                        "stream": False,
                                        "model": "gpt-5.5",
                                    },
                                    "timeout": {"first_byte": 120},
                                }
                            ]
                        }
                    },
                }
            ],
        }
    )

    resolved = main.apply_timeout_policy(
        base_timeout=20,
        timeout_policy=policy,
        provider_name="openai-a",
        endpoint="/v1/responses/compact",
        method="POST",
        stream=False,
        engine="codex",
        original_model="gpt-5.5",
        request_model="gpt-5.5",
    )

    assert resolved["timeout_value"] == 120
    assert resolved["timeout_adjusted_from"] == 20
    assert resolved["timeout_policy_sources"] == [
        "global.default",
        "global.rules[0]",
        "provider.rules[0]",
    ]


def test_compile_runtime_config_resolves_wildcard_and_nested_api_keys():
    config = validate_config_data(
        {
            "providers": [
                {
                    "provider": "openai-a",
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "api": "upstream-key",
                    "model": ["gpt-4.1", "gpt-4.1-mini"],
                }
            ],
            "api_keys": [
                {"api": "sk-parent", "model": ["sk-child/*"]},
                {"api": "sk-child", "model": ["openai-a/*"]},
            ],
        }
    )

    runtime = compile_runtime_config(config, ["sk-parent", "sk-child"])

    assert runtime.api_key_allowed_models["sk-child"] == ["gpt-4.1", "gpt-4.1-mini"]
    assert runtime.api_key_allowed_models["sk-parent"] == ["gpt-4.1", "gpt-4.1-mini"]


def test_expand_config_environment_recurses_through_config_values():
    expanded = expand_config_environment(
        {
            "providers": [
                {
                    "provider": "${PROVIDER_NAME:-openai-a}",
                    "api": "${UPSTREAM_API_KEY}",
                    "headers": {"X-App": "prefix-${APP_NAME:-uni-api}"},
                }
            ],
            "api_keys": [{"api": "${CLIENT_API:-sk-default}"}],
        },
        env={"UPSTREAM_API_KEY": "upstream-key"},
    )

    assert expanded["providers"][0]["provider"] == "openai-a"
    assert expanded["providers"][0]["api"] == "upstream-key"
    assert expanded["providers"][0]["headers"]["X-App"] == "prefix-uni-api"
    assert expanded["api_keys"][0]["api"] == "sk-default"


def test_compile_runtime_config_handles_recursive_api_key_dependencies():
    config = validate_config_data(
        {
            "providers": [
                {
                    "provider": "openai-a",
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "api": "upstream-key",
                    "model": ["gpt-4.1"],
                }
            ],
            "api_keys": [
                {"api": "sk-a", "model": ["sk-b/*"]},
                {"api": "sk-b", "model": ["sk-a/*"]},
            ],
        }
    )

    runtime = compile_runtime_config(config, ["sk-a", "sk-b"])

    assert runtime.api_key_allowed_models["sk-a"] == []
    assert runtime.api_key_allowed_models["sk-b"] == []


async def test_load_config_validates_once(monkeypatch, tmp_path):
    config_path = tmp_path / "api.yaml"
    config_path.write_text(
        """
providers:
  - provider: openai-a
    base_url: https://api.example.com/v1/chat/completions
    api: upstream-key
    model:
      - gpt-4.1
api_keys:
  - api: sk-test
    model:
      - all
""",
        encoding="utf-8",
    )

    calls = {"count": 0}

    def fake_validate(config_data):
        calls["count"] += 1
        return config_data

    monkeypatch.setattr(utils, "API_YAML_PATH", str(config_path))
    monkeypatch.setattr(utils, "validate_config_data", fake_validate)

    config, api_keys_db, api_list = await utils.load_config()

    assert config["providers"][0]["provider"] == "openai-a"
    assert api_keys_db[0]["api"] == "sk-test"
    assert api_list == ["sk-test"]
    assert calls["count"] == 1


async def test_update_config_validates_once(monkeypatch):
    calls = {"count": 0}

    def fake_validate(config_data):
        calls["count"] += 1
        return config_data

    monkeypatch.setattr(utils, "validate_config_data", fake_validate)

    await utils.update_config(
        {
            "providers": [
                {
                    "provider": "openai-a",
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "api": "upstream-key",
                    "model": ["gpt-4.1"],
                }
            ],
            "api_keys": [{"api": "sk-test", "model": ["all"]}],
        }
    )

    assert calls["count"] == 1


async def test_update_config_expands_environment_before_validation(monkeypatch):
    monkeypatch.setenv("CONFIG_PROVIDER", "openai-a")
    monkeypatch.setenv("CONFIG_API", "upstream-key")

    config, _, api_list = await utils.update_config(
        {
            "providers": [
                {
                    "provider": "${CONFIG_PROVIDER}",
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "api": "${CONFIG_API}",
                    "model": ["gpt-4.1"],
                }
            ],
            "api_keys": [{"api": "${CLIENT_API:-sk-test}", "model": ["all"]}],
        }
    )

    assert config["providers"][0]["provider"] == "openai-a"
    assert config["providers"][0]["api"] == "upstream-key"
    assert api_list == ["sk-test"]


async def test_refresh_runtime_state_replaces_snapshot_atomically(monkeypatch):
    app = SimpleNamespace(
        state=SimpleNamespace(
            config={
                "providers": [
                    {
                        "provider": "openai-a",
                        "base_url": "https://api.example.com/v1/chat/completions",
                        "api": "upstream-key",
                        "model": ["gpt-4.1"],
                    }
                ],
                "api_keys": [{"api": "sk-test", "model": ["all"]}],
                "preferences": {
                    "timeout_policy": {
                        "rules": [
                            {
                                "match": {"endpoint": "/v1/responses", "stream": True},
                                "timeout": {"first_byte": 60},
                            }
                        ]
                    }
                },
            },
            api_keys_db=[{"api": "sk-test", "role": "admin"}],
            api_list=["sk-test"],
        )
    )

    await main.refresh_runtime_state(app)
    first_snapshot = app.state.runtime_snapshot

    assert app.state.runtime_config is first_snapshot.runtime_config
    assert app.state.admin_api_key == ["sk-test"]
    assert app.state.timeout_policy is first_snapshot.timeout_policy
    assert first_snapshot.timeout_policy["global"]["rules"][0]["timeout"]["first_byte"] == 60

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(main, "compile_runtime_config", fail_compile)
    app.state.config = {"providers": [], "api_keys": [{"api": "sk-test", "model": ["all"]}]}

    with pytest.raises(RuntimeError):
        await main.refresh_runtime_state(app)

    assert app.state.runtime_snapshot is first_snapshot
    assert app.state.runtime_config is first_snapshot.runtime_config
