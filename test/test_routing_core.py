import asyncio
import pathlib
from collections import defaultdict
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from upstream import UpstreamRunner
from uni_api.routing.core import (
    ProviderAttempt,
    RoutingPlan,
    build_api_key_models_map,
    compute_retry_count,
    compute_start_index,
    get_right_order_providers,
    provider_api_circular_list,
)


class _ProviderKeys:
    def __init__(self, *, count=1, all_rate_limited=False, tpr_exceeded=False):
        self._count = count
        self._all_rate_limited = all_rate_limited
        self._tpr_exceeded = tpr_exceeded

    def get_items_count(self):
        return self._count

    async def is_all_rate_limited(self, model):
        _ = model
        return self._all_rate_limited

    async def is_tpr_exceeded(self, model, tokens=0):
        _ = model, tokens
        return self._tpr_exceeded

    async def next(self, model):
        _ = model
        return "provider-key"


def _routing_config():
    return {
        "providers": [
            {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": "key-a",
                "model": ["gpt-4.1"],
            },
            {
                "provider": "provider-b",
                "base_url": "https://provider-b.example/v1/chat/completions",
                "api": "key-b",
                "model": ["gpt-4.1"],
            },
        ],
        "api_keys": [{"api": "sk-test", "model": ["gpt-4.1"]}],
    }


def test_get_right_order_providers_raises_404_for_missing_provider():
    config = {
        "providers": [],
        "api_keys": [{"api": "sk-test", "model": ["gpt-4.1"]}],
    }

    async def run():
        with pytest.raises(HTTPException) as exc_info:
            await get_right_order_providers(
                "gpt-4.1",
                config,
                0,
                "fixed_priority",
                ["sk-test"],
                {"sk-test": ["gpt-4.1"]},
            )
        assert exc_info.value.status_code == 404

    asyncio.run(run())


def test_get_right_order_providers_raises_413_when_tpr_exceeds_all_providers(monkeypatch):
    config = _routing_config()
    monkeypatch.setitem(provider_api_circular_list, "provider-a", _ProviderKeys(tpr_exceeded=True))
    monkeypatch.setitem(provider_api_circular_list, "provider-b", _ProviderKeys(tpr_exceeded=True))

    async def run():
        with pytest.raises(HTTPException) as exc_info:
            await get_right_order_providers(
                "gpt-4.1",
                config,
                0,
                "fixed_priority",
                ["sk-test"],
                {"sk-test": ["gpt-4.1"]},
                request_total_tokens=1000,
            )
        assert exc_info.value.status_code == 413

    asyncio.run(run())


def test_get_right_order_providers_skips_provider_when_request_body_exceeds_limit():
    config = _routing_config()
    config["providers"][0]["preferences"] = {"max_request_body_bytes": "20MB"}
    config["providers"][1]["preferences"] = {"max_request_body_bytes": "25MiB"}

    async def run():
        providers = await get_right_order_providers(
            "gpt-4.1",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-4.1"]},
            request_body_bytes=20_000_001,
        )
        assert [provider["provider"] for provider in providers] == ["provider-b"]

        providers = await get_right_order_providers(
            "gpt-4.1",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-4.1"]},
            request_body_bytes=20_000_000,
        )
        assert [provider["provider"] for provider in providers] == ["provider-a", "provider-b"]

    asyncio.run(run())


def test_get_right_order_providers_raises_413_when_request_body_exceeds_all_limits():
    config = _routing_config()
    for provider in config["providers"]:
        provider["preferences"] = {"max_request_body_bytes": 100}

    async def run():
        with pytest.raises(HTTPException) as exc_info:
            await get_right_order_providers(
                "gpt-4.1",
                config,
                0,
                "fixed_priority",
                ["sk-test"],
                {"sk-test": ["gpt-4.1"]},
                request_body_bytes=101,
            )
        assert exc_info.value.status_code == 413
        assert "request body" in exc_info.value.detail

    asyncio.run(run())


def test_routing_plan_returns_429_when_all_provider_keys_are_limited(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(provider_api_circular_list, provider_name, _ProviderKeys(all_rate_limited=True))

    async def resolver(request_model_name, config, api_index, scheduling_algorithm, api_list, models_list, **kwargs):
        _ = config, api_index, scheduling_algorithm, api_list, models_list, kwargs
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    async def run():
        app = SimpleNamespace(
            state=SimpleNamespace(
                config={"api_keys": [{"api": "sk-test", "model": ["gpt-4.1"]}]},
                api_list=["sk-test"],
                models_list={"sk-test": ["gpt-4.1"]},
                channel_manager=None,
            )
        )
        plan = await RoutingPlan.create(app, "gpt-4.1", 0, {}, {}, provider_resolver=resolver)
        assert await plan.next_provider() is None
        assert plan.status_code == 429

    asyncio.run(run())


def test_routing_plan_passes_request_body_bytes_to_provider_resolver(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(provider_api_circular_list, provider_name, _ProviderKeys())
    seen = {}

    async def resolver(request_model_name, config, api_index, scheduling_algorithm, api_list, models_list, **kwargs):
        _ = config, api_index, scheduling_algorithm, api_list, models_list
        seen["request_body_bytes"] = kwargs.get("request_body_bytes")
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    async def run():
        app = SimpleNamespace(
            state=SimpleNamespace(
                config={"api_keys": [{"api": "sk-test", "model": ["gpt-4.1"]}]},
                api_list=["sk-test"],
                models_list={"sk-test": ["gpt-4.1"]},
                channel_manager=None,
            )
        )
        plan = await RoutingPlan.create(
            app,
            "gpt-4.1",
            0,
            {},
            {},
            request_body_bytes=123,
            provider_resolver=resolver,
        )
        assert plan.request_body_bytes == 123
        assert seen["request_body_bytes"] == 123

    asyncio.run(run())


def test_routing_plan_and_provider_attempt_are_frozen_lightweight_objects(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(provider_api_circular_list, provider_name, _ProviderKeys())

    async def resolver(request_model_name, config, api_index, scheduling_algorithm, api_list, models_list, **kwargs):
        _ = config, api_index, scheduling_algorithm, api_list, models_list, kwargs
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    async def run():
        app = SimpleNamespace(
            state=SimpleNamespace(
                config={"api_keys": [{"api": "sk-test", "model": ["gpt-4.1"]}]},
                api_list=["sk-test"],
                models_list={"sk-test": ["gpt-4.1"]},
                channel_manager=None,
            )
        )
        plan = await RoutingPlan.create(app, "gpt-4.1", 0, {}, {}, provider_resolver=resolver)
        with pytest.raises(FrozenInstanceError):
            plan.request_model_name = "other"

        attempt = await plan.next_provider()
        assert isinstance(attempt, ProviderAttempt)
        with pytest.raises(FrozenInstanceError):
            attempt.provider_name = "other"

        plan.record_failure(502, "bad upstream")
        assert plan.status_code == 502
        assert plan.error_message == "bad upstream"

    asyncio.run(run())


def test_get_right_order_providers_applies_channel_cooldown_filter():
    class ChannelManager:
        cooldown_period = 300

        async def get_available_providers(self, providers):
            return [provider for provider in providers if provider["provider"] == "provider-b"]

    async def run():
        providers = await get_right_order_providers(
            "gpt-4.1",
            _routing_config(),
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-4.1"]},
            channel_manager=ChannelManager(),
        )
        assert [provider["provider"] for provider in providers] == ["provider-b"]

    asyncio.run(run())


def test_compute_start_index_implements_fixed_priority_and_round_robin():
    async def run():
        locks = defaultdict(asyncio.Lock)
        last_indices = defaultdict(lambda: -1)

        assert await compute_start_index(last_indices, locks, "gpt-4.1", "fixed_priority", 2) == 0
        assert last_indices["gpt-4.1"] == -1

        assert await compute_start_index(last_indices, locks, "gpt-4.1", "round_robin", 2) == 0
        assert await compute_start_index(last_indices, locks, "gpt-4.1", "round_robin", 2) == 1

    asyncio.run(run())


def test_compute_retry_count_uses_provider_key_counts(monkeypatch):
    monkeypatch.setitem(provider_api_circular_list, "provider-a", _ProviderKeys(count=3))
    monkeypatch.setitem(provider_api_circular_list, "provider-b", _ProviderKeys(count=4))

    assert compute_retry_count([{"provider": "provider-a"}, {"provider": "provider-b"}]) == 10
    assert compute_retry_count([{"provider": "provider-a"}]) == 3


def test_provider_package_does_not_depend_on_global_app_state():
    provider_root = pathlib.Path(__file__).resolve().parents[1] / "uni_api" / "providers"
    offenders = []
    for path in provider_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "app.state" in source or "main.app.state" in source:
            offenders.append(path.relative_to(provider_root).as_posix())

    assert offenders == []


def test_upstream_runner_uses_plan_api_snapshot_for_provider_key_selection():
    plan = SimpleNamespace(api_list=("sk-snapshot",), app=SimpleNamespace(state=SimpleNamespace(config={"api_keys": []})))
    runner = UpstreamRunner(plan)

    assert runner._runtime_api_list() == ["sk-snapshot"]


def test_routing_plan_create_prefers_runtime_config_snapshot(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(provider_api_circular_list, provider_name, _ProviderKeys())

    async def resolver(request_model_name, config, api_index, scheduling_algorithm, api_list, models_list, **kwargs):
        _ = config, api_index, scheduling_algorithm, api_list, models_list, kwargs
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    runtime_config = SimpleNamespace(
        api_list=("sk-test",),
        api_key_allowed_models={"sk-test": ["gpt-4.1"]},
        routing_index=None,
        api_key_preferences_by_index=({"AUTO_RETRY": False},),
        api_key_roles_by_index=("sk-test",),
        api_key_model_rules_by_index=(("gpt-4.1",),),
        api_key_weights_by_index=({},),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            config={"api_keys": [{"api": "sk-test", "model": ["gpt-4.1"]}]},
            runtime_config=runtime_config,
            runtime_config_source_id=None,
            api_list=["sk-test"],
            models_list={"sk-test": ["gpt-4.1"]},
            channel_manager=None,
        )
    )
    app.state.runtime_config_source_id = id(app.state.config)

    async def run():
        plan = await RoutingPlan.create(app, "gpt-4.1", 0, {}, {}, provider_resolver=resolver)
        assert plan.api_list == ("sk-test",)
        assert plan.api_key_model_rules == (("gpt-4.1",),)
        assert plan.auto_retry is False

    asyncio.run(run())


def test_get_right_order_providers_combines_weights_and_endpoint_exclusion():
    config = {
        "providers": [
            {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/v1/responses",
                "api": "key-a",
                "model": ["gpt-5.4"],
            },
            {
                "provider": "provider-b",
                "base_url": "https://provider-b.example/v1/responses",
                "api": "key-b",
                "model": ["gpt-5.4"],
                "exclude_endpoints": ["v1/responses/compact"],
            },
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
                "weights": {"provider-a/gpt-5.4": 2, "provider-b/gpt-5.4": 1},
            }
        ],
    }

    async def run():
        providers = await get_right_order_providers(
            "gpt-5.4",
            config,
            0,
            "weighted_round_robin",
            ["sk-test"],
            {"sk-test": ["gpt-5.4"]},
            endpoint="/v1/responses/compact",
        )
        assert [provider["provider"] for provider in providers] == ["provider-a"]

    asyncio.run(run())


def test_get_right_order_providers_smart_round_robin_uses_weighted_provider_order():
    config = {
        "providers": [
            {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": "key-a",
                "model": ["gpt-4.1"],
            },
            {
                "provider": "provider-b",
                "base_url": "https://provider-b.example/v1/chat/completions",
                "api": "key-b",
                "model": ["gpt-4.1"],
            },
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-4.1"],
                "weights": {"provider-a/gpt-4.1": 2, "provider-b/gpt-4.1": 1},
            }
        ],
    }

    async def run():
        providers = await get_right_order_providers(
            "gpt-4.1",
            config,
            0,
            "smart_round_robin",
            ["sk-test"],
            {"sk-test": ["gpt-4.1"]},
        )
        assert [provider["provider"] for provider in providers] == [
            "provider-a",
            "provider-b",
            "provider-a",
        ]

    asyncio.run(run())


def test_get_right_order_providers_resolves_nested_api_key_route():
    config = {
        "providers": [
            {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/v1/chat/completions",
                "api": "key-a",
                "model": ["gpt-4.1"],
            }
        ],
        "api_keys": [
            {"api": "sk-parent", "model": ["sk-child/*"]},
            {"api": "sk-child", "model": ["provider-a/*"]},
        ],
    }
    api_list = ["sk-parent", "sk-child"]
    models_list = build_api_key_models_map(config, api_list)

    async def run():
        providers = await get_right_order_providers(
            "gpt-4.1",
            config,
            0,
            "fixed_priority",
            api_list,
            models_list,
        )
        assert [provider["provider"] for provider in providers] == ["sk-child"]
        assert providers[0]["base_url"] == "http://127.0.0.1:8000/v1/chat/completions"

    asyncio.run(run())
