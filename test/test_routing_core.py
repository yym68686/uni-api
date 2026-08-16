import asyncio
import gc
import pathlib
import weakref
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


def test_routing_plan_refresh_preserves_original_order_budget_and_cursor(
    monkeypatch,
):
    request_model_name = "gpt-5.4"
    provider_names = (
        "provider-a",
        "provider-b",
        "provider-c",
        "provider-d",
        "provider-e",
        "provider-f",
        "provider-g",
    )
    providers = {}
    for provider_name in provider_names:
        monkeypatch.setitem(
            provider_api_circular_list,
            provider_name,
            _ProviderKeys(),
        )
        providers[provider_name] = {
            "provider": provider_name,
            "_model_dict_cache": {
                request_model_name: request_model_name,
            },
            "base_url": (
                f"https://{provider_name}.example/v1/responses"
            ),
            "api": [f"{provider_name}-key"],
            "preferences": {},
        }

    resolver_results = [
        [
            providers["provider-a"],
            providers["provider-b"],
            providers["provider-c"],
            providers["provider-d"],
            providers["provider-e"],
        ],
        [
            providers["provider-e"],
            providers["provider-d"],
            providers["provider-c"],
            providers["provider-b"],
        ],
        [
            providers["provider-g"],
            providers["provider-f"],
            providers["provider-e"],
            providers["provider-d"],
            providers["provider-c"],
            providers["provider-a"],
            providers["provider-b"],
        ],
    ]

    async def resolver(*_args, **_kwargs):
        return resolver_results.pop(0)

    app = SimpleNamespace(
        state=SimpleNamespace(
            config={
                "api_keys": [
                    {
                        "api": "sk-test",
                        "model": [request_model_name],
                    }
                ]
            },
            api_list=["sk-test"],
            models_list={"sk-test": [request_model_name]},
            channel_manager=None,
        )
    )

    async def run():
        plan = await RoutingPlan.create(
            app,
            request_model_name,
            0,
            {},
            {},
            endpoint="/v1/responses",
            provider_resolver=resolver,
        )

        attempts = [(await plan.next_provider()).provider_name]
        await plan.refresh_matching_providers()
        assert plan.retry_count == 10
        assert plan.index == 1
        assert [
            provider["provider"] for provider in plan.matching_providers
        ] == [
            "provider-a",
            "provider-b",
            "provider-c",
            "provider-d",
            "provider-e",
        ]

        attempts.append((await plan.next_provider()).provider_name)
        await plan.refresh_matching_providers()
        assert plan.retry_count == 10
        assert plan.index == 2
        assert plan.num_matching_providers == 5

        while True:
            attempt = await plan.next_provider()
            if attempt is None:
                break
            attempts.append(attempt.provider_name)

        assert attempts == [
            "provider-a",
            "provider-b",
            "provider-c",
            "provider-d",
            "provider-e",
            "provider-a",
            "provider-b",
            "provider-c",
            "provider-d",
            "provider-e",
            "provider-a",
            "provider-b",
            "provider-c",
            "provider-d",
            "provider-e",
        ]
        assert "provider-f" not in attempts
        assert "provider-g" not in attempts
        assert plan.index == 15

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


def test_get_right_order_providers_applies_route_circuit_to_single_provider():
    class ChannelManager:
        cooldown_period = 0
        has_route_circuit = True

        async def get_available_providers(self, providers):
            assert len(providers) == 1
            return []

    config = _routing_config()
    config["providers"] = config["providers"][:1]

    async def run():
        with pytest.raises(HTTPException) as exc_info:
            await get_right_order_providers(
                "gpt-4.1",
                config,
                0,
                "fixed_priority",
                ["sk-test"],
                {"sk-test": ["gpt-4.1"]},
                channel_manager=ChannelManager(),
            )
        assert exc_info.value.status_code == 503

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


async def _routing_plan_with_retry_count(monkeypatch, retry_count):
    providers = []
    for provider_name in ("provider-a", "provider-b"):
        monkeypatch.setitem(
            provider_api_circular_list,
            provider_name,
            _ProviderKeys(),
        )
        providers.append(
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-4.1": "gpt-4.1"},
                "base_url": f"https://{provider_name}.example/v1/chat/completions",
                "api": [f"{provider_name}-key"],
                "preferences": {},
            }
        )

    async def resolver(*_args, **_kwargs):
        return providers

    app = SimpleNamespace(
        state=SimpleNamespace(
            config={
                "api_keys": [
                    {
                        "api": "sk-test",
                        "model": ["gpt-4.1"],
                        "preferences": {"AUTO_RETRY": True},
                    }
                ]
            },
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
        provider_resolver=resolver,
    )
    plan._state.retry_count = retry_count
    return plan


@pytest.mark.parametrize(
    ("retry_count", "expected_attempts"),
    [(0, 2), (10, 12)],
)
def test_routing_plan_attempt_boundary_is_providers_plus_retries(
    monkeypatch,
    retry_count,
    expected_attempts,
):
    async def run():
        plan = await _routing_plan_with_retry_count(monkeypatch, retry_count)
        attempts = []
        while True:
            attempt = await plan.next_provider()
            if attempt is None:
                break
            attempts.append(attempt)
        assert len(attempts) == expected_attempts
        assert plan.index == expected_attempts

    asyncio.run(run())


def test_upstream_runner_stops_after_success_at_exact_attempt_boundary(
    monkeypatch,
):
    async def run():
        plan = await _routing_plan_with_retry_count(monkeypatch, 1)
        calls = []
        routing_attempt_ids = []
        current_info = {}

        async def execute_attempt(attempt):
            calls.append(attempt.provider_name)
            routing_attempt_ids.append(attempt.routing_attempt_id)
            if len(calls) < 3:
                raise RuntimeError("retryable upstream failure")
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(
            "upstream.should_retry_provider",
            lambda *_args, **_kwargs: True,
        )
        response = await UpstreamRunner(
            plan,
            observability_context=current_info,
        ).run(execute_attempt)

        assert response.status_code == 200
        assert calls == ["provider-a", "provider-b", "provider-a"]
        assert len(set(routing_attempt_ids)) == 3
        assert [
            item["routing_attempt_id"]
            for item in current_info["routing_attempts"]
        ] == routing_attempt_ids
        assert current_info["attempt_count"] == 3
        assert current_info["retry_decision_count"] == 2
        assert current_info["retry_transition_count"] == 2
        assert [
            item.get("retry_transition_to_index")
            for item in current_info["routing_attempts"]
        ] == [2, 3, None]
        assert current_info["routing_attempts"][-1]["outcome"] == "succeeded"

    asyncio.run(run())


def test_upstream_runner_releases_failed_task_tracebacks_with_gc_disabled(
    monkeypatch,
):
    class RetainedPayload:
        def __init__(self):
            self.data = bytearray(1024 * 1024)

    async def run():
        attempts = [
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name=f"provider-{index}",
                original_model="gpt-4.1",
            )
            for index in range(6)
        ]

        class Plan:
            auto_retry = True
            api_list = ()
            request_model_name = "gpt-4.1"
            status_code = 500
            error_message = "failed"

            async def next_provider(self):
                return attempts.pop(0) if attempts else None

            def record_failure(self, status_code, error_message):
                self.status_code = status_code
                self.error_message = error_message

        retained = []
        calls = []

        async def fail_after_task_boundary(payload):
            await asyncio.sleep(0)
            raise RuntimeError(f"failed with {len(payload.data)} bytes")

        async def execute_attempt(attempt):
            calls.append(attempt.provider_name)
            payload = RetainedPayload()
            retained.append(weakref.ref(payload))
            task = asyncio.create_task(fail_after_task_boundary(payload))
            await asyncio.wait({task})
            return task.result()

        monkeypatch.setattr(
            "upstream.should_retry_provider",
            lambda *_args, **_kwargs: True,
        )
        response = await UpstreamRunner(Plan()).run(execute_attempt)

        assert response.status_code == 500
        assert calls == [f"provider-{index}" for index in range(6)]
        assert all(reference() is None for reference in retained)

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(run())
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_upstream_runner_treats_precollected_stream_response_as_terminal(
    monkeypatch,
):
    async def run():
        plan = await _routing_plan_with_retry_count(monkeypatch, 0)
        current_info = {}
        response = SimpleNamespace(
            status_code=200,
            body_iterator=iter([b"{}"]),
            _uni_api_response_attempt_terminal_outcome="succeeded",
            headers={},
        )

        async def execute_attempt(_attempt):
            return response

        returned = await UpstreamRunner(
            plan,
            observability_context=current_info,
        ).run(execute_attempt)

        assert returned is response
        assert current_info["routing_attempts"][-1]["outcome"] == "succeeded"

    asyncio.run(run())


def test_retry_decision_without_an_available_attempt_is_not_a_transition(
    monkeypatch,
):
    async def run():
        plan = await _routing_plan_with_retry_count(monkeypatch, 1)
        current_info = {}

        async def execute_attempt(_attempt):
            raise RuntimeError("retryable upstream failure")

        monkeypatch.setattr(
            "upstream.should_retry_provider",
            lambda *_args, **_kwargs: True,
        )
        response = await UpstreamRunner(
            plan,
            observability_context=current_info,
        ).run(execute_attempt)

        assert response.status_code == 500
        assert current_info["attempt_count"] == 3
        assert current_info["retry_decision_count"] == 3
        assert current_info["retry_transition_count"] == 2
        assert (
            "retry_transition_to_index"
            not in current_info["routing_attempts"][-1]
        )
        assert current_info["routing_attempts"][-1]["outcome"] == "retry_exhausted"

    asyncio.run(run())


def test_upstream_runner_honors_retry_decider_and_attempt_cap(monkeypatch):
    async def run():
        plan = await _routing_plan_with_retry_count(monkeypatch, 10)
        calls = []
        decisions = []

        async def execute_attempt(attempt):
            calls.append(attempt.provider_name)
            raise RuntimeError("retryable upstream failure")

        async def retry_decider(
            _exc,
            status_code,
            _error_message,
            _attempt,
            prepare_failure,
        ):
            decisions.append((status_code, prepare_failure))
            return True

        response = await UpstreamRunner(plan).run(
            execute_attempt,
            retry_decider=retry_decider,
            max_attempts=3,
        )

        assert response.status_code == 500
        assert calls == ["provider-a", "provider-b", "provider-a"]
        assert decisions == [(500, False)] * 3

    asyncio.run(run())


def test_routing_attempt_ledger_keeps_first_and_last_sixteen(monkeypatch):
    async def run():
        attempts = [
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name=f"provider-{index}",
                original_model="gpt-4.1",
            )
            for index in range(1, 41)
        ]

        class Plan:
            auto_retry = True
            api_list = ()
            request_model_name = "gpt-4.1"
            status_code = 500
            error_message = "failed"

            async def next_provider(self):
                return attempts.pop(0) if attempts else None

            def record_failure(self, status_code, error_message):
                self.status_code = status_code
                self.error_message = error_message

        async def execute_attempt(_attempt):
            raise RuntimeError("retryable failure")

        monkeypatch.setattr(
            "upstream.should_retry_provider",
            lambda *_args, **_kwargs: True,
        )
        current_info = {}
        response = await UpstreamRunner(
            Plan(),
            observability_context=current_info,
        ).run(execute_attempt)

        assert response.status_code == 500
        assert current_info["attempt_count"] == 40
        assert current_info["routing_attempts_omitted_count"] == 8
        assert [item["index"] for item in current_info["routing_attempts"]] == [
            *range(1, 17),
            *range(25, 41),
        ]
        assert current_info["retry_decision_count"] == 40
        assert current_info["retry_transition_count"] == 39

    asyncio.run(run())


def test_routing_attempt_ledger_fingerprints_nested_api_key_provider():
    async def run():
        secret_provider = "sk-nested-provider-secret"
        attempts = [
            SimpleNamespace(
                provider={"preferences": {}},
                provider_name=secret_provider,
                original_model="gpt-4.1",
            )
        ]

        class Plan:
            auto_retry = False
            api_list = (secret_provider,)
            request_model_name = "gpt-4.1"

            async def next_provider(self):
                return attempts.pop(0) if attempts else None

            def record_failure(self, _status_code, _error_message):
                raise AssertionError("the successful attempt must not fail")

        async def execute_attempt(_attempt):
            return SimpleNamespace(status_code=200)

        current_info = {}
        response = await UpstreamRunner(
            Plan(),
            observability_context=current_info,
        ).run(execute_attempt)

        assert response.status_code == 200
        recorded_provider = current_info["routing_attempts"][0]["provider"]
        assert recorded_provider.startswith("local-api-key:")
        assert secret_provider not in recorded_provider

    asyncio.run(run())


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


def test_get_right_order_providers_filters_semantic_request_types():
    config = {
        "providers": [
            {
                "provider": "provider-default",
                "base_url": "https://default.example/v1/responses",
                "api": "key-default",
                "model": ["gpt-5.6-terra"],
            },
            {
                "provider": "provider-compaction-only",
                "base_url": "https://compact.example/v1/responses",
                "api": "key-compact",
                "model": ["gpt-5.6-terra"],
                "only_request_types": ["compaction"],
            },
            {
                "provider": "provider-no-compaction",
                "base_url": "https://regular.example/v1/responses",
                "api": "key-regular",
                "model": ["gpt-5.6-terra"],
                "exclude_request_types": ["compaction"],
            },
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.6-terra"],
            }
        ],
    }

    async def run():
        regular = await get_right_order_providers(
            "gpt-5.6-terra",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.6-terra"]},
        )
        compact = await get_right_order_providers(
            "gpt-5.6-terra",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.6-terra"]},
            request_type="compaction",
        )
        legacy_compact = await get_right_order_providers(
            "gpt-5.6-terra",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.6-terra"]},
            endpoint="/v1/responses/compact",
        )

        assert [provider["provider"] for provider in regular] == [
            "provider-default",
            "provider-no-compaction",
        ]
        assert [provider["provider"] for provider in compact] == [
            "provider-default",
            "provider-compaction-only",
        ]
        assert [provider["provider"] for provider in legacy_compact] == [
            "provider-default",
            "provider-compaction-only",
        ]

    asyncio.run(run())


def test_get_right_order_providers_filters_provider_request_rules():
    config = {
        "providers": [
            {
                "provider": "mxamaxai006",
                "base_url": "https://mxamaxai.example/v1/responses",
                "api": "key-mxamaxai",
                "model": [{"codex-auto-review": "gpt-5.6-luna"}],
                "exclude_request_rules": [
                    {
                        "match": {
                            "endpoint": "/v1/responses",
                            "request_model": "gpt-5.6-luna",
                            "upstream_model": "codex-auto-review",
                            "reasoning_effort": ["max"],
                        },
                        "reason": "unsupported_reasoning_effort",
                    }
                ],
            },
            {
                "provider": "fallback",
                "base_url": "https://fallback.example/v1/responses",
                "api": "key-fallback",
                "model": ["gpt-5.6-luna"],
            },
        ],
        "api_keys": [{"api": "sk-test", "model": ["gpt-5.6-luna"]}],
    }

    async def run():
        excluded = await get_right_order_providers(
            "gpt-5.6-luna",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.6-luna"]},
            endpoint="/v1/responses",
            reasoning_effort="max",
        )
        accepted = await get_right_order_providers(
            "gpt-5.6-luna",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.6-luna"]},
            endpoint="/v1/responses",
            reasoning_effort="high",
        )

        assert [provider["provider"] for provider in excluded] == ["fallback"]
        assert [provider["provider"] for provider in accepted] == [
            "mxamaxai006",
            "fallback",
        ]

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


def test_alpha_search_defaults_to_all_providers_except_exact_exclusions(
    monkeypatch,
):
    providers = [
        {
            "provider": "provider-default",
            "base_url": "https://default.example/v1/responses",
            "api": "key-default",
            "model": ["gpt-5.4"],
            "engine": "gpt",
        },
        {
            "provider": "provider-compact-only",
            "base_url": "https://compact.example/v1/responses",
            "api": "key-compact",
            "model": ["gpt-5.4"],
            "engine": "codex",
            "exclude_endpoints": ["/v1/responses/compact"],
        },
        {
            "provider": "provider-alpha-top-level",
            "base_url": "https://alpha-top.example/v1/responses",
            "api": "key-alpha-top",
            "model": ["gpt-5.4"],
            "engine": "gemini",
            "exclude_endpoints": ["v1/alpha/search/"],
        },
        {
            "provider": "provider-alpha-preference",
            "base_url": "https://alpha-pref.example/v1/responses",
            "api": "key-alpha-pref",
            "model": ["gpt-5.4"],
            "preferences": {
                "exclude_endpoints": ["/v1/alpha/search"],
            },
        },
    ]
    for provider in providers:
        monkeypatch.setitem(
            provider_api_circular_list,
            provider["provider"],
            _ProviderKeys(),
        )
    config = {
        "providers": providers,
        "api_keys": [{"api": "sk-test", "model": ["gpt-5.4"]}],
    }

    async def run():
        alpha = await get_right_order_providers(
            "gpt-5.4",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.4"]},
            endpoint="/v1/alpha/search",
        )
        responses = await get_right_order_providers(
            "gpt-5.4",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.4"]},
            endpoint="/v1/responses",
        )
        assert [provider["provider"] for provider in alpha] == [
            "provider-default",
            "provider-compact-only",
        ]
        assert [provider["provider"] for provider in responses] == [
            "provider-default",
            "provider-compact-only",
            "provider-alpha-top-level",
            "provider-alpha-preference",
        ]

    asyncio.run(run())


def test_routing_plan_refresh_preserves_endpoint(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(
        provider_api_circular_list,
        provider_name,
        _ProviderKeys(),
    )
    endpoints = []

    async def resolver(
        request_model_name,
        config,
        api_index,
        scheduling_algorithm,
        api_list,
        models_list,
        *,
        endpoint=None,
        **_kwargs,
    ):
        _ = config, api_index, scheduling_algorithm, api_list, models_list
        endpoints.append(endpoint)
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {
                    request_model_name: request_model_name,
                },
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    app = SimpleNamespace(
        state=SimpleNamespace(
            config={
                "api_keys": [
                    {
                        "api": "sk-test",
                        "model": ["gpt-5.4"],
                    }
                ]
            },
            api_list=["sk-test"],
            models_list={"sk-test": ["gpt-5.4"]},
            channel_manager=None,
        )
    )

    async def run():
        plan = await RoutingPlan.create(
            app,
            "gpt-5.4",
            0,
            {},
            {},
            endpoint="/v1/alpha/search",
            provider_resolver=resolver,
        )
        await plan.refresh_matching_providers()
        assert endpoints == ["/v1/alpha/search", "/v1/alpha/search"]
        assert plan.endpoint == "/v1/alpha/search"

    asyncio.run(run())


def test_routing_plan_refresh_preserves_request_context():
    request_contexts = []

    async def resolver(
        request_model_name,
        config,
        api_index,
        scheduling_algorithm,
        api_list,
        models_list,
        *,
        request_type=None,
        reasoning_effort=None,
        **_kwargs,
    ):
        _ = config, api_index, scheduling_algorithm, api_list, models_list
        request_contexts.append((request_type, reasoning_effort))
        return [
            {
                "provider": "provider-a",
                "_model_dict_cache": {request_model_name: request_model_name},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    app = SimpleNamespace(
        state=SimpleNamespace(
            config={"api_keys": [{"api": "sk-test", "model": ["gpt-5.6-terra"]}]},
            api_list=["sk-test"],
            models_list={"sk-test": ["gpt-5.6-terra"]},
            channel_manager=None,
        )
    )

    async def run():
        plan = await RoutingPlan.create(
            app,
            "gpt-5.6-terra",
            0,
            {},
            {},
            endpoint="/v1/responses",
            request_type="compaction",
            reasoning_effort="max",
            provider_resolver=resolver,
        )
        await plan.refresh_matching_providers()
        assert request_contexts == [
            ("compaction", "max"),
            ("compaction", "max"),
        ]
        assert plan.request_type == "compaction"
        assert plan.reasoning_effort == "max"

    asyncio.run(run())
