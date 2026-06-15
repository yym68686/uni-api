import asyncio

import pytest
from fastapi import HTTPException

from uni_api.rate_limit import ProviderKeyPool, RateLimitPolicy, RateLimitState, parse_rate_limit


def test_parse_rate_limit_supports_tpr_and_multiple_windows():
    assert parse_rate_limit("2/min,10/day,1048576/tpr") == [
        (2, 60),
        (10, 86400),
        (1048576, -1),
    ]


def test_rate_limit_state_enforces_multiple_sliding_windows():
    clock = {"now": 1000.0}
    state = RateLimitState(now_func=lambda: clock["now"])
    policy = RateLimitPolicy.from_config("2/min,3/hour")

    assert state.is_rate_limited("key", "gpt-4.1", policy, commit=True) is False
    assert state.is_rate_limited("key", "gpt-4.1", policy, commit=True) is False
    assert state.is_rate_limited("key", "gpt-4.1", policy, commit=True) is True

    clock["now"] += 61
    assert state.is_rate_limited("key", "gpt-4.1", policy, commit=True) is False
    assert state.is_rate_limited("key", "gpt-4.1", policy, commit=True) is True


def test_rate_limit_policy_uses_fuzzy_model_match():
    policy = RateLimitPolicy.from_config(
        {
            "default": "999999/min",
            "gemini-1.5-pro": "1/min",
        }
    )
    state = RateLimitState(now_func=lambda: 1000.0)

    assert state.is_rate_limited("key", "gemini-1.5-pro-latest", policy, commit=True) is False
    assert state.is_rate_limited("key", "gemini-1.5-pro-latest", policy, commit=True) is True


def test_rate_limit_state_enforces_tpr():
    policy = RateLimitPolicy.from_config({"default": "1048576/tpr"})
    state = RateLimitState()

    assert state.is_tpr_exceeded("gpt-4.1", 1048576, policy) is False
    assert state.is_tpr_exceeded("gpt-4.1", 1048577, policy) is True


def test_provider_key_pool_empty_pool_raises_429():
    pool = ProviderKeyPool([])

    async def run():
        with pytest.raises(HTTPException) as exc_info:
            await pool.next("gpt-4.1")
        assert exc_info.value.status_code == 429

    asyncio.run(run())


def test_provider_key_pool_cooldown_expires():
    clock = {"now": 1000.0}
    pool = ProviderKeyPool(["key-1"], now_func=lambda: clock["now"])

    async def run():
        await pool.set_cooling("key-1", cooling_time=10)
        assert await pool.is_all_rate_limited("gpt-4.1") is True
        clock["now"] += 11
        assert await pool.is_all_rate_limited("gpt-4.1") is False

    asyncio.run(run())


def test_provider_key_pool_rollback_restores_last_record():
    clock = {"now": 1000.0}
    pool = ProviderKeyPool(["key-1"], rate_limit={"default": "1/min"}, now_func=lambda: clock["now"])

    async def run():
        assert await pool.is_rate_limited("key-1", "gpt-4.1") is False
        assert await pool.is_rate_limited("key-1", "gpt-4.1") is True
        pool.rollback_rate_limit_record("key-1", "gpt-4.1")
        assert await pool.is_rate_limited("key-1", "gpt-4.1") is False

    asyncio.run(run())


def test_provider_key_pool_concurrent_selection_returns_each_available_key_once():
    pool = ProviderKeyPool(
        ["key-1", "key-2", "key-3"],
        rate_limit={"default": "1/min"},
        schedule_algorithm="round_robin",
    )

    async def run():
        selected = await asyncio.gather(
            pool.next("gpt-4.1"),
            pool.next("gpt-4.1"),
            pool.next("gpt-4.1"),
        )
        assert sorted(selected) == ["key-1", "key-2", "key-3"]
        with pytest.raises(HTTPException):
            await pool.next("gpt-4.1")

    asyncio.run(run())


def test_provider_key_pool_close_cancels_reordering_task():
    class SlowReorderPool(ProviderKeyPool):
        async def _load_reordered_items(self):
            await asyncio.Event().wait()

    async def run():
        pool = SlowReorderPool(["key-1"], schedule_algorithm="smart_round_robin", provider_name="provider-a")
        assert pool.snapshot()["reordering_task_active"] is True

        await pool.close()

        assert pool.reordering_task is None
        assert pool.snapshot()["reordering_task_active"] is False

    asyncio.run(run())
