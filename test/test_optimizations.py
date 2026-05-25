import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
import utils
import core.utils as core_utils
from core.response import _stream_responses_to_chat_completions
from core.utils import ThreadSafeCircularList
from routing import build_routing_index, get_right_order_providers, weighted_round_robin, lottery_scheduling


def test_global_rate_limiter_uses_sliding_windows(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(utils, "time", lambda: clock["now"])

    limiter = utils.InMemoryRateLimiter()

    async def run():
        assert await limiter.is_rate_limited("global", [(1, 10)]) is False
        assert await limiter.is_rate_limited("global", [(1, 10)]) is True
        clock["now"] += 11
        assert await limiter.is_rate_limited("global", [(1, 10)]) is False

    asyncio.run(run())


def test_provider_rate_limit_rollback_restores_last_record(monkeypatch):
    clock = {"now": 2000.0}
    monkeypatch.setattr(core_utils, "time", lambda: clock["now"])

    keys = ThreadSafeCircularList(["key-1"], rate_limit={"default": "1/min"}, schedule_algorithm="fixed_priority", provider_name="provider-a")

    async def run():
        assert await keys.is_rate_limited("key-1", "gpt-4.1") is False
        assert await keys.is_rate_limited("key-1", "gpt-4.1") is True
        keys.rollback_rate_limit_record("key-1", "gpt-4.1")
        assert await keys.is_rate_limited("key-1", "gpt-4.1") is False

    asyncio.run(run())


def test_list_models_uses_cached_response(monkeypatch):
    main.app.state.config = {"api_keys": [{"api": "sk-test"}]}
    main.app.state.api_list = ["sk-test"]
    main.app.state.api_keys_db = [{"api": "sk-test"}]
    main.app.state.models_list = {"sk-test": ["gpt-5.4", "gpt-5.4-mini"]}
    main.app.state.model_response_cache = {
        "sk-test": [
            {"id": "gpt-5.4", "object": "model", "created": 1720524448858, "owned_by": "uni-api"},
            {"id": "gpt-5.4-mini", "object": "model", "created": 1720524448858, "owned_by": "uni-api"},
        ]
    }

    def fail_post_all_models(*args, **kwargs):
        raise AssertionError("post_all_models should not run when cache is present")

    monkeypatch.setattr(main, "post_all_models", fail_post_all_models)

    response = asyncio.run(main.list_models(api_index=0))
    body = json.loads(response.body)

    assert [item["id"] for item in body["data"]] == ["gpt-5.4", "gpt-5.4-mini"]


def test_get_right_order_providers_accepts_prebuilt_routing_index():
    config = {
        "providers": [
            {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/v1/responses",
                "model": ["gpt-5.4"],
                "exclude_endpoints": ["v1/responses/compact"],
            },
            {
                "provider": "provider-b",
                "base_url": "https://provider-b.example/v1/responses",
                "model": ["gpt-5.4"],
            },
        ],
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
            }
        ],
    }
    routing_index = build_routing_index(config, ["sk-test"])

    providers = asyncio.run(
        get_right_order_providers(
            "gpt-5.4",
            config,
            0,
            "fixed_priority",
            ["sk-test"],
            {"sk-test": ["gpt-5.4"]},
            endpoint="/v1/responses",
            routing_index=routing_index,
        )
    )

    assert [provider["provider"] for provider in providers] == ["provider-a", "provider-b"]


def test_shared_sse_parser_handles_split_events():
    async def upstream_iter():
        yield b"event: response.output_text.delt"
        yield b"a\ndata: {\"type\": \"response.output_text.delta\", \"delta\": \"hello\"}\n"
        yield b"\n"
        yield b"data: [DONE]\n\n"

    async def run():
        chunks = []
        async for chunk in _stream_responses_to_chat_completions(upstream_iter(), request_model="gpt-5.4"):
            chunks.append(chunk)
            if chunk.endswith("data: [DONE]\n\n"):
                break
        return chunks

    chunks = asyncio.run(run())
    body = "".join(chunks)

    assert "hello" in body
    assert body.endswith("data: [DONE]\n\n")


def test_weighted_scheduling_helpers_keep_provider_counts():
    weights = {"provider-a": 2, "provider-b": 1}

    weighted = weighted_round_robin(weights)
    lottery = lottery_scheduling(weights)

    assert len(weighted) == 3
    assert sorted(weighted).count("provider-a") == 2
    assert sorted(weighted).count("provider-b") == 1
    assert len(lottery) == 3
    assert set(lottery).issubset({"provider-a", "provider-b"})
