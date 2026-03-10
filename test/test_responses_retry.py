import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
from fastapi import BackgroundTasks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from core.models import ResponsesRequest


class DummyCircularList:
    def __init__(self, items):
        self.items = list(items)
        self.next_calls = []
        self.cooling_calls = []

    async def is_all_rate_limited(self, model):
        return False

    async def next(self, model):
        item = self.items[len(self.next_calls) % len(self.items)]
        self.next_calls.append((model, item))
        return item

    def get_items_count(self):
        return len(self.items)

    async def set_cooling(self, item, cooling_time):
        self.cooling_calls.append((item, cooling_time))


class DummyStreamContext:
    def __init__(self, response, calls):
        self.response = response
        self.calls = calls

    async def __aenter__(self):
        self.calls.append("enter")
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        self.calls.append("exit")


class DummyClient:
    def __init__(self, response, stream_calls):
        self.response = response
        self.stream_calls = stream_calls

    def stream(self, method, url, headers=None, content=None, timeout=None):
        self.stream_calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
            }
        )
        return DummyStreamContext(self.response, [])


class DummyClientManager:
    def __init__(self, response):
        self.response = response
        self.stream_calls = []

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2=None):
        yield DummyClient(self.response, self.stream_calls)


def test_responses_bad_request_does_not_retry_all_keys(monkeypatch):
    provider_name = "codex-like-provider"
    keys = DummyCircularList(["key-1", "key-2", "key-3"])
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://example.com/v1/responses",
                "api": ["key-1", "key-2", "key-3"],
                "preferences": {"api_key_cooldown_period": 60},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

    upstream_error = {
        "error": {
            "message": "Invalid type for 'input[0]': expected an input item, but got a string instead.",
            "type": "invalid_request_error",
            "code": "invalid_type",
        }
    }
    upstream_response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://example.com/v1/responses"),
        json=upstream_error,
    )
    client_manager = DummyClientManager(upstream_response)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
                "preferences": {"AUTO_RETRY": True},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = client_manager

    request_token = main.request_info.set(
        {
            "request_id": "req-test",
            "api_key": "sk-test",
            "disconnect_event": None,
        }
    )
    try:
        handler = main.ResponsesRequestHandler()
        request = ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=True)
        response = asyncio.run(
            handler.request_responses(
                http_request=SimpleNamespace(headers={}),
                request_data=request,
                api_index=0,
                background_tasks=BackgroundTasks(),
            )
        )
    finally:
        main.request_info.reset(request_token)

    assert response.status_code == 400
    assert json.loads(response.body) == upstream_error
    assert len(client_manager.stream_calls) == 1
    assert keys.next_calls == [("gpt-5.4", "key-1")]
    assert keys.cooling_calls == []
