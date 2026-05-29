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


def test_debug_header_pairs_preserves_raw_duplicate_headers():
    headers = SimpleNamespace(
        raw=[
            (b"host", b"example.test"),
            (b"x-repeat", b"a"),
            (b"x-repeat", b"b"),
        ]
    )

    assert main._debug_header_pairs(headers) == [
        {"name": "host", "value": "example.test"},
        {"name": "x-repeat", "value": "a"},
        {"name": "x-repeat", "value": "b"},
    ]


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


class DummyClient:
    def __init__(self, response, post_calls):
        self.response = response
        self.post_calls = post_calls

    def _pick_response(self, url):
        if isinstance(self.response, dict):
            return self.response[url]
        return self.response

    async def post(self, url, headers=None, content=None, timeout=None):
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
            }
        )
        return self._pick_response(url)


class DummyClientManager:
    def __init__(self, response):
        self.response = response
        self.post_calls = []

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2=None):
        _ = base_url, proxy, http2
        yield DummyClient(self.response, self.post_calls)


def _set_messages_state(monkeypatch, providers, *, auto_retry=True):
    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        _ = request_model_name, config, api_index, scheduling_algorithm
        return providers

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["claude-alias"],
                "preferences": {"AUTO_RETRY": auto_retry},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}


def _run_messages_request(body, *, http_headers=None):
    request_token = main.request_info.set(
        {
            "request_id": "req-test",
            "api_key": "sk-test",
            "disconnect_event": None,
        }
    )
    try:
        handler = main.MessagesPassthroughHandler()
        return asyncio.run(
            handler.request_messages(
                http_request=SimpleNamespace(headers=http_headers or {}),
                request_body=body,
                api_index=0,
                background_tasks=BackgroundTasks(),
            )
        )
    finally:
        main.request_info.reset(request_token)


def test_messages_passes_body_through_with_model_mapping_and_anthropic_headers(monkeypatch):
    provider_name = "anthropic"
    keys = DummyCircularList(["upstream-key"])
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)
    _set_messages_state(
        monkeypatch,
        [
            {
                "provider": provider_name,
                "_model_dict_cache": {"claude-alias": "claude-sonnet-4-5-20250929"},
                "base_url": "https://api.anthropic.com/v1/messages",
                "api": ["upstream-key"],
                "preferences": {"headers": {"anthropic-beta": "tools-2024-05-16"}},
            }
        ],
        auto_retry=False,
    )

    upstream_body = {"id": "msg_123", "type": "message", "content": [{"type": "text", "text": "ok"}]}
    main.app.state.client_manager = DummyClientManager(
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            json=upstream_body,
        )
    )
    body = {
        "model": "claude-alias",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "tools": [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "tool", "name": "lookup"},
        "stream": False,
    }

    response = _run_messages_request(body, http_headers={"anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    assert json.loads(response.body) == upstream_body
    assert len(main.app.state.client_manager.post_calls) == 1
    call = main.app.state.client_manager.post_calls[0]
    sent_payload = json.loads(call["content"])
    assert sent_payload == {
        **body,
        "model": "claude-sonnet-4-5-20250929",
    }
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "upstream-key"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["headers"]["anthropic-beta"] == "tools-2024-05-16"
    assert keys.next_calls == [("claude-sonnet-4-5-20250929", "upstream-key")]


def test_messages_last_text_supports_native_anthropic_content_blocks():
    body = {
        "model": "claude-alias",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "tool output"},
                    {"type": "text", "text": "final prompt"},
                ],
            }
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
    }

    assert main._messages_request_last_text(body) == "final prompt"


def test_messages_retries_next_provider_on_upstream_failure(monkeypatch):
    provider_a = "anthropic-a"
    provider_b = "anthropic-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))
    _set_messages_state(
        monkeypatch,
        [
            {
                "provider": provider_a,
                "_model_dict_cache": {"claude-alias": "claude-a"},
                "base_url": "https://provider-a.example/v1/messages",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"claude-alias": "claude-b"},
                "base_url": "https://provider-b.example/v1/messages",
                "api": ["key-b"],
                "preferences": {},
            },
        ],
    )
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/messages": httpx.Response(
                500,
                request=httpx.Request("POST", "https://provider-a.example/v1/messages"),
                json={"error": {"message": "temporary failure"}},
            ),
            "https://provider-b.example/v1/messages": httpx.Response(
                200,
                request=httpx.Request("POST", "https://provider-b.example/v1/messages"),
                json={"id": "msg_b", "type": "message"},
            ),
        }
    )

    response = _run_messages_request(
        {
            "model": "claude-alias",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "msg_b"
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://provider-a.example/v1/messages",
        "https://provider-b.example/v1/messages",
    ]
    assert [json.loads(call["content"])["model"] for call in main.app.state.client_manager.post_calls] == [
        "claude-a",
        "claude-b",
    ]


def test_messages_bad_request_forwards_upstream_error_without_retrying_keys(monkeypatch):
    provider_name = "anthropic"
    keys = DummyCircularList(["key-1", "key-2"])
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)
    _set_messages_state(
        monkeypatch,
        [
            {
                "provider": provider_name,
                "_model_dict_cache": {"claude-alias": "claude-sonnet-4-5"},
                "base_url": "https://api.anthropic.com/v1/messages",
                "api": ["key-1", "key-2"],
                "preferences": {"api_key_cooldown_period": 60},
            }
        ],
    )
    error_body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "messages: field required"},
    }
    main.app.state.client_manager = DummyClientManager(
        httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            json=error_body,
        )
    )

    response = _run_messages_request(
        {
            "model": "claude-alias",
            "max_tokens": 32,
            "messages": [],
        }
    )

    assert response.status_code == 400
    assert json.loads(response.body) == error_body
    assert len(main.app.state.client_manager.post_calls) == 1
    assert keys.next_calls == [("claude-sonnet-4-5", "key-1")]
    assert keys.cooling_calls == []


def test_messages_debug_logs_final_upstream_request_headers_and_body(monkeypatch):
    provider_name = "anthropic"
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList(["upstream-key"]))
    _set_messages_state(
        monkeypatch,
        [
            {
                "provider": provider_name,
                "_model_dict_cache": {"claude-alias": "claude-sonnet-4-5-20250929"},
                "base_url": "https://api.anthropic.com/v1/messages",
                "api": ["upstream-key"],
                "preferences": {
                    "headers": {"anthropic-beta": "debug-beta"},
                    "post_body_parameter_overrides": {"metadata": {"source": "debug-test"}},
                },
            }
        ],
        auto_retry=False,
    )
    main.app.state.client_manager = DummyClientManager(
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            json={"id": "msg_debug", "type": "message"},
        )
    )
    logs = []

    def fake_info(message, *args, **kwargs):
        _ = kwargs
        logs.append(message % args if args else message)

    monkeypatch.setattr(main, "is_debug", True)
    monkeypatch.setattr(main.logger, "info", fake_info)

    response = _run_messages_request(
        {
            "model": "claude-alias",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response.status_code == 200
    upstream_header_logs = [log for log in logs if log.startswith("DEBUG upstream request headers")]
    assert len(upstream_header_logs) == 1
    assert '"name": "x-api-key"' in upstream_header_logs[0]
    assert '"value": "upstream-key"' in upstream_header_logs[0]
    assert '"name": "anthropic-beta"' in upstream_header_logs[0]
    assert '"value": "debug-beta"' in upstream_header_logs[0]

    upstream_logs = [log for log in logs if log.startswith("DEBUG upstream request body")]
    assert len(upstream_logs) == 1
    assert '"model": "claude-sonnet-4-5-20250929"' in upstream_logs[0]
    assert '"metadata": {\n    "source": "debug-test"\n  }' in upstream_logs[0]
