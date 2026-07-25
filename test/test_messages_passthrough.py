import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import BackgroundTasks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


def _anthropic_sse(event_name, payload):
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload)}\n\n"
    ).encode("utf-8")


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


def test_messages_model_pricing_400_retries_next_provider(monkeypatch):
    provider_a = "anthropic-pricing-a"
    provider_b = "anthropic-pricing-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))
    _set_messages_state(
        monkeypatch,
        [
            {
                "provider": provider_a,
                "_model_dict_cache": {"claude-alias": "claude-opus-5"},
                "base_url": "https://pricing-a.example/v1/messages",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"claude-alias": "claude-opus-5"},
                "base_url": "https://pricing-b.example/v1/messages",
                "api": ["key-b"],
                "preferences": {},
            },
        ],
    )
    pricing_error = {
        "error": {
            "type": "upstream_error",
            "code": "model_price_not_configured",
            "message": (
                "模型 claude-opus-5 的价格尚未由管理员配置，暂时无法使用；"
                "Model claude-opus-5 has not been priced by the administrator yet."
            ),
        }
    }
    main.app.state.client_manager = DummyClientManager(
        {
            "https://pricing-a.example/v1/messages": httpx.Response(
                400,
                request=httpx.Request("POST", "https://pricing-a.example/v1/messages"),
                json=pricing_error,
            ),
            "https://pricing-b.example/v1/messages": httpx.Response(
                200,
                request=httpx.Request("POST", "https://pricing-b.example/v1/messages"),
                json={"id": "msg-pricing-fallback", "type": "message"},
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
    assert json.loads(response.body) == {
        "id": "msg-pricing-fallback",
        "type": "message",
    }
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://pricing-a.example/v1/messages",
        "https://pricing-b.example/v1/messages",
    ]


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
    assert '"value": "upst...-key"' in upstream_header_logs[0]
    assert '"value": "upstream-key"' not in upstream_header_logs[0]
    assert '"name": "anthropic-beta"' in upstream_header_logs[0]
    assert '"value": "debug-beta"' in upstream_header_logs[0]

    upstream_logs = [log for log in logs if log.startswith("DEBUG upstream request body")]
    assert len(upstream_logs) == 1
    assert '"model": "claude-sonnet-4-5-20250929"' in upstream_logs[0]
    assert '"metadata": {\n    "source": "debug-test"\n  }' in upstream_logs[0]


class _CloseProbe:
    def __init__(self):
        self.response_closes = 0
        self.context_exits = 0

    async def aclose(self):
        self.response_closes += 1

    async def __aexit__(self, exc_type, exc, tb):
        self.context_exits += 1


def _messages_stream_context(*, disconnect_event=None):
    current_info = {
        "request_id": "messages-stream",
        "api_key": "sk-test",
    }
    return {
        "endpoint": "/v1/messages",
        "request_id": "messages-stream",
        "request_model_name": "claude-alias",
        "current_info": current_info,
        "disconnect_event": disconnect_event,
        "background_tasks": BackgroundTasks(),
    }, SimpleNamespace(
        provider_name="anthropic",
        provider_api_key_raw="provider-key",
        state={"channel_id": "anthropic", "track_channel_stats": True},
    )


def test_messages_stream_records_success_only_after_message_stop(monkeypatch):
    async def scenario():
        results = []

        def record(*_args, success, **_kwargs):
            results.append(success)

        monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
        handler = main.MessagesPassthroughHandler()
        ctx, attempt = _messages_stream_context()
        probe = _CloseProbe()

        async def upstream():
            yield _anthropic_sse(
                "message_stop",
                {"type": "message_stop"},
            )

        stream = handler._messages_proxy_stream(
            ctx,
            attempt,
            [
                _anthropic_sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "delta": {"text": "hi"}},
                )
            ],
            upstream(),
            probe,
            probe,
        )
        assert b"content_block_delta" in await anext(stream)
        assert results == []
        remaining = [chunk async for chunk in stream]

        assert b"message_stop" in b"".join(remaining)
        assert results == [True]
        assert ctx["current_info"]["success"] is True
        assert probe.response_closes == 1
        assert probe.context_exits == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("network_abort", [False, True])
def test_messages_stream_truncation_records_failure_without_false_success(
    monkeypatch,
    network_abort,
):
    async def scenario():
        results = []

        def record(*_args, success, **_kwargs):
            results.append(success)

        monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
        handler = main.MessagesPassthroughHandler()
        ctx, attempt = _messages_stream_context()
        probe = _CloseProbe()

        async def upstream():
            if network_abort:
                raise httpx.ReadError(
                    "messages upstream aborted",
                    request=httpx.Request(
                        "POST",
                        "https://example.com/v1/messages",
                    ),
                )
            if False:
                yield b""

        stream = handler._messages_proxy_stream(
            ctx,
            attempt,
            [
                _anthropic_sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "delta": {"text": "partial"}},
                )
            ],
            upstream(),
            probe,
            probe,
        )
        assert b"content_block_delta" in await anext(stream)
        if network_abort:
            with pytest.raises(httpx.ReadError):
                await anext(stream)
        else:
            terminal = await anext(stream)
            assert terminal.startswith(b"event: error\n")
            assert b'"type":"error"' in terminal
            assert b'"code":"upstream_sse_protocol_error"' in terminal
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert ctx["current_info"][
                "postcommit_sse_protocol_error_isolated"
            ] is True

        assert results == [False]
        assert ctx["current_info"]["success"] is False
        assert ctx["current_info"]["stream_outcome"] == "upstream_stream_abort"
        assert probe.response_closes == 1
        assert probe.context_exits == 1

    asyncio.run(scenario())


def test_messages_real_disconnect_cancels_upstream_without_channel_failure(
    monkeypatch,
):
    async def scenario():
        results = []

        def record(*_args, success, **_kwargs):
            results.append(success)

        monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
        disconnect_event = asyncio.Event()
        handler = main.MessagesPassthroughHandler()
        ctx, attempt = _messages_stream_context(
            disconnect_event=disconnect_event
        )
        probe = _CloseProbe()
        upstream_closed = False

        async def upstream():
            nonlocal upstream_closed
            try:
                await asyncio.Event().wait()
                yield b"unreachable"
            finally:
                upstream_closed = True

        stream = handler._messages_proxy_stream(
            ctx,
            attempt,
            [
                _anthropic_sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "delta": {"text": "partial"}},
                )
            ],
            upstream(),
            probe,
            probe,
        )
        assert b"content_block_delta" in await anext(stream)
        disconnect_event.set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

        assert results == []
        assert upstream_closed is True
        assert ctx["current_info"]["stream_outcome"] == "downstream_disconnected"
        assert ctx["current_info"]["downstream_disconnected"] is True
        assert probe.response_closes == 1
        assert probe.context_exits == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("coalesced", [False, True])
def test_messages_terminal_is_event_bounded_and_does_not_wait_for_eof(
    monkeypatch,
    coalesced,
):
    async def scenario():
        results = []

        def record(*_args, success, **_kwargs):
            results.append(success)

        monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
        handler = main.MessagesPassthroughHandler()
        ctx, attempt = _messages_stream_context()
        probe = _CloseProbe()
        first = _anthropic_sse(
            "content_block_delta",
            {"type": "content_block_delta", "delta": {"text": "zero"}},
        )
        second = _anthropic_sse(
            "content_block_delta",
            {"type": "content_block_delta", "delta": {"text": "one"}},
        )
        terminal = _anthropic_sse("message_stop", {"type": "message_stop"})

        async def upstream():
            if coalesced:
                yield second + terminal
            else:
                yield second
                yield terminal
            raise AssertionError("messages proxy read after message_stop")

        stream = handler._messages_proxy_stream(
            ctx,
            attempt,
            [first],
            upstream(),
            probe,
            probe,
        )
        body = b"".join([chunk async for chunk in stream])

        assert b"zero" in body
        assert b"one" in body
        assert b"message_stop" in body
        assert results == [True]
        assert probe.response_closes == 1
        assert probe.context_exits == 1

    asyncio.run(scenario())


def test_messages_malformed_terminal_data_is_protocol_failure(monkeypatch):
    async def scenario():
        results = []

        def record(*_args, success, **_kwargs):
            results.append(success)

        monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
        handler = main.MessagesPassthroughHandler()
        ctx, attempt = _messages_stream_context()
        probe = _CloseProbe()

        async def upstream():
            if False:
                yield b""

        stream = handler._messages_proxy_stream(
            ctx,
            attempt,
            [b"event: message_stop\ndata: not-json\n\n"],
            upstream(),
            probe,
            probe,
        )
        terminal = await anext(stream)
        assert terminal.startswith(b"event: error\n")
        assert b'"code":"upstream_sse_protocol_error"' in terminal
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

        assert results == [False]
        assert ctx["current_info"]["success"] is False
        assert ctx["current_info"][
            "postcommit_sse_protocol_error_isolated"
        ] is True

    asyncio.run(scenario())
