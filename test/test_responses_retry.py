import asyncio
import gzip
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
    def __init__(self, response, stream_calls, post_calls):
        self.response = response
        self.stream_calls = stream_calls
        self.post_calls = post_calls

    def _pick_response(self, url):
        if isinstance(self.response, dict):
            return self.response[url]
        return self.response

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
        return DummyStreamContext(self._pick_response(url), [])

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
        self.stream_calls = []
        self.post_calls = []

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2=None):
        yield DummyClient(self.response, self.stream_calls, self.post_calls)


class SequencedDummyClient:
    def __init__(self, responses, post_calls):
        self.responses = responses
        self.post_calls = post_calls

    async def post(self, url, headers=None, content=None, timeout=None):
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


class SequencedDummyClientManager:
    def __init__(self, responses):
        self.responses = list(responses)
        self.post_calls = []

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2=None):
        yield SequencedDummyClient(self.responses, self.post_calls)


class DummyStreamingUpstreamResponse:
    def __init__(self, *, chunks=None, stream_error=None, status_code=200, json_data=None, raw_body=None):
        self.status_code = status_code
        self._chunks = list(chunks or [])
        self._stream_error = stream_error
        self._json_data = json_data if json_data is not None else {"ok": True}
        self._raw_body = raw_body

    async def aread(self):
        if self._raw_body is not None:
            return self._raw_body
        return json.dumps(self._json_data).encode("utf-8")

    def json(self):
        return self._json_data

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk
        if self._stream_error is not None:
            raise self._stream_error

    async def aiter_bytes(self):
        async for chunk in self.aiter_raw():
            yield chunk


class EncodedStreamingUpstreamResponse(DummyStreamingUpstreamResponse):
    def __init__(self, *, raw_chunks, decoded_chunks, status_code=200):
        super().__init__(chunks=raw_chunks, status_code=status_code)
        self._decoded_chunks = list(decoded_chunks)

    async def aiter_bytes(self):
        for chunk in self._decoded_chunks:
            yield chunk


def _responses_sse(event_name, payload):
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


def _configure_responses_test(monkeypatch, *, engine, provider_preferences=None):
    provider_name = f"{engine}-provider"
    keys = DummyCircularList(["key-1"])
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://example.com/v1/responses",
                "api": ["key-1"],
                "preferences": provider_preferences or {},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: (engine, None))

    if engine == "codex":
        monkeypatch.setattr(main, "_split_codex_api_key", lambda raw: ("account-1", "refresh-1"))

        async def fake_get_codex_access_token(provider_name, provider_api_key_raw, proxy):
            return "codex-access-token"

        monkeypatch.setattr(main, "_get_codex_access_token", fake_get_codex_access_token)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}

    upstream_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.com/v1/responses"),
        json={"ok": True},
    )
    client_manager = DummyClientManager(upstream_response)
    main.app.state.client_manager = client_manager
    return client_manager


def _run_responses_request(request, *, endpoint="/v1/responses", http_headers=None):
    request_token = main.request_info.set(
        {
            "request_id": "req-test",
            "api_key": "sk-test",
            "disconnect_event": None,
        }
    )
    try:
        handler = main.ResponsesRequestHandler()
        return asyncio.run(
            handler.request_responses(
                http_request=SimpleNamespace(headers=http_headers or {}),
                request_data=request,
                api_index=0,
                background_tasks=BackgroundTasks(),
                endpoint=endpoint,
            )
        )
    finally:
        main.request_info.reset(request_token)


def _run_responses_request_with_stream_body(request, *, endpoint="/v1/responses"):
    request_token = main.request_info.set(
        {
            "request_id": "req-test",
            "api_key": "sk-test",
            "disconnect_event": None,
        }
    )

    async def _run():
        handler = main.ResponsesRequestHandler()
        response = await handler.request_responses(
            http_request=SimpleNamespace(headers={}),
            request_data=request,
            api_index=0,
            background_tasks=BackgroundTasks(),
            endpoint=endpoint,
        )

        body = ""
        if hasattr(response, "body_iterator"):
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                chunks.append(chunk)
            body = b"".join(chunks).decode("utf-8")
        elif hasattr(response, "body"):
            body = response.body.decode("utf-8") if isinstance(response.body, bytes) else str(response.body)

        return response, body

    try:
        return asyncio.run(_run())
    finally:
        main.request_info.reset(request_token)


def test_resolve_codex_upstream_auth_passes_through_plain_bearer(monkeypatch):
    called = False

    async def fake_get_codex_access_token(provider_name, provider_api_key_raw, proxy):
        nonlocal called
        _ = provider_name, provider_api_key_raw, proxy
        called = True
        return "should-not-run"

    monkeypatch.setattr(main, "_get_codex_access_token", fake_get_codex_access_token)

    api_key, account_id = asyncio.run(
        main._resolve_codex_upstream_auth("codex-provider", "change-me", None)
    )

    assert api_key == "change-me"
    assert account_id is None
    assert called is False


def test_resolve_codex_upstream_auth_uses_oauth_for_account_refresh_format(monkeypatch):
    seen = {}

    async def fake_get_codex_access_token(provider_name, provider_api_key_raw, proxy):
        seen["provider_name"] = provider_name
        seen["provider_api_key_raw"] = provider_api_key_raw
        seen["proxy"] = proxy
        return "codex-access-token"

    monkeypatch.setattr(main, "_get_codex_access_token", fake_get_codex_access_token)

    api_key, account_id = asyncio.run(
        main._resolve_codex_upstream_auth(
            "codex-provider",
            "account-1,refresh-1",
            "http://proxy.example",
        )
    )

    assert api_key == "codex-access-token"
    assert account_id == "account-1"
    assert seen == {
        "provider_name": "codex-provider",
        "provider_api_key_raw": "account-1,refresh-1",
        "proxy": "http://proxy.example",
    }


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


def test_responses_codex_chatgpt_model_unsupported_retries_next_key(monkeypatch):
    provider_name = "codex"
    keys = main.ThreadSafeCircularList(
        ["key-1", "key-2"],
        schedule_algorithm="fixed_priority",
        provider_name=provider_name,
    )
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "engine": "codex",
                "_model_dict_cache": {"gpt-5.5": "gpt-5.5"},
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api": ["key-1", "key-2"],
                "preferences": {},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.5"],
                "preferences": {"AUTO_RETRY": True},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = SequencedDummyClientManager(
        [
            httpx.Response(
                400,
                request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
                json={
                    "detail": "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account."
                },
            ),
            httpx.Response(
                200,
                request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
                json={"id": "resp-b", "status": "completed"},
            ),
        ]
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.5",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "resp-b"
    assert [call["headers"]["Authorization"] for call in main.app.state.client_manager.post_calls] == [
        "Bearer key-1",
        "Bearer key-2",
    ]
    assert keys.cooling_until["key-1"] > 0


def test_responses_compact_codex_chatgpt_model_unsupported_retries_next_key(monkeypatch):
    provider_name = "codex"
    keys = main.ThreadSafeCircularList(
        ["key-1", "key-2"],
        schedule_algorithm="fixed_priority",
        provider_name=provider_name,
    )
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "engine": "codex",
                "_model_dict_cache": {"gpt-5.5": "gpt-5.5"},
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api": ["key-1", "key-2"],
                "preferences": {},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.5"],
                "preferences": {"AUTO_RETRY": True},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = SequencedDummyClientManager(
        [
            httpx.Response(
                400,
                request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses/compact"),
                json={
                    "detail": "The 'gpt-5.5' model is not supported when using Codex with a ChatGPT account."
                },
            ),
            httpx.Response(
                200,
                request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses/compact"),
                json={"id": "resp-b", "status": "completed"},
            ),
        ]
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.5",
            input=[{"role": "user", "content": "hello"}],
        ),
        endpoint="/v1/responses/compact",
    )

    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "resp-b"
    assert [call["headers"]["Authorization"] for call in main.app.state.client_manager.post_calls] == [
        "Bearer key-1",
        "Bearer key-2",
    ]
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://chatgpt.com/backend-api/codex/responses/compact",
        "https://chatgpt.com/backend-api/codex/responses/compact",
    ]
    assert keys.cooling_until["key-1"] > 0


def test_responses_codex_strips_max_output_tokens(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            max_output_tokens=123,
        )
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert "max_output_tokens" not in sent_payload


def test_responses_codex_strips_response_format(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
        )
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert "response_format" not in sent_payload


def test_responses_compact_codex_strips_store(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={"post_body_parameter_overrides": {"store": False}},
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            store=False,
        ),
        endpoint="/v1/responses/compact",
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    assert client_manager.post_calls[0]["url"] == "https://example.com/v1/responses/compact"
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert "store" not in sent_payload


def test_responses_compact_non_stream_error_log_uses_compact_endpoint(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList(["key-a"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"friendly-model": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

    error_logs = []

    def fake_error(msg, *args, **kwargs):
        _ = kwargs
        error_logs.append(msg % args if args else msg)

    monkeypatch.setattr(main.trace_logger, "error", fake_error)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["friendly-model"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses/compact": httpx.Response(
                404,
                request=httpx.Request("POST", "https://provider-a.example/v1/responses/compact"),
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Invalid URL (POST /v1/responses/compact)",
                    }
                },
            )
        }
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="friendly-model",
            input=[{"role": "user", "content": "hello"}],
        ),
        endpoint="/v1/responses/compact",
    )

    assert response.status_code == 404
    assert any("/v1/responses/compact upstream error status=404" in log for log in error_logs)
    assert any("request_id=req-test" in log for log in error_logs)
    assert any("request_model=friendly-model" in log for log in error_logs)
    assert any("actual_model=gpt-5.4" in log for log in error_logs)
    assert any("upstream_url=https://provider-a.example/v1/responses/compact" in log for log in error_logs)


def test_responses_split_summary_and_trace_logs(monkeypatch):
    _configure_responses_test(monkeypatch, engine="gpt")

    human_logs = []
    trace_logs = []

    def fake_human_info(msg, *args, **kwargs):
        _ = kwargs
        human_logs.append(msg % args if args else msg)

    def fake_trace_info(msg, *args, **kwargs):
        _ = kwargs
        trace_logs.append(msg % args if args else msg)

    monkeypatch.setattr(main.logger, "info", fake_human_info)
    monkeypatch.setattr(main.trace_logger, "info", fake_trace_info)

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 200
    assert any("model: gpt-5.4" in log and "engine: gpt" in log and "role: sk-test" in log for log in human_logs)
    assert all("request_id=" not in log for log in human_logs)
    assert any("endpoint=/v1/responses" in log and "request_id=req-test" in log for log in trace_logs)
    assert any("upstream_url=https://example.com/v1/responses" in log for log in trace_logs)


def test_responses_gpt_keeps_max_output_tokens(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="gpt")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            max_output_tokens=123,
        )
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["max_output_tokens"] == 123


def test_responses_generic_post_body_overrides_apply(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="gpt",
        provider_preferences={"post_body_parameter_overrides": {"store": False}},
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            store=True,
        )
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["store"] is False


def test_responses_generic_post_body_overrides_can_remove_fields(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="gpt",
        provider_preferences={"post_body_parameter_overrides": {"__remove__": ["store", "response_format"]}},
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            store=True,
            response_format={"type": "json_object"},
        )
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert "store" not in sent_payload
    assert "response_format" not in sent_payload


def test_responses_codex_without_overrides_keeps_client_store_value(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            store=False,
        )
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["store"] is False


def test_responses_codex_generic_post_body_overrides_apply(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={"post_body_parameter_overrides": {"store": True}},
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            store=False,
        )
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["store"] is True


def test_responses_codex_plain_bearer_api_key_skips_oauth(monkeypatch):
    provider_name = "codex-provider"
    keys = DummyCircularList(["change-me"])
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://example.com/v1/responses",
                "api": ["change-me"],
                "preferences": {},
            }
        ]

    async def fail_get_codex_access_token(provider_name, provider_api_key_raw, proxy):
        raise AssertionError("direct bearer codex auth should not refresh tokens")

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))
    monkeypatch.setattr(main, "_get_codex_access_token", fail_get_codex_access_token)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = DummyClientManager(
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://example.com/v1/responses"),
            json={"id": "resp-plain-bearer", "status": "completed"},
        )
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 200
    sent_headers = main.app.state.client_manager.post_calls[0]["headers"]
    assert sent_headers["Authorization"] == "Bearer change-me"
    assert "Chatgpt-Account-Id" not in sent_headers


def test_responses_codex_forces_current_client_headers_after_overrides(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={
            "headers": {
                "version": "0.21.0",
                "User-Agent": "codex_cli_rs/0.50.0",
            }
        },
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        ),
        http_headers={
            "Version": "0.21.0",
            "User-Agent": "yaak",
        },
    )

    assert response.status_code == 200
    sent_headers = client_manager.post_calls[0]["headers"]
    assert sent_headers["Version"] == main.CODEX_CLI_VERSION
    assert sent_headers["User-Agent"] == main.CODEX_USER_AGENT
    assert "version" not in sent_headers


def test_responses_stream_retries_next_provider_before_output(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
                ],
                stream_error=httpx.ReadTimeout(
                    "upstream stalled",
                    request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
                ),
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "b"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "b"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
                    _responses_sse(
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                        },
                    ),
                    _responses_sse(None, "[DONE]"),
                ]
            ),
        }
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert '"provider": "a"' not in body
    assert '"provider": "b"' in body
    assert "hello-b" in body
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
    ]


def test_responses_stream_parses_decoded_upstream_bytes(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList(["key-a"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

    decoded_chunks = [
        _responses_sse("response.created", {"type": "response.created"}),
        _responses_sse("response.in_progress", {"type": "response.in_progress"}),
        _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-decoded"}),
        _responses_sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
            },
        ),
        _responses_sse(None, "[DONE]"),
    ]
    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = DummyClientManager(
        EncodedStreamingUpstreamResponse(
            raw_chunks=[gzip.compress(b"".join(decoded_chunks))],
            decoded_chunks=decoded_chunks,
        )
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert "hello-decoded" in body
    assert len(main.app.state.client_manager.stream_calls) == 1


def test_responses_stream_commits_on_keepalive_and_does_not_retry(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
                    _responses_sse("keepalive", {"type": "keepalive", "sequence_number": 3, "provider": "a"}),
                ],
                stream_error=httpx.ReadTimeout(
                    "upstream stalled",
                    request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
                ),
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "b"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "b"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
                    _responses_sse(None, "[DONE]"),
                ]
            ),
        }
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert "event: keepalive" in body
    assert '"provider": "b"' not in body
    assert "hello-b" not in body
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
    ]


def test_responses_stream_retries_when_structural_events_end_without_output(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
                    _responses_sse(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "item": {"type": "message", "status": "in_progress", "content": []},
                            "provider": "a",
                        },
                    ),
                    _responses_sse(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "part": {"type": "output_text", "text": ""},
                            "provider": "a",
                        },
                    ),
                    _responses_sse(None, "[DONE]"),
                ]
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "b"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "b"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
                    _responses_sse(
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                        },
                    ),
                    _responses_sse(None, "[DONE]"),
                ]
            ),
        }
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert '"provider": "a"' not in body
    assert '"provider": "b"' in body
    assert "hello-b" in body
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
    ]


def test_responses_stream_retries_next_provider_on_semantic_failure(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
                    _responses_sse(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "tokens",
                                "code": "rate_limit_exceeded",
                                "message": "too many requests",
                            },
                        },
                    ),
                    _responses_sse(
                        "response.failed",
                        {
                            "type": "response.failed",
                            "response": {
                                "status": "failed",
                                "error": {
                                    "code": "rate_limit_exceeded",
                                    "message": "too many requests",
                                },
                            },
                        },
                    ),
                ]
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "b"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "b"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
                    _responses_sse(None, "[DONE]"),
                ]
            ),
        }
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert '"provider": "a"' not in body
    assert '"provider": "b"' in body
    assert "hello-b" in body
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
    ]


def test_responses_stream_does_not_retry_after_output_started(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-a"}),
                ],
                stream_error=httpx.ReadTimeout(
                    "upstream stalled",
                    request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
                ),
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
                    _responses_sse(None, "[DONE]"),
                ]
            ),
        }
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert '"provider": "a"' in body
    assert "hello-a" in body
    assert "hello-b" not in body
    assert body.endswith("data: [DONE]\n\n")
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
    ]


def test_responses_compact_stream_abort_log_uses_compact_endpoint(monkeypatch):
    provider_name = "provider-a"
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, DummyCircularList(["key-a"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"friendly-model": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

    warning_logs = []

    def fake_warning(msg, *args, **kwargs):
        _ = kwargs
        warning_logs.append(msg % args if args else msg)

    monkeypatch.setattr(main.trace_logger, "warning", fake_warning)

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["friendly-model"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses/compact": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                    _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-a"}),
                ],
                stream_error=httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body",
                    request=httpx.Request("POST", "https://provider-a.example/v1/responses/compact"),
                ),
            )
        }
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="friendly-model",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        ),
        endpoint="/v1/responses/compact",
    )

    assert response.status_code == 200
    assert "hello-a" in body
    assert body.endswith("data: [DONE]\n\n")
    assert any("/v1/responses/compact upstream stream aborted stage=post-commit" in log for log in warning_logs)
    assert any("error_type=RemoteProtocolError" in log for log in warning_logs)
    assert any("request_model=friendly-model" in log for log in warning_logs)
    assert any("actual_model=gpt-5.4" in log for log in warning_logs)
    assert any("request_id=req-test" in log for log in warning_logs)
    assert any("upstream_url=https://provider-a.example/v1/responses/compact" in log for log in warning_logs)


def test_responses_non_stream_retries_next_provider_on_semantic_failure(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
                json={
                    "id": "resp-a",
                    "status": "failed",
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": "too many requests",
                    },
                },
            ),
            "https://provider-b.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request("POST", "https://provider-b.example/v1/responses"),
                json={
                    "id": "resp-b",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "hello-b",
                                }
                            ],
                        }
                    ],
                },
            ),
        }
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "resp-b"
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
    ]


def test_responses_non_stream_rate_limit_cools_current_key_and_tries_next_key(monkeypatch):
    provider_name = "codex-provider"
    keys = main.ThreadSafeCircularList(
        ["key-1", "key-2"],
        schedule_algorithm="fixed_priority",
        provider_name=provider_name,
    )
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://example.com/v1/responses",
                "api": ["key-1", "key-2"],
                "preferences": {"api_key_rate_limit_cooldown_period": 1},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))
    monkeypatch.setattr(main, "_split_codex_api_key", lambda raw: ("account-1", "refresh-1"))

    async def fake_get_codex_access_token(provider_name, provider_api_key_raw, proxy):
        return provider_api_key_raw

    monkeypatch.setattr(main, "_get_codex_access_token", fake_get_codex_access_token)

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
    main.app.state.client_manager = SequencedDummyClientManager(
        [
            httpx.Response(
                429,
                request=httpx.Request("POST", "https://example.com/v1/responses"),
                json={
                    "error": {
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                        "message": "Rate limit reached for gpt-5.4 on tokens per min (TPM): Limit 40000000, Used 40000000, Requested 72349. Please try again in 108ms.",
                    }
                },
            ),
            httpx.Response(
                200,
                request=httpx.Request("POST", "https://example.com/v1/responses"),
                json={
                    "id": "resp-b",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "hello-b",
                                }
                            ],
                        }
                    ],
                },
            ),
        ]
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "resp-b"
    assert [call["headers"]["Authorization"] for call in main.app.state.client_manager.post_calls] == [
        "Bearer key-1",
        "Bearer key-2",
    ]
    assert keys.cooling_until["key-1"] > 0


def test_responses_prepare_validation_failure_retries_next_provider(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/chat/completions",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-b.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request("POST", "https://provider-b.example/v1/responses"),
                json={"id": "resp-b", "status": "completed"},
            )
        }
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "resp-b"
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://provider-b.example/v1/responses",
    ]


def test_responses_codex_prepare_failure_does_not_cool_key(monkeypatch):
    provider_name = "codex-provider"
    keys = DummyCircularList(["account-1,bad-key-1", "account-2,bad-key-2"])
    monkeypatch.setitem(main.provider_api_circular_list, provider_name, keys)

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://example.com/v1/responses",
                "api": ["account-1,bad-key-1", "account-2,bad-key-2"],
                "preferences": {"api_key_cooldown_period": 60},
            }
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))
    monkeypatch.setattr(main, "_split_codex_api_key", lambda raw: (_ for _ in ()).throw(ValueError("bad codex key")))

    main.app.state.config = {
        "api_keys": [
            {
                "api": "sk-test",
                "model": ["gpt-5.4"],
                "preferences": {"AUTO_RETRY": False},
            }
        ]
    }
    main.app.state.provider_timeouts = {"global": {"default": 30}}
    main.app.state.client_manager = DummyClientManager(
        httpx.Response(
            200,
            request=httpx.Request("POST", "https://example.com/v1/responses"),
            json={"ok": True},
        )
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {"error": "All gpt-5.4 error: bad codex key"}
    assert keys.cooling_calls == []
    assert keys.next_calls == [("gpt-5.4", "account-1,bad-key-1")]
    assert main.app.state.client_manager.post_calls == []


def test_responses_non_stream_semantic_bad_request_does_not_retry(monkeypatch):
    provider_a = "provider-a"
    provider_b = "provider-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://provider-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("gpt", None))

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
    main.app.state.client_manager = DummyClientManager(
        {
            "https://provider-a.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
                json={
                    "id": "resp-a",
                    "status": "failed",
                    "error": {
                        "code": "invalid_type",
                        "message": "bad input",
                    },
                },
            ),
            "https://provider-b.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request("POST", "https://provider-b.example/v1/responses"),
                json={"id": "resp-b", "status": "completed"},
            ),
        }
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": {
            "code": "invalid_type",
            "message": "bad input",
        }
    }
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://provider-a.example/v1/responses",
    ]
