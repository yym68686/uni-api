import asyncio
import errno
import gzip
import hashlib
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import httpcore
import pytest
from fastapi import BackgroundTasks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
import uni_api.runtime as runtime
from core.models import ResponsesRequest
from upstream import normalize_provider_exception, should_retry_provider


def test_oaix_keepalive_classifier_rejects_large_frames_before_sync_parse(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_extract_responses_stream_event",
        lambda _raw: (_ for _ in ()).throw(
            AssertionError("large frame must not be parsed")
        ),
    )

    assert not runtime._is_oaix_precommit_keepalive(b"x" * 1025)


def test_first_byte_deadline_timeout_reports_configured_seconds():
    async def trigger_timeout():
        loop = asyncio.get_running_loop()
        with pytest.raises(httpx.ReadTimeout) as captured:
            await runtime._await_first_byte_deadline(
                asyncio.sleep(60),
                timeout_seconds=20,
                deadline=loop.time(),
            )
        return captured.value

    error = asyncio.run(trigger_timeout())

    assert error.request.extensions["timeout"]["read"] == 20
    assert normalize_provider_exception(error) == (
        504,
        "Request timed out after 20 seconds",
    )


def test_runtime_records_pool_timeout_as_local_without_marking_admission():
    current_info = {}

    classification = runtime._record_transport_failure(
        current_info,
        httpx.PoolTimeout("pool exhausted"),
    )
    assert classification is not None
    assert current_info["transport_error_kind"] == "pool_timeout"
    assert current_info["transport_error_owner"] == "ember_local_overload"
    assert current_info["transport_error_phase"] == "pool_acquire"
    assert current_info["provider_penalty_eligible"] is False
    assert current_info["local_overload"] is True
    assert current_info["status_origin"] == "ember_local_overload"
    assert "admission_rejected" not in current_info

    connect_failure = runtime._record_transport_failure(
        {},
        httpx.ConnectTimeout("connect timed out"),
    )
    assert connect_failure is not None
    assert connect_failure.provider_penalty_eligible is True


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
        if hasattr(self.response, "close_events"):
            self.response.close_events.append("context_enter")
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        self.calls.append("exit")
        if hasattr(self.response, "context_exit_calls"):
            self.response.context_exit_calls += 1
        if hasattr(self.response, "close_events"):
            self.response.close_events.append("context_exit")


class DummyClient:
    def __init__(self, response, stream_calls, post_calls):
        self.response = response
        self.stream_calls = stream_calls
        self.post_calls = post_calls

    def _pick_response(self, url):
        if isinstance(self.response, dict):
            return self.response[url]
        return self.response

    def stream(
        self,
        method,
        url,
        headers=None,
        content=None,
        timeout=None,
        extensions=None,
    ):
        self.stream_calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
                "extensions": extensions,
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
    def __init__(self, *, chunks=None, stream_error=None, status_code=200, json_data=None, raw_body=None, headers=None, extensions=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.extensions = extensions or {}
        self._chunks = list(chunks or [])
        self._stream_error = stream_error
        self._json_data = json_data if json_data is not None else {"ok": True}
        self._raw_body = raw_body
        self.close_calls = 0
        self.context_exit_calls = 0
        self.close_events = []

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

    async def aclose(self):
        self.close_calls += 1
        self.close_events.append("response_aclose")


class BlockingStreamingUpstreamResponse(DummyStreamingUpstreamResponse):
    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk
        await asyncio.Event().wait()


class TerminalReadTrapResponse(DummyStreamingUpstreamResponse):
    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk
        raise AssertionError("proxy read upstream after semantic terminal")


class YieldingStreamingUpstreamResponse(DummyStreamingUpstreamResponse):
    async def aiter_raw(self):
        for index, chunk in enumerate(self._chunks):
            yield chunk
            if index == 0:
                await asyncio.sleep(0)


class DisconnectingLiveStreamingUpstreamResponse(DummyStreamingUpstreamResponse):
    def __init__(self, *, disconnect_event, disconnect_before_index, **kwargs):
        super().__init__(**kwargs)
        self._disconnect_event = disconnect_event
        self._disconnect_before_index = disconnect_before_index

    async def aiter_raw(self):
        for index, chunk in enumerate(self._chunks):
            if index == self._disconnect_before_index:
                self._disconnect_event.set()
            yield chunk


class DelayedFirstChunkStreamingUpstreamResponse(DummyStreamingUpstreamResponse):
    async def aiter_raw(self):
        await asyncio.sleep(0.02)
        async for chunk in super().aiter_raw():
            yield chunk


class EncodedStreamingUpstreamResponse(DummyStreamingUpstreamResponse):
    def __init__(self, *, raw_chunks, decoded_chunks, status_code=200):
        super().__init__(chunks=raw_chunks, status_code=status_code)
        self._decoded_chunks = list(decoded_chunks)

    async def aiter_bytes(self):
        for chunk in self._decoded_chunks:
            yield chunk


class FakeNetworkStream:
    def get_extra_info(self, name):
        return {
            "client_addr": ("10.0.0.8", 42000),
            "server_addr": ("192.0.2.10", 443),
            "socket": None,
        }.get(name)


def _responses_sse(event_name, payload):
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


def _split_responses_sse(event_name, payload):
    return (
        f"event: {event_name}\n\n"
        f"data: {json.dumps(payload)}\n\n\n"
    ).encode("utf-8")


def _data_only_responses_sse(payload):
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _chained_responses_read_error():
    try:
        raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")
    except ConnectionResetError as os_exc:
        try:
            raise httpcore.ReadError("read failed") from os_exc
        except httpcore.ReadError as core_exc:
            try:
                raise httpx.ReadError(
                    "read failed",
                    request=httpx.Request(
                        "POST",
                        "https://example.com/v1/responses",
                    ),
                ) from core_exc
            except httpx.ReadError as exc:
                return exc


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
    main.app.state.timeout_policy = main.init_timeout_policy({})

    upstream_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.com/v1/responses"),
        json={"ok": True},
    )
    client_manager = DummyClientManager(upstream_response)
    main.app.state.client_manager = client_manager
    return client_manager


def _configure_two_provider_responses_test(
    monkeypatch,
    responses,
    *,
    engine="codex",
):
    provider_names = ("provider-a", "provider-b")
    for suffix, provider_name in zip(("a", "b"), provider_names):
        monkeypatch.setitem(
            main.provider_api_circular_list,
            provider_name,
            DummyCircularList([f"key-{suffix}"]),
        )

    async def fake_get_right_order_providers(
        request_model_name,
        config,
        api_index,
        scheduling_algorithm,
    ):
        return [
            {
                "provider": provider_name,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": (
                    f"https://{provider_name}.example/v1/responses"
                ),
                "api": [f"key-{suffix}"],
                "preferences": {},
            }
            for suffix, provider_name in zip(("a", "b"), provider_names)
        ]

    monkeypatch.setattr(
        main,
        "get_right_order_providers",
        fake_get_right_order_providers,
    )
    monkeypatch.setattr(
        main,
        "get_engine",
        lambda provider, endpoint=None, original_model=None: (engine, None),
    )
    if engine == "codex":
        monkeypatch.setattr(
            main,
            "_split_codex_api_key",
            lambda raw: ("account-1", "refresh-1"),
        )

        async def fake_get_codex_access_token(
            provider_name,
            provider_api_key_raw,
            proxy,
        ):
            return "codex-access-token"

        monkeypatch.setattr(
            main,
            "_get_codex_access_token",
            fake_get_codex_access_token,
        )

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
    main.app.state.timeout_policy = main.init_timeout_policy({})
    client_manager = DummyClientManager(responses)
    main.app.state.client_manager = client_manager
    return client_manager


def _run_responses_request(request, *, endpoint="/v1/responses", http_headers=None):
    request_info_value = {
        "request_id": "req-test",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    request_token = main.request_info.set(request_info_value)
    try:
        handler = main.ResponsesRequestHandler()
        return asyncio.run(
            handler.request_responses(
                http_request=SimpleNamespace(
                    headers=http_headers or {},
                    state=SimpleNamespace(uni_api_request_info=request_info_value),
                ),
                request_data=request,
                api_index=0,
                background_tasks=BackgroundTasks(),
                endpoint=endpoint,
            )
        )
    finally:
        main.request_info.reset(request_token)


def _run_responses_request_with_stream_body(
    request,
    *,
    endpoint="/v1/responses",
    current_info=None,
    http_headers=None,
):
    request_info_value = current_info or {
        "request_id": "req-test",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    request_token = main.request_info.set(request_info_value)

    async def _run():
        handler = main.ResponsesRequestHandler()
        response = await handler.request_responses(
            http_request=SimpleNamespace(
                headers=http_headers or {},
                state=SimpleNamespace(uni_api_request_info=request_info_value),
            ),
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

    current_info = {
        "request_id": "req-test",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    request_token = main.request_info.set(current_info)
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


def test_responses_missing_persisted_item_404_does_not_retry():
    upstream_error = {
        "error": {
            "message": (
                "Item with id 'rs_0dc02c5b394c2253016a2c446c9e148191a6595865d06c6054' not found. "
                "Items are not persisted when `store` is set to false. Try again with `store` set to true, "
                "or remove this item from your input."
            ),
            "type": "invalid_request_error",
            "param": "input",
            "code": None,
        }
    }

    assert not should_retry_provider(
        True,
        404,
        {"base_url": "http://oaix.example/v1/responses"},
        error_message=json.dumps(upstream_error),
        endpoint="/v1/responses",
        original_model="gpt-5.5",
    )


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


def test_responses_normalizes_polluted_custom_tool_call_input(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={
            "normalize_responses_custom_tool_call_ids": True,
        },
    )
    request = ResponsesRequest(
        model="gpt-5.4",
        input=[
            {
                "type": "reasoning",
                "id": "item_reasoning123",
            },
            {
                "type": "custom_tool_call",
                "id": "item_7eb1bd749e0a9e692c69ed40",
                "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
                "name": "exec",
                "input": "{}",
            },
            {
                "type": "custom_tool_call_output",
                "id": "ctco_output123",
                "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
                "output": "ok",
            },
        ],
    )

    response = _run_responses_request(request)

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["input"][0]["type"] == "reasoning"
    assert sent_payload["input"][1]["id"] == "ctc_7eb1bd749e0a9e692c69ed40"
    assert sent_payload["input"][1]["call_id"] == "call_foCUR1DBzdZeYyOccLpOmwUF"
    assert sent_payload["input"][2]["call_id"] == "call_foCUR1DBzdZeYyOccLpOmwUF"
    assert request.input[1]["id"] == "item_7eb1bd749e0a9e692c69ed40"


def test_responses_normalizes_non_stream_custom_tool_call_output(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={
            "normalize_responses_custom_tool_call_ids": True,
        },
    )
    client_manager.response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.com/v1/responses"),
        json={
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "custom_tool_call",
                    "id": "item_output123",
                    "call_id": "call_output123",
                    "name": "exec",
                    "input": "{}",
                }
            ],
        },
    )

    response = _run_responses_request(
        ResponsesRequest(model="gpt-5.4", input="hello", stream=False)
    )

    assert response.status_code == 200
    assert json.loads(response.body)["output"][0]["id"] == "ctc_output123"


def test_responses_normalizes_stream_custom_tool_call_ids_consistently(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={
            "normalize_responses_custom_tool_call_ids": True,
        },
    )
    item = {
        "type": "custom_tool_call",
        "id": "item_stream123",
        "call_id": "call_stream123",
        "name": "exec",
        "input": "{}",
    }
    client_manager.response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": dict(item),
                },
            ),
            _responses_sse(
                "response.custom_tool_call_input.delta",
                {
                    "type": "response.custom_tool_call_input.delta",
                    "output_index": 0,
                    "item_id": "item_stream123",
                    "delta": "{}",
                },
            ),
            _responses_sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "item_id": "item_stream123",
                    "delta": "mapped reference",
                },
            ),
            _responses_sse(
                "response.custom_tool_call_input.done",
                {
                    "type": "response.custom_tool_call_input.done",
                    "output_index": 0,
                    "item_id": "item_stream123",
                    "input": "{}",
                },
            ),
            _responses_sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": dict(item),
                },
            ),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [dict(item)],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                },
            ),
        ],
        headers={"content-type": "text/event-stream"},
    )

    current_info = {
        "request_id": "req-test",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input="hello", stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "item_stream123" not in body
    assert body.count("ctc_stream123") == 6
    assert current_info["custom_tool_call_id_normalized"] is True
    assert current_info["custom_tool_call_id_normalization_count"] == 3
    assert current_info["custom_tool_call_id_reference_rewrite_count"] == 3
    assert current_info["responses_delta_fast_path_candidates"] == 1
    assert current_info["responses_delta_fast_path_events"] == 0
    assert current_info["responses_delta_fast_path_fallbacks"] == 1


@pytest.mark.parametrize(
    ("requested_tier", "expected_tier"),
    [
        ("priority", "priority"),
        ("fast", "fast"),
        (" PRIORITY ", " PRIORITY "),
        ("default", "default"),
    ],
)
def test_responses_codex_preserves_service_tier(
    monkeypatch,
    requested_tier,
    expected_tier,
):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            service_tier=requested_tier,
        )
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["service_tier"] == expected_tier


def test_responses_gpt_preserves_service_tier(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="gpt")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            service_tier="fast",
        )
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["service_tier"] == "fast"


@pytest.mark.parametrize(
    ("engine", "endpoint"),
    [
        ("gpt", "/v1/responses"),
        ("codex", "/v1/responses"),
        ("gpt", "/v1/responses/compact"),
        ("codex", "/v1/responses/compact"),
    ],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {
            "service_tier": "default",
            "gpt-5.4": {"__remove__": ["service_tier"]},
        },
        {
            "__remove__": ["service_tier"],
            "gpt-5.4": {"service_tier": "default"},
        },
    ],
)
def test_responses_service_tier_ignores_provider_overrides(
    monkeypatch,
    engine,
    endpoint,
    overrides,
):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine=engine,
        provider_preferences={"post_body_parameter_overrides": overrides},
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            service_tier="priority",
        ),
        endpoint=endpoint,
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert sent_payload["service_tier"] == "priority"


@pytest.mark.parametrize(
    ("engine", "endpoint"),
    [
        ("gpt", "/v1/responses"),
        ("codex", "/v1/responses"),
        ("gpt", "/v1/responses/compact"),
        ("codex", "/v1/responses/compact"),
    ],
)
def test_responses_provider_override_cannot_inject_service_tier(
    monkeypatch,
    engine,
    endpoint,
):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine=engine,
        provider_preferences={
            "post_body_parameter_overrides": {
                "service_tier": "default",
                "gpt-5.4": {"service_tier": "priority"},
            }
        },
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        ),
        endpoint=endpoint,
    )

    assert response.status_code == 200
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert "service_tier" not in sent_payload


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


def test_responses_codex_strips_top_p(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            top_p=0.2,
        )
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    assert "top_p" not in sent_payload


def test_responses_codex_strips_nested_cache_control(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    content_part = sent_payload["input"][0]["content"][0]
    assert content_part["text"] == "hello"
    assert "cache_control" not in content_part


def test_responses_codex_strips_store_false_reasoning_ids(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[
                {"type": "message", "role": "user", "content": "make the image 2k"},
                {
                    "type": "reasoning",
                    "id": "rs_0dc02c5b394c2253016a2c446c9e148191a6595865d06c6054",
                    "summary": [],
                    "encrypted_content": "encrypted-reasoning",
                },
                {
                    "type": "image_generation_call",
                    "id": "ig_123",
                    "status": "completed",
                    "result": "image-b64",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
            store=False,
        )
    )

    assert response.status_code == 200
    assert len(client_manager.post_calls) == 1
    sent_payload = json.loads(client_manager.post_calls[0]["content"])
    input_items = sent_payload["input"]
    assert [item["type"] for item in input_items] == [
        "message",
        "reasoning",
        "image_generation_call",
        "function_call_output",
    ]
    reasoning_item = input_items[1]
    assert "id" not in reasoning_item
    assert reasoning_item["summary"] == []
    assert reasoning_item["encrypted_content"] == "encrypted-reasoning"
    assert input_items[2]["id"] == "ig_123"
    assert sent_payload["store"] is False


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


def test_responses_compact_non_stream_uses_timeout_policy_and_records_attempt(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.provider_timeouts = {"global": {"gpt-5.4": 20, "default": 30}}
    main.app.state.timeout_policy = main.init_timeout_policy(
        {
            "preferences": {
                "timeout_policy": {
                    "rules": [
                        {
                            "match": {
                                "endpoint": "/v1/responses/compact",
                                "stream": False,
                                "model": "gpt-5.4",
                            },
                            "timeout": {"first_byte": 120, "total": 300},
                        }
                    ]
                }
            }
        }
    )

    current_info = {
        "request_id": "req-large-compact",
        "api_key": "sk-test",
        "disconnect_event": None,
        "trace": main.RequestTrace(trace_id="req-large-compact"),
    }
    request_token = main.request_info.set(current_info)
    try:
        handler = main.ResponsesRequestHandler()
        response = asyncio.run(
            handler.request_responses(
                http_request=SimpleNamespace(headers={}),
                request_data=ResponsesRequest(
                    model="gpt-5.4",
                    input=[{"role": "user", "content": "hello"}],
                ),
                api_index=0,
                background_tasks=BackgroundTasks(),
                endpoint="/v1/responses/compact",
            )
        )
    finally:
        main.request_info.reset(request_token)

    assert response.status_code == 200
    assert client_manager.post_calls[0]["timeout"] == 120
    attempt = current_info["upstream_attempts"][0]
    assert attempt["provider"] == "codex-provider"
    assert attempt["payload_bytes"] > 0
    assert attempt["timeout_seconds"] == 120
    assert attempt["timeout_adjusted_from_seconds"] == 20
    assert attempt["timeout_policy_sources"] == ["global.rules[0]"]
    assert attempt["status_code"] == 200
    assert attempt["success"] is True


def test_responses_compaction_trigger_uses_request_type_timeout_policy(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")
    client_manager.response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        }
                    },
                },
            ),
            _responses_sse(None, "[DONE]"),
        ]
    )
    main.app.state.provider_timeouts = {"global": {"gpt-5.4": 20, "default": 30}}
    main.app.state.timeout_policy = main.init_timeout_policy(
        {
            "preferences": {
                "timeout_policy": {
                    "rules": [
                        {
                            "match": {
                                "endpoint": "/v1/responses",
                                "stream": True,
                                "request_type": "compaction",
                                "engine": "codex",
                                "model": "gpt-5.4*",
                            },
                            "timeout": {"first_byte": 300, "total": 3000},
                        }
                    ]
                }
            }
        }
    )
    current_info = {
        "request_id": "req-compaction-trigger",
        "api_key": "sk-test",
        "disconnect_event": None,
        "trace": main.RequestTrace(trace_id="req-compaction-trigger"),
    }

    response, _ = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"type": "compaction_trigger"}],
            stream=True,
        ),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert current_info["semantic_request_type"] == "compaction"
    assert len(client_manager.stream_calls) == 1
    attempt = current_info["upstream_attempts"][0]
    assert attempt["timeout_seconds"] == 300
    assert attempt["timeout_adjusted_from_seconds"] == 20
    assert attempt["timeout_policy_sources"] == ["global.rules[0]"]


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


def test_responses_stdout_request_summary_log_can_be_disabled(monkeypatch):
    _configure_responses_test(monkeypatch, engine="gpt")
    monkeypatch.setenv("STDOUT_REQUEST_SUMMARY_LOG_ENABLED", "false")

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
    assert not any(log.startswith("provider:") for log in human_logs)
    assert any("endpoint=/v1/responses" in log and "request_id=req-test" in log for log in trace_logs)


def test_stdout_request_summary_sample_rate_zero(monkeypatch):
    monkeypatch.setenv("STDOUT_REQUEST_SUMMARY_LOG_ENABLED", "true")
    monkeypatch.setenv("STDOUT_REQUEST_SUMMARY_LOG_SAMPLE_RATE", "0")

    assert main._should_log_stdout_request_summary() is False


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
    assert "Version" not in sent_headers
    assert sent_headers["User-Agent"] == main.CODEX_USER_AGENT
    assert "version" not in sent_headers


def test_responses_applies_configured_passthrough_request_headers(monkeypatch):
    client_manager = _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={
            "headers": {
                "X-OAIX-Selection-Mode": "marketplace",
                "X-OAIX-Exclude-Owner": "stale-owner",
            },
            "passthrough_request_headers": [
                "X-OAIX-Act-As-User",
                "X-OAIX-Selection-Mode",
                "X-OAIX-Exclude-Owner",
            ],
        },
    )

    response = _run_responses_request(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
        ),
        http_headers={
            "X-OAIX-Act-As-User": "1",
            "x-oaix-selection-mode": "marketplace",
        },
    )

    assert response.status_code == 200
    sent_headers = client_manager.post_calls[0]["headers"]
    assert sent_headers["X-OAIX-Act-As-User"] == "1"
    assert sent_headers["X-OAIX-Selection-Mode"] == "marketplace"
    assert "X-OAIX-Exclude-Owner" not in sent_headers


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
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))

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


def test_responses_stream_records_channel_success_only_after_protocol_terminal(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    channel_results = []

    def record(*_args, success, **_kwargs):
        channel_results.append(success)

    monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
    completed_event = _responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": 7,
            "response": {
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                }
            },
        },
    )
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse("response.created", {"type": "response.created"}),
                _responses_sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "ok"},
                ),
                # OAIX may flush the terminal and [DONE] in one HTTP body
                # chunk. The semantic terminal ends processing; the unconsumed
                # [DONE] frame must not replace its diagnostics or hash.
                completed_event + _responses_sse(None, "[DONE]"),
            ],
            headers={"X-OAIX-Connection-ID": "oaixc-terminal-success"},
            extensions={
                "http_version": b"HTTP/2",
                "stream_id": 13,
                "network_stream": FakeNetworkStream(),
            },
        )
    )
    current_info = {
        "request_id": "terminal-success",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    _response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert "response.completed" in body
    assert channel_results == [True]
    assert current_info["success"] is True
    assert current_info["upstream_attempts"][-1]["success"] is True
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["oaix_connection_id"] == "oaixc-terminal-success"
    assert diagnostics["http_version"] == "HTTP/2"
    assert diagnostics["httpcore_stream_id"] == 13
    assert diagnostics["terminal_frame_seen"] is True
    assert diagnostics["upstream_terminal_seen"] is True
    assert diagnostics["upstream_terminal_validated"] is True
    assert diagnostics["usage_seen"] is True
    assert diagnostics["diagnosis"] == "responses_completed_with_usage"
    assert diagnostics["last_event_type"] == "response.completed"
    assert diagnostics["complete_event_count"] == 3
    assert diagnostics["semantic_terminal_type"] == "response.completed"
    assert diagnostics["semantic_terminal_outcome"] == "completed"
    assert diagnostics["semantic_terminal_sequence_number"] == 7
    assert diagnostics["semantic_terminal_sha256"] == hashlib.sha256(
        completed_event
    ).hexdigest()
    assert diagnostics["downstream_terminal_seen"] is False
    assert diagnostics["cleanup_owner"] == "responses_proxy_finally"
    assert diagnostics["cleanup_result"] == "succeeded"
    assert callable(
        main.app.state.client_manager.stream_calls[0]["extensions"]["trace"]
    )


def test_stream_encodes_each_event_once_and_reuses_batch_receive_time(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    observed = []
    encoded = []
    original_observe = runtime.ResponsesStreamDiagnostics.observe_complete_event
    original_encode = runtime._raw_responses_sse_event_bytes

    def observe_complete_event(self, raw_event, *args, **kwargs):
        if "batch-after-commit" in raw_event or kwargs.get("event_type") == (
            "response.completed"
        ):
            observed.append(
                (
                    kwargs.get("event_type"),
                    kwargs.get("wire_bytes"),
                    kwargs.get("received_at"),
                )
            )
        return original_observe(self, raw_event, *args, **kwargs)

    def encode_wire(raw_event):
        if "event: response." in raw_event:
            encoded.append(raw_event)
        return original_encode(raw_event)

    monkeypatch.setattr(
        runtime.ResponsesStreamDiagnostics,
        "observe_complete_event",
        observe_complete_event,
    )
    monkeypatch.setattr(runtime, "_raw_responses_sse_event_bytes", encode_wire)

    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse(
                    "response.created",
                    {"type": "response.created", "sequence_number": 0},
                ),
                _responses_sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 1,
                        "delta": "commit",
                    },
                ),
                _responses_sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 2,
                        "delta": "batch-after-commit",
                    },
                )
                + _responses_sse(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "sequence_number": 3,
                        "response": {
                            "status": "completed",
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 2,
                                "total_tokens": 3,
                            },
                        },
                    },
                ),
            ]
        )
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True)
    )

    assert response.status_code == 200
    assert "batch-after-commit" in body
    assert len(observed) == 2
    assert all(wire_bytes is not None for _, wire_bytes, _ in observed)
    assert observed[0][2] is observed[1][2]
    assert observed[0][2] is not None
    assert len(encoded) == 4
    assert sum("response.created" in event for event in encoded) == 1
    assert sum('"delta": "commit"' in event for event in encoded) == 1
    assert sum("batch-after-commit" in event for event in encoded) == 1
    assert sum("response.completed" in event for event in encoded) == 1


def test_postcommit_delta_fast_path_preserves_wire_and_falls_back_on_status(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    created = _responses_sse(
        "response.created",
        {"type": "response.created", "sequence_number": 0},
    )
    first_delta = _responses_sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "delta": "commit",
        },
    )
    transparent_delta = _responses_sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "delta": "零拷贝",
        },
    )
    fallback_delta = _responses_sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "status": "in_progress",
            "delta": "fallback",
        },
    )
    completed = _responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": 4,
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 3,
                    "total_tokens": 4,
                },
            },
        },
    )
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                created,
                first_delta,
                transparent_delta,
                fallback_delta,
                completed,
            ]
        )
    )
    current_info = {
        "request_id": "responses-fast-path",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    # The Codex adapter may prepend its protocol keepalive. The provider event
    # sequence itself must remain a byte-for-byte suffix.
    assert body.encode("utf-8").endswith(
        created
        + first_delta
        + transparent_delta
        + fallback_delta
        + completed
    )
    assert current_info["responses_delta_fast_path_candidates"] == 2
    assert current_info["responses_delta_fast_path_events"] == 1
    assert current_info["responses_delta_fast_path_fallbacks"] == 1
    assert current_info["responses_delta_fast_path_bytes"] == len(
        transparent_delta
    )


def test_postcommit_delta_fast_path_coexists_with_custom_tool_id_normalizer(
    monkeypatch,
):
    _configure_responses_test(
        monkeypatch,
        engine="codex",
        provider_preferences={
            "normalize_responses_custom_tool_call_ids": True,
        },
    )
    created = _responses_sse(
        "response.created",
        {"type": "response.created", "sequence_number": 0},
    )
    first_delta = _responses_sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "item_message",
            "delta": "commit",
        },
    )
    transparent_delta = _responses_sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": "item_message",
            "delta": "normalizer-compatible",
        },
    )
    completed = _responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            },
        },
    )
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[created, first_delta, transparent_delta, completed]
        )
    )
    current_info = {
        "request_id": "responses-fast-path-normalizer",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    events = [
        json.loads(frame.split("data: ", 1)[1])
        for frame in body.strip().split("\n\n")
        if frame.startswith("event: response.")
    ]
    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[1]["item_id"] == "msg_message"
    assert events[2]["item_id"] == "msg_message"
    assert events[1]["delta"] == "commit"
    assert events[2]["delta"] == "normalizer-compatible"
    assert current_info["responses_delta_fast_path_candidates"] == 1
    assert current_info["responses_delta_fast_path_events"] == 0
    assert current_info["responses_delta_fast_path_fallbacks"] == 1
    assert current_info["responses_delta_fast_path_bytes"] == 0


def test_responses_stream_consumes_exact_oaix_terminal_flush_marker(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    hop_observations = []
    monkeypatch.setattr(
        main.worker_runtime_observer,
        "record_terminal_hop",
        hop_observations.append,
    )
    completed_event = _responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": 9,
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            },
        },
    )
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    marker = (
        ": oaix-terminal-flush-v1 "
        + json.dumps(
            {
                "schema_version": 1,
                "connection_id": "oaixc-terminal-marker",
                "terminal_wire_sha256": hashlib.sha256(
                    completed_event
                ).hexdigest(),
                "flush_attempted_unix_nano": now_ns - 2_001_000_000,
                "flush_completed_unix_nano": now_ns - 2_000_000_000,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n\n"
    ).encode("utf-8")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse(
                    "response.created",
                    {"type": "response.created"},
                ),
                _responses_sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "ok"},
                ),
                completed_event,
                # A distinct transport chunk exercises the bounded
                # post-terminal marker read rather than same-chunk coalescing.
                marker,
            ],
            headers={
                "X-OAIX-Connection-ID": "oaixc-terminal-marker",
                "X-OAIX-Terminal-Flush-Marker": "sse-comment-v1",
            },
        )
    )
    current_info = {
        "request_id": "terminal-marker",
        "trace_id": "trace-terminal-marker",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    _response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert "response.completed" in body
    assert "oaix-terminal-flush-v1" not in body
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["oaix_terminal_flush_marker_expected"] is True
    assert diagnostics["oaix_terminal_flush_marker_seen"] is True
    assert diagnostics["oaix_terminal_flush_marker_hash_matched"] is True
    assert diagnostics[
        "oaix_terminal_flush_to_ember_receive_observed"
    ] is True
    assert diagnostics["oaix_terminal_flush_to_ember_receive_ms"] >= 1900
    assert "oaix_terminal_flush_marker_missing" not in diagnostics
    assert len(hop_observations) == 1
    assert hop_observations[0]["request_id"] == "terminal-marker"


@pytest.mark.parametrize(
    ("terminal_type", "status"),
    [
        ("response.completed", "completed"),
        ("response.incomplete", "incomplete"),
    ],
)
def test_responses_semantic_terminal_closes_without_waiting_for_eof(
    monkeypatch,
    terminal_type,
    status,
):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(
            chunks=[
                _responses_sse(
                    terminal_type,
                    {
                        "type": terminal_type,
                        "response": {"status": status, "output": []},
                    },
                )
            ]
        )
    )

    current_info = {
        "request_id": f"terminal-without-usage-{terminal_type}",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert terminal_type in body
    assert "proxy read upstream after semantic terminal" not in body
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["usage_seen"] is False
    assert diagnostics["diagnosis"] == (
        "responses_completed_without_usage"
        if terminal_type == "response.completed"
        else "responses_incomplete_terminal"
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 7, 1024])
def test_responses_normalizes_split_event_and_data_blocks(
    monkeypatch,
    chunk_size,
):
    _configure_responses_test(monkeypatch, engine="codex")
    wire = b"".join(
        [
            _split_responses_sse(
                "response.created",
                {
                    "type": "response.created",
                    "response": {"status": "in_progress"},
                },
            ),
            _split_responses_sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": "hello",
                },
            ),
            _split_responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 2,
                    "response": {
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 5,
                            "total_tokens": 8,
                        },
                    },
                },
            ),
        ]
    )
    chunks = (
        [wire]
        if chunk_size is None
        else [wire[offset : offset + chunk_size] for offset in range(0, len(wire), chunk_size)]
    )
    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(chunks=chunks)
    )
    current_info = {
        "request_id": f"split-sse-{chunk_size}",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "event: error" not in body
    for event_type in (
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ):
        assert body.count(f"event: {event_type}\n") == 1
        assert f'event: {event_type}\ndata: {{' in body
    assert '"delta": "hello"' in body
    assert current_info["success"] is True
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["upstream_terminal_seen"] is True
    assert diagnostics["upstream_terminal_validated"] is True
    assert diagnostics["usage_seen"] is True
    assert diagnostics["ignored_no_data_event_count"] == 3
    assert diagnostics["canonicalized_data_only_event_count"] == 3
    assert diagnostics["normalization_applied"] is True
    assert diagnostics["diagnosis"] == "responses_completed_with_usage"
    assert diagnostics["semantic_terminal_sha256"] == diagnostics["last_event_sha256"]
    assert diagnostics["declared_terminal_sha256"] == diagnostics["last_event_sha256"]


def test_responses_max_sized_data_only_terminal_survives_canonicalization(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    payload = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        },
        "padding": "",
    }
    initial = _data_only_responses_sse(payload)
    padding_bytes = main.DEFAULT_MAX_EVENT_BYTES - len(initial) + 2
    assert padding_bytes > 0
    payload["padding"] = "x" * padding_bytes
    wire = _data_only_responses_sse(payload)
    assert len(wire) - 2 == main.DEFAULT_MAX_EVENT_BYTES

    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(chunks=[wire])
    )
    current_info = {
        "request_id": "max-sized-data-only-terminal",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    request_token = main.request_info.set(current_info)

    async def run():
        handler = main.ResponsesRequestHandler()
        response = await handler.request_responses(
            http_request=SimpleNamespace(
                headers={},
                state=SimpleNamespace(uni_api_request_info=current_info),
            ),
            request_data=ResponsesRequest(
                model="gpt-5.4",
                input=["hello"],
                stream=True,
            ),
            api_index=0,
            background_tasks=BackgroundTasks(),
        )
        sent_bytes = 0

        async def send(message):
            nonlocal sent_bytes
            sent_bytes += len(message.get("body", b""))

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        wrapped = main.LoggingStreamingResponse(
            response.body_iterator,
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type,
            current_info=current_info,
            usage_buffer_limit_bytes=main.RESPONSES_CANONICAL_EVENT_MAX_BYTES,
        )
        await wrapped(
            {"type": "http", "method": "POST", "path": "/v1/responses"},
            receive,
            send,
        )
        return response.status_code, sent_bytes

    try:
        status_code, sent_bytes = asyncio.run(run())
    finally:
        main.request_info.reset(request_token)

    assert status_code == 200
    assert sent_bytes > main.DEFAULT_MAX_EVENT_BYTES
    assert current_info["success"] is True
    assert current_info["prompt_tokens"] == 1
    assert current_info["completion_tokens"] == 1
    assert current_info["total_tokens"] == 2
    assert current_info["usage_seen"] is True
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["upstream_terminal_validated"] is True
    assert diagnostics["downstream_usage_observer_status"] == "completed"
    assert diagnostics["downstream_usage_seen"] is True


def test_responses_event_only_terminal_is_ignored_but_empty_data_is_rejected(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[b"event: response.completed\n\n"],
        )
    )
    ignored_info = {
        "request_id": "event-only-terminal",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    ignored_response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=ignored_info,
    )

    assert ignored_response.status_code == 502
    ignored_diagnostics = ignored_info["responses_stream_diagnostics"]
    assert ignored_diagnostics["ignored_no_data_event_count"] == 1
    assert ignored_diagnostics["terminal_frame_seen"] is False

    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[b"event: response.completed\ndata:\n\n"],
        )
    )
    empty_info = {
        "request_id": "empty-data-terminal",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    empty_response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=empty_info,
    )

    assert empty_response.status_code == 502
    empty_diagnostics = empty_info["responses_stream_diagnostics"]
    assert empty_diagnostics["ignored_no_data_event_count"] == 0
    assert empty_diagnostics["terminal_frame_seen"] is True
    assert empty_diagnostics["diagnosis"] == "responses_sse_protocol_error"


def test_responses_drops_no_data_control_block_before_data_only_terminal(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    terminal = _data_only_responses_sse(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        }
    )
    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(
            chunks=[
                b"event: response.completed\nid: 42\nretry: 1000\n\n"
                + terminal
            ]
        )
    )
    current_info = {
        "request_id": "no-data-control-block",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "id: 42" not in body
    assert "retry: 1000" not in body
    assert "event: response.completed\ndata:" in body
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["ignored_no_data_event_count"] == 1
    assert diagnostics["canonicalized_data_only_event_count"] == 1


def test_responses_rejects_conflicting_wire_and_payload_event_types(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(
            chunks=[
                _responses_sse(
                    "response.completed",
                    {
                        "type": "response.created",
                        "response": {
                            "status": "in_progress",
                            "usage": None,
                        },
                    },
                )
            ]
        )
    )
    current_info = {
        "request_id": "conflicting-event-types",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 502
    assert "conflicts with data.type" in body
    assert current_info["success"] is False
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["upstream_terminal_validated"] is False
    assert diagnostics["diagnosis"] == "responses_sse_protocol_error"


def test_responses_rejects_explicit_empty_wire_event_type(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                b"event:\n"
                b'data: {"type":"response.completed","response":'
                b'{"status":"completed","output":[]}}\n\n'
            ],
        )
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info={
            "request_id": "empty-wire-event-type",
            "api_key": "sk-test",
            "disconnect_event": None,
        },
    )

    assert response.status_code == 502
    assert "event field must not be empty" in body


@pytest.mark.parametrize(
    "payload",
    [
        {"response": {"status": "in_progress"}},
        {"type": 123, "response": {"status": "in_progress"}},
        {"type": {"name": "response.created"}},
        "not-a-json-object",
    ],
)
def test_responses_rejects_data_only_event_without_valid_string_type(
    monkeypatch,
    payload,
):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[_data_only_responses_sse(payload)],
        )
    )

    response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info={
            "request_id": "invalid-data-only-event-type",
            "api_key": "sk-test",
            "disconnect_event": None,
        },
    )

    assert response.status_code == 502


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"])
def test_responses_normalizes_split_terminal_for_all_sse_line_endings(
    monkeypatch,
    line_ending,
):
    _configure_responses_test(monkeypatch, engine="codex")
    wire = _split_responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            },
        },
    ).replace(b"\n", line_ending)
    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(chunks=[bytes((value,)) for value in wire])
    )
    current_info = {
        "request_id": f"split-terminal-{line_ending!r}",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "event: error" not in body
    assert "event: response.completed\ndata: {" in body
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["upstream_terminal_validated"] is True
    assert diagnostics["usage_seen"] is True
    assert diagnostics["ignored_no_data_event_count"] == 1
    assert diagnostics["canonicalized_data_only_event_count"] == 1


def test_responses_split_error_waits_for_data_payload(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _split_responses_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "status_code": 429,
                            "message": "provider rate limited",
                        },
                    },
                )
            ]
        )
    )
    current_info = {
        "request_id": "split-error",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 429
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["ignored_no_data_event_count"] == 1
    assert diagnostics["semantic_status"] == "failed"


@pytest.mark.parametrize("coalesced", [False, True])
def test_responses_failure_terminal_is_forwarded_at_event_boundaries(
    monkeypatch,
    coalesced,
):
    _configure_responses_test(monkeypatch, engine="codex")
    delta0 = _responses_sse(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "delta": "zero"},
    )
    delta1 = _responses_sse(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "delta": "one"},
    )
    failed = _responses_sse(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "provider terminal failure",
                },
            },
        },
    )
    tail = [delta1 + failed] if coalesced else [delta1, failed]
    main.app.state.client_manager = DummyClientManager(
        TerminalReadTrapResponse(chunks=[delta0, *tail])
    )
    current_info = {
        "request_id": f"failed-boundary-{coalesced}",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert '"delta": "zero"' in body or '"delta":"zero"' in body
    assert '"delta": "one"' in body or '"delta":"one"' in body
    assert "response.failed" in body
    assert "provider terminal failure" in body
    assert "event: error" not in body
    assert current_info["success"] is False
    assert current_info["stream_outcome"] == "upstream_failure_terminal"


def test_oaix_keepalive_then_context_failure_preserves_response_failed_terminal(
    monkeypatch,
):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.config["api_keys"][0]["preferences"]["AUTO_RETRY"] = True
    client_manager.response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse(
                "keepalive",
                {"type": "keepalive", "sequence_number": 0},
            ),
            _responses_sse(
                "response.failed",
                {
                    "type": "response.failed",
                    "sequence_number": 4,
                    "response": {
                        "id": "resp_context_failure",
                        "object": "response",
                        "model": "gpt-5.4",
                        "status": "failed",
                        "error": {
                            "code": "context_length_exceeded",
                            "type": "invalid_request_error",
                            "message": "Your input exceeds the context window.",
                            "param": "input",
                        },
                    },
                },
            ),
        ]
    )
    current_info = {
        "request_id": "oaix-context-failure",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert body.count("event: response.failed") == 1
    assert body.count('"type":"response.failed"') == 1
    assert '"code":"context_length_exceeded"' in body
    assert '"message":"Your input exceeds the context window."' in body
    assert "event: error" not in body
    assert "data: [DONE]" not in body
    assert len(client_manager.stream_calls) == 1
    assert current_info["stream_error_status_code"] == 400
    assert current_info["stream_error_event_type"] == "response.failed"
    assert current_info["stream_outcome"] == "upstream_failure_terminal"
    assert current_info["routing_attempts"][-1]["retry_decision"] is False


@pytest.mark.parametrize("data_only_error", [False, True])
def test_oaix_keepalive_then_provider_error_normalizes_response_failed_terminal(
    monkeypatch,
    data_only_error,
):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.config["api_keys"][0]["preferences"]["AUTO_RETRY"] = True
    error_payload = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "code": "context_length_exceeded",
            "type": "invalid_request_error",
            "message": (
                "Your input exceeds the context window of this model. "
                "Please adjust your input and try again."
            ),
            "param": "input",
        },
    }
    error_frame = (
        f"data: {json.dumps(error_payload, separators=(',', ':'))}\n\n".encode()
        if data_only_error
        else _responses_sse("error", error_payload)
    )
    client_manager.response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse(
                "keepalive",
                {"type": "keepalive", "sequence_number": 0},
            ),
            error_frame,
        ]
    )
    current_info = {
        "request_id": f"oaix-provider-context-error-{data_only_error}",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert body.count("event: response.failed") == 1
    assert body.count('"type":"response.failed"') == 1
    assert '"sequence_number":2' in body
    assert '"code":"context_length_exceeded"' in body
    assert '"type":"invalid_request_error"' in body
    assert '"param":"input"' in body
    assert "event: error" not in body
    assert "data: [DONE]" not in body
    assert len(client_manager.stream_calls) == 1
    assert current_info["stream_error_status_code"] == 400
    assert current_info["stream_error_event_type"] == "response.failed"
    assert current_info["stream_outcome"] == "upstream_failure_terminal"
    routing_attempt = current_info["routing_attempts"][-1]
    assert routing_attempt["wire_status_code"] == 200
    assert routing_attempt["semantic_status_code"] == 400
    assert routing_attempt["terminal_event_type"] == "error"
    assert routing_attempt["retry_decision"] is False
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["declared_terminal_type"] == "error"
    assert diagnostics["provider_error_to_response_failed_count"] == 1
    assert diagnostics["last_normalization_rule"] == (
        "provider_error_to_response_failed"
    )
    assert diagnostics["last_normalized_event_type"] == "error"


def test_context_failure_before_commit_remains_http_400(monkeypatch):
    client_manager = _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.config["api_keys"][0]["preferences"]["AUTO_RETRY"] = True
    client_manager.response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "context_length_exceeded",
                            "type": "invalid_request_error",
                            "message": "input is too long",
                        },
                    },
                },
            )
        ]
    )

    current_info = {
        "request_id": "context-failure-before-commit",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 400
    assert json.loads(body) == {
        "error": {
            "code": "context_length_exceeded",
            "type": "invalid_request_error",
            "message": "input is too long",
        }
    }
    assert len(client_manager.stream_calls) == 1


def test_retry_success_does_not_leak_previous_response_failed_terminal(
    monkeypatch,
):
    client_manager = _configure_two_provider_responses_test(
        monkeypatch,
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse(
                        "keepalive",
                        {"type": "keepalive", "sequence_number": 0},
                    ),
                    _responses_sse(
                        "response.failed",
                        {
                            "type": "response.failed",
                            "response": {
                                "status": "failed",
                                "error": {
                                    "code": "rate_limit_exceeded",
                                    "message": "retry provider a",
                                },
                            },
                        },
                    ),
                ]
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "delta": "provider-b-success",
                        },
                    ),
                    _responses_sse(
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": {
                                "status": "completed",
                                "usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 1,
                                    "total_tokens": 2,
                                },
                            },
                        },
                    ),
                ]
            ),
        },
    )

    current_info = {
        "request_id": "retry-success-clears-failure-terminal",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "provider-b-success" in body
    assert "response.failed" not in body
    assert "retry provider a" not in body
    assert current_info.get("stream_error_status_code") is None
    assert current_info.get("stream_error_event_type") is None
    assert current_info.get("stream_error_code") is None
    assert current_info.get("stream_outcome") != "upstream_failure_terminal"
    assert current_info["success"] is True
    assert [call["url"] for call in client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
    ]


def test_transport_failure_does_not_reuse_previous_response_failed_terminal(
    monkeypatch,
):
    _configure_two_provider_responses_test(
        monkeypatch,
        {
            "https://provider-a.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse(
                        "keepalive",
                        {"type": "keepalive", "sequence_number": 0},
                    ),
                    _responses_sse(
                        "response.failed",
                        {
                            "type": "response.failed",
                            "response": {
                                "status": "failed",
                                "error": {
                                    "code": "rate_limit_exceeded",
                                    "message": "stale provider failure",
                                },
                            },
                        },
                    ),
                ]
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                stream_error=httpx.ReadError(
                    "provider b connection closed",
                    request=httpx.Request(
                        "POST",
                        "https://provider-b.example/v1/responses",
                    ),
                )
            ),
        },
    )

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True)
    )

    assert response.status_code == 200
    assert "event: error" in body
    assert "data: [DONE]" in body
    assert "response.failed" not in body
    assert "stale provider failure" not in body


def test_responses_request_scoped_failure_terminal_does_not_mark_channel_failure(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    local_byte_budget = runtime.RetainedByteBudget(
        capacity_bytes=1 << 20,
        wait_timeout_seconds=0.25,
    )
    monkeypatch.setattr(runtime, "responses_stream_byte_budget", local_byte_budget)
    channel_results = []

    def record(*_args, success, **_kwargs):
        channel_results.append(success)

    monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "partial"},
                )
                + _responses_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "code": "context_length_exceeded",
                            "type": "invalid_request_error",
                            "message": "context window exceeded",
                        },
                    },
                )
            ]
        )
    )
    current_info = {
        "request_id": "request-scoped-failure-terminal",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "context_length_exceeded" in body
    assert body.count("event: response.failed") == 1
    assert "event: error" not in body
    assert "data: [DONE]" not in body
    assert channel_results == []
    assert current_info["stream_error_status_code"] == 400
    assert current_info["stream_error_code"] == "context_length_exceeded"
    assert current_info["stream_error_type"] == "invalid_request_error"
    assert current_info["stream_error_event_type"] == "response.failed"
    routing_attempt = current_info["routing_attempts"][-1]
    assert routing_attempt["wire_status_code"] == 200
    assert routing_attempt["semantic_status_code"] == 400
    assert routing_attempt["terminal_event_type"] == "error"
    assert routing_attempt["error_code"] == "context_length_exceeded"
    assert routing_attempt["error_type"] == "invalid_request_error"
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["declared_terminal_type"] == "error"
    assert diagnostics["provider_error_to_response_failed_count"] == 1
    assert local_byte_budget.snapshot().used_bytes == 0


def test_responses_malformed_terminal_payload_is_protocol_failure(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                b"event: response.completed\ndata: not-json\n\n",
            ]
        )
    )

    current_info = {
        "request_id": "malformed-terminal",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 502
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["terminal_frame_seen"] is True
    assert diagnostics["upstream_terminal_seen"] is False
    assert diagnostics["upstream_terminal_validated"] is False
    assert diagnostics["semantic_status"] == "error"
    assert diagnostics["diagnosis"] == "responses_sse_protocol_error"


def test_responses_completed_label_with_failed_payload_is_diagnosed_as_failure(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse(
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "failed",
                            "error": {
                                "status_code": 503,
                                "message": "provider failed",
                            },
                        },
                    },
                )
            ]
        )
    )
    current_info = {
        "request_id": "completed-label-failed-payload",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 503
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["declared_terminal_type"] == "response.completed"
    assert diagnostics["semantic_terminal_type"] == "response.completed"
    assert diagnostics["semantic_terminal_outcome"] == "failed"
    assert diagnostics["diagnosis"] == "responses_terminal_semantics_inconsistent"
    assert diagnostics["semantic_status"] == "failed"
    assert diagnostics["terminal_consistency_status"] == "inconsistent"
    assert diagnostics["terminal_semantics_inconsistency"] == [
        "declared_outcome_mismatch"
    ]
    assert "exception_type" not in diagnostics
    assert diagnostics["semantic_status"] == "failed"


def test_responses_failed_label_without_failure_semantics_is_not_a_terminal(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {"status": "in_progress"},
                    },
                )
            ]
        )
    )
    current_info = {
        "request_id": "failed-label-nonfailure-payload",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, _body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 502
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["declared_terminal_type"] == "response.failed"
    assert diagnostics["terminal_frame_semantic_outcome"] == "nonterminal"
    assert diagnostics["upstream_terminal_seen"] is False
    assert diagnostics["semantic_status"] == "error"


@pytest.mark.parametrize(
    "stream_error",
    [
        None,
        httpx.ReadError(
            "upstream aborted",
            request=httpx.Request("POST", "https://example.com/v1/responses"),
        ),
        RuntimeError("unexpected upstream iterator failure"),
    ],
)
def test_responses_postcommit_abort_records_only_channel_failure(
    monkeypatch,
    stream_error,
):
    _configure_responses_test(monkeypatch, engine="codex")
    channel_results = []
    warning_logs = []

    def record_warning(message, *args, **_kwargs):
        warning_logs.append(message % args if args else message)

    def record(*_args, success, **_kwargs):
        channel_results.append(success)

    monkeypatch.setattr(main.trace_logger, "warning", record_warning)
    monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse("response.created", {"type": "response.created"}),
                _responses_sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "partial"},
                ),
            ],
            stream_error=stream_error,
        )
    )
    current_info = {
        "request_id": "terminal-failure",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    _response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert "partial" in body
    assert "event: error" not in body
    assert "data: [DONE]" not in body
    assert channel_results == [False]
    assert current_info["success"] is False
    assert current_info["stream_outcome"] == "upstream_stream_abort"
    assert current_info["stream_error_after_response_start"] is True
    assert current_info["postcommit_stream_terminal_suppressed"] is True
    assert current_info["upstream_attempts"][-1]["success"] is False
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["diagnosis"] != "responses_stream_in_progress"
    assert diagnostics["exception_type"] in {
        "ReadError",
        "RuntimeError",
        "SSEProtocolError",
    }
    assert any(
        "upstream stream finished without completed usage" in message
        for message in warning_logs
    )


def test_responses_postcommit_partial_terminal_read_error_is_fully_diagnostic(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    partial_terminal = (
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"usage":'
        b'{"input_tokens":1,"output_tokens":1}'
    )
    main.app.state.client_manager = DummyClientManager(
        DummyStreamingUpstreamResponse(
            chunks=[
                _responses_sse("response.created", {"type": "response.created"}),
                _responses_sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "partial"},
                ),
                partial_terminal,
            ],
            stream_error=_chained_responses_read_error(),
            headers={"X-OAIX-Connection-ID": "oaixc-partial-reset"},
        )
    )
    current_info = {
        "request_id": "partial-terminal-reset",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "partial" in body
    assert "event: error" not in body
    assert "data: [DONE]" not in body
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["oaix_connection_id"] == "oaixc-partial-reset"
    assert diagnostics["upstream_terminal_seen"] is False
    assert diagnostics["usage_seen"] is False
    assert diagnostics["last_event_type"] == "response.output_text.delta"
    assert diagnostics["partial_event_bytes"] == len(partial_terminal)
    assert diagnostics["partial_event_sha256"] == hashlib.sha256(
        partial_terminal
    ).hexdigest()
    assert diagnostics["exception_type"] == "ReadError"
    assert diagnostics["exception_errno"] == errno.ECONNRESET
    assert diagnostics["exception_errno_name"] == "ECONNRESET"
    assert [row["type"] for row in diagnostics["exception_chain"]] == [
        "ReadError",
        "ReadError",
        "ConnectionResetError",
    ]
    assert diagnostics["diagnosis"] == "responses_partial_event_abort"
    assert diagnostics["cleanup_owner"] == "responses_proxy_finally"
    assert diagnostics["cleanup_trigger"] == (
        "after_upstream_read_or_stream_failure"
    )


def test_responses_stream_closes_entered_upstream_response_before_context(monkeypatch):
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

    upstream = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                },
            ),
            _responses_sse(None, "[DONE]"),
        ]
    )
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
    main.app.state.client_manager = DummyClientManager(upstream)

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert response.status_code == 200
    assert "hello" in body
    assert upstream.close_calls == 1
    assert upstream.context_exit_calls == 1
    assert upstream.close_events[-2:] == ["response_aclose", "context_exit"]


def test_responses_stream_closes_upstream_when_downstream_closes_after_commit(monkeypatch):
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

    upstream = BlockingStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
        ]
    )
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
    main.app.state.client_manager = DummyClientManager(upstream)
    current_info = {
        "request_id": "req-test",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    request_token = main.request_info.set(current_info)

    async def _run():
        handler = main.ResponsesRequestHandler()
        response = await handler.request_responses(
            http_request=SimpleNamespace(headers={}),
            request_data=ResponsesRequest(
                model="gpt-5.4",
                input=[{"role": "user", "content": "hello"}],
                stream=True,
            ),
            api_index=0,
            background_tasks=BackgroundTasks(),
        )
        chunks = []
        body_iterator = response.body_iterator
        for _ in range(2):
            chunk = await anext(body_iterator)
            chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
        await asyncio.sleep(0)
        await body_iterator.aclose()
        await asyncio.sleep(0)
        return "".join(chunks)

    try:
        body = asyncio.run(_run())
    finally:
        main.request_info.reset(request_token)

    assert "hello" in body
    assert upstream.close_calls == 1
    assert upstream.context_exit_calls == 1
    assert upstream.close_events[-2:] == ["response_aclose", "context_exit"]
    assert current_info["routing_attempts"][-1]["outcome"] == (
        "consumer_or_shutdown_unknown"
    )
    assert "success" not in current_info["routing_attempts"][-1]


def test_responses_stream_retry_closes_failed_upstream_response(monkeypatch):
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
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))

    upstream_a = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
            _responses_sse("response.in_progress", {"type": "response.in_progress", "provider": "a"}),
        ],
        stream_error=httpx.ReadTimeout(
            "upstream stalled",
            request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
        ),
        headers={"X-OAIX-Connection-ID": "oaixc-retry-a"},
    )
    upstream_b = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created", "provider": "b"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        }
                    },
                },
            ),
            _responses_sse(None, "[DONE]"),
        ],
        headers={"X-OAIX-Connection-ID": "oaixc-retry-b"},
    )
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
            "https://provider-a.example/v1/responses": upstream_a,
            "https://provider-b.example/v1/responses": upstream_b,
        }
    )

    current_info = {
        "request_id": "precommit-retry-isolation",
        "api_key": "sk-test",
        "disconnect_event": None,
    }
    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="gpt-5.4",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        ),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert '"provider": "a"' not in body
    assert '"provider": "b"' in body
    assert upstream_a.close_events[-2:] == ["response_aclose", "context_exit"]
    assert upstream_a.close_calls == 1
    assert upstream_a.context_exit_calls == 1
    assert upstream_b.close_calls == 1
    assert upstream_b.context_exit_calls == 1
    attempts = current_info["upstream_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["stream_diagnostics"]["oaix_connection_id"] == (
        "oaixc-retry-a"
    )
    assert attempts[0]["stream_diagnostics"]["exception_type"] == "ReadTimeout"
    assert attempts[1]["stream_diagnostics"]["oaix_connection_id"] == (
        "oaixc-retry-b"
    )
    assert attempts[1]["stream_diagnostics"]["upstream_terminal_seen"] is True
    assert current_info["responses_stream_diagnostics"] is attempts[1][
        "stream_diagnostics"
    ]
    assert "exception_type" not in current_info["responses_stream_diagnostics"]


def test_responses_stream_keepalive_does_not_commit_and_retries(monkeypatch):
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
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))

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
                        _responses_sse("keepalive", {"type": "keepalive", "sequence_number": 0}),
                    ],
                headers={
                    "X-OAIX-Request-ID": "req_provider_a",
                    "X-OAIX-Token-ID": "111",
                    "X-OAIX-Token-Owner-User-ID": "222",
                },
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
                            "response": {
                                "usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 1,
                                    "total_tokens": 2,
                                }
                            },
                        },
                    ),
                    _responses_sse(None, "[DONE]"),
                ],
                headers={
                    "X-OAIX-Request-ID": "req_provider_b",
                    "X-OAIX-Token-ID": "333",
                    "X-OAIX-Token-Owner-User-ID": "444",
                },
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
    assert response.headers["x-oaix-request-id"] == "req_provider_b"
    assert response.headers["x-oaix-token-id"] == "333"
    assert response.headers["x-oaix-token-owner-user-id"] == "444"
    assert body.startswith('event: keepalive\ndata: {"type":"keepalive","sequence_number":0}')
    assert body.count("event: keepalive") == 1
    assert '"provider": "a"' not in body
    assert '"provider": "b"' in body
    assert "hello-b" in body
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
    ]


def test_responses_stream_forwards_initial_upstream_keepalive_once_and_retries(monkeypatch):
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
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))

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
                    _responses_sse("keepalive", {"type": "keepalive", "sequence_number": 0}),
                    _responses_sse("response.created", {"type": "response.created", "provider": "a"}),
                ],
                stream_error=httpx.ReadTimeout(
                    "upstream stalled",
                    request=httpx.Request("POST", "https://provider-a.example/v1/responses"),
                ),
            ),
            "https://provider-b.example/v1/responses": DummyStreamingUpstreamResponse(
                chunks=[
                    _responses_sse("response.created", {"type": "response.created", "provider": "b"}),
                    _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello-b"}),
                    _responses_sse(
                        "response.completed",
                        {
                            "type": "response.completed",
                            "response": {
                                "usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 1,
                                    "total_tokens": 2,
                                }
                            },
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
    first_event = body.split("\n\n", 1)[0]
    assert first_event.startswith("event: keepalive\ndata: ")
    assert json.loads(first_event.split("data: ", 1)[1]) == {
        "type": "keepalive",
        "sequence_number": 0,
    }
    assert body.count("event: keepalive") == 1
    assert '"provider": "a"' not in body
    assert '"provider": "b"' in body
    assert "hello-b" in body
    assert [call["url"] for call in main.app.state.client_manager.stream_calls] == [
        "https://provider-a.example/v1/responses",
        "https://provider-b.example/v1/responses",
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
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))

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
    monkeypatch.setattr(main, "get_engine", lambda provider, endpoint=None, original_model=None: ("codex", None))

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
    monkeypatch.setattr(
        main.app.state,
        "channel_manager",
        main.ChannelManager(cooldown_period=300),
        raising=False,
    )
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
    assert asyncio.run(
        main.app.state.channel_manager.is_model_excluded(provider_a, "gpt-5.4", 300)
    )


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
    assert "event: error" not in body
    assert "data: [DONE]" not in body
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
    current_info = {
        "request_id": "req-test",
        "api_key": "sk-test",
        "disconnect_event": None,
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(
            model="friendly-model",
            input=[{"role": "user", "content": "hello"}],
            stream=True,
        ),
        endpoint="/v1/responses/compact",
        current_info=current_info,
    )

    assert response.status_code == 200
    assert "hello-a" in body
    assert "event: error" not in body
    assert "data: [DONE]" not in body
    assert any("/v1/responses/compact upstream stream aborted stage=post-commit" in log for log in warning_logs)
    assert any("error_type=RemoteProtocolError" in log for log in warning_logs)
    assert any("request_model=friendly-model" in log for log in warning_logs)
    assert any("actual_model=gpt-5.4" in log for log in warning_logs)
    assert any("request_id=req-test" in log for log in warning_logs)
    assert any("upstream_url=https://provider-a.example/v1/responses/compact" in log for log in warning_logs)
    assert current_info["transport_error_kind"] == "remote_protocol_error"
    assert current_info["transport_error_owner"] == "upstream_transport"
    assert current_info["transport_error_phase"] == "postcommit"
    assert current_info["transport_error_count"] == 1


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


def test_responses_non_stream_preserves_oaix_response_headers(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    upstream_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.com/v1/responses"),
        headers={
            "X-OAIX-Request-ID": "req_123",
            "X-OAIX-Token-ID": "8662",
            "X-OAIX-Token-Owner-User-ID": "63910",
        },
        json={"id": "resp_ok"},
    )
    main.app.state.client_manager = DummyClientManager(upstream_response)

    response = _run_responses_request(
        ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=False)
    )

    assert response.status_code == 200
    assert response.headers["x-oaix-request-id"] == "req_123"
    assert response.headers["x-oaix-token-id"] == "8662"
    assert response.headers["x-oaix-token-owner-user-id"] == "63910"


def test_responses_stream_preserves_oaix_response_headers(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    upstream_response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                },
            ),
            _responses_sse(None, "[DONE]"),
        ],
        headers={
            "X-OAIX-Request-ID": "req_stream",
            "X-OAIX-Token-ID": "8662",
            "X-OAIX-Token-Owner-User-ID": "63910",
        },
    )
    main.app.state.client_manager = DummyClientManager(upstream_response)

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=True)
    )

    assert "hello" in body
    assert response.status_code == 200
    assert response.headers["x-oaix-request-id"] == "req_stream"
    assert response.headers["x-oaix-token-id"] == "8662"
    assert response.headers["x-oaix-token-owner-user-id"] == "63910"


def test_responses_stream_preserves_oaix_headers_after_precommit_yield(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    upstream_response = YieldingStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                },
            ),
            _responses_sse(None, "[DONE]"),
        ],
        headers={
            "X-OAIX-Request-ID": "req_yield",
            "X-OAIX-Token-ID": "8685",
            "X-OAIX-Token-Owner-User-ID": "51851",
        },
    )
    main.app.state.client_manager = DummyClientManager(upstream_response)

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=True)
    )

    assert "hello" in body
    assert response.status_code == 200
    assert response.headers["x-oaix-request-id"] == "req_yield"
    assert response.headers["x-oaix-token-id"] == "8685"
    assert response.headers["x-oaix-token-owner-user-id"] == "51851"


def test_responses_live_disconnect_does_not_rewrite_client_close_as_sse_502(
    monkeypatch,
):
    _configure_responses_test(monkeypatch, engine="codex")
    channel_results = []
    info_logs = []
    warning_logs = []

    def record_info(message, *args, **_kwargs):
        info_logs.append(message % args if args else message)

    def record_warning(message, *args, **_kwargs):
        warning_logs.append(message % args if args else message)

    def record(*_args, success, **_kwargs):
        channel_results.append(success)

    monkeypatch.setattr(main.trace_logger, "info", record_info)
    monkeypatch.setattr(main.trace_logger, "warning", record_warning)
    monkeypatch.setattr(main, "_schedule_channel_stats_bounded", record)
    disconnect_event = asyncio.Event()
    upstream_response = DisconnectingLiveStreamingUpstreamResponse(
        disconnect_event=disconnect_event,
        disconnect_before_index=2,
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "first"},
            ),
            _responses_sse(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "after-close"},
            ),
        ],
    )
    main.app.state.client_manager = DummyClientManager(upstream_response)
    current_info = {
        "request_id": "req-live-disconnect",
        "api_key": "sk-test",
        "disconnect_event": disconnect_event,
    }

    _response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True),
        current_info=current_info,
    )

    assert "first" in body
    assert "after-close" not in body
    assert "event: error" not in body
    assert current_info.get("stream_outcome") != "upstream_stream_abort"
    assert current_info.get("status_code") != 502
    assert current_info["stream_outcome"] == "downstream_disconnected"
    assert current_info["downstream_disconnected"] is True
    assert current_info["success"] is False
    assert channel_results == []
    assert current_info["upstream_attempts"][-1]["status_code"] == 499
    assert upstream_response.close_calls >= 1
    diagnostics = current_info["responses_stream_diagnostics"]
    assert diagnostics["downstream_disconnected"] is True
    assert diagnostics["downstream_disconnect_stage"] == "after-stream-commit"
    assert diagnostics["diagnosis"] == "responses_downstream_disconnect"
    assert diagnostics["cleanup_trigger"] == "downstream_disconnect"
    assert any(
        "upstream read cancelled after downstream disconnect before completed usage"
        in message
        for message in info_logs
    )
    assert not any(
        "upstream stream finished without completed usage" in message
        for message in warning_logs
    )


def test_responses_stream_emits_oaix_keepalive_before_real_output(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.provider_timeouts = {"global": {"gpt-5.4": 20, "default": 30}}
    upstream_response = DelayedFirstChunkStreamingUpstreamResponse(
        chunks=[
            _responses_sse("keepalive", {"type": "keepalive", "sequence_number": 0}),
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                },
            ),
            _responses_sse(None, "[DONE]"),
        ],
        headers={
            "X-OAIX-Request-ID": "req_keepalive",
            "X-OAIX-Token-ID": "8686",
            "X-OAIX-Connection-ID": "oaixc-keepalive",
        },
    )
    client_manager = DummyClientManager(upstream_response)
    main.app.state.client_manager = client_manager
    current_info = {
        "request_id": "req-keepalive",
        "api_key": "sk-test",
        "disconnect_event": None,
        "trace": main.RequestTrace(trace_id="req-keepalive"),
    }

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=True),
        current_info=current_info,
    )

    assert response.status_code == 200
    assert response.headers["x-oaix-request-id"] == "req_keepalive"
    assert response.headers["x-oaix-token-id"] == "8686"
    assert response.headers["x-oaix-connection-id"] == "oaixc-keepalive"
    assert current_info["responses_stream_diagnostics"][
        "oaix_connection_id"
    ] == "oaixc-keepalive"
    assert body.startswith("event: keepalive\ndata: ")
    assert '"type": "keepalive"' in body.split("\n\n", 1)[0]
    assert '"sequence_number": 0' in body.split("\n\n", 1)[0]
    assert body.index("event: keepalive") < body.index("event: response.created")
    assert body.index("event: response.created") < body.index("event: response.output_text.delta")
    assert client_manager.stream_calls[0]["timeout"] is None
    assert current_info["timing_spans"]["upstream_first_chunk"] >= 1


def test_gpt_responses_stream_starts_with_response_created_without_keepalive(monkeypatch):
    _configure_responses_test(monkeypatch, engine="gpt")
    upstream_response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "hello"},
            ),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        }
                    },
                },
            ),
        ]
    )
    main.app.state.client_manager = DummyClientManager(upstream_response)

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello"], stream=True)
    )

    assert response.status_code == 200
    assert body.startswith("event: response.created\ndata: ")
    assert "event: keepalive" not in body


def test_responses_stream_observability_uses_request_state_current_info(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    upstream_response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                },
            ),
            _responses_sse(None, "[DONE]"),
        ]
    )
    main.app.state.client_manager = DummyClientManager(upstream_response)
    state_info = {
        "request_id": "req-state",
        "trace_id": "11111111111111111111111111111111",
        "api_key": "sk-test",
        "disconnect_event": None,
        "timing_spans": {"request_received": 0, "body_parsed": 5},
    }
    context_info = {
        "request_id": "req-context",
        "trace_id": "22222222222222222222222222222222",
        "api_key": "sk-test",
        "disconnect_event": None,
        "timing_spans": {"request_received": 0, "body_parsed": 9},
    }
    request_token = main.request_info.set(context_info)

    async def run_test():
        handler = main.ResponsesRequestHandler()
        response = await handler.request_responses(
            http_request=SimpleNamespace(
                headers={},
                state=SimpleNamespace(uni_api_request_info=state_info),
            ),
            request_data=ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=True),
            api_index=0,
            background_tasks=BackgroundTasks(),
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
        return b"".join(chunks).decode("utf-8")

    try:
        body = asyncio.run(run_test())
    finally:
        main.request_info.reset(request_token)

    assert "hello" in body
    assert state_info["provider"] == "codex-provider"
    assert isinstance(state_info["trace"], main.RequestTrace)
    assert state_info["timing_spans"]["body_parsed"] == 5
    assert state_info["timing_spans"]["provider_selected"] >= 1
    assert state_info["timing_spans"]["upstream_headers_received"] >= 1
    assert state_info["timing_spans"]["upstream_first_chunk"] >= 1
    assert context_info.get("provider") is None


def test_responses_stream_uses_explicit_idle_as_httpx_read_timeout(monkeypatch):
    _configure_responses_test(monkeypatch, engine="codex")
    main.app.state.provider_timeouts = {"global": {"gpt-5.4": 20, "default": 30}}
    main.app.state.timeout_policy = main.init_timeout_policy(
        {
            "preferences": {
                "timeout_policy": {
                    "rules": [
                        {
                            "match": {"endpoint": "/v1/responses", "stream": True, "model": "gpt-5.4"},
                            "timeout": {"first_byte": 20, "idle": 120, "total": 300},
                        }
                    ]
                }
            }
        }
    )
    upstream_response = DummyStreamingUpstreamResponse(
        chunks=[
            _responses_sse("response.created", {"type": "response.created"}),
            _responses_sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hello"}),
            _responses_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
                },
            ),
            _responses_sse(None, "[DONE]"),
        ],
    )
    client_manager = DummyClientManager(upstream_response)
    main.app.state.client_manager = client_manager

    response, body = _run_responses_request_with_stream_body(
        ResponsesRequest(model="gpt-5.4", input=["hello world"], stream=True)
    )

    assert response.status_code == 200
    assert "hello" in body
    timeout = client_manager.stream_calls[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 120


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


def test_responses_non_stream_model_pricing_400_retries_next_provider(monkeypatch):
    provider_a = "responses-pricing-a"
    provider_b = "responses-pricing-b"
    monkeypatch.setitem(main.provider_api_circular_list, provider_a, DummyCircularList(["key-a"]))
    monkeypatch.setitem(main.provider_api_circular_list, provider_b, DummyCircularList(["key-b"]))

    async def fake_get_right_order_providers(request_model_name, config, api_index, scheduling_algorithm):
        return [
            {
                "provider": provider_a,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://responses-pricing-a.example/v1/responses",
                "api": ["key-a"],
                "preferences": {},
            },
            {
                "provider": provider_b,
                "_model_dict_cache": {"gpt-5.4": "gpt-5.4"},
                "base_url": "https://responses-pricing-b.example/v1/responses",
                "api": ["key-b"],
                "preferences": {},
            },
        ]

    monkeypatch.setattr(main, "get_right_order_providers", fake_get_right_order_providers)
    monkeypatch.setattr(
        main,
        "get_engine",
        lambda provider, endpoint=None, original_model=None: ("gpt", None),
    )

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
            "https://responses-pricing-a.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://responses-pricing-a.example/v1/responses",
                ),
                json={
                    "id": "resp-pricing-a",
                    "status": "failed",
                    "error": {
                        "type": "upstream_error",
                        "code": "model_price_not_configured",
                        "status": 400,
                        "message": "Model pricing has not been configured by the administrator yet.",
                    },
                },
            ),
            "https://responses-pricing-b.example/v1/responses": httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://responses-pricing-b.example/v1/responses",
                ),
                json={"id": "resp-pricing-b", "status": "completed"},
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
    assert json.loads(response.body) == {
        "id": "resp-pricing-b",
        "status": "completed",
    }
    assert [call["url"] for call in main.app.state.client_manager.post_calls] == [
        "https://responses-pricing-a.example/v1/responses",
        "https://responses-pricing-b.example/v1/responses",
    ]
