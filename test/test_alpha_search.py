import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import BackgroundTasks

from core.utils import provider_api_circular_list
from uni_api.api.alpha_search import AlphaSearchRequestHandler
from uni_api.middleware.request_decompression import JSON_BODY_PATHS
from uni_api.observability.middleware import StatsMiddleware
from uni_api.rate_limit.key_pool import ProviderKeyPool


class _SequenceClient:
    def __init__(self, manager):
        self.manager = manager

    async def post(self, url, headers=None, content=None, timeout=None):
        if self.manager.delay:
            await asyncio.sleep(self.manager.delay)
        self.manager.calls.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "content": content,
                "timeout": timeout,
            }
        )
        if not self.manager.responses:
            raise AssertionError("unexpected upstream request")
        result = self.manager.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _SequenceClientManager:
    def __init__(self, responses, *, delay=0.0):
        self.responses = list(responses)
        self.delay = delay
        self.calls = []
        self.client_requests = []

    @asynccontextmanager
    async def get_client(self, base_url, proxy=None, http2=None):
        self.client_requests.append(
            {
                "base_url": base_url,
                "proxy": proxy,
                "http2": http2,
            }
        )
        yield _SequenceClient(self)


def _upstream_response(status, body, headers=None):
    return httpx.Response(
        status,
        content=body,
        headers=headers or {},
        request=httpx.Request("POST", "https://upstream.example/v1/alpha/search"),
    )


def _provider(
    name,
    *,
    key,
    base_url=None,
    engine="gpt",
    model=None,
    preferences=None,
    exclude_endpoints=None,
):
    value = {
        "provider": name,
        "base_url": base_url or f"https://{name}.example/v1/responses",
        "api": list(key) if isinstance(key, (list, tuple)) else [key],
        "model": model or ["gpt-5.4"],
        "engine": engine,
        "preferences": dict(preferences or {}),
    }
    if exclude_endpoints is not None:
        value["exclude_endpoints"] = list(exclude_endpoints)
    return value


def _make_handler(
    monkeypatch,
    providers,
    responses,
    *,
    auto_retry=True,
    response_delay=0.0,
):
    for provider in providers:
        pool = ProviderKeyPool(
            list(provider.get("api") or []),
            rate_limit={"default": "999999/min"},
            provider_name=provider["provider"],
        )
        monkeypatch.setitem(
            provider_api_circular_list,
            provider["provider"],
            pool,
        )
    config = {
        "providers": providers,
        "api_keys": [
            {
                "api": "client-api-key",
                "model": ["gpt-5.4"],
                "preferences": {"AUTO_RETRY": auto_retry},
            }
        ],
    }
    manager = _SequenceClientManager(responses, delay=response_delay)
    app = SimpleNamespace(
        state=SimpleNamespace(
            config=config,
            api_list=["client-api-key"],
            models_list={"client-api-key": ["gpt-5.4"]},
            channel_manager=None,
            client_manager=manager,
        )
    )

    async def resolve_codex_upstream_auth(_provider, raw_key, _proxy):
        return f"access-for-{raw_key}", "account-selected"

    handler = AlphaSearchRequestHandler(
        app=app,
        get_runtime_api_list=lambda: ["client-api-key"],
        api_key_has_model_rules=lambda _app, _index: True,
        resolve_codex_upstream_auth=resolve_codex_upstream_auth,
        resolve_timeout=lambda **_kwargs: None,
    )
    return handler, manager


def _request(headers=None):
    return SimpleNamespace(
        headers=dict(headers or {}),
        state=SimpleNamespace(
            uni_api_request_info={
                "request_id": "req-test",
                "api_key": "client-api-key",
            }
        ),
    )


def _run(handler, body, *, headers=None):
    return asyncio.run(
        handler.request_search(
            http_request=_request(headers),
            request_body=body,
            api_index=0,
            background_tasks=BackgroundTasks(),
        )
    )


def test_alpha_search_preserves_request_and_raw_response_contract(monkeypatch):
    raw_response = (
        b'{"encrypted_output":null,"output":"turn0search0","future":true}'
    )
    provider = _provider(
        "provider-codex",
        key="provider-secret",
        engine="codex",
        model=[{"gpt-5.4-upstream": "gpt-5.4"}],
        preferences={
            "headers": {"X-Provider-Configured": "yes"},
            "passthrough_request_headers": [
                "X-Trace-Context",
                "Authorization",
                "X-API-Key",
            ],
        },
    )
    handler, manager = _make_handler(
        monkeypatch,
        [provider],
        [
            _upstream_response(
                200,
                raw_response,
                headers=[
                    ("Content-Type", "application/json"),
                    ("Authorization", "Bearer upstream-secret"),
                    ("X-API-Key", "upstream-api-key"),
                    ("ChatGPT-Account-ID", "upstream-account"),
                    ("X-OAIX-Token-ID", "internal-token-id"),
                    ("X-OAIX-Token-Owner-User-ID", "internal-owner-id"),
                    ("X-OAIX-Request-ID", "oaix-request-id"),
                    ("X-OAIX-Connection-ID", "oaix-connection-id"),
                    ("Set-Cookie", "upstream=must-not-leak"),
                    ("X-Upstream", "preserved"),
                ],
            )
        ],
    )
    body = {
        "id": "session-1",
        "model": "gpt-5.4",
        "commands": {"search_query": [{"q": "news"}]},
        "future_request_field": {"keep": True},
    }

    response = _run(
        handler,
        body,
        headers={
            "Authorization": "Bearer client-secret",
            "X-API-Key": "client-x-api-key",
            "ChatGPT-Account-ID": "client-account",
            "Cookie": "client=cookie",
            "X-OAIX-Selection-Mode": "client-controlled",
            "X-Trace-Context": "trace-me",
        },
    )

    assert response.status_code == 200
    assert response.body == raw_response
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-upstream"] == "preserved"
    assert response.headers["x-oaix-request-id"] == "oaix-request-id"
    assert response.headers["x-oaix-connection-id"] == "oaix-connection-id"
    assert "set-cookie" not in response.headers
    assert "authorization" not in response.headers
    assert "x-api-key" not in response.headers
    assert "chatgpt-account-id" not in response.headers
    assert "x-oaix-token-id" not in response.headers
    assert "x-oaix-token-owner-user-id" not in response.headers
    assert len(manager.calls) == 1
    call = manager.calls[0]
    assert call["url"] == "https://provider-codex.example/v1/alpha/search"
    payload = json.loads(call["content"])
    assert payload == {
        **body,
        "model": "gpt-5.4-upstream",
    }
    assert call["headers"]["Authorization"] == "Bearer access-for-provider-secret"
    assert call["headers"]["Chatgpt-Account-Id"] == "account-selected"
    assert call["headers"]["Session_id"] == "session-1"
    assert call["headers"]["Accept"] == "application/json"
    assert call["headers"]["X-Provider-Configured"] == "yes"
    assert call["headers"]["X-Trace-Context"] == "trace-me"
    assert "client-secret" not in repr(call["headers"])
    assert "client-x-api-key" not in repr(call["headers"])
    assert "client-account" not in repr(call["headers"])
    assert "client-controlled" not in repr(call["headers"])
    assert manager.client_requests[0]["http2"] is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 422])
def test_alpha_search_non_retryable_4xx_is_raw_and_single_attempt(
    monkeypatch,
    status,
):
    raw = f'{{"error":{{"status":{status}}}}}'.encode()
    providers = [
        _provider("provider-a", key="key-a"),
        _provider("provider-b", key="key-b"),
    ]
    handler, manager = _make_handler(
        monkeypatch,
        providers,
        [
            _upstream_response(
                status,
                raw,
                headers=[
                    ("Content-Type", "application/json"),
                    ("Retry-After", "7"),
                    ("Authorization", "Bearer upstream-secret"),
                    ("X-API-Key", "upstream-api-key"),
                    ("ChatGPT-Account-ID", "upstream-account"),
                    ("X-OAIX-Token-ID", "internal-token-id"),
                    ("Set-Cookie", "error=must-not-leak"),
                ],
            )
        ],
    )

    response = _run(
        handler,
        {"id": f"session-{status}", "model": "gpt-5.4"},
    )

    assert response.status_code == status
    assert response.body == raw
    assert response.headers["retry-after"] == "7"
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    assert "authorization" not in response.headers
    assert "x-api-key" not in response.headers
    assert "chatgpt-account-id" not in response.headers
    assert "x-oaix-token-id" not in response.headers
    assert len(manager.calls) == 1


def test_alpha_search_retries_without_binding_subsequent_requests(monkeypatch):
    providers = [
        _provider("provider-a", key="key-a"),
        _provider("provider-b", key="key-b"),
    ]
    handler, manager = _make_handler(
        monkeypatch,
        providers,
        [
            _upstream_response(503, b'{"error":"a unavailable"}'),
            _upstream_response(200, b'{"output":"from-b","future":1}'),
            _upstream_response(200, b'{"output":"from-b-again"}'),
        ],
    )
    body = {"id": "session-bind", "model": "gpt-5.4"}

    first = _run(handler, body)
    second = _run(handler, body)

    assert first.status_code == 200
    assert first.body == b'{"output":"from-b","future":1}'
    assert second.status_code == 200
    assert second.body == b'{"output":"from-b-again"}'
    assert [call["url"] for call in manager.calls] == [
        "https://provider-a.example/v1/alpha/search",
        "https://provider-b.example/v1/alpha/search",
        "https://provider-a.example/v1/alpha/search",
    ]
    assert manager.calls[1]["headers"]["Authorization"] == "Bearer key-b"
    assert manager.calls[2]["headers"]["Authorization"] == "Bearer key-a"


def test_alpha_search_concurrent_requests_rotate_credentials(monkeypatch):
    providers = [
        _provider("provider-a", key=["key-a1", "key-a2"]),
        _provider("provider-b", key="key-b"),
    ]
    handler, manager = _make_handler(
        monkeypatch,
        providers,
        [
            _upstream_response(200, f'{{"output":"result-{index}"}}'.encode())
            for index in range(20)
        ],
        response_delay=0.01,
    )
    body = {"id": "session-concurrent", "model": "gpt-5.4"}

    async def run_all():
        return await asyncio.gather(
            *[
                handler.request_search(
                    http_request=_request(),
                    request_body=body,
                    api_index=0,
                    background_tasks=BackgroundTasks(),
                )
                for _index in range(20)
            ]
        )

    responses = asyncio.run(run_all())

    assert all(response.status_code == 200 for response in responses)
    assert len(manager.calls) == 20
    assert {call["url"] for call in manager.calls} == {
        "https://provider-a.example/v1/alpha/search",
    }
    selected_credentials = {
        call["headers"]["Authorization"] for call in manager.calls
    }
    assert selected_credentials == {
        "Bearer key-a1",
        "Bearer key-a2",
    }


def test_alpha_search_accepts_any_success_body_without_retry(monkeypatch):
    providers = [
        _provider(f"provider-{name}", key=f"key-{name}")
        for name in ("a", "b", "c", "d")
    ]
    handler, manager = _make_handler(
        monkeypatch,
        providers,
        [
            _upstream_response(200, b'{"id":"responses-envelope"}'),
        ],
    )

    response = _run(
        handler,
        {"id": "session-cap", "model": "gpt-5.4"},
    )

    assert response.status_code == 200
    assert response.body == b'{"id":"responses-envelope"}'
    assert len(manager.calls) == 1


def test_alpha_search_accepts_non_special_engine_and_json_body_path(monkeypatch):
    provider = _provider(
        "provider-generic",
        key="generic-key",
        engine="gemini",
    )
    handler, manager = _make_handler(
        monkeypatch,
        [provider],
        [_upstream_response(200, b'{"output":"generic"}')],
    )

    response = _run(
        handler,
        {"id": "session-generic", "model": "gpt-5.4"},
    )

    assert response.status_code == 200
    assert manager.calls[0]["headers"]["Authorization"] == "Bearer generic-key"
    assert "/v1/alpha/search" in JSON_BODY_PATHS


def test_runtime_registers_alpha_search_post_route():
    from uni_api.runtime import app

    matching = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/v1/alpha/search"
    ]
    assert len(matching) == 1
    assert "POST" in matching[0].methods


def test_alpha_search_body_inspection_does_not_log_unknown_request_type(
    monkeypatch,
):
    async def _allow_request(_api_key, _model, _current_info):
        return None

    errors = []
    monkeypatch.setattr(
        "uni_api.observability.middleware.logger.error",
        lambda *args, **_kwargs: errors.append(args),
    )
    middleware = SimpleNamespace(
        dependencies=SimpleNamespace(
            is_video_or_asset_request_path=lambda _path: False,
        ),
        _rate_limit_response=_allow_request,
    )
    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/alpha/search"),
        query_params={},
    )

    moderated_content = asyncio.run(
        StatsMiddleware._rate_limit_and_extract_moderation_text(
            middleware,
            request,
            parsed_body={
                "id": "session-observability",
                "model": "gpt-5.4",
                "commands": {"search_query": [{"q": "news"}]},
            },
            current_info={},
            final_api_key="client-api-key",
        )
    )

    assert moderated_content is None
    assert errors == []


def test_alpha_search_does_not_require_id(monkeypatch):
    handler, manager = _make_handler(
        monkeypatch,
        [_provider("provider-a", key="key-a")],
        [_upstream_response(200, b'{"anything":true}')],
    )

    response = _run(handler, {"model": "gpt-5.4"})

    assert response.status_code == 200
    assert response.body == b'{"anything":true}'
    assert "Session_id" not in manager.calls[0]["headers"]
