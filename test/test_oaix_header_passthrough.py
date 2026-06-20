import asyncio

import httpx

from uni_api.providers.responses import fetch_response, fetch_response_stream


def header_value(headers: dict[str, str], key: str) -> str:
    lower = key.lower()
    for header_key, value in headers.items():
        if header_key.lower() == lower:
            return value
    raise KeyError(key)


class DummyStreamContext:
    def __init__(self, response: httpx.Response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        await self.response.aclose()


class DummyClient:
    def __init__(self, response: httpx.Response):
        self.response = response

    async def post(self, url, headers=None, content=None, timeout=None):
        _ = url, headers, content, timeout
        return self.response

    def stream(self, method, url, headers=None, content=None, timeout=None):
        _ = method, url, headers, content, timeout
        return DummyStreamContext(self.response)


async def _fetch_response_captures_oaix_headers():
    response = httpx.Response(
        200,
        headers={
            "X-OAIX-Request-ID": "req_123",
            "X-OAIX-Token-ID": "456",
            "X-OAIX-Token-Owner-User-ID": "789",
        },
        json={"id": "chatcmpl-test", "choices": []},
    )
    captured = {}

    chunks = [
        chunk
        async for chunk in fetch_response(
            DummyClient(response),
            "https://oaix.example/v1/chat/completions",
            {},
            {"model": "gpt-test", "messages": []},
            "gpt",
            "gpt-test",
            response_headers_sink=captured.update,
        )
    ]

    assert chunks == [{"id": "chatcmpl-test", "choices": []}]
    assert header_value(captured, "X-OAIX-Request-ID") == "req_123"
    assert header_value(captured, "X-OAIX-Token-ID") == "456"
    assert header_value(captured, "X-OAIX-Token-Owner-User-ID") == "789"


def test_fetch_response_captures_oaix_headers():
    asyncio.run(_fetch_response_captures_oaix_headers())


async def _fetch_response_stream_captures_oaix_headers():
    response = httpx.Response(
        200,
        headers={
            "X-OAIX-Request-ID": "req_stream",
            "X-OAIX-Token-ID": "654",
            "X-OAIX-Token-Owner-User-ID": "987",
        },
        content=b"data: [DONE]\n\n",
    )
    captured = {}

    chunks = [
        chunk
        async for chunk in fetch_response_stream(
            DummyClient(response),
            "https://oaix.example/v1/chat/completions",
            {},
            {"model": "gpt-test", "messages": [], "stream": True},
            "gpt",
            "gpt-test",
            response_headers_sink=captured.update,
        )
    ]

    assert chunks
    assert header_value(captured, "X-OAIX-Request-ID") == "req_stream"
    assert header_value(captured, "X-OAIX-Token-ID") == "654"
    assert header_value(captured, "X-OAIX-Token-Owner-User-ID") == "987"


def test_fetch_response_stream_captures_oaix_headers():
    asyncio.run(_fetch_response_stream_captures_oaix_headers())
