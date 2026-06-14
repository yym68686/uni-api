import asyncio
import base64
import hashlib

import main
import httpx
from starlette.requests import Request

from request_review import RequestReviewDispatcher, build_review_event, is_request_review_enabled


def _request(path: str = "/v1/chat/completions") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"debug=1",
        "headers": [(b"content-type", b"application/json")],
        "server": ("testserver", 80),
        "client": ("203.0.113.10", 12345),
        "scheme": "http",
    }
    return Request(scope)


def test_request_review_enabled_requires_base_url_and_api_key():
    assert not is_request_review_enabled({})
    assert not is_request_review_enabled({"preferences": {"request_review": {"base_url": "http://review00"}}})
    assert not is_request_review_enabled({"preferences": {"request_review": {"api_key": "secret"}}})
    assert not is_request_review_enabled(
        {
            "preferences": {
                "request_review": {
                    "enabled": False,
                    "base_url": "http://review00",
                    "api_key": "secret",
                }
            }
        }
    )
    assert is_request_review_enabled(
        {
            "preferences": {
                "request_review": {
                    "base_url": "http://review00",
                    "api_key": "secret",
                }
            }
        }
    )


def test_build_review_event_uses_raw_body_bytes():
    raw_body = b'{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"hello"}]}'
    event = build_review_event(
        request=_request(),
        current_info={
            "request_id": "req_123",
            "api_key": "sk-secret",
            "client_ip": "198.51.100.5",
        },
        raw_body=raw_body,
        parsed_body={"model": "gpt-test", "stream": True},
    )

    assert event["requestId"] == "req_123"
    assert event["path"] == "/v1/chat/completions"
    assert event["queryString"] == "debug=1"
    assert event["model"] == "gpt-test"
    assert event["stream"] is True
    assert event["sourceIp"] == "198.51.100.5"
    assert event["apiKeyId"] == ""
    assert event["payload"]["encoding"] == "base64"
    assert event["payload"]["sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert event["payload"]["bytes"] == len(raw_body)
    assert base64.b64decode(event["payload"]["data"]) == raw_body


def test_build_review_event_marks_truncated_payload():
    raw_body = b"abcdef"
    event = build_review_event(
        request=_request(),
        current_info={"request_id": "req_123", "api_key": "sk-secret"},
        raw_body=raw_body,
        parsed_body={},
        max_payload_bytes=3,
    )

    assert event["payload"]["truncated"] is True
    assert event["payload"]["bytes"] == 3
    assert event["payload"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert base64.b64decode(event["payload"]["data"]) == b"abc"


def test_dispatcher_enqueue_drops_when_queue_is_full():
    async def run():
        dispatcher = RequestReviewDispatcher(queue_size=1)
        config = {
            "preferences": {
                "request_review": {
                    "base_url": "http://review00",
                    "api_key": "secret",
                }
            }
        }
        assert dispatcher.enqueue(config, {"eventId": "event_1"})
        assert not dispatcher.enqueue(config, {"eventId": "event_2"})
        assert dispatcher.dropped_events == 1

    asyncio.run(run())


def test_api_config_preferences_merge_preserves_existing_review_api_key():
    existing = {
        "preferences": {
            "rate_limit": "999999/min",
            "request_review": {
                "enabled": True,
                "base_url": "http://old-review00",
                "api_key": "existing-secret",
            },
        }
    }
    updates = {
        "preferences": {
            "request_review": {
                "enabled": False,
                "base_url": "http://new-review00",
                "api_key": "",
            }
        }
    }

    merged = main._merge_preferences(existing, updates)

    assert merged["rate_limit"] == "999999/min"
    assert merged["request_review"] == {
        "enabled": False,
        "base_url": "http://new-review00",
        "api_key": "existing-secret",
    }


def test_dispatcher_send_test_posts_synthetic_event(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 202
        text = "accepted"

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return DummyResponse()

    monkeypatch.setattr(httpx, "AsyncClient", DummyClient)

    async def run():
        dispatcher = RequestReviewDispatcher(timeout_seconds=1.0)
        status_code, response_text = await dispatcher.send_test("http://review00.internal:8000", "secret-key")
        assert status_code == 202
        assert response_text == "accepted"

    asyncio.run(run())
    assert captured["url"].endswith("/v1/request-reviews/batch")
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["events"][0]["payload"]["encoding"] == "base64"
