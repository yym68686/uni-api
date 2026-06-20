from core.models import RequestModel
from uni_api.providers.payloads import get_gemini_payload, get_vertex_gemini_payload


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _request_with_reasoning_content() -> RequestModel:
    return RequestModel(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "hello",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "thinking answer",
                        "reasoning_content": "nested reasoning",
                    }
                ],
                "reasoning_content": "message reasoning",
            },
        ],
        stream=False,
        reasoning_content="request reasoning",
    )


async def test_gemini_payload_strips_reasoning_content_recursively():
    provider = {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": ["gemini-2.5-flash"],
    }

    _, _, payload = await get_gemini_payload(
        _request_with_reasoning_content(),
        "gemini",
        provider,
        api_key="test-key",
    )

    assert not _contains_key(payload, "reasoning_content")
    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][1]["parts"][0]["text"] == "thinking answer"


async def test_vertex_gemini_payload_strips_reasoning_content_recursively():
    provider = {
        "provider": "vertex-gemini",
        "base_url": "https://aiplatform.googleapis.com",
        "model": ["gemini-2.5-flash"],
    }

    _, _, payload = await get_vertex_gemini_payload(
        _request_with_reasoning_content(),
        "vertex-gemini",
        provider,
        api_key="ab.c",
    )

    assert not _contains_key(payload, "reasoning_content")
    assert payload["contents"][1]["role"] == "model"
    assert payload["contents"][1]["parts"][0]["text"] == "thinking answer"
