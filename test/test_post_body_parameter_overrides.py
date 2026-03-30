import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import RequestModel
from core.request import get_payload


def test_gpt_responses_generic_post_body_overrides_apply():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )
    provider = {
        "provider": "wusan",
        "base_url": "https://example.com/v1/responses",
        "api": "test-key",
        "model": ["gpt-5.4"],
        "preferences": {
            "post_body_parameter_overrides": {
                "store": False,
            }
        },
        "tools": True,
    }

    _, _, payload = asyncio.run(get_payload(request, "gpt", provider, api_key="test-key"))

    assert payload["store"] is False


def test_codex_generic_post_body_overrides_apply():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )
    provider = {
        "provider": "codex",
        "engine": "codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api": "account-id,refresh-token",
        "model": ["gpt-5.4"],
        "preferences": {
            "post_body_parameter_overrides": {
                "store": True,
            }
        },
        "tools": True,
    }

    _, _, payload = asyncio.run(get_payload(request, "codex", provider, api_key="access-token"))

    assert payload["store"] is True


def test_codex_strips_response_format_from_post_body_overrides():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )
    provider = {
        "provider": "codex",
        "engine": "codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api": "account-id,refresh-token",
        "model": ["gpt-5.4"],
        "preferences": {
            "post_body_parameter_overrides": {
                "response_format": {"type": "json_object"},
            }
        },
        "tools": True,
    }

    _, _, payload = asyncio.run(get_payload(request, "codex", provider, api_key="access-token"))

    assert "response_format" not in payload


def test_gemini_reasoning_effort_overrides_post_body_thinking_level():
    request = RequestModel(
        model="gemini-3-flash",
        messages=[{"role": "user", "content": "hello"}],
        reasoning={"effort": "minimal"},
        stream=False,
    )
    provider = {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api": "test-key",
        "model": ["gemini-3-flash"],
        "preferences": {
            "post_body_parameter_overrides": {
                "gemini-3-flash": {
                    "generationConfig": {
                        "thinkingConfig": {
                            "includeThoughts": True,
                            "thinkingLevel": "HIGH",
                        },
                        "maxOutputTokens": 65535,
                    }
                }
            }
        },
    }

    _, _, payload = asyncio.run(get_payload(request, "gemini", provider, api_key="test-key"))

    assert payload["generationConfig"]["maxOutputTokens"] == 65535
    assert payload["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"


def test_vertex_gemini_reasoning_effort_overrides_post_body_thinking_level():
    request = RequestModel(
        model="gemini-3-flash",
        messages=[{"role": "user", "content": "hello"}],
        reasoning={"effort": "minimal"},
        stream=False,
    )
    provider = {
        "provider": "vertex-gemini",
        "base_url": "https://google-vertex-ai.example.com",
        "api": "test-key",
        "project_id": "test-project",
        "model": ["gemini-3-flash"],
        "preferences": {
            "post_body_parameter_overrides": {
                "gemini-3-flash": {
                    "generationConfig": {
                        "thinkingConfig": {
                            "includeThoughts": True,
                            "thinkingLevel": "HIGH",
                        },
                        "maxOutputTokens": 65535,
                    }
                }
            }
        },
    }

    _, _, payload = asyncio.run(get_payload(request, "vertex-gemini", provider, api_key="test-key"))

    assert payload["generationConfig"]["maxOutputTokens"] == 65535
    assert payload["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
