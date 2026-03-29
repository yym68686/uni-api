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
