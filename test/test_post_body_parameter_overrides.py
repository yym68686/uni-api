import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import RequestModel
from core.request import apply_post_body_parameter_overrides, get_payload


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


def test_post_body_parameter_overrides_can_remove_global_fields():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
        temperature=0.2,
        stream=False,
    )
    provider = {
        "provider": "wusan",
        "base_url": "https://example.com/v1/responses",
        "api": "test-key",
        "model": ["gpt-5.4"],
        "preferences": {
            "post_body_parameter_overrides": {
                "__remove__": ["response_format", "temperature"],
            }
        },
        "tools": True,
    }

    _, _, payload = asyncio.run(get_payload(request, "gpt", provider, api_key="test-key"))

    assert "response_format" not in payload
    assert "temperature" not in payload


def test_post_body_parameter_overrides_can_remove_model_specific_fields():
    request = RequestModel(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
        stream=False,
    )
    provider = {
        "provider": "wusan",
        "base_url": "https://example.com/v1/responses",
        "api": "test-key",
        "model": ["gpt-5.4"],
        "preferences": {
            "post_body_parameter_overrides": {
                "gpt-5.4": {
                    "__remove__": ["response_format"],
                }
            }
        },
        "tools": True,
    }

    _, _, payload = asyncio.run(get_payload(request, "gpt", provider, api_key="test-key"))

    assert "response_format" not in payload


def test_post_body_parameter_overrides_deep_merge_nested_dicts_without_mutating_provider_config():
    payload = {
        "generationConfig": {
            "temperature": 1,
            "thinkingConfig": {
                "thinkingLevel": "minimal",
            },
        }
    }
    provider = {
        "model": ["gemini-3-flash"],
        "preferences": {
            "post_body_parameter_overrides": {
                "gemini-3-flash": {
                    "generationConfig": {
                        "thinkingConfig": {
                            "includeThoughts": True,
                        }
                    }
                }
            }
        },
    }

    apply_post_body_parameter_overrides(payload, provider, "gemini-3-flash")

    assert payload["generationConfig"]["temperature"] == 1
    assert payload["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert provider["preferences"]["post_body_parameter_overrides"]["gemini-3-flash"]["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
    }


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


def test_gemini_reasoning_effort_is_preserved_for_aliased_gemini_3_latest_models():
    request = RequestModel(
        model="gemini-3-flash",
        messages=[{"role": "user", "content": "hello"}],
        reasoning={"effort": "minimal"},
        temperature=1,
        stream=False,
    )
    provider = {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api": "test-key",
        "model": [{"gemini-flash-latest": "gemini-3-flash"}],
        "preferences": {
            "post_body_parameter_overrides": {
                "gemini-3-flash": {
                    "generationConfig": {
                        "thinkingConfig": {
                            "includeThoughts": True,
                        }
                    }
                }
            }
        },
    }

    _, _, payload = asyncio.run(get_payload(request, "gemini", provider, api_key="test-key"))

    assert payload["generationConfig"]["temperature"] == 1
    assert payload["generationConfig"]["maxOutputTokens"] == 8192
    assert payload["generationConfig"]["thinkingConfig"]["includeThoughts"] is True
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert provider["preferences"]["post_body_parameter_overrides"]["gemini-3-flash"]["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
    }


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
