from __future__ import annotations

from types import SimpleNamespace

from uni_api.routing.request_rules import (
    provider_accepts_request_rules,
    provider_exclude_request_rules,
    request_reasoning_effort,
)


def _provider():
    return {
        "provider": "mxamaxai006",
        "_model_dict_cache": {"gpt-5.6-luna": "codex-auto-review"},
        "exclude_request_rules": [
            {
                "match": {
                    "endpoint": "/v1/responses",
                    "request_model": "gpt-5.6-luna",
                    "upstream_model": "codex-auto-review",
                    "reasoning_effort": ["max"],
                },
                "reason": "unsupported_reasoning_effort",
            }
        ],
    }


def test_request_reasoning_effort_supports_nested_and_legacy_shapes():
    assert request_reasoning_effort({"reasoning": {"effort": " MAX "}}) == "max"
    assert request_reasoning_effort({"reasoning_effort": "High"}) == "high"
    assert request_reasoning_effort(
        SimpleNamespace(reasoning=SimpleNamespace(effort="low"))
    ) == "low"
    assert request_reasoning_effort({"reasoning": {}}) is None


def test_provider_request_rule_excludes_only_the_matching_request():
    provider = _provider()

    assert not provider_accepts_request_rules(
        provider,
        endpoint="/v1/responses/",
        request_model="gpt-5.6-luna",
        reasoning_effort="MAX",
        request_type=None,
    )
    assert provider_accepts_request_rules(
        provider,
        endpoint="/v1/responses",
        request_model="gpt-5.6-luna",
        reasoning_effort="high",
        request_type=None,
    )
    assert provider_accepts_request_rules(
        provider,
        endpoint="/v1/responses",
        request_model="gpt-5.6-luna",
        reasoning_effort=None,
        request_type=None,
    )
    assert provider_accepts_request_rules(
        provider,
        endpoint="/v1/chat/completions",
        request_model="gpt-5.6-luna",
        reasoning_effort="max",
        request_type=None,
    )


def test_provider_request_rules_merge_preferences_and_ignore_invalid_conditions():
    provider = _provider()
    provider["preferences"] = {
        "exclude_request_rules": {
            "match": {
                "request_type": ["compaction", "future-*"],
            }
        }
    }
    provider["exclude_request_rules"].append(
        {"match": {"unsupported_field": "*"}}
    )

    assert len(provider_exclude_request_rules(provider)) == 3
    assert not provider_accepts_request_rules(
        provider,
        endpoint="/v1/responses",
        request_model="gpt-5.6-luna",
        reasoning_effort="low",
        request_type="future-type",
    )
    assert provider_accepts_request_rules(
        provider,
        endpoint="/v1/responses",
        request_model="gpt-5.6-luna",
        reasoning_effort="low",
        request_type=None,
    )
