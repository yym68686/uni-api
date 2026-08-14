from __future__ import annotations

from types import SimpleNamespace

from uni_api.routing.request_types import (
    detect_request_type,
    provider_accepts_request_type,
    provider_request_type_values,
)


def test_detect_request_type_supports_legacy_and_trigger_based_compaction():
    assert detect_request_type(
        "/v1/responses",
        {"input": [{"type": "message"}, {"type": "compaction_trigger"}]},
    ) == "compaction"
    assert detect_request_type(
        "/v1/responses/compact/",
        {"input": "legacy"},
    ) == "compaction"
    assert detect_request_type(
        "/v1/responses",
        SimpleNamespace(input=[{"type": "message"}]),
    ) is None


def test_request_type_filters_apply_only_and_exclude_rules():
    assert provider_accepts_request_type({}, None)
    assert provider_accepts_request_type({}, "compaction")
    assert provider_accepts_request_type(
        {"only_request_types": ["compaction"]},
        "compaction",
    )
    assert not provider_accepts_request_type(
        {"only_request_types": ["compaction"]},
        None,
    )
    assert not provider_accepts_request_type(
        {"exclude_request_types": ["compaction"]},
        "compaction",
    )
    assert provider_accepts_request_type(
        {"exclude_request_types": ["compaction"]},
        None,
    )


def test_exclude_rule_wins_and_preferences_are_merged():
    provider = {
        "only_request_types": "compaction",
        "preferences": {
            "exclude_request_types": [" COMPACTION ", "future-type"],
        },
    }

    assert provider_request_type_values(provider, "only_request_types") == (
        "compaction",
    )
    assert provider_request_type_values(provider, "exclude_request_types") == (
        "compaction",
        "future-type",
    )
    assert not provider_accepts_request_type(provider, "compaction")
