import copy

import pytest

from uni_api.upstream.responses_normalization import (
    ResponsesCustomToolCallIdCollisionError,
    ResponsesCustomToolCallIdNormalizer,
    responses_custom_tool_call_id_normalization_enabled,
)


def test_normalizes_supported_top_level_response_items_by_type():
    payload = {
        "input": [
            {
                "type": "reasoning",
                "id": "item_reasoning123",
                "content": [
                    {
                        "type": "custom_tool_call",
                        "id": "item_nested123",
                    }
                ],
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_alreadycanonical123",
                "call_id": "call_existing",
            },
            {
                "type": "custom_tool_call",
                "id": "item_7eb1bd749e0a9e692c69ed40",
                "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
            },
            {
                "type": "custom_tool_call_output",
                "id": "ctco_output123",
                "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
            },
            {
                "type": "message",
                "id": "item_message123",
                "role": "assistant",
                "content": [],
            },
            {
                "type": "function_call",
                "id": "item_function123",
                "call_id": "call_function123",
            },
        ]
    }

    normalizer = ResponsesCustomToolCallIdNormalizer()
    result = normalizer.normalize(payload)

    assert payload["input"][0]["id"] == "rs_reasoning123"
    assert payload["input"][0]["content"][0]["id"] == "item_nested123"
    assert payload["input"][1]["id"] == "ctc_alreadycanonical123"
    assert payload["input"][2] == {
        "type": "custom_tool_call",
        "id": "ctc_7eb1bd749e0a9e692c69ed40",
        "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
    }
    assert payload["input"][3]["call_id"] == "call_foCUR1DBzdZeYyOccLpOmwUF"
    assert payload["input"][4]["id"] == "msg_message123"
    assert payload["input"][5]["id"] == "fc_function123"
    assert result.normalized_ids == 4
    assert result.rewritten_references == 0
    assert result.paths == (
        "input[0].id",
        "input[2].id",
        "input[4].id",
        "input[5].id",
    )

    second_result = normalizer.normalize(payload)
    assert not second_result.changed


def test_invalid_or_long_item_ids_are_deterministically_shortened():
    payload = {
        "input": [
            {
                "type": "custom_tool_call",
                "id": "ctc_not-canonical",
                "call_id": "call_1",
            },
            {
                "type": "message",
                "id": "resp_" + "a" * 76 + "_msg",
                "content": [],
            },
        ]
    }

    result = ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert result.normalized_ids == 2
    assert payload["input"][0]["id"].startswith("ctc_")
    assert payload["input"][1]["id"].startswith("msg_")
    assert len(payload["input"][0]["id"]) <= 64
    assert len(payload["input"][1]["id"]) <= 64
    repeated = copy.deepcopy(payload)
    assert not ResponsesCustomToolCallIdNormalizer().normalize(repeated).changed


def test_wrong_canonical_prefix_is_rewritten_for_actual_item_type():
    payload = {
        "input": [
            {
                "type": "custom_tool_call",
                "id": "fc_417ee9f223f346fc97f3c3de2bd18bdb",
            },
            {
                "type": "message",
                "id": "rs_message123",
                "content": [],
            },
        ]
    }

    result = ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert result.normalized_ids == 2
    assert payload["input"][0]["id"] == "ctc_417ee9f223f346fc97f3c3de2bd18bdb"
    assert payload["input"][1]["id"] == "msg_message123"


def test_tool_search_call_uses_tsc_item_id_prefix():
    payload = {
        "input": [
            {
                "type": "tool_search_call",
                "id": "fc_3e812ff47c529a7d687ecf89239a0f57433566d7",
            }
        ]
    }

    result = ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert result.normalized_ids == 1
    assert (
        payload["input"][0]["id"]
        == "tsc_3e812ff47c529a7d687ecf89239a0f57433566d7"
    )


def test_unknown_item_type_is_not_guessed_or_mutated():
    payload = {"input": [{"type": "future_item", "id": "item_future123"}]}

    result = ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert not result.changed
    assert payload["input"][0]["id"] == "item_future123"


def test_collision_is_rejected_before_payload_mutation():
    payload = {
        "input": [
            {
                "type": "custom_tool_call",
                "id": "item_duplicate123",
                "call_id": "call_1",
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_duplicate123",
                "call_id": "call_2",
            },
        ]
    }
    original = copy.deepcopy(payload)

    with pytest.raises(
        ResponsesCustomToolCallIdCollisionError,
        match="would collide with an existing item ID",
    ):
        ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert payload == original


def test_collision_with_item_from_previous_stream_event_is_rejected():
    normalizer = ResponsesCustomToolCallIdNormalizer()
    normalizer.normalize(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_duplicate123",
                "call_id": "call_existing",
            },
        }
    )
    payload = {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "custom_tool_call",
            "id": "item_duplicate123",
            "call_id": "call_new",
        },
    }
    original = copy.deepcopy(payload)

    with pytest.raises(
        ResponsesCustomToolCallIdCollisionError,
        match="would collide with an existing item ID",
    ):
        normalizer.normalize(payload)

    assert payload == original


def test_one_original_id_cannot_map_to_two_item_types():
    payload = {
        "input": [
            {"type": "message", "id": "item_shared123", "content": []},
            {"type": "function_call", "id": "item_shared123", "arguments": "{}"},
        ]
    }
    original = copy.deepcopy(payload)

    with pytest.raises(
        ResponsesCustomToolCallIdCollisionError,
        match="inconsistent mapping",
    ):
        ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert payload == original


def test_stream_event_sequence_uses_one_consistent_id_mapping():
    normalizer = ResponsesCustomToolCallIdNormalizer()
    item = {
        "type": "custom_tool_call",
        "id": "item_stream123",
        "call_id": "call_stream123",
        "name": "exec",
        "input": "{}",
    }
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": copy.deepcopy(item),
        },
        {
            "type": "response.custom_tool_call_input.delta",
            "output_index": 0,
            "item_id": "item_stream123",
            "delta": "{}",
        },
        {
            "type": "response.custom_tool_call_input.done",
            "output_index": 0,
            "item_id": "item_stream123",
            "input": "{}",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": copy.deepcopy(item),
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [copy.deepcopy(item)],
            },
        },
    ]

    results = [normalizer.normalize(event) for event in events]

    assert events[0]["item"]["id"] == "ctc_stream123"
    assert events[1]["item_id"] == "ctc_stream123"
    assert events[2]["item_id"] == "ctc_stream123"
    assert events[3]["item"]["id"] == "ctc_stream123"
    assert events[4]["response"]["output"][0]["id"] == "ctc_stream123"
    assert [result.normalized_ids for result in results] == [1, 0, 0, 1, 1]
    assert [result.rewritten_references for result in results] == [0, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("event_type", "original_id", "expected_id"),
    [
        ("response.output_text.delta", "item_message123", "msg_message123"),
        ("response.reasoning_text.delta", "item_reasoning123", "rs_reasoning123"),
        ("response.function_call_arguments.delta", "item_function123", "fc_function123"),
        ("response.custom_tool_call_input.delta", "fc_custom123", "ctc_custom123"),
    ],
)
def test_stream_reference_can_be_normalized_from_event_type(
    event_type,
    original_id,
    expected_id,
):
    event = {"type": event_type, "item_id": original_id, "delta": "x"}

    result = ResponsesCustomToolCallIdNormalizer().normalize(event)

    assert result.normalized_ids == 0
    assert result.rewritten_references == 1
    assert event["item_id"] == expected_id


@pytest.mark.parametrize(
    ("configured", "models", "expected"),
    [
        (True, ("gpt-5.6-sol",), True),
        (False, ("gpt-5.6-sol",), False),
        (["gpt-5.6-sol"], ("gpt-5.6-sol",), True),
        (["gpt-5.6-sol"], ("gpt-5.6-terra",), False),
        (["*"], ("gpt-5.6-terra",), True),
        ("gpt-5.6-sol", ("gpt-5.6-sol",), False),
    ],
)
def test_provider_model_feature_flag(configured, models, expected):
    provider = {
        "preferences": {
            "normalize_responses_custom_tool_call_ids": configured,
        }
    }

    assert (
        responses_custom_tool_call_id_normalization_enabled(provider, models)
        is expected
    )


@pytest.mark.parametrize(
    ("provider_name", "model", "expected"),
    [
        ("fugue-codex", "gpt-5.6-sol", True),
        ("fugue-codex", "gpt-5.6-terra", True),
        ("fugue-codex", "gpt-5.4", True),
        ("937auth", "gpt-5.6-sol", True),
        ("937auth01", "gpt-5.6-sol", True),
        ("unrelated", "gpt-5.6-sol", False),
    ],
)
def test_default_provider_model_compatibility_matrix(provider_name, model, expected):
    provider = {"provider": provider_name, "preferences": {}}

    assert (
        responses_custom_tool_call_id_normalization_enabled(provider, (model,))
        is expected
    )


def test_provider_setting_can_disable_default_compatibility_matrix():
    provider = {
        "provider": "fugue-codex",
        "preferences": {
            "normalize_responses_custom_tool_call_ids": False,
        },
    }

    assert not responses_custom_tool_call_id_normalization_enabled(
        provider,
        ("gpt-5.6-sol",),
    )


def test_codex_engine_enables_normalization_for_new_provider_names():
    provider = {
        "provider": "new-codex-provider",
        "engine": "codex",
        "preferences": {},
    }

    assert responses_custom_tool_call_id_normalization_enabled(
        provider,
        ("future-codex-model",),
    )
