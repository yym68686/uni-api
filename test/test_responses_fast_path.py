import asyncio

import pytest

from uni_api.admission.json_parsing import ReusableJSONParseWorkspace
from uni_api.streaming.responses_fast_path import (
    can_forward_responses_delta_without_materializing,
    match_canonical_responses_delta_frame,
)
from uni_api.streaming.sse import IncrementalSSEParser, SSEProtocolError
from uni_api.upstream.responses_normalization import (
    ResponsesCustomToolCallIdNormalizer,
)


def _delta_wire(payload: bytes) -> bytes:
    return (
        b"event: response.output_text.delta\n"
        b"data: "
        + payload
        + b"\n\n"
    )


def _is_transparent(
    wire: bytes,
    *,
    max_json_estimated_bytes=64 * 1024 * 1024,
    item_id_requires_full_normalization=None,
) -> bool:
    async def scenario() -> bool:
        frame = match_canonical_responses_delta_frame(wire)
        assert frame is not None
        workspace = await ReusableJSONParseWorkspace.create()
        try:
            return await can_forward_responses_delta_without_materializing(
                frame,
                workspace=workspace,
                max_json_estimated_bytes=max_json_estimated_bytes,
                item_id_requires_full_normalization=(
                    item_id_requires_full_normalization
                ),
            )
        finally:
            await workspace.aclose()

    return asyncio.run(scenario())


def test_canonical_delta_match_borrows_original_wire_bytes():
    payload = (
        b'{"type":"response.output_text.delta",'
        b'"sequence_number":7,"delta":"hello"}'
    )
    wire = _delta_wire(payload)

    frame = match_canonical_responses_delta_frame(wire)

    assert frame is not None
    assert frame.wire_bytes is wire
    assert frame.event_type == "response.output_text.delta"
    assert wire[frame.data_start : frame.data_end] == payload
    assert frame.decode_raw_event().encode("utf-8") + b"\n\n" == wire


@pytest.mark.parametrize(
    "wire",
    [
        b'event: response.output_text.delta\r\ndata: {"delta":"x"}\r\n\r\n',
        b'event: response.output_text.delta\ndata: {"delta":"x"}',
        (
            b'event: response.output_text.delta\ndata: {"delta":"x"}\n\n'
            b'event: response.output_text.delta\ndata: {"delta":"y"}\n\n'
        ),
        b'event: response.output_text.delta\ndata: {"delta":\n"x"}\n\n',
        b'event: response.completed\ndata: {"type":"response.completed"}\n\n',
    ],
)
def test_noncanonical_or_terminal_frame_uses_general_parser(wire):
    assert match_canonical_responses_delta_frame(wire) is None


def test_parser_only_allows_bypass_after_a_clean_complete_frame():
    parser = IncrementalSSEParser()
    wire = _delta_wire(
        b'{"type":"response.output_text.delta","delta":"first"}'
    )

    assert parser.can_bypass_complete_frame is False
    assert len(parser.feed(wire)) == 1
    assert parser.can_bypass_complete_frame is True
    assert parser.feed(b"event: response.output_text.delta\n") == []
    assert parser.can_bypass_complete_frame is False


def test_selective_decoder_accepts_utf8_and_ignores_nested_diagnostic_keys():
    wire = _delta_wire(
        (
            '{"type":"response.output_text.delta","delta":"你好",'
            '"metadata":{"error":{"message":"not a protocol error"}}}'
        ).encode("utf-8")
    )

    assert _is_transparent(wire) is True


def test_item_normalizer_materializes_noncanonical_or_mapped_references():
    normalizer = ResponsesCustomToolCallIdNormalizer()
    requires_full_normalization = (
        normalizer.requires_item_id_full_normalization
    )
    unmapped = _delta_wire(
        b'{"type":"response.output_text.delta",'
        b'"item_id":"item_unmapped","delta":"x"}'
    )

    assert _is_transparent(
        unmapped,
        item_id_requires_full_normalization=requires_full_normalization,
    ) is False

    result = normalizer.normalize(
        {
            "type": "response.output_item.added",
            "item": {"type": "custom_tool_call", "id": "item_mapped"},
        }
    )
    assert result.changed is True
    mapped = _delta_wire(
        b'{"type":"response.output_text.delta",'
        b'"item_id":"item_mapped","delta":"x"}'
    )

    assert _is_transparent(
        mapped,
        item_id_requires_full_normalization=requires_full_normalization,
    ) is False


@pytest.mark.parametrize("field", ["item", "input", "output"])
def test_custom_tool_normalizer_forces_full_path_for_embedded_items(field):
    normalizer = ResponsesCustomToolCallIdNormalizer()
    wire = _delta_wire(
        (
            '{"type":"response.output_text.delta","delta":"x","'
            + field
            + '":null}'
        ).encode()
    )

    assert _is_transparent(
        wire,
        item_id_requires_full_normalization=(
            normalizer.requires_item_id_full_normalization
        ),
    ) is False


@pytest.mark.parametrize(
    "payload",
    [
        b'{"type":"response.output_text.delta","error":null,"delta":"x"}',
        b'{"type":"response.output_text.delta","status":"ok","delta":"x"}',
        b'{"type":"response.output_text.delta","response":null,"delta":"x"}',
        b'{"type":"response.output_text.delta","usage":null,"delta":"x"}',
        b'{"type":"other.event","delta":"x"}',
        b'{"type":12,"delta":"x"}',
        b'{"\\u0065rror":null,"type":"response.output_text.delta","delta":"x"}',
        b'{"type":"response.output_text.delta",BAD}',
        b'["response.output_text.delta"]',
    ],
)
def test_semantically_relevant_or_invalid_payload_falls_back(payload):
    assert _is_transparent(_delta_wire(payload)) is False


def test_selective_decoder_keeps_existing_json_complexity_limit():
    nested = b"[" * 129 + b"0" + b"]" * 129
    wire = _delta_wire(
        b'{"type":"response.output_text.delta","delta":' + nested + b"}"
    )

    with pytest.raises(SSEProtocolError, match="nesting exceeds"):
        _is_transparent(wire)


def test_selective_decoder_uses_caller_resource_budget_not_frame_size():
    payload = (
        b'{"type":"response.output_text.delta","delta":"'
        + b"x" * (64 * 1024)
        + b'"}'
    )
    wire = _delta_wire(payload)

    with pytest.raises(SSEProtocolError, match="materialization exceeds"):
        _is_transparent(wire, max_json_estimated_bytes=64 * 1024)
    assert _is_transparent(
        wire,
        max_json_estimated_bytes=2 * 1024 * 1024,
    ) is True
