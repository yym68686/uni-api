from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import msgspec

from uni_api.admission.json_memory import (
    JSONMemoryComplexityError,
    estimate_json_memory_bytes,
)
from uni_api.admission.json_parsing import ReusableJSONParseWorkspace
from uni_api.streaming.sse import DEFAULT_MAX_EVENT_BYTES, SSEProtocolError


_FAST_DELTA_PREFIXES = (
    (
        b"event: response.output_text.delta\ndata: ",
        "response.output_text.delta",
    ),
    (
        b"event: response.reasoning_summary_text.delta\ndata: ",
        "response.reasoning_summary_text.delta",
    ),
    (
        b"event: response.reasoning_text.delta\ndata: ",
        "response.reasoning_text.delta",
    ),
    (
        b"event: response.function_call_arguments.delta\ndata: ",
        "response.function_call_arguments.delta",
    ),
    (
        b"event: response.refusal.delta\ndata: ",
        "response.refusal.delta",
    ),
    (
        b"event: response.mcp_call_arguments.delta\ndata: ",
        "response.mcp_call_arguments.delta",
    ),
    (
        b"event: response.audio.delta\ndata: ",
        "response.audio.delta",
    ),
    (
        b"event: response.audio_transcript.delta\ndata: ",
        "response.audio_transcript.delta",
    ),
)
MIN_CANONICAL_RESPONSES_DELTA_FRAME_BYTES = min(
    len(prefix) + len(b"{}\n\n")
    for prefix, _event_type in _FAST_DELTA_PREFIXES
)
_JSON_PARSE_WORKSPACE_FIXED_BYTES = 64 * 1024
_JSON_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CanonicalResponsesDeltaFrame:
    """One complete canonical delta whose wire bytes can be forwarded as-is."""

    wire_bytes: bytes
    event_type: str
    data_start: int
    data_end: int

    def decode_raw_event(self) -> str:
        try:
            return self.wire_bytes[:-2].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SSEProtocolError(
                "SSE stream contains invalid UTF-8"
            ) from exc


class _SelectiveResponsesMetadata(msgspec.Struct):
    type: str | msgspec.UnsetType = msgspec.UNSET
    item_id: str | None | msgspec.UnsetType = msgspec.UNSET
    item: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET
    input: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET
    output: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET
    response: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET
    status: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET
    error: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET
    usage: msgspec.Raw | msgspec.UnsetType = msgspec.UNSET


_SELECTIVE_METADATA_DECODER = msgspec.json.Decoder(_SelectiveResponsesMetadata)


def match_canonical_responses_delta_frame(
    wire_bytes: bytes,
    *,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
) -> CanonicalResponsesDeltaFrame | None:
    """Match one exact LF-delimited delta without copying its payload."""

    if not isinstance(wire_bytes, bytes):
        return None
    if len(wire_bytes) < 2 or (
        max_event_bytes and len(wire_bytes) - 2 > max_event_bytes
    ):
        return None
    if not wire_bytes.endswith(b"\n\n"):
        return None

    data_end = len(wire_bytes) - 2
    for prefix, event_type in _FAST_DELTA_PREFIXES:
        if not wire_bytes.startswith(prefix):
            continue
        data_start = len(prefix)
        # A canonical fast frame contains exactly one event line and one data
        # line. Multiline data, coalesced frames, CRLF, and partial chunks keep
        # using the general SSE parser.
        if wire_bytes.find(b"\n", data_start, data_end) >= 0:
            return None
        return CanonicalResponsesDeltaFrame(
            wire_bytes=wire_bytes,
            event_type=event_type,
            data_start=data_start,
            data_end=data_end,
        )
    return None


async def can_forward_responses_delta_without_materializing(
    frame: CanonicalResponsesDeltaFrame,
    *,
    workspace: ReusableJSONParseWorkspace,
    max_json_estimated_bytes: int = _JSON_MAX_ESTIMATED_BYTES,
    item_id_requires_full_normalization: Callable[[str, str], bool]
    | None = None,
) -> bool:
    """Validate a delta while materializing only protocol discriminator fields.

    The existing complexity scan and conservative high-water admission policy
    remain in force. Any field that can affect failure classification, usage,
    or nested response semantics forces the caller back to the full JSON path.
    """

    payload_view = memoryview(frame.wire_bytes)[
        frame.data_start : frame.data_end
    ]
    metadata: _SelectiveResponsesMetadata | None = None
    try:
        try:
            estimate = estimate_json_memory_bytes(
                payload_view,
                raw_memory_multiplier=4,
                token_memory_bytes=128,
                max_estimated_bytes=max_json_estimated_bytes,
            )
        except JSONMemoryComplexityError as exc:
            raise SSEProtocolError(
                f"SSE JSON materialization exceeds local limit: {exc}"
            ) from exc

        workspace_bytes = (
            (len(frame.wire_bytes) - 2) * 8
            + _JSON_PARSE_WORKSPACE_FIXED_BYTES
        )
        await workspace.ensure(workspace_bytes + estimate.estimated_bytes)
        try:
            metadata = _SELECTIVE_METADATA_DECODER.decode(payload_view)
        except msgspec.DecodeError:
            return False

        if metadata.type is not msgspec.UNSET:
            if metadata.type != frame.event_type:
                return False
        if item_id_requires_full_normalization is not None:
            # A configured Responses item normalizer can mutate embedded items or
            # a previously mapped item_id. Keep those frames on the full path;
            # ordinary deltas with an unmapped reference remain byte-transparent.
            if any(
                field is not msgspec.UNSET
                for field in (
                    metadata.item,
                    metadata.input,
                    metadata.output,
                    metadata.response,
                )
            ):
                return False
            if (
                isinstance(metadata.item_id, str)
                and item_id_requires_full_normalization(
                    frame.event_type,
                    metadata.item_id,
                )
            ):
                return False
        return all(
            field is msgspec.UNSET
            for field in (
                metadata.response,
                metadata.status,
                metadata.error,
                metadata.usage,
            )
        )
    finally:
        metadata = None
        payload_view.release()
