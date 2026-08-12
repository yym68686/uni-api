import asyncio

import pytest

from uni_api.streaming.sse import (
    DEFAULT_MAX_EVENT_BYTES,
    IncrementalLineParser,
    IncrementalSSEParser,
    SSEBufferOverflowError,
    SSEIncompleteEventError,
    SSEOutputLimitError,
    SSEProtocolError,
    UNLIMITED_SSE_BYTES,
    is_sse_comment_frame,
    parse_owned_sse_event,
    parse_sse_event,
    sse_event_has_data_field,
    validate_sse_event_type_consistency,
)


def test_streaming_sse_facade_handles_split_events_and_comments():
    parser = IncrementalSSEParser()

    assert parser.feed("data: {\"a\"") == []
    assert parser.feed(": 1}\n\n") == ['data: {"a": 1}']
    assert is_sse_comment_frame(": keepalive")
    assert parse_sse_event("event: done\ndata: {\"ok\": true}") == ("done", {"ok": True})


def test_streaming_sse_facade_preserves_utf8_split_across_byte_chunks():
    parser = IncrementalSSEParser()
    encoded = "data: 你好🙂\n\n".encode("utf-8")
    split_at = encoded.index("🙂".encode("utf-8")) + 2

    assert parser.feed(encoded[:split_at]) == []
    assert parser.pending_data == encoded[:split_at]
    assert parser.feed(encoded[split_at:]) == ["data: 你好🙂"]
    assert parser.finish() == []


def test_streaming_sse_facade_normalizes_crlf_split_across_chunks_once():
    parser = IncrementalSSEParser()

    assert parser.feed(b"event: message\r") == []
    assert parser.feed(b"\ndata: first\r\ndata: second\r") == []
    assert parser.feed(b"\n\r") == ["event: message\ndata: first\ndata: second"]
    assert parser.feed(b"\n") == []
    assert parser.finish() == []


def test_streaming_sse_pending_data_preserves_raw_trailing_cr_state():
    parser = IncrementalSSEParser()
    assert parser.feed(b"event: note\ndata: first\r") == []
    pending = parser.pending_data
    assert pending.endswith(b"\r")

    resumed = IncrementalSSEParser()
    assert resumed.feed(pending) == []
    assert resumed.feed(b"\n\r\n") == ["event: note\ndata: first"]
    assert resumed.finish() == []


def test_streaming_sse_facade_enforces_pending_and_event_byte_limits():
    pending_parser = IncrementalSSEParser(max_pending_bytes=8, max_event_bytes=64)
    with pytest.raises(SSEBufferOverflowError) as pending_error:
        pending_parser.feed("data: abc")
    assert pending_error.value.buffer_name == "pending buffer"
    assert pending_error.value.limit_bytes == 8

    event_parser = IncrementalSSEParser(max_pending_bytes=64, max_event_bytes=6)
    with pytest.raises(SSEBufferOverflowError) as event_error:
        event_parser.feed("data: x\n\n")
    assert event_error.value.buffer_name == "event"
    assert event_error.value.observed_bytes == 7


def test_streaming_sse_zero_limit_accepts_frame_beyond_legacy_bound():
    parser = IncrementalSSEParser(
        max_pending_bytes=UNLIMITED_SSE_BYTES,
        max_event_bytes=UNLIMITED_SSE_BYTES,
        max_feed_bytes=UNLIMITED_SSE_BYTES,
    )
    payload = b"x" * (DEFAULT_MAX_EVENT_BYTES + 1)

    assert parser.feed(b"data: " + payload) == []
    events = parser.feed(b"\n\n")

    assert len(events) == 1
    assert len(events[0].encode()) == len(payload) + len(b"data: ")


def test_streaming_sse_negative_limits_remain_invalid():
    with pytest.raises(ValueError, match="cannot be negative"):
        IncrementalSSEParser(max_event_bytes=-1)


def test_owned_sse_event_zero_limit_disables_only_fixed_frame_bound():
    async def inspect():
        owner = await parse_owned_sse_event(
            'event: note\ndata: {"type":"note"}',
            max_event_bytes=UNLIMITED_SSE_BYTES,
        )
        try:
            return owner.event_name, owner.payload
        finally:
            await owner.aclose()

    assert asyncio.run(inspect()) == ("note", {"type": "note"})


def test_streaming_sse_facade_finish_accepts_complete_stream_and_rejects_incomplete_event():
    complete_parser = IncrementalSSEParser()
    assert complete_parser.feed("data: complete\n\n") == ["data: complete"]
    assert complete_parser.finish() == []
    assert complete_parser.finish() == []

    incomplete_parser = IncrementalSSEParser()
    assert incomplete_parser.feed("data: incomplete") == []
    with pytest.raises(SSEIncompleteEventError) as incomplete_error:
        incomplete_parser.finish()
    assert incomplete_error.value.pending_bytes == len(b"data: incomplete")


def test_streaming_sse_facade_finish_flushes_terminal_cr_event_separator():
    parser = IncrementalSSEParser()

    assert parser.feed("data: complete\r\r") == ["data: complete"]
    assert parser.finish() == []


def test_streaming_sse_facade_finish_rejects_incomplete_utf8():
    parser = IncrementalSSEParser()
    encoded = "data: 🙂".encode("utf-8")

    assert parser.feed(encoded[:-1]) == []
    with pytest.raises(SSEProtocolError, match="incomplete UTF-8"):
        parser.finish()


def test_streaming_sse_facade_preserves_comments_and_multiline_data():
    parser = IncrementalSSEParser()
    frames = parser.feed(
        ": keepalive\r\n\r\nevent: message\r\ndata: first\r\ndata: second\r\n\r\n"
    )

    assert frames == [": keepalive", "event: message\ndata: first\ndata: second"]
    assert is_sse_comment_frame(frames[0])
    assert not is_sse_comment_frame(frames[1])
    assert parse_sse_event(frames[1]) == ("message", "first\nsecond")
    assert parser.finish() == []


def test_incremental_line_parser_is_utf8_crlf_safe_and_bounded():
    parser = IncrementalLineParser(max_line_bytes=16)
    encoded = "你\r\n好\r末".encode("utf-8")

    assert parser.feed(encoded[:2]) == []
    assert parser.feed(encoded[2:7]) == ["你"]
    assert parser.feed(encoded[7:]) == ["好"]
    assert parser.finish() == ["末"]

    oversized = IncrementalLineParser(max_line_bytes=3)
    with pytest.raises(SSEBufferOverflowError):
        oversized.feed("four")


@pytest.mark.parametrize("parser_type", [IncrementalSSEParser, IncrementalLineParser])
def test_feed_limit_counts_utf8_bytes_for_text_and_failure_is_sticky(parser_type):
    parser = parser_type(max_feed_bytes=4)

    with pytest.raises(SSEBufferOverflowError) as exc_info:
        parser.feed("🙂🙂")

    assert exc_info.value.observed_bytes == 8
    assert exc_info.value.limit_bytes == 4
    with pytest.raises(SSEProtocolError, match="after a parse failure"):
        parser.feed("ok")


def test_parse_sse_event_preserves_protocol_data_whitespace_and_empty_fields():
    assert parse_sse_event("event: note\ndata:  leading \ndata") == (
        "note",
        " leading \n",
    )


def test_owned_sse_event_distinguishes_missing_and_empty_data_fields():
    async def inspect(raw_event):
        owner = await parse_owned_sse_event(raw_event)
        try:
            return {
                "event_name": owner.event_name,
                "declared_event_name": owner.declared_event_name,
                "payload": owner.payload,
                "is_comment": owner.is_comment,
                "has_event_field": owner.has_event_field,
                "has_data_field": owner.has_data_field,
            }
        finally:
            await owner.aclose()

    event_only = asyncio.run(inspect("event: response.completed"))
    assert event_only == {
        "event_name": "response.completed",
        "declared_event_name": "response.completed",
        "payload": "",
        "is_comment": False,
        "has_event_field": True,
        "has_data_field": False,
    }
    assert not sse_event_has_data_field("event: response.completed")

    empty_data = asyncio.run(inspect("event: response.completed\ndata:"))
    assert empty_data["event_name"] == "response.completed"
    assert empty_data["payload"] == ""
    assert empty_data["has_data_field"] is True
    assert sse_event_has_data_field("data")

    data_only = asyncio.run(
        inspect('data: {"type":"response.completed","response":{}}')
    )
    assert data_only["event_name"] == "response.completed"
    assert data_only["declared_event_name"] == ""
    assert data_only["has_event_field"] is False
    assert data_only["payload"] == {
        "type": "response.completed",
        "response": {},
    }
    assert data_only["has_data_field"] is True

    comment = asyncio.run(inspect(": keepalive"))
    assert comment["is_comment"] is True
    assert comment["has_event_field"] is False
    assert comment["has_data_field"] is False

    done = asyncio.run(inspect("data: [DONE]"))
    assert done["event_name"] == "[DONE]"
    assert done["has_data_field"] is True


def test_responses_event_type_consistency_rejects_only_ambiguous_records():
    validate_sse_event_type_consistency(
        "response.completed",
        {"type": "response.completed"},
        protocol_name="Responses",
        has_event_field=True,
        require_event_name=True,
    )
    validate_sse_event_type_consistency(
        "",
        {"type": "response.completed"},
        protocol_name="Responses",
        has_event_field=False,
        require_event_name=True,
    )
    validate_sse_event_type_consistency(
        "response.completed",
        {"response": {}},
        protocol_name="Responses",
        has_event_field=True,
        require_event_name=True,
    )

    with pytest.raises(
        SSEProtocolError,
        match="Responses SSE event field conflicts with data.type",
    ):
        validate_sse_event_type_consistency(
            "response.completed",
            {"type": "response.created"},
            protocol_name="Responses",
            has_event_field=True,
            require_event_name=True,
        )

    for invalid_type in (None, 123, {"name": "response.completed"}, "", " bad "):
        with pytest.raises(
            SSEProtocolError,
            match="Responses SSE data.type must be a non-empty string",
        ):
            validate_sse_event_type_consistency(
                "response.completed",
                {"type": invalid_type},
                protocol_name="Responses",
                has_event_field=True,
                require_event_name=True,
            )

    with pytest.raises(SSEProtocolError, match="Responses SSE event type is missing"):
        validate_sse_event_type_consistency(
            "",
            {"response": {}},
            protocol_name="Responses",
            has_event_field=False,
            require_event_name=True,
        )

    with pytest.raises(
        SSEProtocolError,
        match="Responses SSE event field must not be empty",
    ):
        validate_sse_event_type_consistency(
            "",
            {"type": "response.completed"},
            protocol_name="Responses",
            has_event_field=True,
            require_event_name=True,
        )

    with pytest.raises(
        SSEProtocolError,
        match="Responses SSE data must be a JSON object",
    ):
        validate_sse_event_type_consistency(
            "response.completed",
            "not-json-object",
            protocol_name="Responses",
            has_event_field=True,
            require_event_name=True,
        )

    for invalid_scalar in ("\ud800", "x" * 257):
        with pytest.raises(SSEProtocolError, match="Responses SSE data.type is invalid"):
            validate_sse_event_type_consistency(
                "",
                {"type": invalid_scalar},
                protocol_name="Responses",
                has_event_field=False,
                require_event_name=True,
            )


def test_parse_sse_event_does_not_treat_unicode_separators_as_sse_newlines():
    raw = 'data: {"type":"message","delta":"a\u2028b\u2029c\u0085d"}'
    assert parse_sse_event(raw) == (
        "message",
        {"type": "message", "delta": "a\u2028b\u2029c\u0085d"},
    )


def test_sse_and_line_parsers_ignore_one_initial_utf8_bom():
    sse = IncrementalSSEParser()
    assert sse.feed(b"\xef\xbb") == []
    assert sse.feed(b"\xbfdata: ok\n\n") == ["data: ok"]

    lines = IncrementalLineParser()
    assert lines.feed(b"\xef\xbb\xbfline\n") == ["line"]


def test_sse_parser_is_linear_over_many_tiny_chunks_and_preserves_unicode_separators():
    delta = "x" * 50_000 + "a\u2028b\u2029c\u0085d"
    raw_event = f'data: {{"delta":"{delta}"}}\r\n\r\n'.encode()
    parser = IncrementalSSEParser()
    events = []

    for byte in raw_event:
        events.extend(parser.feed(bytes((byte,))))

    assert events == [f'data: {{"delta":"{delta}"}}']
    assert parser.pending_bytes == 0
    assert parser.finish() == []


def test_line_parser_is_linear_over_many_tiny_chunks_and_preserves_unicode_separators():
    first_line = "x" * 50_000 + "alpha\u2028beta\u2029gamma\u0085delta"
    raw_lines = f"{first_line}\r\nomega\rfinal".encode()
    parser = IncrementalLineParser()
    lines = []

    for byte in raw_lines:
        lines.extend(parser.feed(bytes((byte,))))

    lines.extend(parser.finish())
    assert lines == [first_line, "omega", "final"]


def test_sse_parser_rejects_one_chunk_with_too_many_events_before_unbounded_output():
    parser = IncrementalSSEParser(max_events_per_feed=32)
    chunk = "".join(f"data: {index}\n\n" for index in range(10_000))

    with pytest.raises(SSEOutputLimitError) as exc_info:
        parser.feed(chunk)

    assert exc_info.value.output_name == "events"
    assert exc_info.value.limit == 32
    assert exc_info.value.observed == 33
    with pytest.raises(SSEProtocolError, match="after a parse failure"):
        parser.feed("data: later\n\n")


def test_line_parser_rejects_one_chunk_with_too_many_lines_before_unbounded_output():
    parser = IncrementalLineParser(max_lines_per_feed=32)
    chunk = "".join(f"line-{index}\n" for index in range(10_000))

    with pytest.raises(SSEOutputLimitError) as exc_info:
        parser.feed(chunk)

    assert exc_info.value.output_name == "lines"
    assert exc_info.value.limit == 32
    assert exc_info.value.observed == 33
    with pytest.raises(SSEProtocolError, match="after a parse failure"):
        parser.feed("later\n")
