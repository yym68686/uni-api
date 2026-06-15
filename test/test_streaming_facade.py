from uni_api.streaming.sse import IncrementalSSEParser, is_sse_comment_frame, parse_sse_event


def test_streaming_sse_facade_handles_split_events_and_comments():
    parser = IncrementalSSEParser()

    assert parser.feed("data: {\"a\"") == []
    assert parser.feed(": 1}\n\n") == ['data: {"a": 1}']
    assert is_sse_comment_frame(": keepalive")
    assert parse_sse_event("event: done\ndata: {\"ok\": true}") == ("done", {"ok": True})
