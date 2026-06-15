import asyncio
import json

from uni_api.streaming.chat_completion_events import (
    build_chat_completion_chunk_sse,
    responses_usage_to_chat_completion_usage,
)
from uni_api.streaming.responses_events import (
    extract_responses_stream_sse_event,
    stream_responses_to_chat_completions,
)


def test_responses_event_parser_reads_named_event_and_payload():
    event_type, payload = extract_responses_stream_sse_event(
        'event: response.output_text.delta\ndata: {"delta": "hello"}'
    )

    assert event_type == "response.output_text.delta"
    assert payload == {"delta": "hello"}


def test_chat_completion_chunk_builder_outputs_openai_chunk_shape():
    raw = build_chat_completion_chunk_sse(
        response_id="chatcmpl_1",
        created_at=123,
        model_name="gpt-5.4",
        delta={"role": "assistant", "content": "hello"},
    )
    payload = json.loads(raw.removeprefix("data: "))

    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hello"


def test_responses_usage_to_chat_completion_usage_maps_input_output_tokens():
    usage = responses_usage_to_chat_completion_usage(
        {
            "input_tokens": 3,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 4},
        }
    )

    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 8
    assert usage["prompt_tokens_details"]["cached_tokens"] == 2
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 4


def test_responses_stream_to_chat_completions_converts_text_reasoning_and_done():
    async def upstream_iter():
        yield b"event: response.output_text.delt"
        yield b'a\ndata: {"type": "response.output_text.delta", "delta": "hello"}\n\n'
        yield b'event: response.reasoning_summary_text.delta\ndata: {"delta": "why"}\n\n'
        yield b"data: [DONE]\n\n"

    async def run():
        chunks = []
        async for chunk in stream_responses_to_chat_completions(upstream_iter(), request_model="gpt-5.4"):
            chunks.append(chunk)
        return "".join(chunks)

    body = asyncio.run(run())

    assert '"content": "hello"' in body
    assert '"reasoning_content": "why"' in body
    assert body.endswith("data: [DONE]\n\n")

