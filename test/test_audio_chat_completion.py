import os
import sys
import json
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import error_handling_wrapper


async def test_audio_chat_completion_non_stream_not_empty():
    sample = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-audio-2025-08-28",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "audio": {"id": "audio_x", "data": "AAAA", "format": "wav"},
                    "annotations": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    async def gen():
        yield "data: " + json.dumps(sample) + "\n\n"

    wrapped, _ = await error_handling_wrapper(
        gen(),
        channel_id="test",
        engine="gpt",
        stream=False,
        error_triggers=[],
        last_message_role="user",
    )
    first = await wrapped.__anext__()
    assert '"audio"' in first


if __name__ == "__main__":
    asyncio.run(test_audio_chat_completion_non_stream_not_empty())

