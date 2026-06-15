from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.models import UnifiedRequest
from uni_api.observability.request_inspection import inspect_request_body


def _sample_chat_body(message_count: int) -> dict:
    return {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"message {index}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abcd"}},
                ],
            }
            for index in range(message_count)
        ],
        "stream": True,
        "temperature": 0.7,
        "top_p": 1.0,
    }


def _sample_responses_body(message_count: int) -> dict:
    return {
        "model": "gpt-4.1",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"message {index}"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abcd"},
                ],
            }
            for index in range(message_count)
        ],
        "stream": True,
        "temperature": 0.7,
        "top_p": 1.0,
    }


def _time_call(fn, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    return time.perf_counter() - started


def _measure(fn, iterations: int, rounds: int) -> list[float]:
    return [_time_call(fn, iterations) for _ in range(rounds)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--messages", type=int, default=32)
    parser.add_argument("--kind", choices=["chat", "responses"], default="chat")
    args = parser.parse_args()

    body = _sample_chat_body(args.messages) if args.kind == "chat" else _sample_responses_body(args.messages)

    def pydantic_validate() -> None:
        UnifiedRequest.model_validate(body).data.get_last_text_message()

    def lightweight_inspect() -> None:
        inspection = inspect_request_body(body)
        if not inspection.model:
            raise AssertionError("model was not extracted")

    pydantic_times = _measure(pydantic_validate, args.iterations, args.rounds)
    lightweight_times = _measure(lightweight_inspect, args.iterations, args.rounds)

    pydantic_best = min(pydantic_times)
    lightweight_best = min(lightweight_times)
    result = {
        "iterations": args.iterations,
        "kind": args.kind,
        "rounds": args.rounds,
        "messages": args.messages,
        "pydantic_validate_seconds_best": pydantic_best,
        "pydantic_validate_seconds_mean": statistics.mean(pydantic_times),
        "lightweight_inspect_seconds_best": lightweight_best,
        "lightweight_inspect_seconds_mean": statistics.mean(lightweight_times),
        "best_speedup": pydantic_best / lightweight_best if lightweight_best else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
