from __future__ import annotations

from typing import Any

from core.utils import _build_openai_usage, end_of_line
from uni_api.serialization import json


def build_chat_completion_chunk_sse(
    *,
    response_id: str,
    created_at: int,
    model_name: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + end_of_line


def build_chat_completion_usage_chunk_sse(
    *,
    response_id: str,
    created_at: int,
    model_name: str,
    usage: dict,
) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model_name,
        "choices": [],
        "usage": usage,
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + end_of_line


def responses_usage_to_chat_completion_usage(usage_obj: object) -> dict | None:
    if not isinstance(usage_obj, dict):
        return None
    if all(
        usage_obj.get(key) is None
        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
    ):
        return None

    prompt_tokens = usage_obj.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage_obj.get("input_tokens")

    completion_tokens = usage_obj.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage_obj.get("output_tokens")

    total_tokens = usage_obj.get("total_tokens")
    if total_tokens is None:
        try:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        except Exception:
            total_tokens = 0

    prompt_details = usage_obj.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = usage_obj.get("input_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    completion_details = usage_obj.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = usage_obj.get("output_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}

    return _build_openai_usage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        total_tokens=total_tokens or 0,
        cached_tokens=prompt_details.get("cached_tokens", 0) or 0,
        prompt_audio_tokens=prompt_details.get("audio_tokens", 0) or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens", 0) or 0,
        completion_audio_tokens=completion_details.get("audio_tokens", 0) or 0,
        accepted_prediction_tokens=completion_details.get("accepted_prediction_tokens", 0) or 0,
        rejected_prediction_tokens=completion_details.get("rejected_prediction_tokens", 0) or 0,
    )
