from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any

from core.utils import IncrementalSSEParser, end_of_line, parse_sse_event, safe_get
from uni_api.serialization import json

from .chat_completion_events import (
    build_chat_completion_chunk_sse,
    build_chat_completion_usage_chunk_sse,
    responses_usage_to_chat_completion_usage,
)
from .sse import is_sse_comment_frame


def extract_responses_stream_sse_event(raw_event: str) -> tuple[str, object]:
    return parse_sse_event(raw_event)


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def coerce_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def mime_type_from_output_format(output_format: str | None) -> str:
    normalized = str(output_format or "").strip().lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "webp":
        return "image/webp"
    if normalized == "gif":
        return "image/gif"
    return "image/png"


def extract_response_model_name(payload: object) -> str | None:
    for candidate in (
        safe_get(payload, "model_name", default=None),
        safe_get(payload, "model", default=None),
        safe_get(payload, "response", "model_name", default=None),
        safe_get(payload, "response", "model", default=None),
    ):
        normalized = normalize_optional_text(candidate)
        if normalized is not None:
            return normalized
    return None


def chat_completion_response_id_from_payload(payload: object, fallback: str) -> str:
    for candidate in (
        safe_get(payload, "id", default=None),
        safe_get(payload, "response", "id", default=None),
    ):
        normalized = normalize_optional_text(candidate)
        if normalized is not None:
            return normalized
    return fallback


def chat_completion_created_at_from_payload(payload: object, fallback: int) -> int:
    for candidate in (
        safe_get(payload, "created", default=None),
        safe_get(payload, "created_at", default=None),
        safe_get(payload, "response", "created_at", default=None),
        safe_get(payload, "response", "created", default=None),
    ):
        normalized = coerce_positive_int(candidate)
        if normalized is not None:
            return normalized
    return fallback


def chat_completion_tool_calls_from_responses_output(output_items: object) -> list[dict]:
    if not isinstance(output_items, list):
        return []

    tool_calls: list[dict] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if normalize_optional_text(item.get("type")) != "function_call":
            continue

        name = normalize_optional_text(item.get("name"))
        if name is None:
            continue

        tool_calls.append(
            {
                "id": normalize_optional_text(item.get("call_id")) or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": str(item.get("arguments") or ""),
                },
            }
        )

    return tool_calls


def chat_completion_message_from_responses_payload(payload: dict) -> tuple[dict, str]:
    output_items = safe_get(payload, "output", default=None)
    if not isinstance(output_items, list):
        output_items = safe_get(payload, "response", "output", default=[])

    message_parts: list[str] = []
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue

            item_type = normalize_optional_text(item.get("type"))
            if item_type == "message":
                content_items = item.get("content")
                if not isinstance(content_items, list):
                    continue
                text_parts: list[str] = []
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    content_type = normalize_optional_text(content_item.get("type"))
                    if content_type in {"output_text", "input_text", "text"} and content_item.get("text") is not None:
                        text_parts.append(str(content_item.get("text")))
                if text_parts:
                    message_parts.append("".join(text_parts))
                continue

            if item_type == "image_generation_call":
                result_b64 = item.get("result")
                if not isinstance(result_b64, str) or not result_b64:
                    continue
                mime_type = mime_type_from_output_format(
                    normalize_optional_text(item.get("output_format"))
                )
                message_parts.append(f"![image](data:{mime_type};base64,{result_b64})")

    content = "\n\n".join(part for part in message_parts if part)
    tool_calls = chat_completion_tool_calls_from_responses_output(output_items)

    message: dict[str, object] = {
        "role": "assistant",
        "content": content or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not content:
            message["content"] = None
        return message, "tool_calls"
    return message, "stop"


def chat_completion_usage_from_responses_payload(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None

    usage_obj = safe_get(payload, "response", "usage", default=None)
    if usage_obj is None:
        usage_obj = payload.get("usage")
    return responses_usage_to_chat_completion_usage(usage_obj)


def collect_responses_output_item_done(
    event_payload: object,
    *,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> None:
    if not isinstance(event_payload, dict):
        return

    item = event_payload.get("item")
    if not isinstance(item, dict):
        return

    output_index = coerce_positive_int(event_payload.get("output_index"))
    item_copy = copy.deepcopy(item)
    if output_index is None and event_payload.get("output_index") not in (0, "0"):
        output_items_fallback.append(item_copy)
        return
    if output_index is None:
        output_index = 0
    output_items_by_index[output_index] = item_copy


def patch_responses_completed_output(
    payload: object,
    *,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> object:
    if not isinstance(payload, dict):
        return payload

    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        return payload

    output_items = response_payload.get("output")
    should_patch_output = (
        (not isinstance(output_items, list) or not output_items)
        and (output_items_by_index or output_items_fallback)
    )
    if not should_patch_output:
        return payload

    patched_payload = copy.deepcopy(payload)
    patched_response = patched_payload.get("response")
    if not isinstance(patched_response, dict):
        return patched_payload

    patched_output = [
        copy.deepcopy(output_items_by_index[index])
        for index in sorted(output_items_by_index)
    ]
    patched_output.extend(copy.deepcopy(item) for item in output_items_fallback)
    patched_response["output"] = patched_output
    return patched_payload


def build_synthetic_responses_completed_payload(
    *,
    response_id: str,
    model_name: str,
    created_at: int,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> dict | None:
    if not output_items_by_index and not output_items_fallback:
        return None

    payload = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "model": model_name,
            "created_at": created_at,
            "status": "completed",
        },
    }
    patched_payload = patch_responses_completed_output(
        payload,
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
    )
    output_items = safe_get(patched_payload, "response", "output", default=None)
    if not isinstance(output_items, list) or not output_items:
        return None
    return patched_payload


def build_missing_responses_completed_payload(
    *,
    completed_response_seen: bool,
    error_seen: bool,
    response_id: str,
    model_name: str,
    created_at: int,
    output_items_by_index: dict[int, dict],
    output_items_fallback: list[dict],
) -> dict | None:
    if completed_response_seen or error_seen:
        return None
    return build_synthetic_responses_completed_payload(
        response_id=response_id,
        model_name=model_name,
        created_at=created_at,
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
    )


async def stream_responses_to_chat_completions(
    text_iterator,
    *,
    request_model: str,
):
    emitted_content = ""
    role_sent = False
    response_id = f"chatcmpl_{uuid.uuid4().hex}"
    created_at = int(datetime.timestamp(datetime.now()))
    model_name = request_model
    completed_response_seen = False
    error_seen = False
    output_items_by_index: dict[int, dict] = {}
    output_items_fallback: list[dict] = []
    async def emit_content_delta(content: str):
        nonlocal role_sent, emitted_content
        if not content:
            return
        delta = {"content": content}
        if not role_sent:
            delta["role"] = "assistant"
            role_sent = True
        emitted_content += content
        yield build_chat_completion_chunk_sse(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            delta=delta,
        )
    async def emit_reasoning_delta(content: str):
        nonlocal role_sent
        if not content:
            return
        delta = {"content": "", "reasoning_content": content}
        if not role_sent:
            delta["role"] = "assistant"
            role_sent = True
        yield build_chat_completion_chunk_sse(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            delta=delta,
        )
    async def emit_completed_payload(event_payload: dict):
        nonlocal completed_response_seen, role_sent, response_id, created_at, model_name
        completed_response_seen = True

        patched_payload = patch_responses_completed_output(
            event_payload,
            output_items_by_index=output_items_by_index,
            output_items_fallback=output_items_fallback,
        )
        response_id = chat_completion_response_id_from_payload(patched_payload, response_id)
        created_at = chat_completion_created_at_from_payload(patched_payload, created_at)
        model_name = extract_response_model_name(patched_payload) or model_name
        message, finish_reason = chat_completion_message_from_responses_payload(patched_payload)
        final_content = str(message.get("content") or "")
        if final_content:
            suffix = final_content
            if emitted_content and final_content.startswith(emitted_content):
                suffix = final_content[len(emitted_content):]
            async for content_chunk in emit_content_delta(suffix):
                yield content_chunk

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tool_call_deltas = []
            for index, tool_call in enumerate(tool_calls):
                tool_call_delta = copy.deepcopy(tool_call)
                tool_call_delta["index"] = index
                tool_call_deltas.append(tool_call_delta)
            delta = {"tool_calls": tool_call_deltas}
            if not role_sent:
                delta["role"] = "assistant"
                role_sent = True
            yield build_chat_completion_chunk_sse(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                delta=delta,
            )

        if not role_sent:
            yield build_chat_completion_chunk_sse(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                delta={"role": "assistant"},
            )
            role_sent = True

        yield build_chat_completion_chunk_sse(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            delta={},
            finish_reason=finish_reason,
        )
        usage = chat_completion_usage_from_responses_payload(patched_payload)
        if usage is not None:
            yield build_chat_completion_usage_chunk_sse(
                response_id=response_id,
                created_at=created_at,
                model_name=model_name,
                usage=usage,
            )
        yield "data: [DONE]" + end_of_line

    sse_parser = IncrementalSSEParser()
    async for chunk in text_iterator:
        for raw_event in sse_parser.feed(chunk):
            if not raw_event.strip():
                continue

            if is_sse_comment_frame(raw_event):
                yield raw_event + end_of_line
                continue

            event_type, event_payload = extract_responses_stream_sse_event(raw_event)
            if event_type == "[DONE]":
                synthetic_completed_payload = build_missing_responses_completed_payload(
                    completed_response_seen=completed_response_seen,
                    error_seen=error_seen,
                    response_id=response_id,
                    model_name=model_name,
                    created_at=created_at,
                    output_items_by_index=output_items_by_index,
                    output_items_fallback=output_items_fallback,
                )
                if synthetic_completed_payload is not None:
                    async for completed_chunk in emit_completed_payload(synthetic_completed_payload):
                        yield completed_chunk
                    return
                yield "data: [DONE]" + end_of_line
                return

            if event_type == "error":
                error_seen = True
                if isinstance(event_payload, dict):
                    yield "data: " + json.dumps(event_payload, ensure_ascii=False) + end_of_line
                else:
                    yield raw_event + end_of_line
                continue

            if event_type == "keepalive":
                continue

            if isinstance(event_payload, dict):
                response_id = chat_completion_response_id_from_payload(event_payload, response_id)
                created_at = chat_completion_created_at_from_payload(event_payload, created_at)
                model_name = extract_response_model_name(event_payload) or model_name

            if event_type == "response.output_item.done":
                collect_responses_output_item_done(
                    event_payload,
                    output_items_by_index=output_items_by_index,
                    output_items_fallback=output_items_fallback,
                )
                continue

            if event_type == "response.output_text.delta" and isinstance(event_payload, dict):
                delta_text = str(event_payload.get("delta") or "")
                async for content_chunk in emit_content_delta(delta_text):
                    yield content_chunk
                continue

            if event_type == "response.reasoning_summary_text.delta" and isinstance(event_payload, dict):
                delta_text = str(event_payload.get("delta") or "")
                async for reasoning_chunk in emit_reasoning_delta(delta_text):
                    yield reasoning_chunk
                continue

            if event_type == "response.reasoning_summary_text.done":
                async for reasoning_chunk in emit_reasoning_delta("\n\n"):
                    yield reasoning_chunk
                continue

            if event_type != "response.completed" or not isinstance(event_payload, dict):
                continue

            async for completed_chunk in emit_completed_payload(event_payload):
                yield completed_chunk
            return

    synthetic_completed_payload = build_missing_responses_completed_payload(
        completed_response_seen=completed_response_seen,
        error_seen=error_seen,
        response_id=response_id,
        model_name=model_name,
        created_at=created_at,
        output_items_by_index=output_items_by_index,
        output_items_fallback=output_items_fallback,
    )
    if synthetic_completed_payload is not None:
        async for completed_chunk in emit_completed_payload(synthetic_completed_payload):
            yield completed_chunk
        return
    yield "data: [DONE]" + end_of_line
