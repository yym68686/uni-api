from __future__ import annotations

import asyncio
import time as time_module
from typing import Any

import h2.exceptions
import httpx
from fastapi import HTTPException

from core.log_config import logger
from core.utils import safe_get
from uni_api.serialization import json
from uni_api.streaming.cleanup import close_async_iterator_safely
from uni_api.streaming.sse import is_sse_comment_frame


async def ensure_string(item: Any) -> str:
    if isinstance(item, (bytes, bytearray)):
        return item.decode("utf-8")
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        json_str = await asyncio.to_thread(json.dumps, item)
        return f"data: {json_str}\n\n"
    return str(item)


def identify_audio_format(file_bytes: bytes) -> str:
    if file_bytes.startswith(b"\xFF\xFB") or file_bytes.startswith(b"\xFF\xF3"):
        return "MP3"
    if file_bytes.startswith(b"ID3"):
        return "MP3 with ID3"
    if file_bytes.startswith(b"OpusHead"):
        return "OPUS"
    if file_bytes.startswith(b"ADIF"):
        return "AAC (ADIF)"
    if file_bytes.startswith(b"\xFF\xF1") or file_bytes.startswith(b"\xFF\xF9"):
        return "AAC (ADTS)"
    if file_bytes.startswith(b"fLaC"):
        return "FLAC"
    if file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WAVE":
        return "WAV"
    return "Unknown/PCM"


async def wait_for_timeout(wait_for_thing, timeout=3, wait_task=None):
    if wait_task is None:
        first_response_task = asyncio.create_task(wait_for_thing.__anext__())
    else:
        first_response_task = wait_task

    timeout_task = asyncio.create_task(asyncio.sleep(timeout))
    done, _pending = await asyncio.wait(
        [first_response_task, timeout_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if first_response_task in done:
        timeout_task.cancel()
        return first_response_task.result(), "success"
    return first_response_task, "timeout"


async def _close_async_iterator_safely(iterator, channel_id, reason):
    await close_async_iterator_safely(
        iterator,
        label=f"provider: {channel_id} async iterator during {reason}",
    )


async def _cancel_pending_task_safely(task, channel_id, reason):
    if task is None or not hasattr(task, "cancel") or not hasattr(task, "done") or task.done():
        return

    task.cancel()
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        logger.debug(
            "provider: %s pending task cleanup failed during %s",
            channel_id,
            reason,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _infer_openai_like_error_status(error_obj, default_status=500):
    if not isinstance(error_obj, dict):
        return default_status

    raw_status = error_obj.get("status_code") or error_obj.get("status")
    try:
        status_code = int(raw_status)
    except (TypeError, ValueError):
        status_code = None
    if status_code is not None and 100 <= status_code <= 599:
        return status_code

    error_code = str(error_obj.get("code") or "").strip().lower()
    if error_code in {"rate_limit_exceeded", "billing_hard_limit_reached", "insufficient_quota"}:
        return 429
    if error_code in {"invalid_api_key", "incorrect_api_key_provided", "authentication_error"}:
        return 401
    if error_code in {"permission_denied"}:
        return 403
    if error_code in {"invalid_request_error", "invalid_type", "unsupported_parameter", "context_length_exceeded"}:
        return 400
    if error_code in {"model_not_found", "not_found_error"}:
        return 404

    error_type = str(error_obj.get("type") or "").strip().lower()
    if error_type in {"tokens", "rate_limit_error"}:
        return 429
    if error_type == "authentication_error":
        return 401
    if error_type == "permission_error":
        return 403
    if error_type == "invalid_request_error":
        return 400
    if error_type == "not_found_error":
        return 404

    message = str(error_obj.get("message") or "").lower()
    if "rate limit" in message or "too many requests" in message:
        return 429
    if "invalid" in message or "unsupported" in message:
        return 400
    if "not found" in message:
        return 404
    if "permission" in message or "forbidden" in message:
        return 403
    if "auth" in message or "api key" in message or "unauthorized" in message:
        return 401

    return default_status


async def error_handling_wrapper(
    generator,
    channel_id,
    engine,
    stream,
    error_triggers,
    keepalive_interval=None,
    last_message_role=None,
):
    async def new_generator(first_item=None, with_keepalive=False, wait_task=None, timeout=3):
        try:
            if first_item:
                yield await ensure_string(first_item)

            if with_keepalive:
                yield ": keepalive\n\n"
                while True:
                    try:
                        item, status = await wait_for_timeout(generator, timeout=timeout, wait_task=wait_task)
                        if status == "timeout":
                            wait_task = item
                            yield ": keepalive\n\n"
                        else:
                            yield await ensure_string(item)
                            wait_task = None
                    except asyncio.CancelledError:
                        logger.debug("provider: %-11s Stream cancelled by client in main loop", channel_id)
                        break
                    except Exception:
                        break
            else:
                try:
                    async for item in generator:
                        yield await ensure_string(item)
                except asyncio.CancelledError:
                    logger.debug("provider: %-11s Stream cancelled by client", channel_id)
                    return
                except (
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    httpx.ReadTimeout,
                    httpx.WriteError,
                    httpx.ProtocolError,
                    h2.exceptions.ProtocolError,
                ) as exc:
                    logger.error("provider: %-11s Network error in new_generator: %s", channel_id, exc)
                    yield "data: [DONE]\n\n"
                    return
        finally:
            await _cancel_pending_task_safely(wait_task, channel_id, "error_handling_wrapper")
            await _close_async_iterator_safely(generator, channel_id, "error_handling_wrapper")

    start_time = time_module.time()
    first_item_str = None
    try:
        if keepalive_interval and stream:
            first_item, status = await wait_for_timeout(generator, timeout=keepalive_interval)
            if status == "timeout":
                return new_generator(None, with_keepalive=True, wait_task=first_item, timeout=keepalive_interval), 3.1415
        else:
            first_item = await generator.__anext__()

        first_response_time = time_module.time() - start_time
        first_item_str = first_item
        if isinstance(first_item_str, (bytes, bytearray)):
            if identify_audio_format(first_item_str) in ["MP3", "MP3 with ID3", "OPUS", "AAC (ADIF)", "AAC (ADTS)", "FLAC", "WAV"]:
                await _close_async_iterator_safely(generator, channel_id, "direct bytes response")
                return first_item, first_response_time
            first_item_str = first_item_str.decode("utf-8")

        is_named_sse_frame = (
            isinstance(first_item_str, str)
            and stream
            and engine == "dalle"
            and first_item_str.lstrip().startswith("event:")
        )
        is_comment_sse_frame = (
            isinstance(first_item_str, str)
            and stream
            and is_sse_comment_frame(first_item_str)
        )
        if isinstance(first_item_str, str) and not is_comment_sse_frame and not is_named_sse_frame:
            if first_item_str.startswith("data:"):
                first_item_str = first_item_str.lstrip("data: ")
            if first_item_str.startswith("[DONE]"):
                logger.error("provider: %-11s error_handling_wrapper [DONE]!", channel_id)
                raise StopAsyncIteration
            try:
                encode_first_item_str = first_item_str.encode().decode("unicode-escape")
            except UnicodeDecodeError:
                encode_first_item_str = first_item_str
                logger.error("provider: %-11s error UnicodeDecodeError: %s", channel_id, first_item_str)
            if any(trigger in encode_first_item_str for trigger in error_triggers):
                logger.error("provider: %-11s error const string: %s", channel_id, encode_first_item_str)
                raise StopAsyncIteration
            try:
                first_item_str = await asyncio.to_thread(json.loads, first_item_str)
            except json.JSONDecodeError:
                logger.error("provider: %-11s error_handling_wrapper JSONDecodeError! %r", channel_id, first_item_str)
                raise StopAsyncIteration

            status_code = safe_get(first_item_str, "base_resp", "status_code", default=200)
            if status_code != 200:
                if status_code == 2013:
                    status_code = 400
                if status_code == 1008:
                    status_code = 429
                detail = safe_get(first_item_str, "base_resp", "status_msg", default="no error returned")
                raise HTTPException(status_code=status_code, detail=f"{detail}"[:1000])

        if isinstance(first_item_str, dict) and safe_get(first_item_str, "base_resp", "status_msg", default=None) == "success":
            full_audio_hex = safe_get(first_item_str, "data", "audio", default=None)
            audio_bytes = bytes.fromhex(full_audio_hex)
            await _close_async_iterator_safely(generator, channel_id, "direct audio response")
            return audio_bytes, first_response_time

        if isinstance(first_item_str, dict) and "error" in first_item_str and first_item_str.get("error") != {"message": "", "type": "", "param": "", "code": None}:
            status_code = first_item_str.get("status_code") or _infer_openai_like_error_status(first_item_str.get("error"), default_status=500)
            detail = first_item_str.get("details", f"{first_item_str}")
            raise HTTPException(status_code=status_code, detail=f"{detail}"[:1000])

        if isinstance(first_item_str, dict) and safe_get(first_item_str, "choices", 0, "error", default=None):
            status_code = _infer_openai_like_error_status(
                safe_get(first_item_str, "choices", 0, "error", default={}) or {},
                default_status=500,
            )
            detail = safe_get(first_item_str, "choices", 0, "error", "message", default=f"{first_item_str}")
            raise HTTPException(status_code=status_code, detail=f"{detail}"[:1000])

        finish_reason = safe_get(first_item_str, "choices", 0, "finish_reason", default=None)
        if isinstance(first_item_str, dict) and finish_reason == "PROHIBITED_CONTENT":
            raise HTTPException(status_code=400, detail="PROHIBITED_CONTENT")

        if (
            isinstance(first_item_str, dict)
            and finish_reason == "stop"
            and not safe_get(first_item_str, "choices", 0, "message", "content", default=None)
            and not safe_get(first_item_str, "choices", 0, "message", "audio", default=None)
            and not safe_get(first_item_str, "choices", 0, "message", "refusal", default=None)
            and not safe_get(first_item_str, "choices", 0, "message", "tool_calls", default=None)
            and not safe_get(first_item_str, "choices", 0, "delta", "tool_calls", default=None)
            and not safe_get(first_item_str, "choices", 0, "delta", "content", default=None)
            and not safe_get(first_item_str, "choices", 0, "delta", "audio", default=None)
            and last_message_role != "assistant"
        ):
            raise StopAsyncIteration

        if isinstance(first_item_str, dict) and engine not in ["tts", "embedding", "dalle", "moderation", "whisper", "search"] and not stream:
            if any(trigger in str(first_item_str) for trigger in error_triggers):
                logger.error("provider: %-11s error const string: %s", channel_id, first_item_str)
                raise StopAsyncIteration
            content = safe_get(first_item_str, "choices", 0, "message", "content", default=None)
            reasoning_content = safe_get(first_item_str, "choices", 0, "message", "reasoning_content", default=None)
            b64_json = safe_get(first_item_str, "data", 0, "b64_json", default=None)
            tool_calls = safe_get(first_item_str, "choices", 0, "message", "tool_calls", default=None)
            audio = safe_get(first_item_str, "choices", 0, "message", "audio", default=None)
            refusal = safe_get(first_item_str, "choices", 0, "message", "refusal", default=None)
            if (
                (content == "" or content is None)
                and (tool_calls == "" or tool_calls is None)
                and (reasoning_content == "" or reasoning_content is None)
                and b64_json is None
                and (audio == "" or audio is None)
                and (refusal == "" or refusal is None)
            ):
                raise StopAsyncIteration

        return new_generator(first_item), first_response_time

    except StopAsyncIteration:
        await _close_async_iterator_safely(generator, channel_id, "empty first response")
        logger.warning("provider: %-11s empty response [%s]: %s", channel_id, type(first_item_str), first_item_str)
        raise HTTPException(status_code=502, detail="Upstream server returned an empty response.")
    except BaseException:
        await _close_async_iterator_safely(generator, channel_id, "first response handling")
        raise
