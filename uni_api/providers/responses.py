import re
import random
import string
import base64
import uuid
import asyncio
from datetime import datetime
from urllib.parse import urlparse
from typing import Any, AsyncIterator, Callable

from core.log_config import logger
from uni_api.serialization import json

from core.utils import (
    safe_get,
    IncrementalSSEParser,
    generate_sse_response,
    generate_no_stream_response,
    end_of_line,
    parse_json_safely,
    parse_sse_event,
)
from uni_api.providers.normalization import build_openai_audio_object, normalize_gemini_parts
from uni_api.streaming.sse import is_sse_comment_frame
from uni_api.streaming.cleanup import (
    BACKGROUND_STREAM_CLEANUP_TASKS as _BACKGROUND_STREAM_CLEANUP_TASKS,
    await_stream_cleanup_safely as _await_stream_cleanup_safely,
    call_cleanup_safely as _call_cleanup_safely,
    close_async_iterator_safely as _close_async_iterator_safely,
    drain_current_task_cancellation as _drain_current_task_cancellation,
    force_close_response_httpcore_stream_chain_safely as _force_close_response_httpcore_stream_chain_safely,
    force_release_httpcore_pool_request_safely as _force_release_httpcore_pool_request_safely,
    track_background_stream_cleanup_task as _track_background_stream_cleanup_task,
    yield_from_stream as _yield_from_stream,
)
from uni_api.streaming.chat_completion_events import (
    responses_usage_to_chat_completion_usage as _responses_usage_to_chat_completion_usage,
)
from uni_api.streaming.responses_events import (
    mime_type_from_output_format as _mime_type_from_output_format,
    normalize_optional_text as _normalize_optional_text,
    stream_responses_to_chat_completions as _stream_responses_to_chat_completions,
)

ResponseHeadersSink = Callable[[Any], None]


def _capture_response_headers(response_headers_sink: ResponseHeadersSink | None, headers: Any) -> None:
    if response_headers_sink is not None and headers is not None:
        response_headers_sink(headers)


def _normalize_search_item_defaults(item: dict) -> dict:
    normalized = dict(item or {})
    normalized.setdefault("title", "")
    normalized.setdefault("url", "")
    normalized.setdefault("description", "")
    normalized.setdefault("content", "")
    normalized.setdefault("usage", None)
    normalized.setdefault("score", None)
    normalized.setdefault("raw_content", None)
    return normalized

def normalize_search_response(url: str, response_json: object) -> dict:
    """
    Normalizes different search providers into a Jina-like shape:
      { code, status, data: [{title,url,description,content,...}], meta: {...} }
    """
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()

    # Tavily shape:
    # {query, results:[{url,title,content,score,raw_content}], response_time, request_id, ...}
    if isinstance(response_json, dict) and (host.endswith("tavily.com") or "results" in response_json):
        results = response_json.get("results") or []
        data = []
        for r in results:
            if not isinstance(r, dict):
                continue
            title = r.get("title") or ""
            link = r.get("url") or ""
            content = r.get("content") or ""
            description = content
            if isinstance(description, str) and len(description) > 240:
                description = description[:237] + "..."
            item = {
                "title": title,
                "url": link,
                "description": description or "",
                "content": content or "",
            }
            # keep Tavily extra fields at the same level for convenience
            for k, v in r.items():
                if k not in item:
                    item[k] = v
            data.append(_normalize_search_item_defaults(item))

        meta = {
            "provider": "tavily",
            "query": response_json.get("query"),
            "answer": response_json.get("answer"),
            "follow_up_questions": response_json.get("follow_up_questions"),
            "images": response_json.get("images"),
            "response_time": response_json.get("response_time"),
            "request_id": response_json.get("request_id"),
        }
        # preserve any additional top-level fields
        for k, v in response_json.items():
            if k not in meta and k != "results":
                meta[k] = v

        return {
            "code": 200,
            "status": 20000,
            "data": data,
            "meta": meta,
        }

    # Jina (already close to desired format).
    if isinstance(response_json, dict) and "data" in response_json:
        out = dict(response_json)
        out.setdefault("code", 200)
        out.setdefault("status", 20000)
        meta = out.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("provider", "jina")
        out["meta"] = meta
        normalized_data = []
        for item in (out.get("data") or []):
            if isinstance(item, dict):
                normalized_data.append(_normalize_search_item_defaults(item))
        out["data"] = normalized_data
        return out

    # Fallback: wrap unknown shapes without losing data.
    return {
        "code": 200,
        "status": 20000,
        "data": [],
        "meta": {"provider": "unknown", "raw": response_json},
    }

def _responses_output_to_text(response_json: dict) -> tuple[str, str]:
    """
    Best-effort extraction of text + reasoning text from an OpenAI Responses-style response.
    Returns: (content, reasoning_content)
    """
    if not isinstance(response_json, dict):
        return "", ""

    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text:
        content_parts.append(output_text)

    output = response_json.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type in ("output_text", "text") and item.get("text"):
                content_parts.append(str(item.get("text")))
                continue
            if item_type == "image_generation_call":
                result_b64 = item.get("result")
                if isinstance(result_b64, str) and result_b64:
                    mime_type = _mime_type_from_output_format(
                        _normalize_optional_text(item.get("output_format"))
                    )
                    content_parts.append(f"![image](data:{mime_type};base64,{result_b64})")
                continue
            if item_type in ("reasoning_summary_text", "reasoning_text") and item.get("text"):
                reasoning_parts.append(str(item.get("text")))
                continue

            if item_type != "message":
                continue

            role = (item.get("role") or "").lower()
            if role and role not in ("assistant", "model"):
                continue

            msg_content = item.get("content")
            if isinstance(msg_content, str) and msg_content:
                content_parts.append(msg_content)
                continue
            if not isinstance(msg_content, list):
                continue
            for part in msg_content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("output_text", "text") and part.get("text"):
                    content_parts.append(str(part.get("text")))
                elif part_type in ("reasoning_summary_text", "reasoning_text") and part.get("text"):
                    reasoning_parts.append(str(part.get("text")))

    return "".join(content_parts), "".join(reasoning_parts)

def _is_responses_api_call(url: str, payload: dict) -> bool:
    if "v1/responses" in (url or ""):
        return True
    if isinstance(payload, dict) and "input" in payload and "messages" not in payload:
        return True
    return False


async def check_response(response, error_log):
    if response and not (200 <= response.status_code < 300):
        error_message = await response.aread()
        error_str = error_message.decode('utf-8', errors='replace')
        try:
            error_json = await asyncio.to_thread(json.loads, error_str)
        except json.JSONDecodeError:
            error_json = error_str
        return {"error": f"{error_log} HTTP Error", "status_code": response.status_code, "details": error_json}
    return None

async def gemini_json_poccess(response_json):
    promptTokenCount = 0
    candidatesTokenCount = 0
    totalTokenCount = 0
    cachedContentTokenCount = 0
    thoughtsTokenCount = 0

    json_data = safe_get(response_json, "candidates", 0, "content", default=None)
    finishReason = safe_get(response_json, "candidates", 0 , "finishReason", default=None)
    usage_metadata = response_json.get("usageMetadata") if isinstance(response_json, dict) else None
    if finishReason and isinstance(usage_metadata, dict):
        promptTokenCount = usage_metadata.get("promptTokenCount", promptTokenCount) or 0
        candidatesTokenCount = usage_metadata.get("candidatesTokenCount", candidatesTokenCount) or 0
        totalTokenCount = usage_metadata.get("totalTokenCount", totalTokenCount) or 0
        cachedContentTokenCount = usage_metadata.get("cachedContentTokenCount", cachedContentTokenCount) or 0
        thoughtsTokenCount = usage_metadata.get("thoughtsTokenCount", thoughtsTokenCount) or 0
        if finishReason != "STOP":
            logger.error(f"finishReason: {finishReason}")

    parts_list = safe_get(json_data, "parts", default=[])
    normalized = normalize_gemini_parts(parts_list if isinstance(parts_list, list) else [])

    blockReason = safe_get(json_data, 0, "promptFeedback", "blockReason", default=None)

    return (
        normalized.is_thinking,
        normalized.reasoning_content,
        normalized.content,
        normalized.image_base64,
        normalized.audio_wav_base64,
        normalized.function_call.name,
        normalized.function_call.arguments_json,
        normalized.function_call.call_id,
        finishReason,
        blockReason,
        promptTokenCount,
        candidatesTokenCount,
        totalTokenCount,
        cachedContentTokenCount,
        thoughtsTokenCount,
    )

async def fetch_gemini_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_gemini_response_stream")
        if error_message:
            yield error_message
            return
        buffer = ""
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0
        cachedContentTokenCount = 0
        thoughtsTokenCount = 0
        parts_json = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            if buffer and "\n" not in buffer:
                buffer += "\n"

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.startswith("data: "):
                    parts_json = line.lstrip("data: ").strip()
                    try:
                        response_json = await asyncio.to_thread(json.loads, parts_json)
                    except json.JSONDecodeError:
                        # logger.error(f"JSON decode error: {parts_json}")
                        continue
                else:
                    parts_json += line
                    parts_json = parts_json.lstrip("[,")
                    try:
                        response_json = await asyncio.to_thread(json.loads, parts_json)
                    except json.JSONDecodeError:
                        continue

                # https://ai.google.dev/api/generate-content?hl=zh-cn#FinishReason
                (
                    is_thinking,
                    reasoning_content,
                    content,
                    image_base64,
                    audio_b64_wav,
                    function_call_name,
                    function_full_response,
                    tools_id,
                    finishReason,
                    blockReason,
                    promptTokenCount,
                    candidatesTokenCount,
                    totalTokenCount,
                    cachedContentTokenCount,
                    thoughtsTokenCount,
                ) = await gemini_json_poccess(response_json)

                if is_thinking and reasoning_content:
                    sse_string = await generate_sse_response(timestamp, model, reasoning_content=reasoning_content)
                    yield sse_string
                if not image_base64 and content:
                    sse_string = await generate_sse_response(timestamp, model, content=content)
                    yield sse_string

                if image_base64:
                    if "flash-image" not in model and "pro-image" not in model:
                        completion_tokens = candidatesTokenCount + thoughtsTokenCount
                        openai_total_tokens = totalTokenCount or (promptTokenCount + completion_tokens)
                        yield await generate_no_stream_response(
                            timestamp,
                            model,
                            content=content,
                            tools_id=None,
                            function_call_name=None,
                            function_call_content=None,
                            role=None,
                            total_tokens=openai_total_tokens,
                            prompt_tokens=promptTokenCount,
                            completion_tokens=completion_tokens,
                            cached_tokens=cachedContentTokenCount,
                            reasoning_tokens=thoughtsTokenCount,
                            image_base64=image_base64,
                        )
                    else:
                        sse_string = await generate_sse_response(timestamp, model, content=f"\n![image](data:image/png;base64,{image_base64})")
                        yield sse_string
                if audio_b64_wav:
                    audio_obj = build_openai_audio_object(audio_b64_wav, transcript=content or None)
                    yield await generate_no_stream_response(
                        timestamp,
                        model,
                        content=content or None,
                        tools_id=None,
                        function_call_name=None,
                        function_call_content=None,
                        role="assistant",
                        total_tokens=totalTokenCount or (promptTokenCount + candidatesTokenCount + thoughtsTokenCount),
                        prompt_tokens=promptTokenCount,
                        completion_tokens=candidatesTokenCount + thoughtsTokenCount,
                        cached_tokens=cachedContentTokenCount,
                        reasoning_tokens=thoughtsTokenCount,
                        audio=audio_obj,
                    )

                if function_call_name:
                    sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=tools_id, function_call_name=function_call_name)
                    yield sse_string
                if function_full_response:
                    sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=tools_id, function_call_name=None, function_call_content=function_full_response)
                    yield sse_string

                if parts_json == "[]" or blockReason == "PROHIBITED_CONTENT":
                    sse_string = await generate_sse_response(timestamp, model, stop="PROHIBITED_CONTENT")
                    yield sse_string
                elif finishReason:
                    sse_string = await generate_sse_response(timestamp, model, stop="stop")
                    yield sse_string
                    break

                parts_json = ""

        completion_tokens = candidatesTokenCount + thoughtsTokenCount
        openai_total_tokens = totalTokenCount or (promptTokenCount + completion_tokens)
        sse_string = await generate_sse_response(
            timestamp,
            model,
            None,
            None,
            None,
            None,
            None,
            openai_total_tokens,
            promptTokenCount,
            completion_tokens,
            cached_tokens=cachedContentTokenCount,
            reasoning_tokens=thoughtsTokenCount,
        )
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_vertex_claude_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_vertex_claude_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        revicing_function_call = False
        function_full_response = "{"
        need_function_call = False
        is_finish = False
        promptTokenCount = 0
        candidatesTokenCount = 0
        totalTokenCount = 0

        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info(f"{line}")

                if line and '\"finishReason\": \"' in line:
                    is_finish = True
                if is_finish and '\"promptTokenCount\": ' in line:
                    json_data = parse_json_safely( "{" + line + "}")
                    promptTokenCount = json_data.get('promptTokenCount', 0)
                if is_finish and '\"candidatesTokenCount\": ' in line:
                    json_data = parse_json_safely( "{" + line + "}")
                    candidatesTokenCount = json_data.get('candidatesTokenCount', 0)
                if is_finish and '\"totalTokenCount\": ' in line:
                    json_data = parse_json_safely( "{" + line + "}")
                    totalTokenCount = json_data.get('totalTokenCount', 0)

                if line and '\"text\": \"' in line and not is_finish:
                    try:
                        json_data = await asyncio.to_thread(json.loads, "{" + line.strip().rstrip(",") + "}")
                        content = json_data.get('text', '')
                        sse_string = await generate_sse_response(timestamp, model, content=content)
                        yield sse_string
                    except json.JSONDecodeError:
                        logger.error(f"无法解析JSON: {line}")

                if line and ('\"type\": \"tool_use\"' in line or revicing_function_call):
                    revicing_function_call = True
                    need_function_call = True
                    if ']' in line:
                        revicing_function_call = False
                        continue

                    function_full_response += line

        if need_function_call:
            function_call = await asyncio.to_thread(json.loads, function_full_response)
            function_call_name = function_call["name"]
            function_call_id = function_call["id"]
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=function_call_id, function_call_name=function_call_name)
            yield sse_string
            function_full_response = await asyncio.to_thread(json.dumps, function_call["input"])
            sse_string = await generate_sse_response(timestamp, model, content=None, tools_id=function_call_id, function_call_name=None, function_call_content=function_full_response)
            yield sse_string

        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, totalTokenCount, promptTokenCount, candidatesTokenCount)
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_gpt_response_stream(client, url, headers, payload, timeout, response_headers_sink: ResponseHeadersSink | None = None):
    timestamp = int(datetime.timestamp(datetime.now()))
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    json_payload = await asyncio.to_thread(json.dumps, payload)
    response = None
    completed_normally = False
    input_tokens = 0
    output_tokens = 0
    try:
        async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
            _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
            error_message = await check_response(response, "fetch_gpt_response_stream")
            if error_message:
                yield error_message
                return

            if _is_responses_api_call(url, payload):
                async for chunk in _stream_responses_to_chat_completions(
                    response.aiter_text(),
                    request_model=payload["model"],
                ):
                    yield chunk
                completed_normally = True
                return

            buffer = ""
            enter_buffer = ""

            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    # logger.info("line: %s", repr(line))
                    if line.startswith(": keepalive"):
                        yield line + end_of_line
                        continue
                    if line and not is_sse_comment_frame(line) and (result:=line.lstrip("data: ").strip()) and not line.startswith("event: "):
                        if result.strip() == "[DONE]":
                            completed_normally = True
                            break
                        line = await asyncio.to_thread(json.loads, result)
                        line['id'] = f"chatcmpl-{random_str}"

                        # v1/responses
                        if line.get("type") == "response.reasoning_summary_text.delta" and line.get("delta"):
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=line.get("delta"))
                            yield sse_string
                            continue
                        elif line.get("type") == "response.reasoning_summary_text.done":
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content="\n\n")
                            yield sse_string
                            continue
                        elif line.get("type") == "response.output_text.delta" and line.get("delta"):
                            sse_string = await generate_sse_response(timestamp, payload["model"], content=line.get("delta"))
                            yield sse_string
                            continue
                        elif line.get("type") == "response.output_text.done":
                            sse_string = await generate_sse_response(timestamp, payload["model"], stop="stop")
                            yield sse_string
                            continue
                        elif line.get("type") == "response.completed":
                            input_tokens = safe_get(line, "response", "usage", "input_tokens", default=0)
                            output_tokens = safe_get(line, "response", "usage", "output_tokens", default=0)
                            continue
                        elif line.get("type", "").startswith("response."):
                            continue

                        # 处理 <think> 标签
                        content = safe_get(line, "choices", 0, "delta", "content", default="")
                        if "<think>" in content:
                            is_thinking = True
                            ark_tag = True
                            content = content.replace("<think>", "")
                        if "</think>" in content:
                            end_think_reasoning_content = ""
                            end_think_content = ""
                            is_thinking = False

                            if content.rstrip('\n').endswith("</think>"):
                                end_think_reasoning_content = content.replace("</think>", "").rstrip('\n')
                            elif content.lstrip('\n').startswith("</think>"):
                                end_think_content = content.replace("</think>", "").lstrip('\n')
                            else:
                                end_think_reasoning_content = content.split("</think>")[0]
                                end_think_content = content.split("</think>")[1]

                            if end_think_reasoning_content:
                                sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=end_think_reasoning_content)
                                yield sse_string
                            if end_think_content:
                                sse_string = await generate_sse_response(timestamp, payload["model"], content=end_think_content)
                                yield sse_string
                            continue
                        if is_thinking and ark_tag:
                            if not has_send_thinking:
                                content = content.replace("\n\n", "")
                            if content:
                                sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                                yield sse_string
                                has_send_thinking = True
                            continue

                        # 处理 poe thinking 标签
                        if "Thinking..." in content and "\n> " in content:
                            is_thinking = True
                            content = content.replace("Thinking...", "").replace("\n> ", "")
                        if is_thinking and "\n\n" in content and not ark_tag:
                            is_thinking = False
                        if is_thinking and not ark_tag:
                            content = content.replace("\n> ", "")
                            if not has_send_thinking:
                                content = content.replace("\n", "")
                            if content:
                                sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                                yield sse_string
                                has_send_thinking = True
                            continue

                        no_stream_content = safe_get(line, "choices", 0, "message", "content", default=None)
                        openrouter_reasoning = safe_get(line, "choices", 0, "delta", "reasoning", default="")
                        openrouter_base64_image = safe_get(line, "choices", 0, "delta", "images", 0, "image_url", "url", default="")
                        if openrouter_base64_image:
                            image_data_url = openrouter_base64_image if openrouter_base64_image.startswith("data:") else f"data:image/png;base64,{openrouter_base64_image}"
                            sse_string = await generate_sse_response(timestamp, payload["model"], content=f"\n![image]({image_data_url})")
                            yield sse_string
                            continue
                        azure_databricks_claude_summary_content = safe_get(line, "choices", 0, "delta", "content", 0, "summary", 0, "text", default="")
                        azure_databricks_claude_signature_content = safe_get(line, "choices", 0, "delta", "content", 0, "summary", 0, "signature", default="")
                        # print("openrouter_reasoning", repr(openrouter_reasoning), openrouter_reasoning.endswith("\\\\"), openrouter_reasoning.endswith("\\"))
                        if azure_databricks_claude_signature_content:
                            pass
                        elif azure_databricks_claude_summary_content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=azure_databricks_claude_summary_content)
                            yield sse_string
                        elif openrouter_reasoning:
                            if openrouter_reasoning.endswith("\\"):
                                enter_buffer += openrouter_reasoning
                                continue
                            elif enter_buffer.endswith("\\") and openrouter_reasoning == 'n':
                                enter_buffer += "n"
                                continue
                            elif enter_buffer.endswith("\\n") and openrouter_reasoning == '\\n':
                                enter_buffer += "\\n"
                                continue
                            elif enter_buffer.endswith("\\n\\n"):
                                openrouter_reasoning = '\n\n' + openrouter_reasoning
                                enter_buffer = ""
                            elif enter_buffer:
                                openrouter_reasoning = enter_buffer + openrouter_reasoning
                                enter_buffer = ''
                            openrouter_reasoning = openrouter_reasoning.replace("\\n", "\n")

                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=openrouter_reasoning)
                            yield sse_string
                        elif no_stream_content and not has_send_thinking:
                            sse_string = await generate_sse_response(safe_get(line, "created", default=None), safe_get(line, "model", default=None), content=no_stream_content)
                            yield sse_string
                        else:
                            if no_stream_content:
                                del line["choices"][0]["message"]
                            json_line = await asyncio.to_thread(json.dumps, line)
                            yield "data: " + json_line.strip() + end_of_line
    finally:
        if response is not None and not completed_normally:
            await _force_close_response_httpcore_stream_chain_safely(
                response,
                label="gpt upstream response stream",
            )

    if input_tokens and output_tokens:
        sse_string = await generate_sse_response(timestamp, payload["model"], None, None, None, None, None, total_tokens=input_tokens + output_tokens, prompt_tokens=input_tokens, completion_tokens=output_tokens)
        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_azure_response_stream(client, url, headers, payload, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    is_thinking = False
    has_send_thinking = False
    ark_tag = False
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_azure_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        sse_string = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line and not is_sse_comment_frame(line) and (result:=line.lstrip("data: ").strip()):
                    if result.strip() == "[DONE]":
                        break
                    line = await asyncio.to_thread(json.loads, result)
                    no_stream_content = safe_get(line, "choices", 0, "message", "content", default="")
                    content = safe_get(line, "choices", 0, "delta", "content", default="")

                    # 处理 <think> 标签
                    if "<think>" in content:
                        is_thinking = True
                        ark_tag = True
                        content = content.replace("<think>", "")
                    if "</think>" in content:
                        is_thinking = False
                        content = content.replace("</think>", "")
                        if not content:
                            continue
                    if is_thinking and ark_tag:
                        if not has_send_thinking:
                            content = content.replace("\n\n", "")
                        if content:
                            sse_string = await generate_sse_response(timestamp, payload["model"], reasoning_content=content)
                            yield sse_string
                            has_send_thinking = True
                        continue

                    if no_stream_content or content or sse_string:
                        input_tokens = safe_get(line, "usage", "prompt_tokens", default=0)
                        output_tokens = safe_get(line, "usage", "completion_tokens", default=0)
                        total_tokens = safe_get(line, "usage", "total_tokens", default=0)
                        sse_string = await generate_sse_response(timestamp, safe_get(line, "model", default=None), content=no_stream_content or content, total_tokens=total_tokens, prompt_tokens=input_tokens, completion_tokens=output_tokens)
                        yield sse_string
                    else:
                        if no_stream_content:
                            del line["choices"][0]["message"]
                        json_line = await asyncio.to_thread(json.dumps, line)
                        yield "data: " + json_line.strip() + end_of_line
    yield "data: [DONE]" + end_of_line

async def fetch_cloudflare_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_cloudflare_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                if line.startswith("data:"):
                    line = line.lstrip("data: ")
                    if line == "[DONE]":
                        break
                    resp: dict = await asyncio.to_thread(json.loads, line)
                    message = resp.get("response")
                    if message:
                        sse_string = await generate_sse_response(timestamp, model, content=message)
                        yield sse_string
    yield "data: [DONE]" + end_of_line

async def fetch_cohere_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_cohere_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info("line: %s", repr(line))
                resp: dict = await asyncio.to_thread(json.loads, line)
                if resp.get("is_finished"):
                    break
                if resp.get("event_type") == "text-generation":
                    message = resp.get("text")
                    sse_string = await generate_sse_response(timestamp, model, content=message)
                    yield sse_string
    yield "data: [DONE]" + end_of_line

async def fetch_claude_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_claude_response_stream")
        if error_message:
            yield error_message
            return
        buffer = ""
        input_tokens = 0
        cache_read_input_tokens = 0
        async for chunk in response.aiter_text():
            # logger.info(f"chunk: {repr(chunk)}")
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                # logger.info(line)

                if line.startswith("data:") and (line := line.lstrip("data: ")):
                    resp: dict = await asyncio.to_thread(json.loads, line)

                    input_tokens = input_tokens or safe_get(resp, "message", "usage", "input_tokens", default=0) or safe_get(resp, "usage", "input_tokens", default=0)
                    cache_read_input_tokens = cache_read_input_tokens or safe_get(resp, "message", "usage", "cache_read_input_tokens", default=0) or safe_get(resp, "usage", "cache_read_input_tokens", default=0)
                    output_tokens = safe_get(resp, "usage", "output_tokens", default=0)
                    if output_tokens:
                        total_tokens = input_tokens + output_tokens
                        sse_string = await generate_sse_response(
                            timestamp,
                            model,
                            None,
                            None,
                            None,
                            None,
                            None,
                            total_tokens,
                            input_tokens,
                            output_tokens,
                            cached_tokens=cache_read_input_tokens,
                        )
                        yield sse_string
                        break

                    text = safe_get(resp, "delta", "text", default="")
                    if text:
                        sse_string = await generate_sse_response(timestamp, model, text)
                        yield sse_string
                        continue

                    function_call_name = safe_get(resp, "content_block", "name", default=None)
                    tools_id = safe_get(resp, "content_block", "id", default=None)
                    if tools_id and function_call_name:
                        sse_string = await generate_sse_response(timestamp, model, None, tools_id, function_call_name, None)
                        yield sse_string

                    thinking_content = safe_get(resp, "delta", "thinking", default="")
                    if thinking_content:
                        sse_string = await generate_sse_response(timestamp, model, reasoning_content=thinking_content)
                        yield sse_string

                    function_call_content = safe_get(resp, "delta", "partial_json", default="")
                    if function_call_content:
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, function_call_content)
                        yield sse_string

    yield "data: [DONE]" + end_of_line

async def fetch_aws_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_aws_response_stream")
        if error_message:
            yield error_message
            return

        buffer = ""
        async for line in response.aiter_text():
            buffer += line
            while "\r" in buffer:
                line, buffer = buffer.split("\r", 1)
                if not line or \
                line.strip() == "" or \
                line.strip().startswith(':content-type') or \
                line.strip().startswith(':event-type'): # 过滤掉完全空的行或只有空白的行
                    continue

                json_match = re.search(r'event{.*?}', line)
                if not json_match:
                    continue
                try:
                    chunk_data = await asyncio.to_thread(json.loads, json_match.group(0).lstrip('event'))
                except json.JSONDecodeError:
                    logger.error(f"DEBUG json.JSONDecodeError: {json_match.group(0).lstrip('event')!r}")
                    continue

                # --- 后续处理逻辑不变 ---
                if "bytes" in chunk_data:
                    # 解码 Base64 编码的字节
                    decoded_bytes = base64.b64decode(chunk_data["bytes"])
                    # 将解码后的字节再次解析为 JSON
                    payload_chunk = await asyncio.to_thread(json.loads, decoded_bytes.decode('utf-8'))
                    # print(f"DEBUG payload_chunk: {payload_chunk!r}")

                    text = safe_get(payload_chunk, "delta", "text", default="")
                    if text:
                        sse_string = await generate_sse_response(timestamp, model, text, None, None)
                        yield sse_string

                    usage = safe_get(payload_chunk, "amazon-bedrock-invocationMetrics", default="")
                    if usage:
                        input_tokens = usage.get("inputTokenCount", 0)
                        output_tokens = usage.get("outputTokenCount", 0)
                        total_tokens = input_tokens + output_tokens
                        sse_string = await generate_sse_response(timestamp, model, None, None, None, None, None, total_tokens, input_tokens, output_tokens)
                        yield sse_string

    yield "data: [DONE]" + end_of_line

def _pop_multipart_payload(payload):
    if not isinstance(payload, dict) or "__multipart_files__" not in payload:
        return None
    files = payload.pop("__multipart_files__", None) or []
    data = payload.pop("__multipart_data__", None) or []
    return data, files

def _quote_multipart_header_value(value) -> str:
    text = str(value or "")
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', "%22")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )

def _read_multipart_file_content(content) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    if hasattr(content, "seek"):
        try:
            content.seek(0)
        except Exception:
            pass
    if hasattr(content, "read"):
        value = content.read()
        if isinstance(value, str):
            return value.encode("utf-8")
        return bytes(value or b"")
    return bytes(content or b"")

def _build_multipart_content(headers: dict, data: list, files: list) -> tuple[dict, bytes]:
    boundary = f"----uniapi-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for key, value in data:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{_quote_multipart_header_value(key)}"\r\n\r\n'.encode("utf-8")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for key, file_value in files:
        filename = "upload"
        content_type = "application/octet-stream"
        content = file_value
        if isinstance(file_value, (tuple, list)):
            if len(file_value) >= 1 and file_value[0]:
                filename = str(file_value[0])
            if len(file_value) >= 2:
                content = file_value[1]
            if len(file_value) >= 3 and file_value[2]:
                content_type = str(file_value[2])

        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{_quote_multipart_header_value(key)}"; '
                f'filename="{_quote_multipart_header_value(filename)}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(_read_multipart_file_content(content))
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    request_headers = dict(headers or {})
    for key in list(request_headers.keys()):
        if str(key).lower() == "content-type":
            request_headers.pop(key, None)
    request_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    return request_headers, b"".join(chunks)


async def _fetch_search_response(client, url, headers, payload, timeout, response_headers_sink: ResponseHeadersSink | None = None):
    content_type = None
    for key in ("Content-Type", "content-type"):
        if key in (headers or {}):
            content_type = headers.get(key)
            break
    if content_type and "application/json" in str(content_type).lower():
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)
    else:
        response = await client.get(url, headers=headers, params=payload, timeout=timeout)
    _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
    return response


async def _fetch_post_response(client, url, headers, payload, timeout, response_headers_sink: ResponseHeadersSink | None = None):
    multipart_payload = _pop_multipart_payload(payload)
    if multipart_payload is not None:
        data, files = multipart_payload
        multipart_headers, multipart_content = _build_multipart_content(headers, data, files)
        response = await client.post(url, headers=multipart_headers, content=multipart_content, timeout=timeout)
        _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
        return response
    if payload.get("file"):
        file = payload.pop("file")
        response = await client.post(url, headers=headers, data=payload, files={"file": file}, timeout=timeout)
        _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
        return response
    json_payload = await asyncio.to_thread(json.dumps, payload)
    response = await client.post(url, headers=headers, content=json_payload, timeout=timeout)
    _capture_response_headers(response_headers_sink, getattr(response, "headers", None))
    return response


async def _yield_search_response(response, url):
    try:
        response_json = response.json()
    except Exception:
        response_json = {"text": response.text}
    normalized = normalize_search_response(url, response_json)
    yield await asyncio.to_thread(json.dumps, normalized, ensure_ascii=False)


async def _yield_responses_api_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    usage = _responses_usage_to_chat_completion_usage(safe_get(response_json, "usage", default=None))
    prompt_tokens = safe_get(usage, "prompt_tokens", default=0) or 0
    completion_tokens = safe_get(usage, "completion_tokens", default=0) or 0
    total_tokens = safe_get(usage, "total_tokens", default=0) or 0
    content, reasoning_content = _responses_output_to_text(response_json)
    timestamp = safe_get(response_json, "created", default=int(datetime.timestamp(datetime.now())))
    yield await generate_no_stream_response(
        timestamp,
        model,
        content=content or None,
        role="assistant",
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_content=reasoning_content or None,
        cached_tokens=safe_get(usage, "prompt_tokens_details", "cached_tokens", default=0) or 0,
        prompt_audio_tokens=safe_get(usage, "prompt_tokens_details", "audio_tokens", default=0) or 0,
        reasoning_tokens=safe_get(usage, "completion_tokens_details", "reasoning_tokens", default=0) or 0,
        completion_audio_tokens=safe_get(usage, "completion_tokens_details", "audio_tokens", default=0) or 0,
        accepted_prediction_tokens=safe_get(usage, "completion_tokens_details", "accepted_prediction_tokens", default=0) or 0,
        rejected_prediction_tokens=safe_get(usage, "completion_tokens_details", "rejected_prediction_tokens", default=0) or 0,
    )


def _parse_provider_json_response(response_json):
    if isinstance(response_json, str):
        import ast

        return ast.literal_eval(str(response_json))
    if isinstance(response_json, list):
        return response_json
    if isinstance(response_json, dict):
        return [response_json]
    logger.error("error fetch_response: Unknown response_json type: %s", type(response_json))
    return response_json


async def _yield_gemini_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    parsed_data = _parse_provider_json_response(response_json)
    parts_list = safe_get(parsed_data, 0, "candidates", 0, "content", "parts", default=[])
    normalized_parts = normalize_gemini_parts(parts_list if isinstance(parts_list, list) else [])
    usage_metadata = safe_get(parsed_data, -1, "usageMetadata")
    prompt_tokens = safe_get(usage_metadata, "promptTokenCount", default=0)
    candidates_tokens = safe_get(usage_metadata, "candidatesTokenCount", default=0)
    total_tokens = safe_get(usage_metadata, "totalTokenCount", default=0)
    cached_tokens = safe_get(usage_metadata, "cachedContentTokenCount", default=0)
    reasoning_tokens = safe_get(usage_metadata, "thoughtsTokenCount", default=0)
    completion_tokens = candidates_tokens + reasoning_tokens
    role = safe_get(parsed_data, -1, "candidates", 0, "content", "role")
    if role == "model":
        role = "assistant"
    else:
        logger.error("Unknown role: %s, parsed_data: %s", role, parsed_data)
        role = "assistant"

    content = normalized_parts.content
    audio_obj = build_openai_audio_object(normalized_parts.audio_wav_base64, transcript=content or None)
    if audio_obj and not content:
        content = None
    yield await generate_no_stream_response(
        int(datetime.timestamp(datetime.now())),
        model,
        content=content,
        tools_id=normalized_parts.function_call.call_id,
        function_call_name=normalized_parts.function_call.name,
        function_call_content=normalized_parts.function_call.arguments,
        role=role,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_content=normalized_parts.reasoning_content,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        image_base64=normalized_parts.image_base64 or "",
        audio=audio_obj,
    )


async def _yield_claude_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    prompt_tokens = safe_get(response_json, "usage", "input_tokens", default=0)
    output_tokens = safe_get(response_json, "usage", "output_tokens", default=0)
    yield await generate_no_stream_response(
        int(datetime.timestamp(datetime.now())),
        model,
        content=safe_get(response_json, "content", 0, "text"),
        tools_id=safe_get(response_json, "content", 1, "id", default=None),
        function_call_name=safe_get(response_json, "content", 1, "name", default=None),
        function_call_content=safe_get(response_json, "content", 1, "input", default=None),
        role=safe_get(response_json, "role"),
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        cached_tokens=safe_get(response_json, "usage", "cache_read_input_tokens", default=0),
    )


async def _yield_doubao_translation_chat_completion(response, model):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    if isinstance(response_json, dict) and response_json.get("error"):
        yield {
            "error": "doubao-translation upstream error",
            "status_code": 502,
            "details": response_json.get("error"),
        }
        return

    output_text = None
    for out in safe_get(response_json, "output", default=[]) or []:
        if not isinstance(out, dict) or out.get("type") != "message" or out.get("role") != "assistant":
            continue
        for content_item in (out.get("content") or []):
            if isinstance(content_item, dict) and content_item.get("type") == "output_text" and content_item.get("text"):
                output_text = content_item.get("text")
                break
        if output_text:
            break

    if not output_text:
        yield {
            "error": "doubao-translation empty output",
            "status_code": 502,
            "details": response_json,
        }
        return

    usage_obj = safe_get(response_json, "usage", default={}) or {}
    prompt_tokens = usage_obj.get("input_tokens") or usage_obj.get("prompt_tokens") or 0
    completion_tokens = usage_obj.get("output_tokens") or usage_obj.get("completion_tokens") or 0
    yield await generate_no_stream_response(
        int(datetime.timestamp(datetime.now())),
        model,
        content=output_text,
        role="assistant",
        total_tokens=usage_obj.get("total_tokens") or (prompt_tokens + completion_tokens),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _yield_azure_response(response):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    if "choices" in response_json:
        for choice in response_json["choices"]:
            if "content_filter_results" in choice:
                del choice["content_filter_results"]
    if "prompt_filter_results" in response_json:
        del response_json["prompt_filter_results"]
    yield response_json


async def _yield_dashscope_multimodal_response(response):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    yield safe_get(response_json, "output", "choices", 0, "message", "content", 0, default=None)


async def _yield_embedding_response(response, model):
    response_bytes = await response.aread()
    response_json = await asyncio.to_thread(json.loads, response_bytes)
    content = safe_get(response_json, "embedding", "values", default=[])
    yield {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": content,
                "index": 0,
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


async def fetch_response(client, url, headers, payload, engine, model, timeout=200, response_headers_sink: ResponseHeadersSink | None = None):
    if engine == "search":
        response = await _fetch_search_response(client, url, headers, payload, timeout, response_headers_sink=response_headers_sink)
        error_message = await check_response(response, "fetch_response")
        if error_message:
            yield error_message
            return
        async for item in _yield_search_response(response, url):
            yield item
        return

    response = await _fetch_post_response(client, url, headers, payload, timeout, response_headers_sink=response_headers_sink)
    error_message = await check_response(response, "fetch_response")
    if error_message:
        yield error_message
        return

    if engine == "tts":
        yield response.read()

    elif engine in ("gpt", "codex") and _is_responses_api_call(url, payload):
        async for item in _yield_responses_api_chat_completion(response, model):
            yield item

    elif engine == "gemini" or engine == "vertex-gemini" or engine == "aws":
        async for item in _yield_gemini_chat_completion(response, model):
            yield item

    elif engine == "claude" or engine == "vertex-claude":
        async for item in _yield_claude_chat_completion(response, model):
            yield item

    elif engine == "azure":
        async for item in _yield_azure_response(response):
            yield item

    elif "dashscope.aliyuncs.com" in url and "multimodal-generation" in url:
        async for item in _yield_dashscope_multimodal_response(response):
            yield item

    elif "embedContent" in url:
        async for item in _yield_embedding_response(response, model):
            yield item
    elif engine == "doubao-translation":
        async for item in _yield_doubao_translation_chat_completion(response, model):
            yield item
    else:
        response_bytes = await response.aread()
        response_json = await asyncio.to_thread(json.loads, response_bytes)
        yield response_json

async def fetch_doubao_translation_response_stream(client, url, headers, payload, model, timeout):
    timestamp = int(datetime.timestamp(datetime.now()))
    json_payload = await asyncio.to_thread(json.dumps, payload)

    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        error_message = await check_response(response, "fetch_doubao_translation_response_stream")
        if error_message:
            yield error_message
            return

        sse_parser = IncrementalSSEParser()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        async for chunk in response.aiter_text():
            for raw_event in sse_parser.feed(chunk):
                if not raw_event.strip():
                    continue

                event_name, event_data = parse_sse_event(raw_event)
                if not event_name and not event_data:
                    continue
                if event_name == "[DONE]":
                    yield "data: [DONE]" + end_of_line
                    return

                if not isinstance(event_data, dict):
                    continue

                if event_name == "response.output_text.delta":
                    delta_text = safe_get(event_data, "delta", default=None)
                    if not delta_text:
                        continue
                    yield await generate_sse_response(timestamp, model, content=delta_text)
                    continue

                if event_name == "response.completed":
                    usage_obj = safe_get(event_data, "response", "usage", default={}) or {}
                    prompt_tokens = usage_obj.get("input_tokens") or 0
                    completion_tokens = usage_obj.get("output_tokens") or 0
                    total_tokens = usage_obj.get("total_tokens") or (prompt_tokens + completion_tokens)

                    yield await generate_sse_response(timestamp, model, stop="stop")
                    if total_tokens:
                        yield await generate_sse_response(
                            timestamp,
                            model,
                            total_tokens=total_tokens,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                    yield "data: [DONE]" + end_of_line
                    return

        yield "data: [DONE]" + end_of_line

async def fetch_dalle_response_stream(client, url, headers, payload, timeout=200):
    multipart_payload = _pop_multipart_payload(payload)
    if multipart_payload is not None:
        data, files = multipart_payload
        headers, multipart_content = _build_multipart_content(headers, data, files)
        stream_kwargs = {"content": multipart_content}
    else:
        json_payload = await asyncio.to_thread(json.dumps, payload)
        stream_kwargs = {"content": json_payload}

    async with client.stream("POST", url, headers=headers, timeout=timeout, **stream_kwargs) as response:
        error_message = await check_response(response, "fetch_dalle_response_stream")
        if error_message:
            yield error_message
            return
        async for chunk in response.aiter_text():
            yield chunk

async def fetch_response_stream(client, url, headers, payload, engine, model, timeout=200, response_headers_sink: ResponseHeadersSink | None = None):
    if engine == "gemini" or engine == "vertex-gemini":
        stream = fetch_gemini_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "claude" or engine == "vertex-claude":
        stream = fetch_claude_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "aws":
        stream = fetch_aws_response_stream(client, url, headers, payload, model, timeout)
    elif engine in ("gpt", "codex", "openrouter", "azure-databricks"):
        stream = fetch_gpt_response_stream(client, url, headers, payload, timeout, response_headers_sink=response_headers_sink)
    elif engine == "azure":
        stream = fetch_azure_response_stream(client, url, headers, payload, timeout)
    elif engine == "cloudflare":
        stream = fetch_cloudflare_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "cohere":
        stream = fetch_cohere_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "doubao-translation":
        stream = fetch_doubao_translation_response_stream(client, url, headers, payload, model, timeout)
    elif engine == "dalle":
        stream = fetch_dalle_response_stream(client, url, headers, payload, timeout)
    else:
        raise ValueError("Unknown response")

    async for chunk in _yield_from_stream(stream, label=f"{engine} upstream response stream"):
        yield chunk
