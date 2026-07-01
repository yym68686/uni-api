import re
import json
import copy
import uuid
import hmac
import time
import httpx
import base64
import asyncio
import hashlib
import datetime
import urllib.parse
from io import IOBase
from typing import Tuple
from datetime import timezone
from urllib.parse import urlparse

from fastapi import HTTPException


from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from core.models import RequestModel, Message
from core.utils import (
    c3s,
    c3o,
    c3h,
    c35s,
    c4,
    gemini1,
    gemini_preview,
    gemini2_5_pro_exp,
    BaseAPI,
    safe_get,
    get_engine,
    get_model_dict,
    get_text_message,
    get_image_message,
)
from uni_api.providers.overrides import apply_post_body_parameter_overrides

gemini_max_token_65k_models = ["gemini-3-pro", "gemini-2.5-pro", "gemini-2.0-pro", "gemini-2.0-flash-thinking", "gemini-2.5-flash"]
CODEX_CLI_VERSION = "0.125.0"
CODEX_USER_AGENT = f"codex_cli_rs/{CODEX_CLI_VERSION}"
_FORCED_CODEX_CLIENT_HEADER_KEYS = {"version", "user-agent"}

def force_codex_client_headers(headers: dict) -> dict:
    for key in list(headers.keys()):
        if str(key).lower() in _FORCED_CODEX_CLIENT_HEADER_KEYS:
            headers.pop(key, None)
    headers["Version"] = CODEX_CLI_VERSION
    headers["User-Agent"] = CODEX_USER_AGENT
    return headers

def _decode_gemini_thought_signature_from_tool_call_id(tool_call_id: str | None) -> str | None:
    if not tool_call_id or not tool_call_id.startswith("call_"):
        return None
    encoded = tool_call_id.removeprefix("call_")
    # Allow a nonce suffix (call_<b64url(thoughtSignature)>.<nonce>) to keep tool_call_id unique.
    # Only the first segment encodes the thoughtSignature.
    if "." in encoded:
        encoded = encoded.split(".", 1)[0]
    if not encoded:
        return None
    padded = encoded + ("=" * ((4 - (len(encoded) % 4)) % 4))
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception:
        return None

def _gemini_response_modalities(original_model: str, request_modalities: list[str] | None, has_audio: bool) -> list[str] | None:
    # For Gemini preview TTS models, request AUDIO-only to match official API behavior.
    if "preview-tts" in (original_model or "").lower():
        return ["AUDIO"]
    if not request_modalities and not has_audio:
        return None
    modalities = request_modalities or []
    mapped = []
    for m in modalities:
        if not m:
            continue
        if str(m).lower() == "text":
            mapped.append("TEXT")
        elif str(m).lower() == "audio":
            mapped.append("AUDIO")
    if has_audio and "AUDIO" not in mapped:
        mapped.append("AUDIO")
    return mapped or None

def _get_extra_fields(obj) -> dict:
    return getattr(obj, 'model_extra', None) or {}

_CODEX_UNSUPPORTED_MESSAGE_EXTRA_FIELDS = {"reasoning", "reasoning_content"}

def _get_codex_message_extra_fields(obj) -> dict:
    extra = dict(_get_extra_fields(obj))
    for key in _CODEX_UNSUPPORTED_MESSAGE_EXTRA_FIELDS:
        extra.pop(key, None)
    return extra

def _remove_key_recursive(value, key: str) -> None:
    if isinstance(value, dict):
        value.pop(key, None)
        for nested in value.values():
            _remove_key_recursive(nested, key)
    elif isinstance(value, list):
        for item in value:
            _remove_key_recursive(item, key)

def _strip_gemini_unsupported_fields(payload: dict) -> None:
    _remove_key_recursive(payload, "reasoning_content")

def _build_input_audio_item(item):
    input_audio = getattr(item, "input_audio", None)
    if not input_audio or not getattr(input_audio, "data", None):
        return None
    audio_item = {
        "type": "input_audio",
        "input_audio": {
            "data": input_audio.data,
        }
    }
    if getattr(input_audio, "format", None):
        audio_item["input_audio"]["format"] = input_audio.format
    return audio_item

def _normalize_audio_base64(data: str) -> tuple[str, str | None]:
    if not data:
        return "", None
    cleaned = data.strip()
    mime_type = None
    if cleaned.startswith("data:") and ";base64," in cleaned:
        header, cleaned = cleaned.split(",", 1)
        mime_type = header[5:header.index(";")]
    cleaned = re.sub(r"\s+", "", cleaned)
    pad_len = (-len(cleaned)) % 4
    if pad_len:
        cleaned += "=" * pad_len
    return cleaned, mime_type

def _is_uri(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)

def _format_to_mime(format_value: str | None) -> str | None:
    if not format_value:
        return None
    fmt = format_value.strip().lower()
    if "/" in fmt:
        return fmt
    mapping = {
        "mp4": "video/mp4",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "opus": "audio/opus",
        "webm": "audio/webm",
    }
    return mapping.get(fmt, f"audio/{fmt}")

def _request_reasoning_effort(request: RequestModel) -> str | None:
    reasoning = getattr(request, "reasoning", None)
    effort = getattr(reasoning, "effort", None) if reasoning else None
    if not effort:
        return None
    effort = str(effort).strip().lower()
    return effort or None

def _gemini_2_5_thinking_budget_from_request_model(request_model: str, original_model: str) -> int | None:
    match = re.match(r".*-think-(-?\d+)", request_model)
    if not match:
        return None

    try:
        val = int(match.group(1))
    except ValueError:
        return None

    if "gemini-2.5-pro" in original_model:
        if val < 128:
            return 128
        if val > 32768:
            return 32768
        return val

    if "gemini-2.5-flash-lite" in original_model:
        if val > 0 and val < 512:
            return 512
        if val > 24576:
            return 24576
        return val if val >= 0 else 0

    if val > 24576:
        return 24576
    return val if val >= 0 else 0

def _is_gemini_3_model_name(model_name: str | None) -> bool:
    if not model_name:
        return False
    name = str(model_name).strip().lower()
    if not name:
        return False
    if re.match(r"^gemini-3(?:\.\d+)?-", name):
        return True
    return name.startswith("gemini-flash-latest") or name.startswith("gemini-flash-lite-latest")

def _is_gemini_3_pro_model_name(model_name: str | None) -> bool:
    if not model_name:
        return False
    name = str(model_name).strip().lower()
    return bool(re.match(r"^gemini-3(?:\.\d+)?-pro", name))

def _is_gemini_3_model(request_model: str, original_model: str) -> bool:
    return _is_gemini_3_model_name(request_model) or _is_gemini_3_model_name(original_model)

def _is_gemini_3_pro_model(request_model: str, original_model: str) -> bool:
    return _is_gemini_3_pro_model_name(request_model) or _is_gemini_3_pro_model_name(original_model)

def _gemini_3_thinking_level_from_request(request: RequestModel, original_model: str) -> str | None:
    is_gemini_3_pro = _is_gemini_3_pro_model(request.model, original_model)
    reasoning_effort = _request_reasoning_effort(request)
    if reasoning_effort:
        if is_gemini_3_pro:
            if reasoning_effort == "high":
                return "high"
            if reasoning_effort in {"minimal", "low", "medium"}:
                return "low"
        elif reasoning_effort in {"minimal", "low", "medium", "high"}:
            return reasoning_effort

    match = re.match(r".*-think-(-?\d+)", request.model)
    if match:
        try:
            val = int(match.group(1))
            if is_gemini_3_pro:
                if val <= 32768 * 0.4:
                    return "low"
                return "high"
            if val <= 32768 * 0.1:
                return "minimal"
            if val <= 32768 * 0.3:
                return "low"
            if val <= 32768 * 0.6:
                return "medium"
            return "high"
        except ValueError:
            pass

    level_match = re.search(r"-(minimal|low|medium|high)$", request.model.lower())
    if not level_match:
        return None

    level_str = level_match.group(1)
    if is_gemini_3_pro:
        if level_str in {"minimal", "low", "medium"}:
            return "low"
        return "high"
    return level_str

def _gemini_service_tier_from_request(request: RequestModel) -> str | None:
    service_tier = getattr(request, "service_tier", None)
    if service_tier is None:
        return None
    service_tier = str(service_tier).strip()
    if not service_tier:
        return None
    normalized = service_tier.lower()
    tier_map = {
        "default": "STANDARD",
        "standard": "STANDARD",
        "priority": "PRIORITY",
        "flex": "FLEX",
    }
    return tier_map.get(normalized, service_tier.upper())

def _apply_explicit_gemini_request_controls(
    payload: dict,
    request: RequestModel,
    original_model: str,
    *,
    include_service_tier: bool = False,
) -> None:
    generation_config = payload.setdefault("generationConfig", {})

    if "gemini-2.5" in original_model and "-image" not in original_model and "preview-tts" not in original_model.lower():
        budget = _gemini_2_5_thinking_budget_from_request_model(request.model, original_model)
        if budget is not None:
            thinking_config = generation_config.get("thinkingConfig")
            if not isinstance(thinking_config, dict):
                thinking_config = {}
                generation_config["thinkingConfig"] = thinking_config
            thinking_config["includeThoughts"] = bool(budget)
            thinking_config["thinkingBudget"] = budget

    if _is_gemini_3_model(request.model, original_model):
        thinking_level = _gemini_3_thinking_level_from_request(request, original_model)
        if thinking_level:
            thinking_config = generation_config.get("thinkingConfig")
            if not isinstance(thinking_config, dict):
                thinking_config = {}
                generation_config["thinkingConfig"] = thinking_config
            thinking_config["thinkingLevel"] = thinking_level

    if include_service_tier:
        service_tier = _gemini_service_tier_from_request(request)
        if service_tier:
            payload.pop("service_tier", None)
            payload["serviceTier"] = service_tier

def _build_gemini_input_audio_part(item):
    input_audio = getattr(item, "input_audio", None)
    if not input_audio or not getattr(input_audio, "data", None):
        return None
    audio_data = input_audio.data
    mime_type = _format_to_mime(getattr(input_audio, "format", None))
    if _is_uri(audio_data):
        return {
            "file_data": {
                "file_uri": audio_data,
                "mime_type": mime_type or "application/octet-stream",
            }
        }
    data, mime_from_data = _normalize_audio_base64(audio_data)
    return {
        "inline_data": {
            "mime_type": mime_type or mime_from_data or "audio/wav",
            "data": data,
        }
    }


async def _build_gemini_content_parts(msg, engine: str, provider: dict) -> list[dict] | None:
    if isinstance(msg.content, list):
        content = []
        file_parts = []
        for item in msg.content:
            if item.type == "text":
                text_message = await get_text_message(item.text, engine)
                text_message.update(_get_extra_fields(item))
                content.append(text_message)
            elif item.type == "image_url" and provider.get("image", True):
                image_message = await get_image_message(item.image_url.url, engine)
                image_message.update(_get_extra_fields(item))
                content.append(image_message)
            elif item.type == "input_audio":
                audio_part = _build_gemini_input_audio_part(item)
                if audio_part:
                    audio_part.update(_get_extra_fields(item))
                    if "file_data" in audio_part:
                        file_parts.append(audio_part)
                    else:
                        content.append(audio_part)
        return file_parts + content if file_parts else content
    if msg.content:
        return [{"text": msg.content}]
    return None


def _gemini_function_call_part(tool_call) -> dict:
    try:
        args = json.loads(tool_call.function.arguments)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tool_call arguments for '{tool_call.function.name}': {e}",
        ) from e
    function_arguments = {
        "functionCall": {
            "name": tool_call.function.name,
            "args": args,
        }
    }
    thought_signature = _decode_gemini_thought_signature_from_tool_call_id(tool_call.id)
    if thought_signature:
        function_arguments["thoughtSignature"] = thought_signature
    return function_arguments


def _gemini_function_response_message(function_call_name: str, result) -> dict:
    return {
        "role": "function",
        "parts": [
            {
                "functionResponse": {
                    "name": function_call_name,
                    "response": {
                        "name": function_call_name,
                        "content": {"result": result},
                    },
                }
            }
        ],
    }


async def _build_gemini_messages(
    request_messages: list,
    *,
    engine: str,
    provider: dict,
    normalize_system_text: bool = False,
) -> tuple[list[dict], dict | None]:
    messages = []
    system_prompt = ""
    tool_call_id_to_function_name: dict[str, str] = {}
    for msg in request_messages:
        if msg.role == "assistant":
            msg.role = "model"

        role = getattr(msg, "role", None)
        tool_calls = getattr(msg, "tool_calls", None) or []
        content = await _build_gemini_content_parts(msg, engine, provider)
        if role == "system":
            if content and safe_get(content, 0, "text", default=None) is not None:
                text = content[0]["text"]
                if normalize_system_text:
                    text = re.sub(r"_+", "_", text)
                system_prompt = system_prompt + "\n\n" + text
            continue

        if tool_calls:
            if role != "model":
                raise HTTPException(status_code=400, detail=f"tool_calls only supported for role 'assistant', got '{role}'")
            parts = []
            if content:
                parts.extend(content)
            for tool_call in tool_calls:
                tool_call_id_to_function_name[tool_call.id] = tool_call.function.name
                parts.append(_gemini_function_call_part(tool_call))
            messages.append({"role": "model", "parts": parts})
        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_call_id = str(tool_call_id).strip() if tool_call_id else ""
            function_call_name = tool_call_id_to_function_name.get(tool_call_id)
            if not function_call_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tool message: tool_call_id '{tool_call_id}' has no matching prior assistant tool_calls",
                )
            messages.append(_gemini_function_response_message(function_call_name, msg.content))
        elif content:
            msg_dict = {"role": role, "parts": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)

    system_instruction = {"parts": [{"text": system_prompt}]} if system_prompt.strip() else None
    return messages, system_instruction

async def get_gemini_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }

    # 获取映射后的实际模型ID
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    if request.stream:
        gemini_stream = "streamGenerateContent"
    else:
        gemini_stream = "generateContent"
    url = provider['base_url']
    parsed_url = urlparse(url)
    if "/v1beta" in parsed_url.path:
        api_version = "v1beta"
    else:
        api_version = "v1"

    url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path.split('/models')[0].rstrip('/')}/models/{original_model}:{gemini_stream}?key={api_key}"

    try:
        request_messages = [Message(role="user", content=request.prompt)]
    except Exception:
        request_messages = copy.deepcopy(request.messages)
    messages, systemInstruction = await _build_gemini_messages(request_messages, engine=engine, provider=provider, normalize_system_text=True)

    if any(off_model in original_model for off_model in gemini_max_token_65k_models) or "-image" in original_model:
        safety_settings = "OFF"
    else:
        safety_settings = "BLOCK_NONE"

    payload = {
        "contents": messages or [{"role": "user", "parts": [{"text": "No messages"}]}],
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                "threshold": "BLOCK_NONE"
            },
        ]
    }

    if systemInstruction:
        if api_version == "v1beta":
            payload["systemInstruction"] = systemInstruction
        if api_version == "v1":
            first_message = safe_get(payload, "contents", 0, "parts", 0, "text", default=None)
            system_instruction = safe_get(systemInstruction, "parts", 0, "text", default=None)
            if first_message and system_instruction:
                payload["contents"][0]["parts"][0]["text"] = system_instruction + "\n" + first_message

    miss_fields = [
        'model',
        'messages',
        'stream',
        'tool_choice',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'response_format',
        'stream_options',
        'prompt',
        'size',
        # OpenAI-style audio fields (mapped into generationConfig for Gemini)
        'modalities',
        'audio',
        'reasoning',
    ]
    generation_config = {}

    def process_tool_parameters(data):
        if isinstance(data, dict):
            # 移除 Gemini 不支持的 'additionalProperties'
            data.pop("additionalProperties", None)

            # 将 'default' 值移入 'description'
            if "default" in data:
                default_value = data.pop("default")
                description = data.get("description", "")
                data["description"] = f"{description}\nDefault: {default_value}"

            # 递归处理
            for value in data.values():
                process_tool_parameters(value)
        elif isinstance(data, list):
            for item in data:
                process_tool_parameters(item)

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "tools" and ("gemini-2.0-flash-thinking" in original_model or "-image" in original_model):
                continue
            if field == "tools":
                # 处理每个工具的 function 定义
                processed_tools = []
                for tool in value:
                    function_def = tool["function"]
                    if "parameters" in function_def:
                        process_tool_parameters(function_def["parameters"])

                    if function_def["name"] != "googleSearch" and function_def["name"] != "googleSearch":
                        processed_tools.append({"function": function_def})

                if processed_tools:
                    payload.update({
                        "tools": [{
                            "function_declarations": [tool["function"] for tool in processed_tools]
                        }],
                        "tool_config": {
                            "function_calling_config": {
                                "mode": "AUTO"
                            }
                        }
                    })
            elif field == "temperature":
                if "-image" in original_model:
                    value = 1
                generation_config["temperature"] = value
            elif field == "max_tokens":
                if value > 65536:
                    value = 65536
                generation_config["maxOutputTokens"] = value
            elif field == "top_p":
                generation_config["topP"] = value
            else:
                payload[field] = value

    payload["generationConfig"] = generation_config
    if "maxOutputTokens" not in generation_config:
        if any(pro_model in original_model for pro_model in gemini_max_token_65k_models):
            payload["generationConfig"]["maxOutputTokens"] = 65536
        else:
            payload["generationConfig"]["maxOutputTokens"] = 8192

    # Map OpenAI-style audio request fields to Gemini TTS/generateContent config.
    response_modalities = _gemini_response_modalities(
        original_model=original_model,
        request_modalities=getattr(request, "modalities", None),
        has_audio=bool(getattr(request, "audio", None)),
    )
    if response_modalities:
        payload["generationConfig"]["responseModalities"] = response_modalities
        if "AUDIO" in response_modalities:
            voice_name = getattr(getattr(request, "audio", None), "voice", None) or "Kore"
            payload["generationConfig"]["speechConfig"] = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            }
            payload.setdefault("model", original_model)

    apply_post_body_parameter_overrides(payload, provider, request.model)
    _apply_explicit_gemini_request_controls(
        payload,
        request,
        original_model,
        include_service_tier=True,
    )
    _strip_gemini_unsupported_fields(payload)

    return url, headers, payload

def create_jwt(client_email, private_key):
    # JWT Header
    header = json.dumps({
        "alg": "RS256",
        "typ": "JWT"
    }).encode()

    # JWT Payload
    now = int(time.time())
    payload = json.dumps({
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now + 3600,
        "iat": now
    }).encode()

    # Encode header and payload
    segments = [
        base64.urlsafe_b64encode(header).rstrip(b'='),
        base64.urlsafe_b64encode(payload).rstrip(b'=')
    ]

    # Create signature
    signing_input = b'.'.join(segments)
    private_key = load_pem_private_key(private_key.encode(), password=None)
    signature = private_key.sign(
        signing_input,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    segments.append(base64.urlsafe_b64encode(signature).rstrip(b'='))
    return b'.'.join(segments).decode()

async def get_access_token(client_email, private_key):
    jwt = await asyncio.to_thread(create_jwt, client_email, private_key)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt
            },
            headers={'Content-Type': "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        return response.json()["access_token"]

async def get_vertex_gemini_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    if provider.get("client_email") and provider.get("private_key"):
        access_token = await get_access_token(provider['client_email'], provider['private_key'])
        headers['Authorization'] = f"Bearer {access_token}"
    if provider.get("project_id"):
        project_id = provider.get("project_id")

    if request.stream:
        gemini_stream = "streamGenerateContent"
    else:
        gemini_stream = "generateContent"
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    # search_tool = None

    # https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash?hl=zh-cn
    pro_models = ["gemini-2.5"]
    global_models = ["gemini-2.5-flash-image-preview", "gemini-3-pro", "gemini-3-flash", "gemini-3.1-pro"]
    if any(global_model in original_model for global_model in global_models):
        location = gemini_preview
    elif any(pro_model in original_model for pro_model in pro_models):
        location = gemini2_5_pro_exp
    else:
        location = gemini1

    if api_key is not None and len(api_key) > 2 and api_key[2] == ".":
        base = (provider.get("base_url") or "https://aiplatform.googleapis.com").rstrip("/")
        url = f"{base}/v1/publishers/google/models/{original_model}:{gemini_stream}?key={api_key}"
        headers.pop("Authorization", None)
    elif "google-vertex-ai" in provider.get("base_url", "") or any(global_model in original_model for global_model in global_models):
        url = provider.get("base_url").rstrip('/') + "/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:{stream}".format(
            LOCATION=await location.next(),
            PROJECT_ID=project_id,
            MODEL_ID=original_model,
            stream=gemini_stream
        )
    else:
        url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:{stream}".format(
            LOCATION=await location.next(),
            PROJECT_ID=project_id,
            MODEL_ID=original_model,
            stream=gemini_stream
        )

    request_messages = copy.deepcopy(request.messages)
    messages, systemInstruction = await _build_gemini_messages(request_messages, engine=engine, provider=provider)

    if any(off_model in original_model for off_model in gemini_max_token_65k_models):
        safety_settings = "OFF"
    else:
        safety_settings = "BLOCK_NONE"

    payload = {
        "contents": messages,
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": safety_settings
            },
            {
                "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                "threshold": "BLOCK_NONE"
            },
        ]
    }
    if systemInstruction:
        payload["system_instruction"] = systemInstruction

    miss_fields = [
        'model',
        'messages',
        'stream',
        'tool_choice',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'stream_options',
        'prompt',
        'size',
        # OpenAI-style audio fields (mapped into generationConfig for Gemini)
        'modalities',
        'audio',
        'reasoning',
    ]
    generation_config = {}

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "tools":
                payload.update({
                    "tools": [{
                        "function_declarations": [tool["function"] for tool in value]
                    }],
                    "tool_config": {
                        "function_calling_config": {
                            "mode": "AUTO"
                        }
                    }
                })
            elif field == "temperature":
                generation_config["temperature"] = value
            elif field == "max_tokens":
                if value > 65535:
                    value = 65535
                generation_config["max_output_tokens"] = value
            elif field == "top_p":
                generation_config["top_p"] = value
            else:
                payload[field] = value

    payload["generationConfig"] = generation_config
    if "max_output_tokens" not in generation_config:
        if any(pro_model in original_model for pro_model in gemini_max_token_65k_models):
            payload["generationConfig"]["max_output_tokens"] = 65535
        else:
            payload["generationConfig"]["max_output_tokens"] = 8192

    # Gemini 2.5 系列的 thinkingConfig 处理
    # Note: preview TTS models do not support thinkingConfig.
    if "gemini-2.5" in original_model and "preview-tts" not in original_model.lower():
        budget = _gemini_2_5_thinking_budget_from_request_model(request.model, original_model)
        if budget is not None:
            payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True if budget else False,
                "thinkingBudget": budget
            }
        else:
            # gemini-2.5-flash-lite 默认不启用 thinking，不能单独设置 includeThoughts
            if "gemini-2.5-flash-lite" not in original_model:
                payload["generationConfig"]["thinkingConfig"] = {
                    "includeThoughts": True,
                }

    # Gemini 3 系列的 thinkingLevel 处理
    if _is_gemini_3_model(request.model, original_model):
        thinking_level = _gemini_3_thinking_level_from_request(request, original_model)
        if thinking_level:
            if "thinkingConfig" not in payload["generationConfig"]:
                payload["generationConfig"]["thinkingConfig"] = {}
            payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] = thinking_level

    apply_post_body_parameter_overrides(payload, provider, request.model)
    _apply_explicit_gemini_request_controls(payload, request, original_model)

    # Map OpenAI-style audio request fields to Gemini TTS/generateContent config.
    if "generationConfig" not in payload:
        payload["generationConfig"] = {}
    response_modalities = _gemini_response_modalities(
        original_model=original_model,
        request_modalities=getattr(request, "modalities", None),
        has_audio=bool(getattr(request, "audio", None)),
    )
    if response_modalities:
        payload["generationConfig"]["responseModalities"] = response_modalities
        if "AUDIO" in response_modalities:
            voice_name = getattr(getattr(request, "audio", None), "voice", None) or "Kore"
            payload["generationConfig"]["speechConfig"] = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            }
            payload.setdefault("model", original_model)

    _strip_gemini_unsupported_fields(payload)
    return url, headers, payload

async def get_vertex_claude_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json',
    }
    if provider.get("client_email") and provider.get("private_key"):
        access_token = await get_access_token(provider['client_email'], provider['private_key'])
        headers['Authorization'] = f"Bearer {access_token}"
    if provider.get("project_id"):
        project_id = provider.get("project_id")

    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if "claude-3-5-sonnet" in original_model or "claude-3-7-sonnet" in original_model or "4-5@" in original_model:
        location = c35s
    elif "claude-3-opus" in original_model:
        location = c3o
    elif "claude-sonnet-4" in original_model or "claude-opus-4" in original_model:
        location = c4
    elif "claude-3-sonnet" in original_model:
        location = c3s
    elif "claude-3-haiku" in original_model:
        location = c3h

    claude_stream = "streamRawPredict"
    url = "https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:{stream}".format(
        LOCATION=await location.next(),
        PROJECT_ID=project_id,
        MODEL=original_model,
        stream=claude_stream
    )

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            msg_dict = {"role": msg.role, "content": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    if "claude-3-7-sonnet" in original_model:
        max_tokens = 20000
    elif "claude-3-5-sonnet" in original_model:
        max_tokens = 8192
    else:
        max_tokens = 4096

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
        "max_tokens": max_tokens,
    }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") is False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    return url, headers, payload

def sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, 'aws4_request')
    return k_signing

def get_signature(request_body, model_id, aws_access_key, aws_secret_key, aws_region, host, content_type, accept_header):
    request_body = json.dumps(request_body)
    SERVICE = "bedrock"
    canonical_querystring = ''
    method = 'POST'
    raw_path = f'/model/{model_id}/invoke-with-response-stream'
    canonical_uri = urllib.parse.quote(raw_path, safe='/-_.~')
    # Create a date for headers and the credential string
    t = datetime.datetime.now(timezone.utc)
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d') # Date YYYYMMDD

    # --- Task 1: Create a Canonical Request ---
    payload_hash = hashlib.sha256(request_body.encode('utf-8')).hexdigest()

    canonical_headers = f'accept:{accept_header}\n' \
                        f'content-type:{content_type}\n' \
                        f'host:{host}\n' \
                        f'x-amz-bedrock-accept:{accept_header}\n' \
                        f'x-amz-content-sha256:{payload_hash}\n' \
                        f'x-amz-date:{amz_date}\n'
    # 注意：头名称需要按字母顺序排序

    signed_headers = 'accept;content-type;host;x-amz-bedrock-accept;x-amz-content-sha256;x-amz-date' # 按字母顺序排序

    canonical_request = f'{method}\n' \
                        f'{canonical_uri}\n' \
                        f'{canonical_querystring}\n' \
                        f'{canonical_headers}\n' \
                        f'{signed_headers}\n' \
                        f'{payload_hash}'

    # --- Task 2: Create the String to Sign ---
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f'{date_stamp}/{aws_region}/{SERVICE}/aws4_request'
    string_to_sign = f'{algorithm}\n' \
                    f'{amz_date}\n' \
                    f'{credential_scope}\n' \
                    f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'

    # --- Task 3: Calculate the Signature ---
    signing_key = get_signature_key(aws_secret_key, date_stamp, aws_region, SERVICE)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    # --- Task 4: Add Signing Information to the Request ---
    authorization_header = f'{algorithm} Credential={aws_access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}'
    return amz_date, payload_hash, authorization_header

async def get_aws_payload(request, engine, provider, api_key=None):
    CONTENT_TYPE = "application/json"
    # AWS_REGION = "us-east-1"
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    # MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    base_url = provider.get('base_url')
    AWS_REGION = base_url.split('.')[1]
    HOST = f"bedrock-runtime.{AWS_REGION}.amazonaws.com"
    # url = f"{base_url}/model/{original_model}/invoke"
    url = f"{base_url}/model/{original_model}/invoke-with-response-stream"

    messages = []
    # system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            msg_dict = {"role": msg.role, "content": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)
        # elif msg.role == "system":
        #     system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    # if "claude-3-7-sonnet" in original_model:
    #     max_tokens = 20000
    # elif "claude-3-5-sonnet" in original_model:
    #     max_tokens = 8192
    # else:
    #     max_tokens = 4096
    max_tokens = 4096

    payload = {
        "messages": messages,
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
    }

    # payload = {
    #     "anthropic_version": "vertex-2023-10-16",
    #     "messages": messages,
    #     "system": system_prompt or "You are Claude, a large language model trained by Anthropic.",
    #     "max_tokens": max_tokens,
    # }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'stream_options',
        'stream',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") is False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if provider.get("aws_access_key") and provider.get("aws_secret_key"):
        ACCEPT_HEADER = "application/vnd.amazon.bedrock.payload+json" # 指定接受 Bedrock 流格式
        amz_date, payload_hash, authorization_header = await asyncio.to_thread(
            get_signature, payload, original_model, provider.get("aws_access_key"), provider.get("aws_secret_key"), AWS_REGION, HOST, CONTENT_TYPE, ACCEPT_HEADER
        )
        headers = {
            'Accept': ACCEPT_HEADER,
            'Content-Type': CONTENT_TYPE,
            'X-Amz-Date': amz_date,
            'X-Amz-Bedrock-Accept': ACCEPT_HEADER, # Bedrock 特定头
            'X-Amz-Content-Sha256': payload_hash,
            'Authorization': authorization_header,
            # Add 'X-Amz-Security-Token': SESSION_TOKEN if using temporary credentials
        }

    return url, headers, payload

async def get_gpt_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json',
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']
    use_responses_api = engine == "codex" or "v1/responses" in url
    if "openrouter.ai" in url:
        headers['HTTP-Referer'] = "https://github.com/yym68686/uni-api"
        headers['X-Title'] = "Uni API"

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    if use_responses_api:
                        text_message["type"] = "input_text"
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    if use_responses_api:
                        image_message = {
                            "type": "input_image",
                            "image_url": image_message["image_url"]["url"]
                        }
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
                elif item.type == "input_audio":
                    audio_item = _build_input_audio_item(item)
                    if audio_item:
                        audio_item.update(_get_extra_fields(item))
                        content.append(audio_item)
        else:
            content = msg.content
            if msg.role == "system" and "o3-mini" in original_model and not content.startswith("Formatting re-enabled"):
                content = "Formatting re-enabled. " + content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            msg_dict = {"role": msg.role, "content": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)

    if use_responses_api:
        payload = {
            "model": original_model,
            "input": messages,
        }
    else:
        payload = {
            "model": original_model,
            "messages": messages,
        }

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and use_responses_api:
                payload["max_output_tokens"] = value
            elif field == "max_tokens" and "gpt-5" in original_model:
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") is False or "chatgpt-4o-latest" in original_model or "grok" in original_model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if "api.x.ai" in url:
        payload.pop("stream_options", None)
        payload.pop("presence_penalty", None)
        payload.pop("frequency_penalty", None)

    if "gpt-5.2" in original_model:
        payload.pop("top_p", None)

    if "grok-3-mini" in original_model:
        if request.model.endswith("high"):
            payload["reasoning_effort"] = "high"
        elif request.model.endswith("low"):
            payload["reasoning_effort"] = "low"

    if "gpt-oss" in original_model or "gpt-5" in original_model:
        if request.model.endswith("high"):
            if use_responses_api:
                payload["reasoning"] = {"effort": "high"}
            else:
                payload["reasoning_effort"] = "high"
        elif request.model.endswith("low"):
            if use_responses_api:
                payload["reasoning"] = {"effort": "low"}
            else:
                payload["reasoning_effort"] = "low"
        # else:
        #     if "v1/responses" in url:
        #         payload["reasoning"] = {"effort": "medium"}
        #     else:
        #         payload["reasoning_effort"] = "medium"

        if "temperature" in payload:
            payload.pop("temperature")

        if use_responses_api:
            payload.pop("stream_options", None)

    # 代码生成/数学解题  0.0
    # 数据抽取/分析	     1.0
    # 通用对话          1.3
    # 翻译	           1.3
    # 创意类写作/诗歌创作 1.5
    if "deepseek-r" in original_model.lower():
        if "temperature" not in payload:
            payload["temperature"] = 0.6

    if request.model.endswith("-search") and "gemini" in original_model:
        if "tools" not in payload:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": "googleSearch",
                    "description": "googleSearch"
                }
            }]
        else:
            if not any(tool["function"]["name"] == "googleSearch" for tool in payload["tools"]):
                payload["tools"].append({
                    "type": "function",
                    "function": {
                        "name": "googleSearch",
                        "description": "googleSearch"
                    }
                })

    apply_post_body_parameter_overrides(payload, provider, request.model)

    return url, headers, payload

def _codex_responses_url(base_url: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return "https://chatgpt.com/backend-api/codex/responses"

    base = base.rstrip("/")
    if base.endswith("/v1/responses") or base.endswith("/responses"):
        return base
    return f"{base}/responses"

def _strip_key_recursive(value, key: str):
    if isinstance(value, dict):
        value.pop(key, None)
        for child in value.values():
            _strip_key_recursive(child, key)
    elif isinstance(value, list):
        for child in value:
            _strip_key_recursive(child, key)
    return value


def _strip_codex_store_false_reasoning_input_ids(payload: dict) -> dict:
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return payload

    changed = False
    for item in input_items:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            changed = changed or "id" in item
            item.pop("id", None)

    if changed:
        payload["input"] = input_items
    return payload


def _strip_unsupported_codex_message_item_reasoning_fields(payload: dict) -> dict:
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return payload

    for item in input_items:
        if isinstance(item, dict) and item.get("type") == "message":
            for key in _CODEX_UNSUPPORTED_MESSAGE_EXTRA_FIELDS:
                item.pop(key, None)

    return payload


def strip_unsupported_codex_payload_fields(payload: dict, *, strip_store: bool = False) -> dict:
    # Codex rejects these fields; drop them on any Codex-bound request.
    payload.pop("max_output_tokens", None)
    payload.pop("response_format", None)
    payload.pop("top_p", None)
    payload.pop("truncation", None)
    _strip_key_recursive(payload, "cache_control")
    # Chat-style histories may carry provider-private reasoning deltas that
    # Responses input objects reject as unknown parameters.
    _strip_key_recursive(payload, "reasoning_content")
    _strip_unsupported_codex_message_item_reasoning_fields(payload)
    # ChatGPT Codex upstream requires store=false. Preserve encrypted reasoning
    # state, but do not replay rs_* ids because non-persisted items 404.
    _strip_codex_store_false_reasoning_input_ids(payload)
    if strip_store:
        payload.pop("store", None)
    return payload

def strip_codex_image_generation_defaults(payload: dict, model: str) -> dict:
    # gpt-image-2 accepts Responses payloads, but not the Codex chat defaults.
    if model == "gpt-image-2":
        payload.pop("parallel_tool_calls", None)
        payload.pop("reasoning", None)
        payload.pop("include", None)
    return payload

def _codex_system_messages_to_instructions(request: RequestModel) -> str:
    instructions: list[str] = []
    for msg in request.messages or []:
        role = (getattr(msg, "role", None) or "").strip()
        if role != "system":
            continue

        content_value = getattr(msg, "content", None)
        if isinstance(content_value, str):
            if content_value:
                instructions.append(content_value)
        elif isinstance(content_value, list):
            text_parts: list[str] = []
            for item in content_value:
                if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                    text_parts.append(str(item.text))
            if text_parts:
                instructions.append("\n".join(text_parts))
    return "\n\n".join(instructions)

_CODEX_EMPTY_TOOL_NAME_INFERENCE_LIMIT = 64
_CODEX_TOOL_ARGUMENTS_INFERENCE_MAX_BYTES = 64 * 1024


def _codex_tool_parameter_schema_index(request: RequestModel) -> list[dict]:
    tools: list[dict] = []
    for tool in request.tools or []:
        if getattr(tool, "type", None) != "function":
            continue

        fn = getattr(tool, "function", None)
        name = str(getattr(fn, "name", "") or "").strip() if fn else ""
        if not name:
            continue

        parameters = getattr(fn, "parameters", None)
        if isinstance(parameters, dict):
            properties = parameters.get("properties")
            required = parameters.get("required")
        else:
            properties = getattr(parameters, "properties", None)
            required = getattr(parameters, "required", None)

        property_names = set(str(key) for key in properties.keys()) if isinstance(properties, dict) else set()
        required_names = set(str(key) for key in required) if isinstance(required, list) else set()
        tools.append({"name": name, "properties": property_names, "required": required_names})
    return tools


def _codex_tool_argument_keys(arguments) -> tuple[set[str] | None, str | None]:
    if arguments is None:
        return None, "arguments is missing"
    raw = str(arguments)
    if not raw.strip():
        return None, "arguments is empty"
    if len(raw.encode("utf-8")) > _CODEX_TOOL_ARGUMENTS_INFERENCE_MAX_BYTES:
        return None, "arguments is too large to infer safely"
    try:
        parsed = json.loads(raw)
    except Exception:
        return None, "arguments is not valid JSON"
    if not isinstance(parsed, dict):
        return None, "arguments is not a JSON object"
    return set(str(key) for key in parsed.keys()), None


def _infer_codex_tool_call_name_from_arguments(
    tool_schema_index: list[dict],
    arguments,
) -> tuple[str | None, str | None]:
    argument_keys, reason = _codex_tool_argument_keys(arguments)
    if argument_keys is None:
        return None, reason

    matches = [
        item["name"]
        for item in tool_schema_index
        if argument_keys.issubset(item["properties"]) and item["required"].issubset(argument_keys)
    ]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "arguments do not match any declared tool schema"
    return None, "arguments match multiple declared tool schemas"


def _codex_chat_messages_to_responses_input(request: RequestModel, provider: dict) -> list[dict]:
    input_items: list[dict] = []
    tool_schema_index: list[dict] | None = None
    inferred_empty_tool_name_count = 0

    for message_index, msg in enumerate(request.messages or []):
        role = (getattr(msg, "role", None) or "").strip()
        if not role:
            continue
        if role == "system":
            continue

        # Tool results are top-level items in Codex/Responses format.
        if role == "tool":
            tool_call_id = (getattr(msg, "tool_call_id", None) or "").strip()
            if not tool_call_id:
                continue

            output_value = getattr(msg, "content", None)
            if isinstance(output_value, list):
                text_parts: list[str] = []
                for part in output_value:
                    if getattr(part, "type", None) == "text" and getattr(part, "text", None):
                        text_parts.append(str(part.text))
                output_value = "\n".join(text_parts) if text_parts else json.dumps(
                    [p.model_dump(exclude_unset=True) for p in output_value],
                    ensure_ascii=False,
                )
            if output_value is None:
                output_value = ""

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": output_value,
                }
            )
            continue

        part_type = "output_text" if role == "assistant" else "input_text"

        content_parts: list[dict] = []
        content_value = getattr(msg, "content", None)
        if isinstance(content_value, str):
            if content_value:
                content_parts.append({"type": part_type, "text": content_value})
        elif isinstance(content_value, list):
            for item in content_value:
                item_type = getattr(item, "type", None)
                if item_type == "text" and getattr(item, "text", None):
                    text_part = {"type": part_type, "text": str(item.text)}
                    text_part.update(_get_extra_fields(item))
                    content_parts.append(text_part)
                elif item_type == "image_url" and provider.get("image", True) and role == "user":
                    image_url_obj = getattr(item, "image_url", None)
                    image_url = getattr(image_url_obj, "url", None) if image_url_obj else None
                    if image_url:
                        image_part = {"type": "input_image", "image_url": image_url}
                        image_part.update(_get_extra_fields(item))
                        content_parts.append(image_part)
                elif item_type == "input_audio" and role == "user":
                    audio_item = _build_input_audio_item(item)
                    if audio_item:
                        audio_item.update(_get_extra_fields(item))
                        content_parts.append(audio_item)

        message_item = {
            "type": "message",
            "role": role,
            "content": content_parts,
        }
        message_item.update(_get_codex_message_extra_fields(msg))
        input_items.append(message_item)

        # Tool calls are separate top-level objects in Codex payloads.
        if role == "assistant":
            for tool_call_index, tool_call in enumerate(getattr(msg, "tool_calls", None) or []):
                if getattr(tool_call, "type", None) != "function":
                    continue
                func = getattr(tool_call, "function", None)
                if not func:
                    continue
                function_name = str(getattr(func, "name", "") or "").strip()
                if not function_name:
                    inferred_empty_tool_name_count += 1
                    if inferred_empty_tool_name_count > _CODEX_EMPTY_TOOL_NAME_INFERENCE_LIMIT:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "Too many assistant tool calls have an empty function.name; "
                                f"limit={_CODEX_EMPTY_TOOL_NAME_INFERENCE_LIMIT}"
                            ),
                        )
                    if tool_schema_index is None:
                        tool_schema_index = _codex_tool_parameter_schema_index(request)
                    function_name, reason = _infer_codex_tool_call_name_from_arguments(
                        tool_schema_index,
                        getattr(func, "arguments", None),
                    )
                    if not function_name:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"messages[{message_index}].tool_calls[{tool_call_index}].function.name "
                                f"is empty and cannot be uniquely inferred from tools: {reason}"
                            ),
                        )
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": getattr(tool_call, "id", None),
                        "name": function_name,
                        "arguments": getattr(func, "arguments", None),
                    }
                )

    return input_items

async def get_codex_payload(request, engine, provider, api_key=None):
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    url = _codex_responses_url(provider.get("base_url", ""))

    payload: dict = {
        "model": original_model,
        "instructions": _codex_system_messages_to_instructions(request),
        "input": _codex_chat_messages_to_responses_input(request, provider),
        # CLIProxyAPI defaults that Codex commonly expects.
        "parallel_tool_calls": True,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "store": False,
    }

    if request.stream is not None:
        payload["stream"] = bool(request.stream)

    if request.tools and provider.get("tools") is not False:
        tools_out: list[dict] = []
        for tool in request.tools:
            tool_type = getattr(tool, "type", None)
            if tool_type != "function":
                # Pass through non-function tools when present.
                tools_out.append(tool.model_dump(exclude_unset=True))
                continue
            fn = getattr(tool, "function", None)
            if not fn:
                continue
            item: dict = {
                "type": "function",
                "name": getattr(fn, "name", None),
            }
            if getattr(fn, "description", None):
                item["description"] = fn.description
            params = getattr(fn, "parameters", None)
            if params is not None:
                try:
                    item["parameters"] = params.model_dump(by_alias=True, exclude_none=True)
                except Exception:
                    item["parameters"] = params
            tools_out.append(item)
        if tools_out:
            payload["tools"] = tools_out

    if request.tool_choice is not None and provider.get("tools") is not False:
        tc = request.tool_choice
        if isinstance(tc, str):
            payload["tool_choice"] = tc
        else:
            tc_type = getattr(tc, "type", None)
            if tc_type == "function":
                name = getattr(getattr(tc, "function", None), "name", None)
                choice: dict = {"type": "function"}
                if name:
                    choice["name"] = name
                payload["tool_choice"] = choice
            elif tc_type:
                payload["tool_choice"] = {"type": tc_type}

    # Match CLIProxyAPI Codex executor hardening.
    payload.pop("previous_response_id", None)
    payload.pop("prompt_cache_retention", None)
    payload.pop("safety_identifier", None)

    # Required / commonly expected Codex headers.
    headers.setdefault("Openai-Beta", "responses=experimental")
    headers.setdefault("Originator", "codex_cli_rs")
    headers.setdefault("Version", CODEX_CLI_VERSION)
    session_id = str(uuid.uuid4())
    headers.setdefault("Session_id", session_id)
    headers.setdefault("Conversation_id", session_id)
    headers.setdefault("User-Agent", CODEX_USER_AGENT)
    headers.setdefault("Connection", "Keep-Alive")
    headers.setdefault("Accept", "text/event-stream" if request.stream else "application/json")
    force_codex_client_headers(headers)

    apply_post_body_parameter_overrides(payload, provider, request.model)

    strip_codex_image_generation_defaults(payload, original_model)
    strip_unsupported_codex_payload_fields(payload)
    return url, headers, payload

def build_azure_endpoint(base_url, deployment_id, api_version="2025-01-01-preview"):
    # 移除base_url末尾的斜杠(如果有)
    base_url = base_url.rstrip('/')
    final_url = base_url

    if "models/chat/completions" not in final_url:
        # 构建路径
        path = f"/openai/deployments/{deployment_id}/chat/completions"
        # 使用urljoin拼接base_url和path
        final_url = urllib.parse.urljoin(base_url, path)

    if "?api-version=" not in final_url:
        # 添加api-version查询参数
        final_url = f"{final_url}?api-version={api_version}"

    return final_url

async def get_azure_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json',
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers['api-key'] = f"{api_key}"

    url = build_azure_endpoint(
        base_url=provider['base_url'],
        deployment_id=original_model,
    )

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
                elif item.type == "input_audio":
                    audio_item = _build_input_audio_item(item)
                    if audio_item:
                        audio_item.update(_get_extra_fields(item))
                        content.append(audio_item)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            msg_dict = {"role": msg.role, "content": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)

    payload = {
        "model": original_model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and "v1/responses" in url:
                payload["max_output_tokens"] = value
            elif field == "max_tokens" and "gpt-5" in original_model:
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") is False or "chatgpt-4o-latest" in original_model or "grok" in original_model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    apply_post_body_parameter_overrides(payload, provider, request.model)

    return url, headers, payload

async def get_azure_databricks_payload(request, engine, provider, api_key=None):
    api_key = base64.b64encode(f"token:{api_key}".encode()).decode()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Basic {api_key}",
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    base_url=provider['base_url']
    url = urllib.parse.urljoin(base_url, f"/serving-endpoints/{original_model}/invocations")

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
                elif item.type == "input_audio":
                    audio_item = _build_input_audio_item(item)
                    if audio_item:
                        audio_item.update(_get_extra_fields(item))
                        content.append(audio_item)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            msg_dict = {"role": msg.role, "content": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)

    if "claude-3-7-sonnet" in original_model:
        max_tokens = 128000
    elif "claude-3-5-sonnet" in original_model:
        max_tokens = 8192
    elif "claude-sonnet-4" in original_model:
        max_tokens = 64000
    elif "claude-opus-4" in original_model:
        max_tokens = 32000
    else:
        max_tokens = 4096

    payload = {
        "model": original_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            if field == "max_tokens" and "v1/responses" in url:
                payload["max_output_tokens"] = value
            elif field == "max_tokens" and "gpt-5" in original_model:
                payload["max_completion_tokens"] = value
            else:
                payload[field] = value

    if provider.get("tools") is False or "chatgpt-4o-latest" in original_model or "grok" in original_model:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if "think" in request.model.lower():
        payload["thinking"] = {
            "budget_tokens": 4096,
            "type": "enabled"
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
        if request.model.split("-")[-1].isdigit():
            think_tokens = int(request.model.split("-")[-1])
            if think_tokens < max_tokens:
                payload["thinking"] = {
                    "budget_tokens": think_tokens,
                    "type": "enabled"
                }

    if request.thinking:
        payload["thinking"] = {
            "budget_tokens": request.thinking.budget_tokens,
            "type": request.thinking.type
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)

    apply_post_body_parameter_overrides(payload, provider, request.model)

    return url, headers, payload

async def get_openrouter_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']
    if "openrouter.ai" in url:
        headers['HTTP-Referer'] = "https://github.com/yym68686/uni-api"
        headers['X-Title'] = "Uni API"

    messages = []
    for msg in request.messages:
        tool_calls = None
        tool_call_id = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
                elif item.type == "input_audio":
                    audio_item = _build_input_audio_item(item)
                    if audio_item:
                        audio_item.update(_get_extra_fields(item))
                        content.append(audio_item)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            for tool_call in tool_calls:
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                if provider.get("tools"):
                    messages.append({"role": msg.role, "tool_calls": tool_calls_list})
        elif tool_call_id:
            if provider.get("tools"):
                messages.append({"role": msg.role, "tool_call_id": tool_call_id, "content": content})
        else:
            # print("content", content)
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        msg_dict = {"role": msg.role, "content": item["text"]}
                        msg_dict.update(_get_extra_fields(msg))
                        messages.append(msg_dict)
                    elif item["type"] == "image_url":
                        messages.append({"role": msg.role, "content": [await get_image_message(item["image_url"]["url"], engine)]})
                    elif item["type"] == "input_audio":
                        messages.append({"role": msg.role, "content": [item]})
            else:
                msg_dict = {"role": msg.role, "content": content}
                msg_dict.update(_get_extra_fields(msg))
                messages.append(msg_dict)

    payload = {
        "model": original_model,
        "messages": messages,
    }

    miss_fields = [
        'model',
        'messages',
        'n',
        'user',
        'include_usage',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    apply_post_body_parameter_overrides(payload, provider, request.model)

    return url, headers, payload

async def get_cohere_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']

    role_map = {
        "user": "USER",
        "assistant" : "CHATBOT",
        "system": "SYSTEM"
    }

    messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    content.append(text_message)
        else:
            content = msg.content

        if isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    messages.append({"role": role_map[msg.role], "message": item["text"]})
        else:
            messages.append({"role": role_map[msg.role], "message": content})

    chat_history = messages[:-1]
    query = messages[-1].get("message")
    payload = {
        "model": original_model,
        "message": query,
    }

    if chat_history:
        payload["chat_history"] = chat_history

    miss_fields = [
        'model',
        'messages',
        'tools',
        'tool_choice',
        'temperature',
        'top_p',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def get_cloudflare_payload(request, engine, provider, api_key=None):
    headers = {
        'Content-Type': 'application/json'
    }
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = "https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{cf_model_id}".format(cf_account_id=provider['cf_account_id'], cf_model_id=original_model)

    msg = request.messages[-1]
    content = None
    if isinstance(msg.content, list):
        for item in msg.content:
            if item.type == "text":
                content = await get_text_message(item.text, engine)
    else:
        content = msg.content

    payload = {
        "prompt": content,
    }

    miss_fields = [
        'model',
        'messages',
        'tools',
        'tool_choice',
        'temperature',
        'top_p',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'logprobs',
        'top_logprobs',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    return url, headers, payload

async def gpt2claude_tools_json(json_dict):
    import copy
    json_dict = copy.deepcopy(json_dict)

    # 处理 $ref 引用
    def resolve_refs(obj, defs):
        if isinstance(obj, dict):
            # 如果有 $ref 引用，替换为实际定义
            if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    # 完全替换为引用的对象
                    ref_obj = copy.deepcopy(defs[ref_name])
                    # 保留原始对象中的其他属性
                    for k, v in obj.items():
                        if k != "$ref":
                            ref_obj[k] = v
                    return ref_obj

            # 递归处理所有属性
            for key, value in list(obj.items()):
                obj[key] = resolve_refs(value, defs)

        elif isinstance(obj, list):
            # 递归处理列表中的每个元素
            for i, item in enumerate(obj):
                obj[i] = resolve_refs(item, defs)

        return obj

    # 提取 $defs 定义
    defs = {}
    if "parameters" in json_dict and "defs" in json_dict["parameters"]:
        defs = json_dict["parameters"]["defs"]
        # 从参数中删除 $defs，因为 Claude 不需要它
        del json_dict["parameters"]["defs"]

    # 解析所有引用
    json_dict = resolve_refs(json_dict, defs)

    # 继续原有的键名转换逻辑
    keys_to_change = {
        "parameters": "input_schema",
    }
    for old_key, new_key in keys_to_change.items():
        if old_key in json_dict:
            if new_key:
                if json_dict[old_key] is None:
                    json_dict[old_key] = {
                        "type": "object",
                        "properties": {}
                    }
                json_dict[new_key] = json_dict.pop(old_key)
            else:
                json_dict.pop(old_key)
    return json_dict

async def get_claude_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    if "claude-3-7-sonnet" in original_model:
        anthropic_beta = "output-128k-2025-02-19"
    elif "claude-3-5-sonnet" in original_model:
        anthropic_beta = "max-tokens-3-5-sonnet-2024-07-15"
    else:
        anthropic_beta = "tools-2024-05-16"

    headers = {
        "content-type": "application/json",
        "x-api-key": f"{api_key}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": anthropic_beta,
    }
    url = provider['base_url']

    messages = []
    system_prompt = None
    tool_id = None
    for msg in request.messages:
        tool_call_id = None
        tool_calls = None
        if isinstance(msg.content, list):
            content = []
            for item in msg.content:
                if item.type == "text":
                    text_message = await get_text_message(item.text, engine)
                    text_message.update(_get_extra_fields(item))
                    content.append(text_message)
                elif item.type == "image_url" and provider.get("image", True):
                    image_message = await get_image_message(item.image_url.url, engine)
                    image_message.update(_get_extra_fields(item))
                    content.append(image_message)
        else:
            content = msg.content
            tool_calls = msg.tool_calls
            tool_id = tool_calls[0].id if tool_calls else None or tool_id
            tool_call_id = msg.tool_call_id

        if tool_calls:
            tool_calls_list = []
            tool_call = tool_calls[0]
            tool_calls_list.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments),
            })
            messages.append({"role": msg.role, "content": tool_calls_list})
        elif tool_call_id:
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content
            }]})
        elif msg.role == "function":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "name": msg.name,
                "input": {"prompt": "..."}
            }]})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_017r5miPMV6PGSNKmhvHPic4",
                "content": msg.content
            }]})
        elif msg.role != "system":
            msg_dict = {"role": msg.role, "content": content}
            msg_dict.update(_get_extra_fields(msg))
            messages.append(msg_dict)
        elif msg.role == "system":
            system_prompt = content

    conversation_len = len(messages) - 1
    message_index = 0
    while message_index < conversation_len:
        if messages[message_index]["role"] == messages[message_index + 1]["role"]:
            if messages[message_index].get("content"):
                if isinstance(messages[message_index]["content"], list):
                    messages[message_index]["content"].extend(messages[message_index + 1]["content"])
                elif isinstance(messages[message_index]["content"], str) and isinstance(messages[message_index + 1]["content"], list):
                    content_list = [{"type": "text", "text": messages[message_index]["content"]}]
                    content_list.extend(messages[message_index + 1]["content"])
                    messages[message_index]["content"] = content_list
                else:
                    messages[message_index]["content"] += messages[message_index + 1]["content"]
            messages.pop(message_index + 1)
            conversation_len = conversation_len - 1
        else:
            message_index = message_index + 1

    if "claude-3-7-sonnet" in original_model:
        max_tokens = 128000
    elif "claude-3-5-sonnet" in original_model:
        max_tokens = 8192
    elif "claude-sonnet-4" in original_model:
        max_tokens = 64000
    elif "claude-opus-4" in original_model:
        max_tokens = 32000
    else:
        max_tokens = 4096

    payload = {
        "model": original_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        payload["system"] = system_prompt

    if request.max_tokens:
        payload["max_tokens"] = int(request.max_tokens)

    miss_fields = [
        'model',
        'messages',
        'presence_penalty',
        'frequency_penalty',
        'n',
        'user',
        'include_usage',
        'stream_options',
    ]

    for field, value in request.model_dump(exclude_unset=True).items():
        if field not in miss_fields and value is not None:
            payload[field] = value

    if request.tools and provider.get("tools"):
        tools = []
        for tool in request.tools:
            # print("tool", type(tool), tool)
            json_tool = await gpt2claude_tools_json(tool.dict()["function"])
            tools.append(json_tool)
        payload["tools"] = tools
        if "tool_choice" in payload:
            if isinstance(payload["tool_choice"], dict):
                if payload["tool_choice"]["type"] == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": payload["tool_choice"]["function"]["name"]
                    }
            if isinstance(payload["tool_choice"], str):
                if payload["tool_choice"] == "auto":
                    payload["tool_choice"] = {
                        "type": "auto"
                    }
                if payload["tool_choice"] == "none":
                    payload["tool_choice"] = {
                        "type": "any"
                    }

    if provider.get("tools") is False:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

    if "think" in request.model.lower():
        payload["thinking"] = {
            "budget_tokens": 4096,
            "type": "enabled"
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
        if request.model.split("-")[-1].isdigit():
            think_tokens = int(request.model.split("-")[-1])
            if think_tokens < max_tokens:
                payload["thinking"] = {
                    "budget_tokens": think_tokens,
                    "type": "enabled"
                }

    if request.thinking:
        payload["thinking"] = {
            "budget_tokens": request.thinking.budget_tokens,
            "type": request.thinking.type
        }
        payload["temperature"] = 1
        payload.pop("top_p", None)
        payload.pop("top_k", None)
    # print("payload", json.dumps(payload, indent=2, ensure_ascii=False))

    apply_post_body_parameter_overrides(payload, provider, request.model)

    return url, headers, payload

async def get_dalle_payload(request, engine, provider, api_key=None, endpoint=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    multipart_files = getattr(request, "multipart_files", None)
    is_multipart = multipart_files is not None
    headers = {} if is_multipart else {"Content-Type": "application/json"}
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    base_api = BaseAPI(url)
    url = base_api.image_edit_url if endpoint == "/v1/images/edits" else base_api.image_url

    if is_multipart:
        multipart_data = list(getattr(request, "multipart_data", None) or [])
        multipart_data = [(key, value) for key, value in multipart_data if key != "model"]
        multipart_data.append(("model", original_model))
        payload = {
            "__multipart_data__": multipart_data,
            "__multipart_files__": list(multipart_files or []),
        }
        return url, headers, payload

    # Keep image payload minimal: only include fields explicitly provided by the client
    # (so we don't inject defaults like n/size/response_format), and pass through
    # newer optional fields (e.g. image_size/aspect_ratio) when present.
    payload = request.model_dump(exclude_unset=True)
    payload["model"] = original_model
    payload.setdefault("prompt", request.prompt)

    return url, headers, payload

async def get_upload_certificate(client: httpx.AsyncClient, api_key: str, model: str) -> dict:
    """第一步：获取文件上传凭证"""
    # print("步骤 1: 正在获取上传凭证...")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"action": "getPolicy", "model": model}
    try:
        response = await client.get("https://dashscope.aliyuncs.com/api/v1/uploads", headers=headers, params=params)
        response.raise_for_status()  # 如果请求失败则抛出异常
        cert_data = response.json()
        # print("凭证获取成功。")
        return cert_data.get("data")
    except httpx.HTTPStatusError as e:
        print(f"获取凭证失败: HTTP {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"获取凭证时发生未知错误: {e}")
        return None

async def upload_file_to_oss(client: httpx.AsyncClient, certificate: dict, file: Tuple[str, IOBase, str]) -> str:
    """第二步：使用凭证将文件内容上传到OSS"""
    upload_host = certificate.get("upload_host")
    upload_dir = certificate.get("upload_dir")
    object_key = f"{upload_dir}/{file[0]}"

    form_data = {
        "key": object_key,
        "policy": certificate.get("policy"),
        "OSSAccessKeyId": certificate.get("oss_access_key_id"),
        "signature": certificate.get("signature"),
        "success_action_status": "200",
        "x-oss-object-acl": certificate.get("x_oss_object_acl"),
        "x-oss-forbid-overwrite": certificate.get("x_oss_forbid_overwrite"),
    }

    files = {"file": file}

    try:
        response = await client.post(upload_host, data=form_data, files=files, timeout=3600)
        response.raise_for_status()
        # print("文件上传成功！")
        oss_url = f"oss://{object_key}"
        # print(f"文件OSS URL: {oss_url}")
        return oss_url
    except httpx.HTTPStatusError as e:
        print(f"上传文件失败: HTTP {e.response.status_code}")
        print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"上传文件时发生未知错误: {e}")
        return None

async def get_whisper_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {}
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).audio_transcriptions

    if "dashscope.aliyuncs.com" in url:
        client = httpx.AsyncClient()
        certificate = await get_upload_certificate(client, api_key, original_model)
        if not certificate:
            return

        # 步骤 2: 上传文件
        oss_url = await upload_file_to_oss(client, certificate, request.file)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-OssResourceResolve": "enable"
        }
        payload = {
            "model": original_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"audio": oss_url}]
                    }
                ]
            }
        }
    else:
        payload = {
            "model": original_model,
            "file": request.file,
        }

    if request.prompt:
        payload["prompt"] = request.prompt
    if request.response_format:
        payload["response_format"] = request.response_format
    if request.temperature:
        payload["temperature"] = request.temperature
    if request.language:
        payload["language"] = request.language

    # https://platform.openai.com/docs/api-reference/audio/createTranscription
    if request.timestamp_granularities:
        payload["timestamp_granularities[]"] = request.timestamp_granularities

    return url, headers, payload

async def get_moderation_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).moderations

    payload = {
        "model": original_model,
        "input": request.input,
    }

    return url, headers, payload

async def get_embedding_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"

    url = provider['base_url']
    url = BaseAPI(url).embeddings
    payload = {
        "input": request.input,
        "model": original_model,
    }

    if request.encoding_format:
        if url.startswith("https://api.jina.ai"):
            payload["embedding_type"] = request.encoding_format
        else:
            payload["encoding_format"] = request.encoding_format

    return url, headers, payload

async def get_tts_payload(request, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    url = provider['base_url']
    url = BaseAPI(url).audio_speech

    if "api.minimaxi.com" in url:
        payload = {
            "model": original_model,
            "text": request.input,
            "voice_setting": {
                "voice_id": request.voice
            }
        }
    else:
        payload = {
            "model": original_model,
            "input": request.input,
            "voice": request.voice,
        }

    if request.response_format:
        payload["response_format"] = request.response_format
    if request.speed:
        payload["speed"] = request.speed
    if request.stream is not None:
        payload["stream"] = request.stream

    return url, headers, payload


def _doubao_extract_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if hasattr(item, "type") and getattr(item, "type") == "text" and getattr(item, "text", None):
                parts.append(item.text)
                continue
            if hasattr(item, "content"):
                sub = _doubao_extract_text(getattr(item, "content", None))
                if sub:
                    parts.append(sub)
                continue
            if isinstance(item, dict):
                sub = _doubao_extract_text(item.get("text") or item.get("content") or item.get("input"))
                if sub:
                    parts.append(sub)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return _doubao_extract_text(content.get("text") or content.get("content") or content.get("input"))
    if hasattr(content, "text") and isinstance(getattr(content, "text", None), str):
        return content.text
    if hasattr(content, "content"):
        return _doubao_extract_text(getattr(content, "content", None))
    return ""

def _doubao_merge_translation_options(base: dict, override: dict | None) -> dict:
    if not isinstance(override, dict):
        return base
    source = override.get("source_language")
    target = override.get("target_language")
    if isinstance(source, str):
        source = source.strip()
    if isinstance(target, str):
        target = target.strip()
    if source:
        base["source_language"] = source
    if target:
        base["target_language"] = target
    return base

async def get_doubao_translation_payload(request: RequestModel, engine, provider, api_key=None):
    model_dict = get_model_dict(provider)
    original_model = model_dict[request.model]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = provider["base_url"]

    post_overrides = safe_get(provider, "preferences", "post_body_parameter_overrides", default={}) or {}
    model_overrides = post_overrides.get(request.model) if isinstance(post_overrides, dict) else None
    model_translation_overrides = (
        model_overrides.get("translation_options")
        if isinstance(model_overrides, dict)
        else None
    )

    default_target_language = safe_get(model_translation_overrides, "target_language", default=None) or "zh"
    translation_options = {"target_language": default_target_language}
    _doubao_merge_translation_options(translation_options, model_translation_overrides)

    user_text = None
    for msg in reversed(request.messages or []):
        if getattr(msg, "role", None) != "user":
            continue
        text = _doubao_extract_text(getattr(msg, "content", None))
        if text:
            user_text = text
            break
    if not user_text:
        raise ValueError("No user message")

    content_item = {
        "type": "input_text",
        "text": user_text,
        "translation_options": translation_options,
    }

    payload = {
        "model": original_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        **content_item
                    }
                ],
            }
        ],
    }
    if request.stream:
        payload["stream"] = True

    apply_post_body_parameter_overrides(
        payload,
        provider,
        request.model,
        skip_keys={"translation_options"},
    )

    return url, headers, payload

def _get_search_query(request: RequestModel) -> str:
    q = (request.get_last_text_message() or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing search query")
    return q

async def get_search_payload(request: RequestModel, provider: dict, api_key: str | None = None):
    """
    Search payload builder.
    Keep this as a small router so we can add other search providers later.
    """
    q = _get_search_query(request)

    provider_name = str(provider.get("provider") or "").lower()
    provider_base_url = str(provider.get("base_url") or "").lower()

    # Tavily: https://api.tavily.com/search (POST JSON)
    if provider_name == "tavily" or "api.tavily.com" in provider_base_url:
        url = provider.get("base_url") or "https://api.tavily.com/search"
        defaults = safe_get(provider, "preferences", "search_defaults", default={}) or {}
        payload = {
            "query": q,
            "topic": defaults.get("topic", "general"),
            "search_depth": defaults.get("search_depth", "basic"),
            "chunks_per_source": defaults.get("chunks_per_source", 3),
            "max_results": defaults.get("max_results", 7),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return url, headers, payload

    # Default implementation: Jina search endpoint (GET with query params).
    if provider_name == "jina" or "api.jina.ai" in provider_base_url:
        url = "https://s.jina.ai/"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Respond-With": "no-content",
        }
        payload = {"q": q}
        return url, headers, payload

    raise HTTPException(status_code=400, detail=f"Unsupported search provider: {provider.get('provider')}")

async def get_payload(request: RequestModel, engine, provider, api_key=None, endpoint=None):
    if engine == "gemini":
        return await get_gemini_payload(request, engine, provider, api_key)
    elif engine == "vertex-gemini":
        return await get_vertex_gemini_payload(request, engine, provider, api_key)
    elif engine == "aws":
        return await get_aws_payload(request, engine, provider, api_key)
    elif engine == "vertex-claude":
        return await get_vertex_claude_payload(request, engine, provider, api_key)
    elif engine == "azure":
        return await get_azure_payload(request, engine, provider, api_key)
    elif engine == "azure-databricks":
        return await get_azure_databricks_payload(request, engine, provider, api_key)
    elif engine == "claude":
        return await get_claude_payload(request, engine, provider, api_key)
    elif engine == "codex":
        return await get_codex_payload(request, engine, provider, api_key)
    elif engine == "gpt":
        provider['base_url'] = BaseAPI(provider['base_url']).chat_url
        return await get_gpt_payload(request, engine, provider, api_key)
    elif engine == "openrouter":
        return await get_openrouter_payload(request, engine, provider, api_key)
    elif engine == "cloudflare":
        return await get_cloudflare_payload(request, engine, provider, api_key)
    elif engine == "cohere":
        return await get_cohere_payload(request, engine, provider, api_key)
    elif engine == "dalle":
        return await get_dalle_payload(request, engine, provider, api_key, endpoint=endpoint)
    elif engine == "whisper":
        return await get_whisper_payload(request, engine, provider, api_key)
    elif engine == "tts":
        return await get_tts_payload(request, engine, provider, api_key)
    elif engine == "moderation":
        return await get_moderation_payload(request, engine, provider, api_key)
    elif engine == "embedding":
        return await get_embedding_payload(request, engine, provider, api_key)
    elif engine == "doubao-translation":
        return await get_doubao_translation_payload(request, engine, provider, api_key)
    elif engine == "search":
        return await get_search_payload(request, provider, api_key)
    else:
        raise ValueError("Unknown payload")

async def prepare_request_payload(provider, request_data):

    model_dict = get_model_dict(provider)
    request = RequestModel(**request_data)

    original_model = model_dict[request.model]
    engine, _ = get_engine(provider, endpoint=None, original_model=original_model)

    url, headers, payload = await get_payload(request, engine, provider, api_key=provider['api'])

    return url, headers, payload, engine
