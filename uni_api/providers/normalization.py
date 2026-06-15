from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from core.utils import (
    cache_put_gemini_image_thought_signature,
    gemini_audio_inline_data_to_wav_base64,
    safe_get,
)


@dataclass(frozen=True)
class GeminiInlinePart:
    mime_type: str
    data_base64: str
    thought_signature: str | None = None


@dataclass(frozen=True)
class GeminiFunctionCall:
    name: str | None
    arguments: Any = None
    call_id: str | None = None

    @property
    def arguments_json(self) -> str | None:
        if self.arguments in (None, ""):
            return None
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass(frozen=True)
class GeminiPartsNormalization:
    content: str
    reasoning_content: str
    image_base64: str | None
    audio_wav_base64: str | None
    is_thinking: bool
    function_call: GeminiFunctionCall


def _part_inline_data(part: dict[str, Any]) -> GeminiInlinePart | None:
    mime_type = safe_get(part, "inlineData", "mimeType", default=None)
    if not mime_type:
        mime_type = safe_get(part, "inline_data", "mime_type", default=None)
    data_base64 = safe_get(part, "inlineData", "data", default=None)
    if not data_base64:
        data_base64 = safe_get(part, "inline_data", "data", default=None)
    if not mime_type or not data_base64:
        return None

    thought_signature = safe_get(part, "thoughtSignature", default=None)
    if not thought_signature:
        thought_signature = safe_get(part, "thought_signature", default=None)
    return GeminiInlinePart(
        mime_type=str(mime_type),
        data_base64=str(data_base64),
        thought_signature=str(thought_signature) if thought_signature else None,
    )


def _cache_image_thought_signature(inline: GeminiInlinePart) -> None:
    if inline.thought_signature:
        cache_put_gemini_image_thought_signature(inline.data_base64, inline.thought_signature)


def _function_call_id_from_thought_signature(thought_signature: Any) -> str | None:
    if not thought_signature:
        return None
    encoded = base64.urlsafe_b64encode(str(thought_signature).encode("utf-8")).decode("ascii").rstrip("=")
    return f"call_{encoded}.{uuid.uuid4().hex}"


def normalize_gemini_parts(parts: Iterable[Any]) -> GeminiPartsNormalization:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    image_base64: str | None = None
    audio_wav_base64: str | None = None
    is_thinking = False
    function_call = GeminiFunctionCall(name=None)

    for part in parts or []:
        if not isinstance(part, dict):
            continue

        text = safe_get(part, "text", default=None)
        part_is_thinking = bool(safe_get(part, "thought", default=False))
        if part_is_thinking:
            is_thinking = True
        if text:
            if part_is_thinking:
                reasoning_parts.append(str(text))
            else:
                content_parts.append(str(text))

        inline = _part_inline_data(part)
        if inline is not None:
            mime_type = inline.mime_type.lower()
            if mime_type.startswith("image/"):
                image_base64 = inline.data_base64
                _cache_image_thought_signature(inline)
            elif mime_type.startswith("audio/"):
                audio_wav_base64 = (
                    gemini_audio_inline_data_to_wav_base64(inline.mime_type, inline.data_base64)
                    or audio_wav_base64
                )

        function_name = safe_get(part, "functionCall", "name", default=None)
        if function_name and not function_call.name:
            thought_signature = safe_get(part, "thoughtSignature", default=None)
            if not thought_signature:
                thought_signature = safe_get(part, "thought_signature", default=None)
            function_call = GeminiFunctionCall(
                name=str(function_name),
                arguments=safe_get(part, "functionCall", "args", default=None),
                call_id=_function_call_id_from_thought_signature(thought_signature),
            )

    return GeminiPartsNormalization(
        content="".join(content_parts),
        reasoning_content="".join(reasoning_parts),
        image_base64=image_base64,
        audio_wav_base64=audio_wav_base64,
        is_thinking=is_thinking,
        function_call=function_call,
    )


def build_openai_audio_object(audio_wav_base64: str | None, *, transcript: str | None = None) -> dict | None:
    if not audio_wav_base64:
        return None
    return {
        "id": f"audio_{uuid.uuid4().hex[:24]}",
        "data": audio_wav_base64,
        "expires_at": None,
        "transcript": transcript or None,
        "format": "wav",
    }
