from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RequestInspection:
    model: Optional[str]
    moderated_content: Optional[str]
    request_type: Optional[str]


def inspect_request_body(parsed_body: Any) -> RequestInspection:
    if not isinstance(parsed_body, dict):
        return RequestInspection(model=None, moderated_content=None, request_type=None)

    model = _clean_text(parsed_body.get("model"))

    if "messages" in parsed_body:
        return RequestInspection(
            model=model,
            moderated_content=_last_text_from_messages(parsed_body.get("messages")),
            request_type="chat",
        )

    if _looks_like_video_request(parsed_body):
        return RequestInspection(
            model=model,
            moderated_content=_last_text_from_video_body(parsed_body),
            request_type="video",
        )

    if "prompt" in parsed_body:
        return RequestInspection(
            model=model,
            moderated_content=_clean_text(parsed_body.get("prompt")),
            request_type="image",
        )

    if "file" in parsed_body:
        return RequestInspection(model=model, moderated_content=None, request_type="audio")

    model_lower = (model or "").lower()
    if "tts" in model_lower:
        return RequestInspection(
            model=model,
            moderated_content=_clean_text(parsed_body.get("input")),
            request_type="tts",
        )

    if "text-embedding" in model_lower:
        return RequestInspection(
            model=model,
            moderated_content=_embedding_text(parsed_body.get("input")),
            request_type="embedding",
        )

    if "input" in parsed_body:
        if "moderation" in model_lower or not model:
            return RequestInspection(model=model, moderated_content=None, request_type="moderation")
        return RequestInspection(
            model=model,
            moderated_content=_last_text_from_responses_input(parsed_body.get("input")),
            request_type="chat",
        )

    return RequestInspection(model=model, moderated_content=None, request_type=None)


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _last_text_from_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = _text_from_content(content, text_types={"text"})
        if text:
            return text
    return None


def _text_from_content(content: Any, *, text_types: set[str]) -> Optional[str]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    for item in reversed(content):
        if not isinstance(item, dict):
            continue
        if item.get("type") in text_types and item.get("text"):
            return str(item.get("text"))
    return None


def _last_text_from_responses_input(user_input: Any) -> Optional[str]:
    if isinstance(user_input, str):
        return user_input
    if not isinstance(user_input, list):
        return None
    for item in reversed(user_input):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").lower() != "user":
            continue
        text = _text_from_content(item.get("content"), text_types={"input_text", "text"})
        if text:
            return text
    return None


def _embedding_text(user_input: Any) -> Optional[str]:
    if isinstance(user_input, list) and user_input and isinstance(user_input[0], str):
        return "\n".join(user_input)
    return _clean_text(user_input)


def _looks_like_video_request(values: dict[str, Any]) -> bool:
    if "model" not in values:
        return False
    if "content" in values and isinstance(values.get("content"), list):
        return True
    video_keys = {
        "duration",
        "ratio",
        "resolution",
        "resources",
        "provider",
        "provider_options",
        "audio",
        "generate_audio",
        "watermark",
        "seed",
    }
    return any(key in values for key in video_keys)


def _last_text_from_video_body(values: dict[str, Any]) -> Optional[str]:
    prompt = _clean_text(values.get("prompt"))
    if prompt:
        return prompt
    return _text_from_content(values.get("content"), text_types={"text"})
