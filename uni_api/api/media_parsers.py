from __future__ import annotations

import io
import json
from typing import Any, Optional

from fastapi import HTTPException, Request, UploadFile

from core.models import AudioTranscriptionRequest, ImageEditRequest


def is_form_upload(value: Any) -> bool:
    return hasattr(value, "filename") and hasattr(value, "file")


def form_text(value: Any) -> Optional[str]:
    if value is None or is_form_upload(value):
        return None
    text = str(value).strip()
    return text or None


def form_bool(value: Any, default: bool = False) -> bool:
    text = form_text(value)
    if text is None:
        return default
    return text.lower() in ("1", "true", "yes", "on")


async def parse_image_edit_request(http_request: Request) -> ImageEditRequest:
    content_type = (http_request.headers.get("content-type") or "").strip().lower()
    if content_type.startswith("application/json"):
        try:
            body = await http_request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        request = ImageEditRequest(**body)
        request.request_type = "image"
        return request

    if content_type and not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail=f"Unsupported Content-Type {content_type!r}")

    try:
        form = await http_request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid multipart form: {exc}") from exc

    prompt = form_text(form.get("prompt"))
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")

    model = form_text(form.get("model")) or "gpt-image-2"
    multipart_data: list[tuple[str, Any]] = []
    multipart_files: list[tuple[str, Any]] = []
    form_items = form.multi_items() if hasattr(form, "multi_items") else (
        (key, value) for key in form.keys() for value in form.getlist(key)
    )
    for key, value in form_items:
        if is_form_upload(value):
            try:
                file_content = await value.read()
            finally:
                try:
                    await value.close()
                except Exception:
                    pass
            multipart_files.append(
                (
                    key,
                    (
                        value.filename or "upload",
                        file_content,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
        else:
            multipart_data.append((key, str(value)))

    request = ImageEditRequest(
        prompt=prompt,
        model=model,
        stream=form_bool(form.get("stream"), False),
        multipart_data=multipart_data,
        multipart_files=multipart_files,
    )
    request.request_type = "image"
    return request


async def build_audio_transcription_request(
    *,
    http_request: Request,
    file: UploadFile,
    model: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    temperature: Optional[float] = None,
) -> AudioTranscriptionRequest:
    try:
        form_data = await http_request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = None
        content = await file.read()
        return AudioTranscriptionRequest(
            file=(file.filename, io.BytesIO(content), file.content_type),
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid audio file encoding") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(exc)}") from exc
