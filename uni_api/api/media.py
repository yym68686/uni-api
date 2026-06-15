from __future__ import annotations

from typing import Any, Optional

from fastapi import Request, UploadFile

from core.models import (
    AudioTranscriptionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    ModerationRequest,
    TextToSpeechRequest,
)
from uni_api.api.media_parsers import build_audio_transcription_request, parse_image_edit_request


async def image_generation_response(model_handler: Any, request: ImageGenerationRequest, api_index: int, background_tasks: Any):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/images/generations")


async def image_edit_response(model_handler: Any, http_request: Request, api_index: int, background_tasks: Any):
    request = await parse_image_edit_request(http_request)
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/images/edits")


async def embeddings_response(model_handler: Any, request: EmbeddingRequest, api_index: int, background_tasks: Any):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/embeddings")


async def audio_speech_response(model_handler: Any, request: TextToSpeechRequest, api_index: int, background_tasks: Any):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/audio/speech")


async def moderation_response(model_handler: Any, request: ModerationRequest, api_index: int, background_tasks: Any):
    return await model_handler.request_model(request, api_index, background_tasks, endpoint="/v1/moderations")


async def audio_transcription_response(
    *,
    model_handler: Any,
    http_request: Request,
    background_tasks: Any,
    file: UploadFile,
    model: str,
    api_index: int,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    temperature: Optional[float] = None,
):
    request_obj = await build_audio_transcription_request(
        http_request=http_request,
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
    )
    return await model_handler.request_model(
        request_obj,
        api_index,
        background_tasks,
        endpoint="/v1/audio/transcriptions",
    )
