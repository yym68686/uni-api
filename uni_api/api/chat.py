from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, Request

from core.models import RequestModel, ResponsesRequest


async def search_response(
    *,
    model_handler: Any,
    http_request: Request,
    background_tasks: BackgroundTasks,
    query: str,
    api_index: int,
):
    search_request = RequestModel(
        model="search",
        messages=[{"role": "user", "content": query}],
        stream=False,
    )
    return await model_handler.request_model(
        search_request,
        api_index,
        background_tasks,
        endpoint=str(http_request.url.path),
    )


async def chat_completions_response(
    *,
    model_handler: Any,
    request: RequestModel,
    background_tasks: BackgroundTasks,
    api_index: int,
):
    return await model_handler.request_model(request, api_index, background_tasks)


async def responses_api_response(
    *,
    responses_handler: Any,
    http_request: Request,
    request: ResponsesRequest,
    background_tasks: BackgroundTasks,
    api_index: int,
    endpoint: str = "/v1/responses",
):
    return await responses_handler.request_responses(
        http_request,
        request,
        api_index,
        background_tasks,
        endpoint=endpoint,
    )


async def messages_response(
    *,
    messages_handler: Any,
    http_request: Request,
    request_body: dict[str, Any],
    background_tasks: BackgroundTasks,
    api_index: int,
):
    return await messages_handler.request_messages(
        http_request,
        request_body,
        api_index,
        background_tasks,
    )

