from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, Request


async def video_task_create_response(
    *,
    video_task_handler: Any,
    http_request: Request,
    request_body: dict[str, Any],
    api_index: int,
    background_tasks: BackgroundTasks,
):
    return await video_task_handler.create_task(
        http_request,
        request_body,
        api_index,
        background_tasks,
    )


async def video_task_get_response(
    *,
    video_task_handler: Any,
    http_request: Request,
    task_id: str,
    api_index: int,
    background_tasks: BackgroundTasks,
    model: Optional[str] = None,
):
    return await video_task_handler.get_or_delete_task(
        http_request,
        task_id,
        api_index,
        background_tasks,
        method="GET",
        model=model,
    )


async def asset_groups_create_response(
    *,
    lingjing_openapi_handler: Any,
    http_request: Request,
    request_body: dict[str, Any],
    api_index: int,
    background_tasks: BackgroundTasks,
    endpoint: str,
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        request_body,
        api_index,
        background_tasks,
        method="POST",
        openapi_path="/material/asset-groups",
        endpoint=endpoint,
    )


async def asset_group_get_response(
    *,
    lingjing_openapi_handler: Any,
    http_request: Request,
    group_id: str,
    api_index: int,
    background_tasks: BackgroundTasks,
    endpoint: str,
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        None,
        api_index,
        background_tasks,
        method="GET",
        openapi_path=f"/material/asset-groups/{quote(group_id, safe='')}",
        endpoint=endpoint,
    )


async def assets_create_response(
    *,
    lingjing_openapi_handler: Any,
    http_request: Request,
    request_body: dict[str, Any],
    api_index: int,
    background_tasks: BackgroundTasks,
    endpoint: str,
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        request_body,
        api_index,
        background_tasks,
        method="POST",
        openapi_path="/material/assets/create",
        endpoint=endpoint,
    )


async def asset_get_response(
    *,
    lingjing_openapi_handler: Any,
    http_request: Request,
    asset_id: str,
    api_index: int,
    background_tasks: BackgroundTasks,
    endpoint: str,
):
    return await lingjing_openapi_handler.request_openapi(
        http_request,
        None,
        api_index,
        background_tasks,
        method="GET",
        openapi_path=f"/material/assets/{quote(asset_id, safe='')}",
        endpoint=endpoint,
    )

