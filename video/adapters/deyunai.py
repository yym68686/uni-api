from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote, urlparse

from .base import (
    content_resources,
    estimated_video_usage,
    join_url,
    json_bytes,
    maybe_json_object,
    path_with_task,
    provider_options,
    prompt_from_body,
    resource_value,
    usage_to_video_usage,
)
from video.schema import NormalizedVideoResponse, UpstreamVideoRequest
from .base import VideoProviderAdapter


def _normalize_tasks_url(base_url: str, task_id: Optional[str] = None) -> str:
    base = (base_url or "").strip()
    if not base:
        return base
    base = base.rstrip("/")
    parsed = urlparse(base)
    path = parsed.path.rstrip("/")

    if path.endswith("/contents/generations/tasks"):
        tasks_url = base
    elif path in ("", "/"):
        tasks_url = f"{base}/api/v3/contents/generations/tasks"
    else:
        tasks_url = f"{base}/contents/generations/tasks"

    if task_id is not None:
        tasks_url = f"{tasks_url}/{quote(str(task_id), safe='')}"
    return tasks_url


def _content_part_from_resource(resource: Any) -> Optional[dict[str, Any]]:
    if not isinstance(resource, dict):
        return None
    resource_type = str(resource.get("type") or "image").strip().lower()
    if resource_type not in {"image", "video", "audio"}:
        return None
    value = resource_value(resource)
    if not value:
        return None

    key = f"{resource_type}_url"
    part: dict[str, Any] = {
        "type": key,
        key: {"url": str(value)},
    }
    role = resource.get("role") or resource.get("usage")
    if role:
        part["role"] = role
    return part


def _convert_request_body(
    request_body: dict[str, Any],
    *,
    model_name: str,
    provider_name: str,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in request_body.items()
        if key not in {"provider", "provider_options", "route", "prompt", "resources", "audio"}
    }
    payload["model"] = model_name

    if not isinstance(payload.get("content"), list):
        content: list[dict[str, Any]] = []
        prompt = prompt_from_body(request_body)
        if prompt:
            content.append({"type": "text", "text": prompt})
        for resource in content_resources(request_body):
            part = _content_part_from_resource(resource)
            if part:
                content.append(part)
        if content:
            payload["content"] = content

    if "audio" in request_body and "generate_audio" not in payload:
        payload["generate_audio"] = bool(request_body.get("audio"))

    for key, value in provider_options(request_body, provider_name).items():
        if value is not None:
            payload[key] = value

    return payload


class DeyunaiVideoAdapter(VideoProviderAdapter):
    name = "deyunai"

    def build_request(
        self,
        *,
        method: str,
        task_id: Optional[str],
        request_body: Optional[dict[str, Any]],
        request_model_name: str,
        original_model: str,
        provider: dict[str, Any],
        provider_name: str,
        provider_api_key_raw: Optional[str],
    ) -> UpstreamVideoRequest:
        _ = request_model_name
        method_upper = method.upper()
        base_url = self._base_url(provider)
        if self.config:
            if method_upper == "POST":
                route_method, path, _query = self._route(
                    "create_task",
                    fallback_method="POST",
                    fallback_path="/contents/generations/tasks",
                )
                method_upper = route_method or method_upper
            else:
                route_method, path, _query = self._route(
                    "get_task",
                    fallback_method=method_upper,
                    fallback_path="/contents/generations/tasks/{task_id}",
                )
                method_upper = route_method if route_method in {"GET", "DELETE"} else method_upper
                path = path_with_task(path, str(task_id or ""))
            url = join_url(base_url, path)
        else:
            url = _normalize_tasks_url(base_url, task_id if method_upper != "POST" else None)

        payload = None
        if request_body is not None:
            payload = _convert_request_body(
                request_body,
                model_name=original_model,
                provider_name=provider_name,
            )

        headers = self._bearer_headers(provider_api_key_raw, include_content_type=method_upper in {"POST", "PUT"})
        headers.update(provider.get("preferences", {}).get("headers", {}) or {})
        return UpstreamVideoRequest(method=method_upper, url=url, headers=headers, payload=payload)

    def normalize_response(
        self,
        *,
        method: str,
        raw: bytes,
        task_id: Optional[str],
        request_model_name: str,
        provider_name: str,
        estimated_usage: Optional[dict[str, Any]] = None,
    ) -> NormalizedVideoResponse:
        obj = maybe_json_object(raw)
        if not obj:
            return NormalizedVideoResponse(raw=raw)

        method_upper = method.upper()
        if method_upper == "POST":
            upstream_task_id = obj.get("id")
            if not upstream_task_id:
                return NormalizedVideoResponse(raw=raw)
            return NormalizedVideoResponse(
                raw=json_bytes(
                    {
                        "id": str(upstream_task_id),
                        "model": request_model_name,
                        "provider": provider_name,
                        "status": str(obj.get("status") or "queued"),
                        "created_at": obj.get("created_at") or self.created_now(),
                    }
                ),
                task_id=str(upstream_task_id),
            )

        if method_upper == "GET":
            upstream_task_id = str(obj.get("id") or task_id or "")
            status = obj.get("status")
            if not upstream_task_id or not status:
                return NormalizedVideoResponse(raw=raw, task_id=upstream_task_id or None)

            video: dict[str, Any] = {}
            content = obj.get("content")
            if isinstance(content, dict) and content.get("video_url"):
                video["url"] = content.get("video_url")
            if obj.get("duration") is not None:
                video["duration"] = obj.get("duration")
            if obj.get("resolution") is not None:
                video["resolution"] = obj.get("resolution")
            if obj.get("ratio") is not None:
                video["ratio"] = obj.get("ratio")
            fps = obj.get("fps", obj.get("framespersecond"))
            if fps is not None:
                video["fps"] = fps

            normalized = {
                "id": upstream_task_id,
                "model": request_model_name,
                "provider": provider_name,
                "status": str(status),
                "video": video,
            }
            usage = usage_to_video_usage(obj.get("usage"))
            if not usage and normalized["status"] == "succeeded":
                usage = usage_to_video_usage(estimated_usage)
            if usage:
                normalized["usage"] = usage
            for key in ("created_at", "updated_at", "seed"):
                if obj.get(key) is not None:
                    normalized[key] = obj[key]
            return NormalizedVideoResponse(raw=json_bytes(normalized), task_id=upstream_task_id)

        return NormalizedVideoResponse(raw=raw)


__all__ = ["DeyunaiVideoAdapter", "estimated_video_usage"]
