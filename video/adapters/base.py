from __future__ import annotations

import json
from time import time
from typing import Any, Optional
from urllib.parse import quote, urlencode, urlparse, urlunparse

from video.schema import NormalizedVideoResponse, UpstreamVideoRequest, VideoAdapterError, VideoProviderConfig


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def maybe_json_object(raw: bytes) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def join_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    parsed = urlparse(base)
    base_path = parsed.path.rstrip("/")
    route_path = "/" + str(path or "").strip("/")
    upstream_path = f"{base_path}{route_path}"
    return urlunparse(parsed[:2] + (upstream_path,) + ("",) * 3)


def append_query(url: str, query: dict[str, Any]) -> str:
    filtered = {
        str(k): str(v)
        for k, v in query.items()
        if v is not None and str(v) != ""
    }
    if not filtered:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(filtered, doseq=True)}"


def path_with_task(path: str, task_id: str) -> str:
    return str(path or "").replace("{task_id}", quote(str(task_id), safe="")).replace("{id}", quote(str(task_id), safe=""))


def provider_options(request_body: dict[str, Any], provider_name: str) -> dict[str, Any]:
    options = request_body.get("provider_options")
    if not isinstance(options, dict):
        return {}
    provider_specific = options.get(provider_name)
    if isinstance(provider_specific, dict):
        return dict(provider_specific)
    return {
        key: value
        for key, value in options.items()
        if not isinstance(value, dict)
    }


def prompt_from_body(request_body: dict[str, Any]) -> str:
    prompt = str(request_body.get("prompt") or "").strip()
    if prompt:
        return prompt
    prompt_parts: list[str] = []
    content = request_body.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "").strip() == "text":
                text = str(part.get("text") or "").strip()
                if text:
                    prompt_parts.append(text)
    return "\n".join(prompt_parts).strip()


def resource_value(resource: dict[str, Any]) -> str:
    value = resource.get("url") or resource.get("value")
    source = resource.get("source")
    if not value and isinstance(source, dict):
        value = source.get("value")
    if not value:
        asset_id = resource.get("asset_id") or resource.get("assetId")
        if asset_id:
            value = f"asset://{asset_id}"
    return str(value or "").strip()


def content_resources(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    raw_resources = request_body.get("resources")
    if isinstance(raw_resources, list):
        for item in raw_resources:
            if isinstance(item, dict):
                resources.append(dict(item))

    content = request_body.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip()
            resource_type = ""
            value = ""
            if part_type == "image_url":
                resource_type = "image"
                value = _content_url(part, "image_url")
            elif part_type == "video_url":
                resource_type = "video"
                value = _content_url(part, "video_url")
            elif part_type == "audio_url":
                resource_type = "audio"
                value = _content_url(part, "audio_url")
            if resource_type and value:
                resource: dict[str, Any] = {"type": resource_type, "url": value}
                if part.get("role"):
                    resource["role"] = part.get("role")
                resources.append(resource)
    return resources


def _content_url(part: dict[str, Any], key: str) -> str:
    value = part.get(key)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("url") or "").strip()
    return ""


def usage_to_video_usage(usage: Any) -> Optional[dict[str, Any]]:
    if not isinstance(usage, dict):
        return None
    total_tokens = usage.get("total_tokens")
    completion_tokens = usage.get("completion_tokens")
    video_tokens = usage.get("video_tokens")
    if video_tokens is None:
        video_tokens = completion_tokens if completion_tokens is not None else total_tokens
    if total_tokens is None:
        total_tokens = video_tokens
    normalized: dict[str, Any] = {}
    if video_tokens is not None:
        normalized["video_tokens"] = video_tokens
        normalized["completion_tokens"] = video_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    for key, value in usage.items():
        normalized.setdefault(key, value)
    return normalized or None


def positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(float(str(value).strip().rstrip("pP")))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def estimated_video_usage(request_body: Optional[dict[str, Any]]) -> Optional[dict[str, int]]:
    if not isinstance(request_body, dict):
        return None
    duration = positive_int(request_body.get("duration")) or positive_int(request_body.get("seconds")) or 5
    fps = positive_int(request_body.get("fps")) or positive_int(request_body.get("framespersecond")) or 24
    resolution = request_body.get("quality")
    if resolution is None:
        resolution = request_body.get("resolution")
    resolution_height = positive_int(resolution) or 720
    tokens_per_frame_720p = 907.5
    video_tokens = max(1, int(round(duration * fps * tokens_per_frame_720p * (resolution_height / 720) ** 2)))
    return {
        "video_tokens": video_tokens,
        "completion_tokens": video_tokens,
        "total_tokens": video_tokens,
    }


class VideoProviderAdapter:
    name = "base"

    def __init__(self, config: Optional[VideoProviderConfig] = None) -> None:
        self.config = config

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
        raise NotImplementedError

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
        raise NotImplementedError

    def _base_url(self, provider: dict[str, Any]) -> str:
        configured = self.config.base_url if self.config else ""
        return configured or str(provider.get("base_url") or "")

    def _route(self, name: str, *, fallback_method: str, fallback_path: str) -> tuple[str, str, dict[str, str]]:
        if self.config and name in self.config.routes:
            route = self.config.routes[name]
            return route.method.upper(), route.path, route.query
        return fallback_method.upper(), fallback_path, {}

    def _model(self, request_model_name: str, original_model: str) -> tuple[str, str, dict[str, Any]]:
        if self.config and request_model_name in self.config.models:
            model = self.config.models[request_model_name]
            return model.upstream_model, model.protocol, model.capabilities
        return original_model, "", {}

    def _bearer_headers(
        self,
        provider_api_key_raw: Optional[str],
        *,
        include_content_type: bool,
    ) -> dict[str, str]:
        token = ""
        if self.config:
            auth = self.config.auth
            if str(auth.get("type") or "").strip().lower() == "bearer":
                token = str(auth.get("token") or "").strip()
        token = token or str(provider_api_key_raw or "").strip()
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def created_now() -> int:
        return int(time())


def replace_query_templates(query: dict[str, str], *, task_id: str) -> dict[str, str]:
    return {
        key: value.replace("{task_id}", str(task_id)).replace("{id}", str(task_id))
        for key, value in query.items()
    }
