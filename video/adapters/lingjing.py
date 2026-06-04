from __future__ import annotations

import json
from typing import Any, Optional
from urllib.parse import quote, urlencode, urlparse, urlunparse

from .base import json_bytes, maybe_json_object, provider_options, prompt_from_body, usage_to_video_usage
from .base import VideoProviderAdapter
from video.schema import NormalizedVideoResponse, UpstreamVideoRequest, VideoAdapterError


LINGJING_UPSTREAM_OPENAPI_PREFIX = "/api/entrance/openapi"


def _normalize_openapi_url(base_url: str, openapi_path: str, query: str = "") -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return base

    path = "/" + str(openapi_path or "").strip("/")
    if path.startswith("/v1/openapi/"):
        path = path[len("/v1"):]
    if path.startswith("/api/entrance/openapi/"):
        upstream_openapi_path = path
    else:
        if not path.startswith("/openapi/"):
            path = "/openapi" + path
        upstream_openapi_path = LINGJING_UPSTREAM_OPENAPI_PREFIX + path[len("/openapi"):]

    parsed = urlparse(base)
    base_path = parsed.path.rstrip("/")
    if upstream_openapi_path.startswith("/api/entrance/openapi"):
        if base_path.endswith("/api/entrance/openapi"):
            upstream_path = base_path + upstream_openapi_path[len("/api/entrance/openapi"):]
        elif base_path.endswith("/api/entrance"):
            upstream_path = base_path + upstream_openapi_path[len("/api/entrance"):]
        else:
            upstream_path = base_path + upstream_openapi_path
    else:
        upstream_path = base_path + upstream_openapi_path

    url = urlunparse(parsed[:2] + (upstream_path,) + ("",) * 3)
    return f"{url}?{query}" if query else url


def _parse_credentials(provider: dict[str, Any], provider_api_key_raw: Optional[str], config_auth: dict[str, Any]) -> tuple[str, str]:
    headers = config_auth.get("headers") if isinstance(config_auth, dict) else None
    if isinstance(headers, dict):
        access_key = str(headers.get("X-Access-Key") or headers.get("x-access-key") or "").strip()
        secret_key = str(headers.get("X-Secret-Key") or headers.get("x-secret-key") or "").strip()
        if access_key and secret_key:
            return access_key, secret_key

    preferences = provider.get("preferences") if isinstance(provider.get("preferences"), dict) else {}
    access_key = str(preferences.get("access_key") or preferences.get("accessKey") or "").strip()
    secret_key = str(preferences.get("secret_key") or preferences.get("secretKey") or "").strip()
    raw = str(provider_api_key_raw or "").strip()

    if (not access_key or not secret_key) and raw:
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                access_key = access_key or str(parsed.get("access_key") or parsed.get("accessKey") or "").strip()
                secret_key = secret_key or str(parsed.get("secret_key") or parsed.get("secretKey") or "").strip()
        for sep in (":", ",", "|"):
            if access_key and secret_key:
                break
            if sep in raw:
                left, right = raw.split(sep, 1)
                access_key = access_key or left.strip()
                secret_key = secret_key or right.strip()

    if not access_key or not secret_key:
        raise VideoAdapterError("Lingjing provider requires access and secret keys")
    return access_key, secret_key


def _source_from_value(value: Any) -> dict[str, Any]:
    raw = str(value or "").strip()
    if raw.startswith("asset://"):
        return {"kind": "asset_id", "value": raw[len("asset://"):]}
    if raw.startswith("Asset-"):
        return {"kind": "asset_id", "value": raw}
    return {"kind": "url", "value": raw}


def _extract_content_url(part: dict[str, Any], type_name: str) -> str:
    value = part.get(type_name)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("url") or "").strip()
    return ""


def _usage_from_role(role: Any, resource_type: str, resource_index: int) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in {"first_frame", "last_frame", "reference", "keyframe", "source"}:
        return normalized
    if normalized in {"reference_image", "reference_video", "reference_audio"}:
        return "reference"
    if resource_type == "image" and resource_index == 0:
        return "first_frame"
    return "reference"


def _resource_from_unified(resource: Any, resource_index: int) -> Optional[dict[str, Any]]:
    if not isinstance(resource, dict):
        return None

    resource_type = str(resource.get("type") or "image").strip().lower()
    if resource_type not in {"image", "video", "audio"}:
        return None

    usage = resource.get("usage", resource.get("role"))
    source = resource.get("source")
    if not isinstance(source, dict):
        value = (
            resource.get("url")
            or resource.get("asset_id")
            or resource.get("assetId")
            or resource.get("value")
            or resource.get("data_url")
        )
        source = _source_from_value(value)

    normalized: dict[str, Any] = {
        "type": resource_type,
        "usage": _usage_from_role(usage, resource_type, resource_index),
        "source": source,
    }
    reference_key = resource.get("reference_key") or resource.get("referenceKey")
    if reference_key:
        normalized["reference_key"] = reference_key
    return normalized


def _resources_from_unified(resources: Any) -> list[dict[str, Any]]:
    if not isinstance(resources, list):
        return []
    normalized_resources: list[dict[str, Any]] = []
    for resource in resources:
        normalized = _resource_from_unified(resource, len(normalized_resources))
        if normalized:
            normalized_resources.append(normalized)
    return normalized_resources


def _convert_request_body(
    request_body: dict[str, Any],
    *,
    model_code: str,
    provider_name: str,
) -> dict[str, Any]:
    if "taskParams" in request_body or "modelCode" in request_body:
        payload = dict(request_body)
        payload["modelCode"] = model_code
        for key in ("model", "request_model", "provider", "provider_options", "route"):
            payload.pop(key, None)
        return payload

    resources: list[dict[str, Any]] = []
    content = request_body.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip()
            resource_type = ""
            url = ""
            if part_type == "image_url":
                resource_type = "image"
                url = _extract_content_url(part, "image_url")
            elif part_type == "video_url":
                resource_type = "video"
                url = _extract_content_url(part, "video_url")
            elif part_type == "audio_url":
                resource_type = "audio"
                url = _extract_content_url(part, "audio_url")

            if resource_type and url:
                resource: dict[str, Any] = {
                    "type": resource_type,
                    "usage": _usage_from_role(part.get("role"), resource_type, len(resources)),
                    "source": _source_from_value(url),
                }
                reference_key = part.get("reference_key")
                if reference_key:
                    resource["reference_key"] = reference_key
                resources.append(resource)

    input_payload: dict[str, Any] = {
        "prompt": prompt_from_body(request_body),
    }

    quality = request_body.get("quality")
    if quality is None:
        resolution = str(request_body.get("resolution") or "").strip().lower()
        quality = resolution[:-1] if resolution.endswith("p") else resolution
    if quality:
        input_payload["quality"] = str(quality)

    for key in ("duration", "ratio", "generate_num", "prompt_optimizer"):
        if key in request_body and request_body.get(key) is not None:
            input_payload[key] = request_body[key]

    unified_resources = _resources_from_unified(request_body.get("resources"))
    if unified_resources:
        input_payload["resources"] = unified_resources
    elif resources:
        input_payload["resources"] = resources

    for key, value in provider_options(request_body, provider_name).items():
        if value is not None:
            input_payload[key] = value

    if "generate_audio" in request_body:
        input_payload["need_audio"] = bool(request_body.get("generate_audio"))
    if "need_audio" in request_body:
        input_payload["need_audio"] = bool(request_body.get("need_audio"))
    if "audio" in request_body:
        input_payload["need_audio"] = bool(request_body.get("audio"))

    return {"modelCode": model_code, "taskParams": {"input": input_payload}}


def _task_id_from_submit_response(obj: dict[str, Any]) -> Optional[str]:
    data = obj.get("data")
    if isinstance(data, dict):
        for key in ("taskId", "task_id", "id"):
            value = data.get(key)
            if value:
                return str(value)
    return None


def _status_to_unified(status: Any) -> str:
    normalized = str(status or "").strip().upper()
    if normalized == "SUCCESS":
        return "succeeded"
    if normalized == "CANCELED":
        return "cancelled"
    if normalized in {"FAIL", "FAILED", "UNKNOWN"}:
        return "failed"
    if normalized in {"WAITING", "QUEUED", "SUBMITTED", "RUNNING", "PROCESSING"}:
        return "running"
    return normalized.lower() if normalized else "running"


def _first_result_url(result: Any) -> Optional[str]:
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"])
    if isinstance(result, dict) and result.get("url"):
        return str(result["url"])
    return None


class LingjingVideoAdapter(VideoProviderAdapter):
    name = "lingjing"

    def _headers(
        self,
        provider: dict[str, Any],
        provider_api_key_raw: Optional[str],
        *,
        include_content_type: bool,
    ) -> dict[str, str]:
        config_auth = self.config.auth if self.config else {}
        access_key, secret_key = _parse_credentials(provider, provider_api_key_raw, config_auth)
        headers = {
            "X-Access-Key": access_key,
            "X-Secret-Key": secret_key,
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        headers.update(provider.get("preferences", {}).get("headers", {}) or {})
        return headers

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
        if method_upper == "DELETE":
            raise VideoAdapterError("Lingjing video tasks do not support DELETE", status_code=405)

        base_url = self._base_url(provider)
        if method_upper == "POST":
            route_method, path, query = self._route("create_task", fallback_method="POST", fallback_path="/draw/task/submit")
            method_upper = route_method or method_upper
        else:
            route_method, path, query = self._route("get_task", fallback_method="GET", fallback_path="/draw/task/query")
            method_upper = route_method or method_upper
            query = dict(query) if query else {"taskId": "{task_id}"}
        query_string = urlencode(
            {
                key: value.replace("{task_id}", str(task_id or "")).replace("{id}", str(task_id or ""))
                for key, value in query.items()
            },
            doseq=True,
        ) if query else ""
        url = _normalize_openapi_url(base_url, path, query=query_string)

        payload = None
        if request_body is not None:
            payload = _convert_request_body(
                request_body,
                model_code=original_model,
                provider_name=provider_name,
            )
        headers = self._headers(provider, provider_api_key_raw, include_content_type=method_upper in {"POST", "PUT"})
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
            upstream_task_id = _task_id_from_submit_response(obj)
            if not upstream_task_id:
                return NormalizedVideoResponse(raw=raw)
            return NormalizedVideoResponse(
                raw=json_bytes(
                    {
                        "id": upstream_task_id,
                        "model": request_model_name,
                        "provider": provider_name,
                        "status": "queued",
                        "created_at": self.created_now(),
                    }
                ),
                task_id=upstream_task_id,
            )

        if method_upper == "GET":
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            upstream_task_id = str(data.get("task_id") or data.get("taskId") or task_id or "")
            result_url = _first_result_url(data.get("result"))
            normalized: dict[str, Any] = {
                "id": upstream_task_id,
                "model": request_model_name,
                "provider": provider_name,
                "status": _status_to_unified(data.get("status")),
                "video": {},
            }
            if result_url:
                normalized["video"]["url"] = result_url
            usage = usage_to_video_usage(data.get("usage") if isinstance(data, dict) else None)
            if not usage and normalized["status"] == "succeeded":
                usage = usage_to_video_usage(estimated_usage)
            if usage:
                normalized["usage"] = usage
            if data.get("external_error"):
                normalized["error"] = {"message": data.get("external_error")}
            return NormalizedVideoResponse(raw=json_bytes(normalized), task_id=upstream_task_id or None)

        return NormalizedVideoResponse(raw=raw)


__all__ = ["LingjingVideoAdapter"]
