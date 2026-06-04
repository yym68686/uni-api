from __future__ import annotations

import math
import re
from typing import Any, Optional

from .base import (
    content_resources,
    estimated_video_usage,
    join_url,
    json_bytes,
    maybe_json_object,
    path_with_task,
    positive_int,
    provider_options,
    prompt_from_body,
    resource_value,
    usage_to_video_usage,
)
from .base import VideoProviderAdapter
from video.schema import NormalizedVideoResponse, UpstreamVideoRequest, VideoAdapterError


SORA_2_RATIOS = {"16:9", "9:16"}
SORA_V3_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}
SORA_2_SECONDS = {"4", "8", "12"}
VEO_SIZES = {"1280x720", "720x1280", "1920x1080", "1080x1920"}


def _clean_resolution(value: Any, *, default: str = "720p") -> str:
    raw = str(value or default).strip().lower()
    if not raw:
        return default
    if raw.isdigit():
        return f"{raw}p"
    return raw


def _duration_string(value: Any, *, default: str) -> str:
    if value is None or str(value).strip() == "":
        return default
    parsed = positive_int(value)
    if parsed is None:
        raise VideoAdapterError("duration/seconds must be an integer")
    return str(parsed)


def _ratio_from_body(request_body: dict[str, Any], options: dict[str, Any], *, default: str = "16:9") -> str:
    return str(
        request_body.get("aspect_ratio")
        or request_body.get("ratio")
        or options.get("aspect_ratio")
        or options.get("ratio")
        or default
    ).strip()


def _size_for_ratio_resolution(ratio: str, resolution: str) -> str:
    height = positive_int(resolution) or 720
    ratio_parts = str(ratio or "16:9").split(":", 1)
    if len(ratio_parts) != 2:
        raise VideoAdapterError(f"Unsupported ratio: {ratio}")
    try:
        width_ratio = float(ratio_parts[0])
        height_ratio = float(ratio_parts[1])
    except ValueError as exc:
        raise VideoAdapterError(f"Unsupported ratio: {ratio}") from exc
    if width_ratio <= 0 or height_ratio <= 0:
        raise VideoAdapterError(f"Unsupported ratio: {ratio}")
    width = int(round(height * width_ratio / height_ratio))
    return f"{width}x{height}"


def _ratio_for_size(size: str) -> Optional[str]:
    try:
        width_text, height_text = str(size).lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"


def _resource_url(resource: dict[str, Any]) -> str:
    value = resource_value(resource) or str(resource.get("data_url") or resource.get("dataUrl") or "").strip()
    if not value:
        return ""
    if value.startswith("asset://") or value.startswith("Asset-"):
        raise VideoAdapterError("callxyq resources require public URL or data URL; asset_id resources are not supported")
    source = resource.get("source")
    if isinstance(source, dict):
        kind = str(source.get("kind") or "").strip().lower()
        if kind in {"asset_id", "asset"}:
            raise VideoAdapterError("callxyq resources require public URL or data URL; asset_id resources are not supported")
    return value


def _group_resource_urls(request_body: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    for resource in content_resources(request_body):
        resource_type = str(resource.get("type") or "image").strip().lower()
        url = _resource_url(resource)
        if not url:
            continue
        if resource_type == "image":
            images.append(url)
        elif resource_type == "video":
            videos.append(url)
        elif resource_type == "audio":
            audios.append(url)
        else:
            raise VideoAdapterError(f"Unsupported callxyq resource type: {resource_type}")
    return images, videos, audios


def _require_prompt_refs(prompt: str, *, images: int, videos: int, audios: int) -> None:
    if images <= 1 and videos == 0 and audios == 0:
        return
    for index in range(1, images + 1):
        if f"@Image{index}" not in prompt:
            raise VideoAdapterError(f"callxyq Sora multi-resource prompts must reference @Image{index}")
    for index in range(1, videos + 1):
        if f"@Video{index}" not in prompt:
            raise VideoAdapterError(f"callxyq Sora multi-resource prompts must reference @Video{index}")
    for index in range(1, audios + 1):
        if f"@Audio{index}" not in prompt:
            raise VideoAdapterError(f"callxyq Sora audio prompts must reference @Audio{index}")


def _infer_protocol(model: str, configured_protocol: str) -> str:
    protocol = str(configured_protocol or "").strip().lower()
    if protocol:
        return protocol
    model_lower = str(model or "").lower()
    if model_lower.startswith("gemini-veo"):
        return "veo"
    return "sora"


def _is_sora_2(model: str) -> bool:
    return str(model or "").strip().lower() == "sora-2"


def _is_sora_v3(model: str) -> bool:
    return str(model or "").strip().lower() in {"sora-v3-fast", "sora-v3-pro"}


def _veo_duration_from_model(model: str) -> Optional[int]:
    match = re.search(r"-(4|6|8)s$", str(model or ""))
    return int(match.group(1)) if match else None


def _status_to_unified(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "succeeded"
    if normalized in {"in_progress", "processing"}:
        return "running"
    if normalized in {"queued", "failed"}:
        return normalized
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    return normalized or "queued"


def _video_metadata(obj: dict[str, Any]) -> dict[str, Any]:
    video: dict[str, Any] = {}
    if obj.get("video_url"):
        video["url"] = obj.get("video_url")
    seconds = obj.get("seconds")
    parsed_seconds = positive_int(seconds)
    if parsed_seconds is not None:
        video["duration"] = parsed_seconds
    size = obj.get("size")
    if size:
        video["size"] = size
        ratio = _ratio_for_size(str(size))
        if ratio:
            video["ratio"] = ratio
        try:
            height = str(size).lower().split("x", 1)[1]
            video["resolution"] = f"{height}p"
        except Exception:
            pass
    return video


class CallxyqVideoAdapter(VideoProviderAdapter):
    name = "callxyq"

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
        method_upper = method.upper()
        if method_upper == "DELETE":
            raise VideoAdapterError("callxyq video tasks do not support DELETE", status_code=405)

        route_name = "create_task" if method_upper == "POST" else "get_task"
        fallback_path = "/v1/videos" if method_upper == "POST" else "/v1/videos/{task_id}"
        route_method, path, _query = self._route(route_name, fallback_method=method_upper, fallback_path=fallback_path)
        method_upper = route_method or method_upper
        if task_id:
            path = path_with_task(path, task_id)
        url = join_url(self._base_url(provider), path)

        payload = None
        if request_body is not None:
            upstream_model, configured_protocol, _capabilities = self._model(request_model_name, original_model)
            protocol = _infer_protocol(upstream_model, configured_protocol)
            if protocol == "veo":
                payload = self._build_veo_payload(request_body, upstream_model, provider_name)
            else:
                payload = self._build_sora_payload(request_body, upstream_model, provider_name)

        headers = self._bearer_headers(provider_api_key_raw, include_content_type=method_upper in {"POST", "PUT"})
        headers.update(provider.get("preferences", {}).get("headers", {}) or {})
        return UpstreamVideoRequest(method=method_upper, url=url, headers=headers, payload=payload)

    def _build_sora_payload(self, request_body: dict[str, Any], upstream_model: str, provider_name: str) -> dict[str, Any]:
        options = provider_options(request_body, provider_name)
        prompt = prompt_from_body(request_body)
        if not prompt:
            raise VideoAdapterError("callxyq Sora requests require prompt")

        is_sora_2 = _is_sora_2(upstream_model)
        if not is_sora_2 and not _is_sora_v3(upstream_model):
            raise VideoAdapterError(f"Unsupported callxyq Sora model: {upstream_model}")

        default_seconds = "4" if is_sora_2 else "5"
        aspect_ratio = _ratio_from_body(request_body, options)
        resolution = _clean_resolution(request_body.get("resolution") or options.get("resolution"), default="720p")
        seconds = _duration_string(
            options.get("seconds", request_body.get("seconds", request_body.get("duration"))),
            default=default_seconds,
        )
        size = str(request_body.get("size") or options.get("size") or _size_for_ratio_resolution(aspect_ratio, resolution)).strip()

        allowed_ratios = SORA_2_RATIOS if is_sora_2 else SORA_V3_RATIOS
        if aspect_ratio not in allowed_ratios:
            raise VideoAdapterError(f"{upstream_model} does not support ratio/aspect_ratio {aspect_ratio}")
        if is_sora_2 and resolution != "720p":
            raise VideoAdapterError("sora-2 only supports 720p resolution")
        if not is_sora_2 and resolution not in {"480p", "720p"}:
            raise VideoAdapterError("sora-v3 models only support 480p or 720p resolution")
        if is_sora_2 and seconds not in SORA_2_SECONDS:
            raise VideoAdapterError("sora-2 seconds must be one of 4, 8, 12")
        if not is_sora_2:
            parsed_seconds = int(seconds)
            if parsed_seconds < 5 or parsed_seconds > 15:
                raise VideoAdapterError("sora-v3 seconds must be between 5 and 15")

        images, videos, audios = _group_resource_urls(request_body)
        if is_sora_2 and (videos or audios):
            raise VideoAdapterError("sora-2 does not support video or audio references")
        if len(images) > (1 if is_sora_2 else 4):
            raise VideoAdapterError(f"{upstream_model} supports at most {'1' if is_sora_2 else '4'} image references")
        if len(videos) > 3:
            raise VideoAdapterError("sora-v3 supports at most 3 video references")
        if len(audios) > 1:
            raise VideoAdapterError("sora-v3 supports at most 1 audio reference")
        if audios and not images:
            raise VideoAdapterError("callxyq Sora audio references require at least one image reference")
        _require_prompt_refs(prompt, images=len(images), videos=len(videos), audios=len(audios))

        payload: dict[str, Any] = {
            "model": upstream_model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "size": size,
            "seconds": seconds,
        }
        if images:
            if len(images) == 1 or is_sora_2:
                payload["image_url"] = images[0]
            else:
                payload["reference_image_urls"] = images
        if videos:
            payload["reference_video"] = videos[0]
            payload["reference_videos"] = videos
        if audios:
            payload["audio_url"] = audios[0]
            payload["video_config"] = options.get("video_config") or {
                "reference_mode": "image_reference",
                "motion_has_audio": True,
            }

        for key, value in options.items():
            if key in {"ratio", "duration", "seconds"} or value is None:
                continue
            payload[key] = value
        return payload

    def _build_veo_payload(self, request_body: dict[str, Any], upstream_model: str, provider_name: str) -> dict[str, Any]:
        options = provider_options(request_body, provider_name)
        prompt = prompt_from_body(request_body)
        if not prompt:
            raise VideoAdapterError("callxyq Veo requests require prompt")

        images, videos, audios = _group_resource_urls(request_body)
        if videos or audios:
            raise VideoAdapterError("callxyq Veo models do not support video or audio resources")
        max_images = 3 if "-ref-" in upstream_model else 2
        if len(images) > max_images:
            raise VideoAdapterError(f"{upstream_model} supports at most {max_images} image references")

        model_duration = _veo_duration_from_model(upstream_model)
        requested_duration = positive_int(request_body.get("duration")) or positive_int(request_body.get("seconds"))
        if requested_duration is not None and model_duration is not None and requested_duration != model_duration:
            raise VideoAdapterError(
                f"Veo duration is encoded in model name ({model_duration}s); request duration {requested_duration}s does not match"
            )

        size = str(request_body.get("size") or options.get("size") or "").strip()
        aspect_ratio = _ratio_from_body(request_body, options)
        if not size:
            resolution = _clean_resolution(request_body.get("resolution") or options.get("resolution"), default="720p")
            if aspect_ratio == "9:16":
                size = "1080x1920" if resolution == "1080p" else "720x1280"
            elif aspect_ratio == "16:9":
                size = "1920x1080" if resolution == "1080p" else "1280x720"
            else:
                raise VideoAdapterError("callxyq Veo only supports aspect_ratio 16:9 or 9:16")
        if size not in VEO_SIZES:
            raise VideoAdapterError(f"Unsupported callxyq Veo size: {size}")

        payload: dict[str, Any] = {
            "model": upstream_model,
            "prompt": prompt,
            "size": size,
        }
        if "generate_audio" in request_body:
            payload["generate_audio"] = bool(request_body.get("generate_audio"))
        elif "audio" in request_body:
            payload["generate_audio"] = bool(request_body.get("audio"))
        elif "generate_audio" in options:
            payload["generate_audio"] = bool(options.get("generate_audio"))
        if request_body.get("negative_prompt") is not None:
            payload["negative_prompt"] = request_body.get("negative_prompt")
        elif options.get("negative_prompt") is not None:
            payload["negative_prompt"] = options.get("negative_prompt")
        if images:
            if len(images) == 1:
                payload["image_url"] = images[0]
            else:
                payload["images"] = images

        for key, value in options.items():
            if key in {"ratio", "aspect_ratio", "resolution", "duration", "seconds"} or value is None:
                continue
            payload[key] = value
        return payload

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
        upstream_task_id = str(obj.get("id") or obj.get("task_id") or task_id or "")
        status = _status_to_unified(obj.get("status"))

        if method_upper == "POST":
            if not upstream_task_id:
                return NormalizedVideoResponse(raw=raw)
            normalized: dict[str, Any] = {
                "id": upstream_task_id,
                "model": request_model_name,
                "provider": provider_name,
                "status": status or "queued",
                "created_at": obj.get("created_at") or self.created_now(),
            }
            if obj.get("progress") is not None:
                normalized["progress"] = obj.get("progress")
            return NormalizedVideoResponse(raw=json_bytes(normalized), task_id=upstream_task_id)

        if method_upper == "GET":
            if not upstream_task_id:
                return NormalizedVideoResponse(raw=raw)
            normalized = {
                "id": upstream_task_id,
                "model": request_model_name,
                "provider": provider_name,
                "status": status,
                "video": _video_metadata(obj),
            }
            if obj.get("progress") is not None:
                normalized["progress"] = obj.get("progress")
            for source_key, normalized_key in (("created_at", "created_at"), ("completed_at", "updated_at"), ("updated_at", "updated_at")):
                if obj.get(source_key) is not None:
                    normalized[normalized_key] = obj[source_key]
            if obj.get("error"):
                normalized["error"] = obj.get("error")
            usage = usage_to_video_usage(obj.get("usage"))
            if not usage and status == "succeeded":
                usage = usage_to_video_usage(estimated_usage)
            if usage:
                normalized["usage"] = usage
            return NormalizedVideoResponse(raw=json_bytes(normalized), task_id=upstream_task_id)

        return NormalizedVideoResponse(raw=raw)


__all__ = ["CallxyqVideoAdapter", "estimated_video_usage"]
