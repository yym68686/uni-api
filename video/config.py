from __future__ import annotations

import os
import re
from typing import Any, Optional

from .schema import VideoModelConfig, VideoProviderConfig, VideoRouteConfig


_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value.strip())
        if match:
            return os.getenv(match.group(1), "")
    if isinstance(value, dict):
        return {str(k): expand_env_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_value(v) for v in value]
    return value


def _default_route_method(route_name: str) -> str:
    if route_name in {"create_task", "create_asset", "create_asset_group"}:
        return "POST"
    return "GET"


def _route_from_raw(route_name: str, raw: Any) -> Optional[VideoRouteConfig]:
    if isinstance(raw, str):
        return VideoRouteConfig(method=_default_route_method(route_name), path=raw)
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "").strip()
    if not path:
        return None
    method = str(raw.get("method") or _default_route_method(route_name)).strip().upper()
    query_raw = raw.get("query")
    query = {
        str(k): str(v)
        for k, v in query_raw.items()
    } if isinstance(query_raw, dict) else {}
    return VideoRouteConfig(method=method, path=path, query=query)


def _models_from_raw(raw: Any) -> dict[str, VideoModelConfig]:
    models: dict[str, VideoModelConfig] = {}
    if not isinstance(raw, dict):
        return models
    for request_model, item in raw.items():
        request_model_name = str(request_model).strip()
        if not request_model_name:
            continue
        if isinstance(item, str):
            models[request_model_name] = VideoModelConfig(
                request_model=request_model_name,
                upstream_model=item,
            )
            continue
        if not isinstance(item, dict):
            continue
        upstream_model = str(item.get("upstream_model") or request_model_name).strip()
        protocol = str(item.get("protocol") or "").strip()
        capabilities = item.get("capabilities")
        models[request_model_name] = VideoModelConfig(
            request_model=request_model_name,
            upstream_model=upstream_model,
            protocol=protocol,
            capabilities=dict(capabilities) if isinstance(capabilities, dict) else {},
        )
    return models


def _provider_config_from_raw(raw: dict[str, Any]) -> Optional[VideoProviderConfig]:
    name = str(raw.get("name") or raw.get("provider") or "").strip()
    if not name:
        return None
    routes_raw = raw.get("routes")
    routes = {
        str(route_name): route
        for route_name, route in (
            (route_name, _route_from_raw(str(route_name), route_raw))
            for route_name, route_raw in (routes_raw.items() if isinstance(routes_raw, dict) else [])
        )
        if route is not None
    }
    return VideoProviderConfig(
        name=name,
        adapter=str(raw.get("adapter") or "http_json").strip() or "http_json",
        base_url=str(raw.get("base_url") or "").strip(),
        auth=expand_env_value(raw.get("auth") if isinstance(raw.get("auth"), dict) else {}),
        routes=routes,
        models=_models_from_raw(raw.get("models")),
        raw=raw,
    )


def load_video_provider_configs(config: dict[str, Any]) -> dict[str, VideoProviderConfig]:
    raw_list = config.get("video_providers")
    if not isinstance(raw_list, list):
        return {}
    providers: dict[str, VideoProviderConfig] = {}
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        provider_config = _provider_config_from_raw(raw)
        if provider_config is not None:
            providers[provider_config.name] = provider_config
    return providers


def find_video_provider_config(config: dict[str, Any], provider_name: str) -> Optional[VideoProviderConfig]:
    return load_video_provider_configs(config).get(str(provider_name or "").strip())
