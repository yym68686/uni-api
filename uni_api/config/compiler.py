from __future__ import annotations

from collections import defaultdict
from typing import Any

from uni_api.routing.core import (
    build_api_key_model_response_cache,
    build_api_key_models_map,
    build_routing_index,
)
from uni_api.config.runtime import RuntimeConfig
from video.config import load_video_provider_configs


def compile_runtime_config(
    config: dict[str, Any],
    api_list: list[str],
    *,
    models_list: dict[str, list[str]] | None = None,
    default_timeout: int = 100,
) -> RuntimeConfig:
    runtime_api_list = tuple(str(api_key) for api_key in api_list)
    models_list = models_list or build_api_key_models_map(config, api_list)
    routing_index = build_routing_index(config, api_list)
    api_keys = list(config.get("api_keys", []) or [])

    return RuntimeConfig(
        api_list=runtime_api_list,
        api_key_by_token={
            str(item.get("api")): item
            for item in api_keys
            if item.get("api")
        },
        api_key_index_by_token={
            str(api_key): index
            for index, api_key in enumerate(api_list)
        },
        api_key_model_rules_by_index=_api_key_model_rules(api_keys, len(runtime_api_list)),
        api_key_preferences_by_index=_api_key_preferences(api_keys, len(runtime_api_list)),
        api_key_roles_by_index=_api_key_roles(api_keys, runtime_api_list),
        api_key_weights_by_index=_api_key_weights(api_keys, len(runtime_api_list)),
        provider_by_name=dict(routing_index.provider_by_name),
        models_by_provider=dict(routing_index.models_by_provider),
        providers_by_model=dict(routing_index.providers_by_model),
        api_key_allowed_models=dict(models_list),
        api_key_model_response_cache=build_api_key_model_response_cache(api_list, models_list),
        endpoint_exclusions_by_provider=_endpoint_exclusions(config),
        weights_by_api_key_model=_weights_by_api_key(api_keys),
        provider_preferences=_provider_preferences(config),
        model_timeout_index=_preference_index(config, "model_timeout", default_timeout),
        keepalive_interval_index=_preference_index(config, "keepalive_interval", 99999),
        video_provider_configs=load_video_provider_configs(config),
        routing_index=routing_index,
    )


def _api_key_model_rules(api_keys: list[dict[str, Any]], count: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple((api_keys[index] or {}).get("model") or ())
        if index < len(api_keys)
        else ()
        for index in range(count)
    )


def _api_key_preferences(api_keys: list[dict[str, Any]], count: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict((api_keys[index] or {}).get("preferences") or {})
        if index < len(api_keys)
        else {}
        for index in range(count)
    )


def _api_key_roles(api_keys: list[dict[str, Any]], api_list: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        str((api_keys[index] or {}).get("role") or api_key[:8] or "None")
        if index < len(api_keys)
        else str(api_key[:8] or "None")
        for index, api_key in enumerate(api_list)
    )


def _api_key_weights(api_keys: list[dict[str, Any]], count: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict((api_keys[index] or {}).get("weights") or {})
        if index < len(api_keys)
        else {}
        for index in range(count)
    )


def _provider_preferences(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(provider.get("provider")): dict(provider.get("preferences") or {})
        for provider in config.get("providers", []) or []
        if provider.get("provider")
    }


def _weights_by_api_key(api_keys: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("api")): dict(item.get("weights") or {})
        for item in api_keys
        if item.get("api")
    }


def _endpoint_exclusions(config: dict[str, Any]) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for provider in config.get("providers", []) or []:
        provider_name = provider.get("provider")
        if not provider_name:
            continue
        excluded: set[str] = set()
        for value in _endpoint_values(provider.get("exclude_endpoints")):
            excluded.add(_normalize_endpoint_path(value))
        for value in _endpoint_values((provider.get("preferences") or {}).get("exclude_endpoints")):
            excluded.add(_normalize_endpoint_path(value))
        result[str(provider_name)] = frozenset(item for item in excluded if item)
    return result


def _endpoint_values(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_endpoint_path(value: Any) -> str:
    endpoint_path = str(value or "").strip()
    if not endpoint_path:
        return ""
    endpoint_path = endpoint_path.rstrip("/")
    if not endpoint_path.startswith("/"):
        endpoint_path = f"/{endpoint_path}"
    return endpoint_path or "/"


def _preference_index(
    config: dict[str, Any],
    preference_key: str,
    default_value: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = defaultdict(lambda: {"default": default_value})
    global_preferences = config.get("preferences") or {}
    global_value = global_preferences.get(preference_key)
    if isinstance(global_value, int):
        result["global"] = {"default": global_value}
    elif isinstance(global_value, dict):
        result["global"] = dict(global_value)
        result["global"].setdefault("default", default_value)
    else:
        result["global"] = {"default": default_value}

    for provider in config.get("providers", []) or []:
        provider_name = provider.get("provider")
        if not provider_name:
            continue
        provider_value = (provider.get("preferences") or {}).get(preference_key)
        if isinstance(provider_value, dict):
            result[str(provider_name)] = dict(provider_value)
        elif isinstance(provider_value, int):
            result[str(provider_name)] = {"default": provider_value}

    return dict(result)
