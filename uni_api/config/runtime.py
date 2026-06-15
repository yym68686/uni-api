from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uni_api.routing.core import RoutingIndex


@dataclass(frozen=True)
class RuntimeConfig:
    api_list: tuple[str, ...]
    api_key_by_token: dict[str, dict[str, Any]]
    api_key_index_by_token: dict[str, int]
    api_key_model_rules_by_index: tuple[tuple[Any, ...], ...]
    api_key_preferences_by_index: tuple[dict[str, Any], ...]
    api_key_roles_by_index: tuple[str, ...]
    api_key_weights_by_index: tuple[dict[str, Any], ...]
    provider_by_name: dict[str, dict[str, Any]]
    models_by_provider: dict[str, tuple[str, ...]]
    providers_by_model: dict[str, tuple[dict[str, Any], ...]]
    api_key_allowed_models: dict[str, list[str]]
    api_key_model_response_cache: dict[str, list[dict[str, Any]]]
    endpoint_exclusions_by_provider: dict[str, frozenset[str]]
    weights_by_api_key_model: dict[str, dict[str, Any]]
    provider_preferences: dict[str, dict[str, Any]]
    model_timeout_index: dict[str, dict[str, Any]]
    keepalive_interval_index: dict[str, dict[str, Any]]
    video_provider_configs: dict[str, Any]
    routing_index: RoutingIndex
