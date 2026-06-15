from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uni_api.config.runtime import RuntimeConfig


@dataclass(frozen=True, slots=True)
class AppRuntimeSnapshot:
    runtime_config: RuntimeConfig
    provider_registry: Any
    user_api_keys_rate_limit: dict[str, Any]
    global_rate_limit: list[tuple[int, int]]
    admin_api_key: list[str]
    provider_timeouts: dict[str, Any]
    keepalive_interval: dict[str, Any]
