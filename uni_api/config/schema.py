from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    base_url: str = ""
    api: Any = None
    model: Any = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    exclude_endpoints: Any = None


class ApiKeyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    api: str
    model: list[Any] = Field(default_factory=lambda: ["all"])
    preferences: dict[str, Any] = Field(default_factory=dict)
    weights: dict[str, Any] = Field(default_factory=dict)


class UniApiConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    providers: list[ProviderConfig] = Field(default_factory=list)
    api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    video_providers: list[dict[str, Any]] = Field(default_factory=list)


def validate_config_data(config_data: Any) -> dict[str, Any]:
    UniApiConfig.model_validate(config_data)
    if not isinstance(config_data, dict):
        raise TypeError("Configuration must be a mapping")
    return config_data
