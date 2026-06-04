from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class VideoAdapterError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class VideoRouteConfig:
    method: str
    path: str
    query: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoModelConfig:
    request_model: str
    upstream_model: str
    protocol: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoProviderConfig:
    name: str
    adapter: str
    base_url: str = ""
    auth: dict[str, Any] = field(default_factory=dict)
    routes: dict[str, VideoRouteConfig] = field(default_factory=dict)
    models: dict[str, VideoModelConfig] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpstreamVideoRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    payload: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class NormalizedVideoResponse:
    raw: bytes
    task_id: Optional[str] = None
    media_type: Optional[str] = "application/json"
