from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol


@dataclass(frozen=True)
class ProviderRequestContext:
    request: Any
    provider: dict[str, Any]
    engine: str
    original_model: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None


@dataclass(frozen=True)
class ProviderResponseContext:
    response: Any
    provider: dict[str, Any]
    engine: str
    model: str
    url: str


@dataclass(frozen=True)
class ProviderStreamContext:
    chunks: AsyncIterator[Any]
    provider: dict[str, Any]
    engine: str
    model: str
    url: str


@dataclass(frozen=True)
class UpstreamRequest:
    url: str
    headers: dict[str, Any]
    payload: Any
    method: str = "POST"
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedResponse:
    body: Any
    status_code: int = 200
    media_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        retryable: bool = False,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details


class ProviderAdapter(Protocol):
    name: str
    supported_engines: set[str]

    async def build_request(self, context: ProviderRequestContext) -> UpstreamRequest:
        ...

    async def parse_response(self, context: ProviderResponseContext) -> NormalizedResponse:
        ...

    async def parse_stream(self, context: ProviderStreamContext) -> AsyncIterator[bytes | str]:
        ...
