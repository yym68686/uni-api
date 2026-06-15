from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextvars import ContextVar
from typing import Any


request_info: ContextVar[dict[str, Any]] = ContextVar("request_info", default={})


@dataclass(slots=True)
class RequestContext:
    request_id: str
    trace_id: str
    span_id: str | None = None
    parent_span_id: str | None = None
    trace_flags: str = "01"
    tracestate: str = ""
    x_request_id: str | None = None
    start_time: float = 0.0
    endpoint: str = ""
    client_ip: str = "unknown"
    process_time: float = 0.0
    first_response_time: float = -1.0
    provider: str | None = None
    model: str | None = None
    success: bool = False
    api_key: str | None = None
    is_flagged: bool = False
    text: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timing_spans: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        extras = values.pop("extras", {}) or {}
        values.update(extras)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RequestContext":
        field_names = set(cls.__dataclass_fields__) - {"extras"}
        base = {key: values[key] for key in field_names if key in values}
        extras = {key: value for key, value in values.items() if key not in field_names}
        return cls(**base, extras=extras)


def get_request_info() -> dict[str, Any]:
    info = request_info.get()
    return info if isinstance(info, dict) else {}


def set_request_info(info: dict[str, Any]):
    return request_info.set(info)


def reset_request_info(token) -> None:
    request_info.reset(token)
