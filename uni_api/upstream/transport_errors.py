from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


HTTPX_TRANSPORT_ERRORS = (httpx.TransportError,)


@dataclass(frozen=True, slots=True)
class TransportErrorClassification:
    """Stable, policy-neutral facts for one HTTPX transport failure."""

    kind: str
    owner: str
    phase: str
    status_code: int
    message: str
    timeout_extension_key: str | None = None
    provider_penalty_eligible: bool = True

    @property
    def local_overload(self) -> bool:
        return self.owner == "ember_local_overload"

    def observability_facts(self) -> dict[str, Any]:
        return {
            "transport_error_kind": self.kind,
            "transport_error_owner": self.owner,
            "transport_error_phase": self.phase,
            "transport_error_status_code": self.status_code,
            "provider_penalty_eligible": self.provider_penalty_eligible,
            "local_overload": self.local_overload,
        }


def _normalized_stage(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _read_timeout_phase(
    failure_stage: Any,
    first_byte_observed: bool | None,
) -> str:
    if first_byte_observed is False:
        return "before_first_byte"
    if first_byte_observed is True:
        return "after_first_byte"

    stage = _normalized_stage(failure_stage)
    if stage in {
        "send_headers",
        "send_body",
        "upstream_headers",
        "wait_response_headers",
    }:
        return "before_response_headers"
    if stage in {"precommit", "preflight"}:
        return "before_first_byte"
    if stage in {"read_body", "response_body"}:
        return "response_body"
    if stage in {"postcommit", "post_commit"}:
        return "after_first_byte"
    return "unknown"


def classify_httpx_transport_error(
    exc: BaseException,
    *,
    failure_stage: Any = None,
    first_byte_observed: bool | None = None,
) -> TransportErrorClassification | None:
    """Classify every HTTPX TransportError without changing retry policy."""

    stage = _normalized_stage(failure_stage)

    if isinstance(exc, httpx.ConnectTimeout):
        return TransportErrorClassification(
            kind="connect_timeout",
            owner="upstream_transport",
            phase="connect",
            status_code=504,
            message="Connection timed out",
            timeout_extension_key="connect",
        )
    if isinstance(exc, httpx.ConnectError):
        return TransportErrorClassification(
            kind="connect_error",
            owner="upstream_transport",
            phase="connect",
            status_code=503,
            message="Unable to connect to service",
        )
    if isinstance(exc, httpx.PoolTimeout):
        return TransportErrorClassification(
            kind="pool_timeout",
            owner="ember_local_overload",
            phase="pool_acquire",
            status_code=503,
            message="Local upstream connection pool timed out",
            timeout_extension_key="pool",
            provider_penalty_eligible=False,
        )
    if isinstance(exc, httpx.ReadTimeout):
        return TransportErrorClassification(
            kind="read_timeout",
            owner="upstream_transport",
            phase=_read_timeout_phase(failure_stage, first_byte_observed),
            status_code=504,
            message="Request timed out",
            timeout_extension_key="read",
        )
    if isinstance(exc, httpx.WriteTimeout):
        return TransportErrorClassification(
            kind="write_timeout",
            owner="upstream_transport",
            phase=stage or "send_body",
            status_code=504,
            message="Request write timed out",
            timeout_extension_key="write",
        )
    if isinstance(exc, httpx.ReadError):
        return TransportErrorClassification(
            kind="read_error",
            owner="upstream_transport",
            phase=stage or "read_body",
            status_code=502,
            message="Network read error",
        )
    if isinstance(exc, httpx.WriteError):
        return TransportErrorClassification(
            kind="write_error",
            owner="upstream_transport",
            phase=stage or "send_body",
            status_code=502,
            message="Network write error",
        )
    if isinstance(exc, httpx.CloseError):
        return TransportErrorClassification(
            kind="close_error",
            owner="upstream_transport",
            phase=stage or "close",
            status_code=502,
            message="Network close error",
        )
    if isinstance(exc, httpx.ProxyError):
        return TransportErrorClassification(
            kind="proxy_error",
            owner="upstream_transport",
            phase=stage or "proxy_connect",
            status_code=502,
            message="Unable to connect to upstream proxy",
        )
    if isinstance(exc, httpx.UnsupportedProtocol):
        return TransportErrorClassification(
            kind="unsupported_protocol",
            owner="ember_local_configuration",
            phase="configuration",
            status_code=500,
            message="Unsupported upstream protocol",
        )
    if isinstance(exc, httpx.LocalProtocolError):
        return TransportErrorClassification(
            kind="local_protocol_error",
            owner="upstream_transport",
            phase=stage or "protocol",
            status_code=502,
            message="Local protocol error",
        )
    if isinstance(exc, httpx.RemoteProtocolError):
        return TransportErrorClassification(
            kind="remote_protocol_error",
            owner="upstream_transport",
            phase=stage or "protocol",
            status_code=502,
            message="Remote protocol error",
        )
    if isinstance(exc, httpx.TransportError):
        return TransportErrorClassification(
            kind="transport_error",
            owner="upstream_transport",
            phase=stage or "unknown",
            status_code=502,
            message="Upstream transport error",
        )
    return None
