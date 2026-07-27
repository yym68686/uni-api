from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from time import monotonic
from typing import Any, Awaitable, Callable

from uni_api.observability.exceptions import exception_diagnostics
from uni_api.upstream.transport_errors import (
    TransportErrorClassification,
    classify_httpx_transport_error,
)


_MAX_TRACE_EVENTS = 32
_MAX_CONNECTIONS_SCANNED = 256


def _safe_text(value: Any, *, limit: int = 256) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text[:limit] or None


def _enum_text(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name:
        return _safe_text(name, limit=128)
    return _safe_text(value, limit=128)


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _trace_stage(name: str, current: str) -> str:
    if "send_request_headers" in name:
        return "send_headers"
    if "send_request_body" in name:
        return "send_body"
    if "receive_response_headers" in name:
        return "wait_response_headers"
    if "receive_response_body" in name:
        return "read_body"
    return current


def _next_stage_after(name: str, current: str) -> str:
    if name.endswith("send_request_headers.complete"):
        return "send_body"
    if name.endswith("send_request_body.complete"):
        return "wait_response_headers"
    if name.endswith("receive_response_headers.complete"):
        return "read_body"
    return _trace_stage(name, current)


def _network_stream_alpn(network_stream: Any) -> str | None:
    get_extra_info = getattr(network_stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return None
    try:
        ssl_object = get_extra_info("ssl_object")
    except Exception:
        return None
    selected = getattr(ssl_object, "selected_alpn_protocol", None)
    if not callable(selected):
        return None
    try:
        return _safe_text(selected(), limit=32)
    except Exception:
        return None


def _client_transports(client: Any) -> tuple[Any, ...]:
    transports: list[Any] = []
    direct = getattr(client, "_transport", None)
    if direct is not None:
        transports.append(direct)
    mounts = getattr(client, "_mounts", None)
    if isinstance(mounts, dict):
        transports.extend(
            transport
            for transport in mounts.values()
            if transport is not None and transport not in transports
        )
    return tuple(transports)


def _pool_connections(client: Any) -> tuple[Any, ...]:
    found: list[Any] = []
    for transport in _client_transports(client):
        pool = getattr(transport, "_pool", None)
        connections = getattr(pool, "_connections", None)
        if not isinstance(connections, (list, tuple)):
            continue
        for connection in connections:
            if connection not in found:
                found.append(connection)
            if len(found) >= _MAX_CONNECTIONS_SCANNED:
                return tuple(found)
    return tuple(found)


def _inner_connection(connection: Any) -> Any:
    current = connection
    seen: set[int] = set()
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        nested = getattr(current, "_connection", None)
        if nested is None:
            break
        current = nested
    return current


def _origin_matches(connection: Any, origin: Any) -> bool:
    if origin is None:
        return True
    candidate = getattr(connection, "_origin", None)
    try:
        return candidate == origin
    except Exception:
        return False


def _matching_connection(
    client: Any,
    *,
    origin: Any,
    stream_id: int | None,
    network_stream: Any,
) -> Any:
    exact: list[Any] = []
    active: list[Any] = []
    for wrapper in _pool_connections(client):
        inner = _inner_connection(wrapper)
        if inner is None or not _origin_matches(inner, origin):
            continue
        candidate_stream = getattr(inner, "_network_stream", None)
        events = getattr(inner, "_events", None)
        if network_stream is not None and candidate_stream is network_stream:
            exact.append(inner)
            continue
        if (
            stream_id is not None
            and isinstance(events, dict)
            and stream_id in events
        ):
            exact.append(inner)
            continue
        state = _enum_text(getattr(inner, "_state", None))
        if state == "ACTIVE":
            active.append(inner)
    unique_exact = {id(item): item for item in exact}
    if len(unique_exact) == 1:
        return next(iter(unique_exact.values()))
    unique_active = {id(item): item for item in active}
    if stream_id is None and network_stream is None and len(unique_active) == 1:
        return next(iter(unique_active.values()))
    return None


def _connection_snapshot(
    client: Any,
    *,
    origin: Any,
    stream_id: int | None,
    network_stream: Any,
) -> dict[str, Any]:
    connection = _matching_connection(
        client,
        origin=origin,
        stream_id=stream_id,
        network_stream=network_stream,
    )
    if connection is None:
        return {"connection_snapshot_match": "not_unique_or_unavailable"}
    events = getattr(connection, "_events", None)
    h2_state = getattr(connection, "_h2_state", None)
    connection_terminated = getattr(connection, "_connection_terminated", None)
    read_exception = getattr(connection, "_read_exception", None)
    write_exception = getattr(connection, "_write_exception", None)
    resolved_stream = stream_id
    stream_state = None
    if h2_state is not None and resolved_stream is not None:
        streams = getattr(h2_state, "streams", None)
        stream = streams.get(resolved_stream) if isinstance(streams, dict) else None
        state_machine = getattr(stream, "state_machine", None)
        stream_state = _enum_text(getattr(state_machine, "state", None))
    state_machine = getattr(h2_state, "state_machine", None)
    network = getattr(connection, "_network_stream", None) or network_stream
    result: dict[str, Any] = {
        "connection_snapshot_match": "exact",
        "connection_protocol": type(connection).__name__,
        "connection_request_count": _int_value(
            getattr(connection, "_request_count", None)
        ),
        "connection_local_state": _enum_text(
            getattr(connection, "_state", None)
        ),
        "connection_error": bool(
            getattr(connection, "_connection_error", False)
        ),
        "http2_stream_id": resolved_stream,
        "http2_concurrent_streams": len(events)
        if isinstance(events, dict)
        else None,
        "http2_max_concurrent_streams": _int_value(
            getattr(connection, "_max_streams", None)
        ),
        "http2_local_connection_state": _enum_text(
            getattr(state_machine, "state", None)
        ),
        "http2_local_stream_state": stream_state,
        "alpn_protocol": _network_stream_alpn(network),
        "goaway_error_code": _int_value(
            getattr(connection_terminated, "error_code", None)
        ),
        "goaway_error_code_name": _enum_text(
            getattr(connection_terminated, "error_code", None)
        ),
        "goaway_last_stream_id": _int_value(
            getattr(connection_terminated, "last_stream_id", None)
        ),
        "transport_read_exception_type": type(read_exception).__name__
        if isinstance(read_exception, BaseException)
        else None,
        "transport_write_exception_type": type(write_exception).__name__
        if isinstance(write_exception, BaseException)
        else None,
    }
    return {key: value for key, value in result.items() if value is not None}


class UpstreamTransportDiagnostics:
    """Payload-free transport facts for one provider routing attempt."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry
        self._started = monotonic()
        self._stage = "send_headers"
        self._origin: Any = None
        self._stream_id: int | None = None
        self._network_stream: Any = None
        self._client: Any = None
        self._raw_exception_captured = False
        self.facts: dict[str, Any] = {
            "schema_version": 1,
            "failure_stage": self._stage,
            "httpcore_events": [],
        }
        entry["transport_diagnostics"] = self.facts

    def bind_client(self, client: Any) -> None:
        self._client = client

    async def httpcore_trace(self, name: str, info: dict[str, Any]) -> None:
        """Capture raw httpcore facts without ever changing request behavior."""

        try:
            info = info if isinstance(info, dict) else {}
            request = info.get("request")
            request_url = getattr(request, "url", None)
            if request_url is not None:
                self._origin = getattr(request_url, "origin", None)
            stream_id = info.get("stream_id")
            if isinstance(stream_id, int) and not isinstance(stream_id, bool):
                self._stream_id = stream_id
                self.facts["http2_stream_id"] = stream_id
            if name.startswith("http2."):
                self.facts["httpcore_protocol"] = "HTTP/2"
            elif name.startswith("http11."):
                self.facts["httpcore_protocol"] = "HTTP/1.1"
            stage = _trace_stage(name, self._stage)
            self._stage = stage
            self.facts["failure_stage"] = stage
            row: dict[str, Any] = {
                "name": _safe_text(name, limit=160),
                "elapsed_ms": max(0, int((monotonic() - self._started) * 1000)),
            }
            if self._stream_id is not None:
                row["stream_id"] = self._stream_id
            exc = info.get("exception")
            if isinstance(exc, BaseException):
                if isinstance(
                    exc,
                    (
                        asyncio.CancelledError,
                        GeneratorExit,
                        KeyboardInterrupt,
                        SystemExit,
                    ),
                ):
                    # httpcore reports GeneratorExit when Ember deliberately
                    # closes a fully-consumed streaming response. Preserve the
                    # trace fact without turning normal cleanup into a provider
                    # transport failure.
                    row["control_flow_exception_type"] = type(exc).__name__
                else:
                    diagnostics = exception_diagnostics(exc)
                    row.update(
                        {
                            "exception_type": diagnostics["exception_type"],
                            "exception_module": diagnostics["exception_module"],
                            "exception_repr": diagnostics["exception_repr"],
                            "protocol_error_reason": diagnostics[
                                "protocol_error_reason"
                            ],
                        }
                    )
                    self._capture_exception(
                        diagnostics,
                        prefix="httpcore_",
                        stage=stage,
                    )
                    self._raw_exception_captured = True
            if self._client is not None and (
                name.endswith("send_request_headers.started")
                or name.endswith(".failed")
                or name.endswith("receive_response_headers.started")
            ):
                self._capture_connection(self._client)
            events = self.facts.get("httpcore_events")
            if isinstance(events, list) and len(events) < _MAX_TRACE_EVENTS:
                events.append(row)
            else:
                self.facts["httpcore_events_truncated"] = True
            if name.endswith(".complete"):
                self._stage = _next_stage_after(name, self._stage)
                self.facts["failure_stage"] = self._stage
            self._sync_entry()
        except Exception:
            return

    def capture_response(self, response: Any, *, client: Any) -> None:
        try:
            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int) and not isinstance(status_code, bool):
                self.facts["upstream_http_status_code"] = status_code
            extensions = getattr(response, "extensions", None)
            extensions = extensions if isinstance(extensions, dict) else {}
            raw_version = extensions.get("http_version")
            if isinstance(raw_version, bytes):
                raw_version = raw_version.decode("ascii", errors="replace")
            actual_version = _safe_text(
                raw_version or getattr(response, "http_version", None),
                limit=32,
            )
            if actual_version:
                self.facts["http_version"] = actual_version
            stream_id = extensions.get("stream_id")
            if isinstance(stream_id, int) and not isinstance(stream_id, bool):
                self._stream_id = stream_id
                self.facts["http2_stream_id"] = stream_id
            self._network_stream = extensions.get("network_stream")
            alpn = _network_stream_alpn(self._network_stream)
            if alpn:
                self.facts["alpn_protocol"] = alpn
            request = getattr(response, "request", None)
            request_url = getattr(request, "url", None)
            if request_url is not None:
                response_origin = getattr(request_url, "origin", None)
                if response_origin is not None:
                    self._origin = response_origin
            self._capture_connection(client)
            self._stage = "read_body"
            self.facts["failure_stage"] = self._stage
        except Exception as exc:
            self.facts["response_metadata_error"] = type(exc).__name__

    def observe_exception(self, exc: BaseException, *, client: Any) -> None:
        if isinstance(
            exc,
            (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit),
        ):
            return
        try:
            diagnostics = exception_diagnostics(exc)
            prefix = "outer_" if self._raw_exception_captured else ""
            self._capture_exception(diagnostics, prefix=prefix, stage=self._stage)
            transport_failure = classify_httpx_transport_error(
                exc,
                failure_stage=self._stage,
            )
            if transport_failure is not None:
                self.record_transport_failure(transport_failure)
            self._capture_connection(client)
        except Exception:
            return

    def record_transport_failure(
        self,
        classification: TransportErrorClassification,
    ) -> None:
        self.facts.update(classification.observability_facts())
        self._sync_entry()

    def _capture_exception(
        self,
        diagnostics: dict[str, Any],
        *,
        prefix: str,
        stage: str,
    ) -> None:
        for key in (
            "exception_type",
            "exception_module",
            "exception_message",
            "exception_repr",
            "exception_chain_json",
            "exception_chain_depth",
            "exception_chain_truncated",
            "protocol_error_reason",
        ):
            self.facts[f"{prefix}{key}"] = diagnostics.get(key)
        self.facts["failure_stage"] = stage
        reason = diagnostics.get("protocol_error_reason")
        diagnostic_text = str(diagnostics.get("exception_chain_json") or "").lower()
        if reason == "LOCAL_PROTOCOL_UNCLASSIFIED":
            closed_state = any(
                marker in diagnostic_text
                for marker in (
                    "state=closed",
                    "state=done",
                    "connectionstate.closed",
                    "connection is closed",
                )
            )
            if closed_state and stage == "send_headers":
                reason = "SEND_HEADERS_ON_CLOSED"
            elif closed_state and stage == "send_body":
                reason = "SEND_BODY_ON_CLOSED"
            self.facts[f"{prefix}protocol_error_reason"] = reason
        if prefix == "httpcore_":
            if reason and reason != "NOT_PROTOCOL_ERROR":
                self.facts["protocol_error_reason"] = reason
        self._sync_entry()

    def _capture_connection(self, client: Any) -> None:
        snapshot = _connection_snapshot(
            client,
            origin=self._origin,
            stream_id=self._stream_id,
            network_stream=self._network_stream,
        )
        if (
            snapshot.get("connection_snapshot_match") != "exact"
            and self.facts.get("connection_snapshot_match") == "exact"
        ):
            return
        self.facts.update(snapshot)
        try:
            self.facts["connection_snapshot_json"] = json.dumps(
                snapshot,
                separators=(",", ":"),
                sort_keys=True,
            )[:4096]
        except Exception:
            pass
        self._sync_entry()

    def _sync_entry(self) -> None:
        for key, value in self.facts.items():
            if key == "httpcore_events":
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                self._entry[key] = value
        events = self.facts.get("httpcore_events")
        if isinstance(events, list):
            try:
                self._entry["httpcore_events_json"] = json.dumps(
                    events,
                    separators=(",", ":"),
                    sort_keys=True,
                )[:8192]
            except Exception:
                pass

    def finalize(self, outcome: str) -> None:
        self.facts["outcome"] = _safe_text(outcome, limit=80)
        self.facts["duration_ms"] = max(
            0,
            int((monotonic() - self._started) * 1000),
        )
        self._sync_entry()


_current_upstream_transport: contextvars.ContextVar[
    UpstreamTransportDiagnostics | None
] = contextvars.ContextVar("uni_api_upstream_transport", default=None)


def bind_upstream_transport_diagnostics(
    diagnostics: UpstreamTransportDiagnostics,
) -> contextvars.Token[UpstreamTransportDiagnostics | None]:
    return _current_upstream_transport.set(diagnostics)


def reset_upstream_transport_diagnostics(
    token: contextvars.Token[UpstreamTransportDiagnostics | None],
) -> None:
    _current_upstream_transport.reset(token)


def current_upstream_transport_diagnostics() -> (
    UpstreamTransportDiagnostics | None
):
    return _current_upstream_transport.get()


def compose_httpcore_trace(
    existing: Callable[[str, dict[str, Any]], Any] | None,
    diagnostics: UpstreamTransportDiagnostics | None,
) -> Callable[[str, dict[str, Any]], Awaitable[None]] | None:
    if diagnostics is None:
        return existing  # type: ignore[return-value]

    async def combined(name: str, info: dict[str, Any]) -> None:
        try:
            await diagnostics.httpcore_trace(name, info)
        except Exception:
            pass
        if existing is None:
            return
        try:
            result = existing(name, info)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # httpcore promotes trace callback failures into request failures.
            # Every observability callback must therefore be fail-open.
            return

    return combined


def inject_transport_trace(
    kwargs: dict[str, Any],
    diagnostics: UpstreamTransportDiagnostics | None,
) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    extensions = dict(request_kwargs.get("extensions") or {})
    combined = compose_httpcore_trace(extensions.get("trace"), diagnostics)
    if combined is not None:
        extensions["trace"] = combined
        request_kwargs["extensions"] = extensions
    return request_kwargs
