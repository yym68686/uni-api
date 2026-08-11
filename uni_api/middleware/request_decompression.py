from __future__ import annotations

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sys import maxsize
from time import monotonic
from typing import Any, Awaitable, Callable, Iterable

import zstandard as zstd
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.log_config import logger
from uni_api.admission import get_request_admission_lease
from uni_api.admission.cpu import run_cpu_phase
from uni_api.admission.json_memory import (
    DEFAULT_JSON_MAX_ESTIMATED_BYTES,
    IncrementalJSONMemoryEstimator,
    JSONMemoryComplexityError,
    JSONMemoryComplexityObservation,
    JSONMemorySnapshot,
)
from uni_api.admission.resources import startup_cpu_worker_count
from uni_api.observability.threadpool_tasks import register_dedicated_threadpool
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.http_content import is_json_media_type


DEFAULT_MAX_ZSTD_REQUEST_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_IDENTITY_REQUEST_BODY_BYTES = DEFAULT_MAX_ZSTD_REQUEST_BODY_BYTES

IDENTITY_REQUEST_MAX_BODY_BYTES_ENV = "REQUEST_MAX_BODY_BYTES"
ZSTD_REQUEST_MAX_BODY_BYTES_ENV = "ZSTD_REQUEST_MAX_BODY_BYTES"
ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES_ENV = (
    "ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES"
)
ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES_ENV = (
    "ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES"
)
REQUEST_BODY_IDLE_TIMEOUT_SECONDS_ENV = "REQUEST_BODY_IDLE_TIMEOUT_SECONDS"
REQUEST_BODY_TOTAL_TIMEOUT_SECONDS_ENV = "REQUEST_BODY_TOTAL_TIMEOUT_SECONDS"
DEFAULT_REQUEST_BODY_IDLE_TIMEOUT_SECONDS = 15.0
DEFAULT_REQUEST_BODY_TOTAL_TIMEOUT_SECONDS = 120.0
DEFAULT_NON_JSON_MEMORY_RESERVATION_MULTIPLIER = 4
JSON_BODY_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/responses/compact",
        "/v1/alpha/search",
        "/v1/messages",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/video/tasks",
        "/v1/asset-groups",
        "/v1/assets",
        "/v1/embeddings",
        "/v1/audio/speech",
        "/v1/moderations",
        "/v1/api_config/update",
    }
)
_DEFAULT_REQUEST_BODY_CPU_WORKERS = startup_cpu_worker_count()
try:
    REQUEST_BODY_CPU_WORKERS = max(
        1,
        int(
            os.getenv(
                "REQUEST_BODY_CPU_WORKERS",
                str(_DEFAULT_REQUEST_BODY_CPU_WORKERS),
            )
            or str(_DEFAULT_REQUEST_BODY_CPU_WORKERS)
        ),
    )
except (TypeError, ValueError):
    REQUEST_BODY_CPU_WORKERS = _DEFAULT_REQUEST_BODY_CPU_WORKERS
_REQUEST_BODY_CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=REQUEST_BODY_CPU_WORKERS,
    thread_name_prefix="uni-api-body",
)
register_dedicated_threadpool(
    "request_body_decode",
    _REQUEST_BODY_CPU_EXECUTOR,
)

BodyBytesReservationCallback = Callable[[int], Awaitable[None]]
BodyBytesReleaseCallback = Callable[[int], Awaitable[Any]]
ResourceOnlyBodyAdmissionPredicate = Callable[[Scope], bool]
BODY_BYTES_RESERVATION_SCOPE_KEY = "uni_api_reserve_body_bytes"
BODY_BYTES_RELEASE_SCOPE_KEY = "uni_api_release_body_bytes"
BODY_REJECTION_RECORDER_SCOPE_KEY = "uni_api_record_body_rejection"
BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY = "uni_api_observe_body_early_response"
BODY_RELEASE_AT_RESPONSE_START_SCOPE_KEY = (
    "uni_api_release_body_at_response_start"
)
BODY_RESPONSE_STARTED_SCOPE_KEY = "uni_api_body_response_started"
BODY_COMPLEXITY_DIAGNOSTICS_SCOPE_KEY = "uni_api_request_body_complexity"
BODY_OBSERVATION_SCOPE_KEY = "uni_api_request_body_observation"
REQUEST_BODY_COMPLEXITY_INFO_KEY = "request_body_complexity"


def initialize_request_body_observation(scope: Scope) -> dict[str, int]:
    state = scope.setdefault("state", {})
    if not isinstance(state, dict):
        return {}
    existing = state.get(BODY_OBSERVATION_SCOPE_KEY)
    if isinstance(existing, dict):
        return existing
    observation: dict[str, int] = {
        "wire_bytes": 0,
        "decoded_bytes": 0,
        "decoder_workspace_bytes": 0,
    }
    try:
        declared = _content_length(scope.get("headers") or [])
    except InvalidContentLength:
        declared = None
    if declared is not None:
        observation["declared_content_length_bytes"] = declared
    state[BODY_OBSERVATION_SCOPE_KEY] = observation
    return observation


def observe_request_wire_bytes(scope: Scope, additional_bytes: int) -> None:
    observation = initialize_request_body_observation(scope)
    if observation:
        observation["wire_bytes"] = max(
            0,
            int(observation.get("wire_bytes", 0)) + max(0, int(additional_bytes)),
        )


def observe_request_decoder_workspace_bytes(
    scope: Scope,
    additional_bytes: int,
) -> None:
    observation = initialize_request_body_observation(scope)
    if observation:
        observation["decoder_workspace_bytes"] = max(
            0,
            int(observation.get("decoder_workspace_bytes", 0))
            + max(0, int(additional_bytes)),
        )


def _observe_request_decoded_bytes(
    scope: Scope,
    additional_bytes: int,
    *,
    json_snapshot: JSONMemorySnapshot | None = None,
    complexity_observation: JSONMemoryComplexityObservation | None = None,
) -> None:
    observation = initialize_request_body_observation(scope)
    if not observation:
        return
    observation["decoded_bytes"] = max(
        0,
        int(observation.get("decoded_bytes", 0)) + max(0, int(additional_bytes)),
    )
    if complexity_observation is not None:
        observation.update(
            {
                "json_raw_bytes": complexity_observation.raw_bytes,
                "json_structural_item_count": (
                    complexity_observation.structural_item_count
                ),
                "json_depth": complexity_observation.depth,
                "json_peak_depth": complexity_observation.peak_depth,
                "json_scalar_bytes": complexity_observation.scalar_bytes,
                "json_estimated_bytes": complexity_observation.estimated_bytes,
                "json_raw_memory_multiplier": (
                    complexity_observation.raw_memory_multiplier
                ),
                "json_structural_item_memory_bytes": (
                    complexity_observation.structural_item_memory_bytes
                ),
            }
        )
    elif json_snapshot is not None:
        observation.update(
            {
                "json_raw_bytes": json_snapshot.raw_bytes,
                "json_structural_item_count": json_snapshot.tokens,
                "json_depth": json_snapshot.depth,
                "json_peak_depth": json_snapshot.peak_depth,
                "json_scalar_bytes": json_snapshot.scalar_bytes,
                "json_estimated_bytes": json_snapshot.estimated_bytes,
                "json_raw_memory_multiplier": json_snapshot.raw_memory_multiplier,
                "json_structural_item_memory_bytes": (
                    json_snapshot.structural_item_memory_bytes
                ),
            }
        )


def request_body_observation_from_scope(scope: Scope) -> dict[str, int]:
    state = scope.get("state")
    if not isinstance(state, dict):
        return {}
    raw = state.get(BODY_OBSERVATION_SCOPE_KEY)
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): int(value)
        for key, value in raw.items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }


def mark_request_body_releasable_at_response_start(scope: Scope) -> None:
    state = scope.setdefault("state", {})
    if isinstance(state, dict):
        state[BODY_RELEASE_AT_RESPONSE_START_SCOPE_KEY] = True


def _request_body_releasable_at_response_start(scope: Scope) -> bool:
    state = scope.get("state")
    return bool(
        isinstance(state, dict)
        and state.get(BODY_RELEASE_AT_RESPONSE_START_SCOPE_KEY)
    )


def request_body_response_started(scope: Scope) -> bool:
    state = scope.get("state")
    return bool(
        isinstance(state, dict)
        and state.get(BODY_RESPONSE_STARTED_SCOPE_KEY)
    )


def _mark_request_body_response_started(scope: Scope) -> None:
    state = scope.setdefault("state", {})
    if isinstance(state, dict):
        state[BODY_RESPONSE_STARTED_SCOPE_KEY] = True


class RequestBodyTooComplex(JSONMemoryComplexityError):
    """A request body exceeds the structural JSON admission envelope."""

    def __init__(
        self,
        message: str,
        *,
        observation: JSONMemoryComplexityObservation | None = None,
        json_memory_reserved_target_bytes: int = 0,
    ) -> None:
        super().__init__(message, observation=observation)
        self.json_memory_reserved_target_bytes = max(
            0,
            int(json_memory_reserved_target_bytes),
        )

    @classmethod
    def from_json_memory_error(
        cls,
        exc: JSONMemoryComplexityError,
        *,
        json_memory_reserved_target_bytes: int,
    ) -> RequestBodyTooComplex:
        return cls(
            str(exc),
            observation=exc.observation,
            json_memory_reserved_target_bytes=(
                json_memory_reserved_target_bytes
            ),
        )


def capture_request_body_complexity_diagnostics(
    scope: Scope,
    exc: RequestBodyTooComplex,
) -> dict[str, int | str]:
    """Snapshot body-free rejection facts before the request lease is freed."""

    observation = exc.observation
    if observation is None:
        return {}
    diagnostics: dict[str, int | str] = {
        "schema_version": observation.schema_version,
        "reason": str(observation.reason),
        "trigger_phase": str(observation.trigger_phase),
        "raw_bytes": observation.raw_bytes,
        "structural_item_count": observation.structural_item_count,
        "depth": observation.depth,
        "peak_depth": observation.peak_depth,
        "scalar_bytes": observation.scalar_bytes,
        "estimated_bytes": observation.estimated_bytes,
        "configured_limit": observation.configured_limit,
        "max_depth": observation.max_depth,
        "max_scalar_bytes": observation.max_scalar_bytes,
        "max_estimated_bytes": observation.max_estimated_bytes,
        "raw_memory_multiplier": observation.raw_memory_multiplier,
        "structural_item_memory_bytes": (
            observation.structural_item_memory_bytes
        ),
        "json_memory_reserved_target_bytes_at_rejection": (
            exc.json_memory_reserved_target_bytes
        ),
    }
    lease = get_request_admission_lease()
    if lease is not None:
        diagnostics["reserved_weighted_bytes_at_rejection"] = max(
            0,
            int(lease.reserved_body_bytes),
        )
    state = scope.setdefault("state", {})
    if isinstance(state, dict):
        state[BODY_COMPLEXITY_DIAGNOSTICS_SCOPE_KEY] = dict(diagnostics)
    return diagnostics


def request_body_complexity_diagnostics_from_scope(
    scope: Scope,
) -> dict[str, int | str]:
    state = scope.get("state")
    if not isinstance(state, dict):
        return {}
    diagnostics = state.get(BODY_COMPLEXITY_DIAGNOSTICS_SCOPE_KEY)
    if not isinstance(diagnostics, dict):
        return {}
    return {
        str(key): value
        for key, value in diagnostics.items()
        if isinstance(value, (int, str)) and not isinstance(value, bool)
    }


class _RequestBodyMemoryBudget:
    """Translate observed body bytes into a conservative live-memory charge."""

    def __init__(
        self,
        *,
        scope: Scope,
        json_body: bool,
        json_max_estimated_bytes: int,
    ) -> None:
        self._scope = scope
        self._json_estimator = (
            IncrementalJSONMemoryEstimator(
                max_estimated_bytes=json_max_estimated_bytes,
            )
            if json_body
            else None
        )
        self._observed_bytes = 0
        self._reserved_target = 0

    async def reserve_chunk(
        self,
        chunk: bytes | bytearray | memoryview,
        callback: BodyBytesReservationCallback | None,
    ) -> None:
        if self._json_estimator is not None:
            try:
                target = await _run_body_cpu(self._json_estimator.feed, chunk)
            except JSONMemoryComplexityError as exc:
                _observe_request_decoded_bytes(
                    self._scope,
                    len(chunk),
                    complexity_observation=exc.observation,
                )
                if callback is not None:
                    # No additional weighted reservation is committed, but the
                    # holder/release event must retain the estimator state that
                    # caused this terminal 413.
                    await _reserve_body_bytes(callback, 0)
                # Give request-body control flow a distinct type.  The generic
                # JSON complexity error is also used for upstream/SSE payloads
                # and those must never be mislabeled as a client-body 413.
                raise RequestBodyTooComplex.from_json_memory_error(
                    exc,
                    json_memory_reserved_target_bytes=self._reserved_target,
                ) from exc
            json_snapshot = self._json_estimator.snapshot()
        else:
            self._observed_bytes += len(chunk)
            target = (
                self._observed_bytes
                * DEFAULT_NON_JSON_MEMORY_RESERVATION_MULTIPLIER
            )
            json_snapshot = None
        _observe_request_decoded_bytes(
            self._scope,
            len(chunk),
            json_snapshot=json_snapshot,
        )
        additional = target - self._reserved_target
        if additional < 0:
            raise RuntimeError("request body memory reservation regressed")
        if callback is not None and additional:
            await _reserve_body_bytes(callback, additional)
        self._reserved_target = target


class RequestBodyDecompressionMiddleware:
    """Bound and decode request bodies before FastAPI reads them.

    ``max_body_bytes`` is retained as the backwards-compatible constructor
    setting. When supplied, it is the fallback for identity, compressed zstd,
    and decompressed zstd bodies. The more specific settings can be used to
    give compressed input and decoded output separate budgets.

    ``ZSTD_REQUEST_MAX_BODY_BYTES`` remains the fallback for both zstd limits.
    New deployments can override either side independently with
    ``ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES`` and
    ``ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES``. Identity requests use
    ``REQUEST_MAX_BODY_BYTES``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int | None = None,
        max_identity_body_bytes: int | None = None,
        max_zstd_compressed_body_bytes: int | None = None,
        max_zstd_decompressed_body_bytes: int | None = None,
        json_max_estimated_bytes: int = DEFAULT_JSON_MAX_ESTIMATED_BYTES,
        body_idle_timeout_seconds: float | None = None,
        body_total_timeout_seconds: float | None = None,
        resource_only_body_admission: (
            ResourceOnlyBodyAdmissionPredicate | None
        ) = None,
    ) -> None:
        self.app = app

        legacy_zstd_limit = _bounded_limit(
            max_body_bytes
            if max_body_bytes is not None
            else _env_int(
                ZSTD_REQUEST_MAX_BODY_BYTES_ENV,
                DEFAULT_MAX_ZSTD_REQUEST_BODY_BYTES,
            )
        )
        self.max_body_bytes = legacy_zstd_limit

        if max_identity_body_bytes is None:
            max_identity_body_bytes = (
                legacy_zstd_limit
                if max_body_bytes is not None
                else _env_int(
                    IDENTITY_REQUEST_MAX_BODY_BYTES_ENV,
                    DEFAULT_MAX_IDENTITY_REQUEST_BODY_BYTES,
                )
            )
        if max_zstd_compressed_body_bytes is None:
            max_zstd_compressed_body_bytes = (
                legacy_zstd_limit
                if max_body_bytes is not None
                else _env_int(
                    ZSTD_REQUEST_MAX_COMPRESSED_BODY_BYTES_ENV,
                    legacy_zstd_limit,
                )
            )
        if max_zstd_decompressed_body_bytes is None:
            max_zstd_decompressed_body_bytes = (
                legacy_zstd_limit
                if max_body_bytes is not None
                else _env_int(
                    ZSTD_REQUEST_MAX_DECOMPRESSED_BODY_BYTES_ENV,
                    legacy_zstd_limit,
                )
            )

        self.max_identity_body_bytes = _bounded_limit(max_identity_body_bytes)
        self.max_zstd_compressed_body_bytes = _bounded_limit(
            max_zstd_compressed_body_bytes
        )
        self.max_zstd_decompressed_body_bytes = _bounded_limit(
            max_zstd_decompressed_body_bytes
        )
        self.json_max_estimated_bytes = _bounded_limit(
            json_max_estimated_bytes
        )
        self.body_idle_timeout_seconds = _positive_float(
            body_idle_timeout_seconds
            if body_idle_timeout_seconds is not None
            else _env_float(
                REQUEST_BODY_IDLE_TIMEOUT_SECONDS_ENV,
                DEFAULT_REQUEST_BODY_IDLE_TIMEOUT_SECONDS,
            )
        )
        self.body_total_timeout_seconds = _positive_float(
            body_total_timeout_seconds
            if body_total_timeout_seconds is not None
            else _env_float(
                REQUEST_BODY_TOTAL_TIMEOUT_SECONDS_ENV,
                DEFAULT_REQUEST_BODY_TOTAL_TIMEOUT_SECONDS,
            )
        )
        self.resource_only_body_admission = (
            resource_only_body_admission or (lambda scope: False)
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers") or []
        resource_only_body_admission = bool(
            self.resource_only_body_admission(scope)
        )
        disconnect_event = _disconnect_event(scope)
        content_type_count = sum(
            name.lower() == b"content-type" for name, _value in headers
        )
        if content_type_count > 1:
            await _json_error(
                scope,
                receive,
                send,
                400,
                "multiple content-type headers are not allowed",
                reason="invalid_content_type",
            )
            return
        reserve_body_bytes = _body_bytes_reservation_callback(scope)
        release_body_bytes = _body_bytes_release_callback(scope)
        body_memory_budget = _request_body_memory_budget(
            scope,
            headers,
            json_max_estimated_bytes=(
                maxsize
                if resource_only_body_admission
                else self.json_max_estimated_bytes
            ),
        )
        encodings = _content_encodings(headers)
        if not encodings or _is_identity(encodings):
            await self._handle_identity(
                scope,
                receive,
                send,
                disconnect_event=disconnect_event,
                resource_only_body_admission=(
                    resource_only_body_admission
                ),
            )
            return
        if encodings != ["zstd"]:
            await _json_error(
                scope,
                receive,
                send,
                415,
                f"unsupported content encoding: {', '.join(encodings)}",
                reason="unsupported_content_encoding",
            )
            return

        try:
            if not resource_only_body_admission:
                _ensure_content_length_within(
                    headers,
                    self.max_zstd_compressed_body_bytes,
                )
            compressed = await _read_body(
                receive,
                (
                    None
                    if resource_only_body_admission
                    else self.max_zstd_compressed_body_bytes
                ),
                reserve_body_bytes,
                idle_timeout_seconds=self.body_idle_timeout_seconds,
                total_timeout_seconds=(
                    None
                    if resource_only_body_admission
                    else self.body_total_timeout_seconds
                ),
            )
            body = await _decompress_zstd(
                compressed,
                (
                    None
                    if resource_only_body_admission
                    else self.max_zstd_decompressed_body_bytes
                ),
                reserve_body_bytes,
                release_body_bytes,
                memory_budget=body_memory_budget,
            )
            compressed_bytes = len(compressed)
            compressed.clear()
            del compressed
            if release_body_bytes is not None and compressed_bytes:
                await _release_body_bytes(
                    release_body_bytes,
                    compressed_bytes,
                )
        except _RequestBodyHardLimitExceeded:
            _record_body_rejection(scope, "body_too_large")
            await _json_error(
                scope,
                receive,
                send,
                413,
                "request body too large",
                reason="body_too_large",
            )
            return
        except RequestBodyTooComplex as exc:
            capture_request_body_complexity_diagnostics(scope, exc)
            _record_body_rejection(scope, "body_too_complex")
            await _json_error(
                scope,
                receive,
                send,
                413,
                "request body too complex",
                reason="body_too_complex",
            )
            return
        except InvalidContentLength:
            await _json_error(
                scope,
                receive,
                send,
                400,
                "invalid content-length",
                reason="invalid_content_length",
            )
            return
        except RequestBodyDisconnected:
            disconnect_event.set()
            await _observe_body_early_response(
                scope,
                499,
                "request_body_disconnected",
            )
            return
        except RequestBodyReadTimeout:
            await _json_error(
                scope,
                receive,
                send,
                408,
                "request body upload timed out",
                reason="request_body_timeout",
            )
            return
        except zstd.ZstdError:
            await _json_error(
                scope,
                receive,
                send,
                400,
                "invalid zstd body",
                reason="invalid_zstd_body",
            )
            return

        decompressed_scope = dict(scope)
        decompressed_scope["headers"] = [
            (name, value)
            for name, value in (scope.get("headers") or [])
            if name.lower() not in {b"content-encoding", b"content-length"}
        ]

        body_sent = False
        disconnect_monitor = asyncio.create_task(
            _monitor_disconnect(receive, disconnect_event),
            name="uni-api-request-disconnect-monitor",
        )

        async def decompressed_receive() -> Message:
            nonlocal body
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                outgoing_body = body
                body = b""
                return {
                    "type": "http.request",
                    "body": outgoing_body,
                    "more_body": False,
                }
            await disconnect_event.wait()
            return {"type": "http.disconnect"}

        async def tracking_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                _mark_request_body_response_started(scope)
                if _request_body_releasable_at_response_start(scope):
                    await _release_remaining_body_bytes(scope)
            await send(message)

        try:
            await self.app(
                decompressed_scope,
                decompressed_receive,
                tracking_send,
            )
        finally:
            await _cancel_disconnect_monitor(disconnect_monitor)

    async def _handle_identity(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        disconnect_event: asyncio.Event,
        resource_only_body_admission: bool,
    ) -> None:
        try:
            if not resource_only_body_admission:
                _ensure_content_length_within(
                    scope.get("headers") or [],
                    self.max_identity_body_bytes,
                )
        except _RequestBodyHardLimitExceeded:
            _record_body_rejection(scope, "body_too_large")
            await _json_error(
                scope,
                receive,
                send,
                413,
                "request body too large",
                reason="body_too_large",
            )
            return
        except InvalidContentLength:
            await _json_error(
                scope,
                receive,
                send,
                400,
                "invalid content-length",
                reason="invalid_content_length",
            )
            return

        total = 0
        body_deadline = (
            None
            if resource_only_body_admission
            else monotonic() + self.body_total_timeout_seconds
        )
        body_complete = False
        empty_request_pending = False
        disconnect_monitor: asyncio.Task[None] | None = None
        response_started = False
        reserve_body_bytes = _body_bytes_reservation_callback(scope)
        body_memory_budget = _request_body_memory_budget(
            scope,
            scope.get("headers") or [],
            json_max_estimated_bytes=(
                maxsize
                if resource_only_body_admission
                else self.json_max_estimated_bytes
            ),
        )

        async def limited_receive() -> Message:
            nonlocal total, body_complete, empty_request_pending, disconnect_monitor
            if empty_request_pending:
                empty_request_pending = False
                return {
                    "type": "http.request",
                    "body": b"",
                    "more_body": False,
                }
            if body_complete:
                await disconnect_event.wait()
                return {"type": "http.disconnect"}
            message = await _receive_body_message(
                receive,
                idle_timeout_seconds=self.body_idle_timeout_seconds,
                total_deadline=body_deadline,
            )
            if message["type"] == "http.disconnect":
                disconnect_event.set()
                raise RequestBodyDisconnected()
            if message["type"] != "http.request":
                return message

            chunk = message.get("body", b"")
            if chunk:
                total += len(chunk)
                if (
                    not resource_only_body_admission
                    and total > self.max_identity_body_bytes
                ):
                    _observe_request_decoded_bytes(scope, len(chunk))
                    if reserve_body_bytes is not None:
                        await _reserve_body_bytes(reserve_body_bytes, 0)
                    raise _RequestBodyHardLimitExceeded()
                await body_memory_budget.reserve_chunk(
                    chunk,
                    reserve_body_bytes,
                )
            if not message.get("more_body", False):
                body_complete = True
                if disconnect_monitor is None:
                    disconnect_monitor = asyncio.create_task(
                        _monitor_disconnect(receive, disconnect_event),
                        name="uni-api-request-disconnect-monitor",
                    )
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                _mark_request_body_response_started(scope)
                if (
                    body_complete
                    and _request_body_releasable_at_response_start(scope)
                ):
                    await _release_remaining_body_bytes(scope)
            await send(message)

        try:
            if _request_has_no_body(scope):
                body_complete = True
                # ASGI applications are still entitled to exactly one empty
                # http.request message.  The raw monitor may consume the
                # server's equivalent message while watching for disconnect,
                # so replay a synthetic equivalent to the downstream app.
                empty_request_pending = True
                disconnect_monitor = asyncio.create_task(
                    _monitor_disconnect(receive, disconnect_event),
                    name="uni-api-request-disconnect-monitor",
                )
            await self.app(scope, limited_receive, tracking_send)
        except _RequestBodyHardLimitExceeded:
            if response_started:
                # It is illegal to send a second response start. Terminating
                # the response is safer than corrupting an in-flight protocol.
                raise
            _record_body_rejection(scope, "body_too_large")
            await _json_error(
                scope,
                receive,
                send,
                413,
                "request body too large",
                reason="body_too_large",
            )
        except RequestBodyTooComplex as exc:
            if response_started:
                raise
            capture_request_body_complexity_diagnostics(scope, exc)
            _record_body_rejection(scope, "body_too_complex")
            await _json_error(
                scope,
                receive,
                send,
                413,
                "request body too complex",
                reason="body_too_complex",
            )
        except RequestBodyDisconnected:
            await _observe_body_early_response(
                scope,
                499,
                "request_body_disconnected",
            )
            return
        except RequestBodyReadTimeout:
            if response_started:
                # A second response.start would violate ASGI after commit.
                raise
            await _json_error(
                scope,
                receive,
                send,
                408,
                "request body upload timed out",
                reason="request_body_timeout",
            )
            return
        finally:
            if disconnect_monitor is not None:
                await _cancel_disconnect_monitor(disconnect_monitor)


class RequestBodyTooLarge(Exception):
    pass


class _RequestBodyHardLimitExceeded(RequestBodyTooLarge):
    pass


class RequestBodyDisconnected(Exception):
    pass


class RequestBodyReadTimeout(Exception):
    pass


class InvalidContentLength(Exception):
    pass


async def _json_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    status_code: int,
    detail: str,
    *,
    reason: str,
) -> None:
    await _observe_body_early_response(scope, status_code, reason)
    headers: dict[str, str] = {}
    if status_code == 413:
        headers["x-uni-api-admission-reason"] = str(reason)
    if str(scope.get("http_version") or "") in {"1.0", "1.1"}:
        # Returning before the ASGI app has consumed the complete request body
        # makes HTTP/1.x keep-alive reuse unsafe: unread bytes can be mistaken
        # for the next request.  Closing only the rejected connection is
        # bounded and protocol-honest.  HTTP/2 forbids this hop-by-hop header.
        headers["connection"] = "close"
    response = JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers=headers or None,
    )
    await response(scope, receive, send)


async def _observe_body_early_response(
    scope: Scope,
    status_code: int,
    reason: str,
) -> None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return
    callback = state.get(BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY)
    if not callable(callback):
        return
    result = callback(status_code, reason)
    if hasattr(result, "__await__"):
        await result


async def _read_body(
    receive: Receive,
    max_body_bytes: int | None,
    reserve_body_bytes: BodyBytesReservationCallback | None = None,
    *,
    idle_timeout_seconds: float,
    total_timeout_seconds: float | None,
) -> bytearray:
    body = bytearray()
    total = 0
    more_body = True
    total_deadline = (
        None
        if total_timeout_seconds is None
        else monotonic() + total_timeout_seconds
    )
    while more_body:
        message = await _receive_body_message(
            receive,
            idle_timeout_seconds=idle_timeout_seconds,
            total_deadline=total_deadline,
        )
        if message["type"] == "http.disconnect":
            raise RequestBodyDisconnected()
        if message["type"] != "http.request":
            continue
        chunk = message.get("body", b"")
        if chunk:
            total += len(chunk)
            if max_body_bytes is not None and total > max_body_bytes:
                raise _RequestBodyHardLimitExceeded()
            if reserve_body_bytes is not None:
                await _reserve_body_bytes(reserve_body_bytes, len(chunk))
            body.extend(chunk)
        more_body = bool(message.get("more_body", False))
    return body


async def _receive_body_message(
    receive: Receive,
    *,
    idle_timeout_seconds: float,
    total_deadline: float | None,
) -> Message:
    timeout = idle_timeout_seconds
    if total_deadline is not None:
        remaining_total = total_deadline - monotonic()
        if remaining_total <= 0:
            raise RequestBodyReadTimeout()
        timeout = min(timeout, remaining_total)
    try:
        return await asyncio.wait_for(receive(), timeout=timeout)
    except TimeoutError as exc:
        raise RequestBodyReadTimeout() from exc


async def _decompress_zstd(
    body: bytes | bytearray,
    max_body_bytes: int | None,
    reserve_body_bytes: BodyBytesReservationCallback | None = None,
    release_body_bytes: BodyBytesReleaseCallback | None = None,
    *,
    memory_budget: _RequestBodyMemoryBudget,
) -> bytes:
    max_window_size = await _run_body_cpu(_ensure_complete_zstd_frames, body)
    declared_size = await _run_body_cpu(zstd.frame_content_size, body)
    if (
        max_body_bytes is not None
        and declared_size >= 0
        and declared_size > max_body_bytes
    ):
        raise _RequestBodyHardLimitExceeded()
    allowed_window_size = (
        max(1024, max_window_size)
        if max_body_bytes is None
        else max(
            1024,
            max_body_bytes + max(1024, max_body_bytes // 8),
        )
    )
    if (
        max_body_bytes is not None
        and max_window_size > allowed_window_size
    ):
        raise zstd.ZstdError(
            "zstd frame window exceeds decoded-body envelope"
        )
    if reserve_body_bytes is not None and max_window_size:
        # The decoder retains its history window independently of compressed
        # input and emitted output.  Account it before constructing/reading the
        # decoder so advertised-window memory cannot bypass body admission.
        observe_request_decoder_workspace_bytes(
            memory_budget._scope,
            max_window_size,
        )
        await _reserve_body_bytes(reserve_body_bytes, max_window_size)

    # A tiny unknown-size frame can advertise a huge decode window.  Bound the
    # decoder workspace itself, not only emitted output bytes, so zstd cannot
    # allocate hundreds of MiB before our decoded-body limit is consulted.
    decoder = zstd.ZstdDecompressor(max_window_size=allowed_window_size)
    decoded = bytearray()
    # ``read()`` is allowed to return fewer bytes than requested at a frame
    # boundary. Keep reading across every concatenated frame instead of
    # treating a short first read as evidence that the next byte exceeds the
    # limit. Each read is capped by the remaining budget plus one byte, so a
    # compression bomb cannot allocate its declared output before rejection.
    try:
        with decoder.stream_reader(body, read_across_frames=True) as reader:
            while True:
                remaining = (
                    None
                    if max_body_bytes is None
                    else max_body_bytes - len(decoded)
                )
                chunk = await _run_body_cpu(
                    reader.read,
                    (
                        64 * 1024
                        if remaining is None
                        else min(64 * 1024, remaining + 1)
                    ),
                )
                if not chunk:
                    break
                if remaining is not None and len(chunk) > remaining:
                    # Record the attempted decoded output even though this
                    # byte cannot be retained. The early-response callback
                    # syncs the observation before release.
                    _observe_request_decoded_bytes(
                        memory_budget._scope,
                        len(chunk),
                    )
                    if reserve_body_bytes is not None:
                        await _reserve_body_bytes(reserve_body_bytes, 0)
                    raise _RequestBodyHardLimitExceeded()
                # Reserve each bounded output chunk before retaining it. JSON
                # uses raw bytes plus structural tokens; non-JSON uses a
                # conservative multi-copy weight.
                await memory_budget.reserve_chunk(chunk, reserve_body_bytes)
                decoded.extend(chunk)
        result = await _run_body_cpu(bytes, decoded)
        decoded.clear()
        return result
    finally:
        if release_body_bytes is not None and max_window_size:
            await _release_body_bytes(release_body_bytes, max_window_size)


async def _run_body_cpu(callback: Callable[..., object], *args: object):
    return await run_cpu_phase(
        _REQUEST_BODY_CPU_EXECUTOR,
        callback,
        *args,
        phase="request_body",
    )


def _ensure_complete_zstd_frames(
    body: bytes | bytearray,
) -> int:
    """Validate complete standard/skippable zstd frame boundaries.

    The Python decompression reader deliberately treats a truncated frame with
    unknown content size as EOF.  Parsing the public zstd frame block headers
    proves structural completeness without a second decompression pass (which
    could otherwise allocate bomb output outside the byte budget).
    """
    view = memoryview(body)
    offset = 0
    max_window_size = 0
    while offset < len(view):
        if len(view) - offset < 4:
            raise zstd.ZstdError("incomplete zstd frame magic")
        magic = int.from_bytes(view[offset : offset + 4], "little")

        # Zstandard skippable frames use magic 0x184D2A50..0x184D2A5F and a
        # four-byte little-endian payload length.
        if 0x184D2A50 <= magic <= 0x184D2A5F:
            if len(view) - offset < 8:
                raise zstd.ZstdError("incomplete zstd skippable frame header")
            payload_size = int.from_bytes(view[offset + 4 : offset + 8], "little")
            frame_end = offset + 8 + payload_size
            if frame_end > len(view):
                raise zstd.ZstdError("incomplete zstd skippable frame")
            offset = frame_end
            continue

        if magic != 0xFD2FB528:
            raise zstd.ZstdError("invalid zstd frame magic")
        try:
            header_size = zstd.frame_header_size(view[offset:])
            parameters = zstd.get_frame_parameters(view[offset:])
        except (ValueError, zstd.ZstdError) as exc:
            raise zstd.ZstdError("incomplete zstd frame header") from exc
        max_window_size = max(
            max_window_size,
            int(parameters.window_size or 0),
        )

        cursor = offset + header_size
        while True:
            if len(view) - cursor < 3:
                raise zstd.ZstdError("incomplete zstd block header")
            block_header = int.from_bytes(view[cursor : cursor + 3], "little")
            cursor += 3
            last_block = bool(block_header & 1)
            block_type = (block_header >> 1) & 0b11
            block_size = block_header >> 3
            if block_type == 0b11:
                raise zstd.ZstdError("reserved zstd block type")
            payload_size = 1 if block_type == 0b01 else block_size
            if cursor + payload_size > len(view):
                raise zstd.ZstdError("incomplete zstd block payload")
            cursor += payload_size
            if last_block:
                break

        if parameters.has_checksum:
            if cursor + 4 > len(view):
                raise zstd.ZstdError("incomplete zstd frame checksum")
            cursor += 4
        offset = cursor
    return max_window_size


def _ensure_content_length_within(
    headers: Iterable[tuple[bytes, bytes]],
    max_body_bytes: int,
) -> None:
    content_length = _content_length(headers)
    if content_length is not None and content_length > max_body_bytes:
        raise _RequestBodyHardLimitExceeded()


def _request_body_memory_budget(
    scope: Scope,
    headers: Iterable[tuple[bytes, bytes]],
    *,
    json_max_estimated_bytes: int,
) -> _RequestBodyMemoryBudget:
    content_types = [
        value.decode("latin-1", errors="replace").strip().lower()
        for name, value in headers
        if name.lower() == b"content-type"
    ]
    json_body = False
    if len(content_types) != 1:
        # Duplicate/malformed or absent headers must not downgrade a known
        # FastAPI Body/Pydantic route to raw-byte accounting.
        json_body = str(scope.get("path") or "") in JSON_BODY_PATHS
    else:
        json_body = is_json_media_type(content_types[0])
    return _RequestBodyMemoryBudget(
        scope=scope,
        json_body=json_body,
        json_max_estimated_bytes=json_max_estimated_bytes,
    )


def _body_bytes_reservation_callback(
    scope: Scope,
) -> BodyBytesReservationCallback | None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return None
    callback = state.get(BODY_BYTES_RESERVATION_SCOPE_KEY)
    return callback if callable(callback) else None


def _body_bytes_release_callback(
    scope: Scope,
) -> BodyBytesReleaseCallback | None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return None
    callback = state.get(BODY_BYTES_RELEASE_SCOPE_KEY)
    return callback if callable(callback) else None


async def _release_body_bytes(
    callback: BodyBytesReleaseCallback,
    released_bytes: int,
) -> None:
    result = callback(max(0, int(released_bytes)))
    if hasattr(result, "__await__"):
        await result


async def _release_remaining_body_bytes(scope: Scope) -> None:
    callback = _body_bytes_release_callback(scope)
    lease = get_request_admission_lease()
    if callback is None or lease is None:
        return
    remaining = max(0, int(lease.reserved_body_bytes))
    if remaining:
        await _release_body_bytes(callback, remaining)


def _disconnect_event(scope: Scope) -> asyncio.Event:
    state = scope.setdefault("state", {})
    event = state.get(DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY)
    if isinstance(event, asyncio.Event):
        return event
    event = asyncio.Event()
    state[DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY] = event
    return event


def _request_has_no_body(scope: Scope) -> bool:
    headers = scope.get("headers") or []
    content_length = None
    has_transfer_encoding = False
    for name, value in headers:
        lowered = name.lower()
        if lowered == b"content-length":
            content_length = value.strip()
        elif lowered == b"transfer-encoding" and value.strip():
            has_transfer_encoding = True
    if content_length == b"0":
        return True
    return (
        str(scope.get("method") or "").upper() in {"GET", "HEAD", "OPTIONS"}
        and content_length is None
        and not has_transfer_encoding
    )


async def _monitor_disconnect(
    receive: Receive,
    disconnect_event: asyncio.Event,
) -> None:
    try:
        while not disconnect_event.is_set():
            message = await receive()
            if message.get("type") == "http.disconnect":
                disconnect_event.set()
                return
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # A receive-adapter failure is not evidence that the peer closed the
        # socket.  Keep disconnect sticky only for an actual ASGI event.
        logger.warning(
            "request disconnect monitor stopped after receive failure: %s",
            exc,
        )


async def _cancel_disconnect_monitor(task: asyncio.Task[None]) -> None:
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _record_body_rejection(scope: Scope, reason: str) -> None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return
    callback = state.get(BODY_REJECTION_RECORDER_SCOPE_KEY)
    if callable(callback):
        callback(reason)


async def _reserve_body_bytes(
    callback: BodyBytesReservationCallback,
    size: int,
) -> None:
    await callback(size)


def _content_length(headers: Iterable[tuple[bytes, bytes]]) -> int | None:
    values: list[str] = []
    for name, value in headers:
        if name.lower() != b"content-length":
            continue
        values.extend(part.strip() for part in value.decode("latin-1").split(","))

    if not values:
        return None
    if any(
        not value or not value.isascii() or not value.isdecimal()
        for value in values
    ):
        raise InvalidContentLength()

    try:
        parsed = [int(value) for value in values]
    except ValueError as exc:
        # Python rejects extremely long integer strings; they are malformed
        # Content-Length values rather than application failures.
        raise InvalidContentLength() from exc
    if any(value != parsed[0] for value in parsed[1:]):
        raise InvalidContentLength()
    return parsed[0]


def _content_encodings(headers: Iterable[tuple[bytes, bytes]]) -> list[str]:
    values = [
        value.decode("latin-1")
        for name, value in headers
        if name.lower() == b"content-encoding"
    ]
    encodings: list[str] = []
    for value in values:
        encodings.extend(
            part.strip().lower()
            for part in value.split(",")
            if part.strip()
        )
    return encodings


def _is_identity(encodings: list[str]) -> bool:
    return all(encoding == "identity" for encoding in encodings)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _bounded_limit(value: int) -> int:
    return max(0, int(value))


def _positive_float(value: float) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("request body timeout must be positive")
    return parsed
