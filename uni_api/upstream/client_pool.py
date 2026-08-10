from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import ceil
from time import monotonic
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from core.log_config import logger
from core.utils import get_proxy
from uni_api.admission.json_memory import (
    JSONMemoryComplexityError,
    estimate_json_memory_bytes,
)
from uni_api.admission.json_parsing import run_json_cpu
from uni_api.admission import (
    AdmissionLease,
    AdmissionRejected,
    BoundedAdmissionGate,
    get_request_admission_lease,
)
from uni_api.admission.network import (
    AdaptiveNetworkGovernor,
    NetworkGovernorClosed,
    NetworkResourceLease,
)
from uni_api.observability.responses_stream import (
    observe_client_pool_shutdown_connection,
    observe_client_pool_shutdown_completed,
)
from uni_api.observability.upstream_transport import (
    current_upstream_transport_diagnostics,
    inject_transport_trace,
    install_transport_phase_observability,
)
from uni_api.streaming.cleanup import (
    await_isolated_transport_cleanup_safely,
    await_stream_cleanup_safely,
    force_close_response_httpcore_stream_chain_safely,
)
from uni_api.upstream.response_limits import (
    read_limited_response_body,
    upstream_error_body_max_bytes,
    upstream_success_body_max_bytes,
)


class UpstreamAdmissionRejected(RuntimeError):
    """The bounded per-upstream queue could not admit a network operation."""

    status_code = 503
    local_admission_rejection = True

    def __init__(
        self,
        reason: str,
        *,
        retry_after_seconds: int,
        client_key_id: str,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.client_key_id = client_key_id


class UpstreamResponseTooLarge(RuntimeError):
    status_code = 502
    reason = "upstream_response_too_large"

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upstream response exceeded {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


class UpstreamUnsupportedContentEncoding(RuntimeError):
    status_code = 502
    reason = "upstream_stream_content_encoding_unsupported"

    def __init__(self, content_encoding: str) -> None:
        normalized = str(content_encoding or "").strip() or "unknown"
        super().__init__(
            "streaming upstream ignored Accept-Encoding: identity and returned "
            f"unsupported Content-Encoding: {normalized}"
        )
        self.content_encoding = normalized


class UpstreamResponseJSONEncodingUnsupported(RuntimeError):
    status_code = 502
    reason = "upstream_json_encoding_unsupported"

    def __init__(self) -> None:
        super().__init__("upstream JSON response must use UTF-8")


_ERROR_RESPONSE_MATERIALIZATION_MULTIPLIER = 72
_TRUNCATED_ERROR_SUFFIX = b"\n...[upstream error body truncated]"


def _json_prefix_kind(content: bytes) -> str:
    if content.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "unsupported"
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "unsupported"
    index = 3 if content.startswith(b"\xef\xbb\xbf") else 0
    while index < len(content) and content[index] in {0x20, 0x09, 0x0A, 0x0D}:
        index += 1
    if index < len(content) and content[index] in {ord("{"), ord("[")}:
        return "json"
    return "other"


def _force_identity_accept_encoding(kwargs: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    headers = httpx.Headers(request_kwargs.get("headers"))
    headers["Accept-Encoding"] = "identity"
    request_kwargs["headers"] = headers
    return request_kwargs


def _validate_stream_content_encoding(response: httpx.Response) -> None:
    if not (200 <= response.status_code < 300):
        return
    content_encoding = str(response.headers.get("content-encoding") or "").strip()
    if content_encoding and content_encoding.lower() != "identity":
        raise UpstreamUnsupportedContentEncoding(content_encoding)


@dataclass(slots=True)
class _AdmissionStats:
    attempts_observed: int = 0
    wait_ms_last: float = 0.0
    wait_ms_total: float = 0.0
    wait_ms_max: float = 0.0

    def record(self, wait_ms: float) -> None:
        wait_ms = max(0.0, float(wait_ms))
        self.attempts_observed += 1
        self.wait_ms_last = wait_ms
        self.wait_ms_total += wait_ms
        self.wait_ms_max = max(self.wait_ms_max, wait_ms)


async def _finish_cleanup_despite_cancellation(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


class _UpstreamLease:
    """An upstream operation plus its provisional connection resources."""

    def __init__(
        self,
        gate_lease: AdmissionLease | NetworkResourceLease,
        *,
        release_on_connected: bool,
        on_release: Callable[[], None],
    ) -> None:
        self.wait_ms = gate_lease.wait_ms
        self._gate_lease = gate_lease
        self._release_on_connected = bool(release_on_connected)
        self._on_release = on_release
        self._connected_task: asyncio.Task[None] | None = None
        self._release_task: asyncio.Task[None] | None = None

    @property
    def released(self) -> bool:
        return self._release_task is not None and self._release_task.done()

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release_once())
        release_task = self._release_task
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(release_task)
            raise

    async def mark_connected(self) -> None:
        """Hand provisional FD/port ownership back to live kernel counters."""

        if not self._release_on_connected:
            return
        if self._connected_task is None:
            self._connected_task = asyncio.create_task(
                self._gate_lease.release()
            )
        connected_task = self._connected_task
        try:
            await asyncio.shield(connected_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(connected_task)
            raise

    async def _release_once(self) -> None:
        if not self._release_on_connected:
            # Preserve legacy gauge ordering before FIFO capacity promotion.
            self._on_release()
            await self._gate_lease.release()
            return
        try:
            await self.mark_connected()
        finally:
            self._on_release()


class _ResponseLeaseBinding:
    """Release admission when an HTTPX streaming response is really closed."""

    def __init__(self, response: httpx.Response, lease: _UpstreamLease) -> None:
        self.response = response
        self.lease = lease
        self._original_aclose = response.aclose
        self._close_task: asyncio.Task[None] | None = None
        self._abort_task: asyncio.Task[None] | None = None
        # HTTPX Response intentionally has an instance dictionary. Keeping the
        # original Response avoids breaking callers that rely on its exact API.
        response.aclose = self.aclose  # type: ignore[method-assign]

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_once())
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(close_task)
            raise

    async def _close_once(self) -> None:
        try:
            await self._original_aclose()
        finally:
            # Break the Response -> bound method -> binding -> Response cycle
            # once the streaming lifecycle is over.
            self.response.aclose = self._original_aclose  # type: ignore[method-assign]
            await self.lease.release()

    async def abort_transport(self) -> None:
        """Release upstream admission after the transport is evicted.

        Callers must remove the associated pool request/connection before
        invoking this method.  The request-level admission lease remains the
        fail-closed owner for any non-cooperative cleanup task, while this
        per-upstream lease can be handed to a new request because the evicted
        connection can no longer be reused.
        """

        if self._abort_task is None:
            self._abort_task = asyncio.create_task(self._abort_once())
        abort_task = self._abort_task
        try:
            await asyncio.shield(abort_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(abort_task)
            raise

    async def _abort_once(self) -> None:
        self.response.aclose = self._original_aclose  # type: ignore[method-assign]
        await self.lease.release()


class _ManagedStreamContext:
    def __init__(
        self,
        client: _ManagedAsyncClient,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._client = client
        self._args = args
        self._kwargs = kwargs
        self._lease: _UpstreamLease | None = None
        self._raw_context: Any = None
        self._binding: _ResponseLeaseBinding | None = None
        self._exit_task: asyncio.Task[bool | None] | None = None

    async def __aenter__(self) -> httpx.Response:
        if self._lease is not None:
            raise RuntimeError("upstream stream context cannot be entered twice")
        lease = await self._client._acquire()
        self._lease = lease
        diagnostics = current_upstream_transport_diagnostics()
        if diagnostics is not None:
            diagnostics.bind_client(self._client._client)
        request_kwargs = inject_transport_trace(
            _force_identity_accept_encoding(self._kwargs),
            diagnostics,
        )
        raw_context = self._client._client.stream(*self._args, **request_kwargs)
        self._raw_context = raw_context
        response: httpx.Response | None = None
        try:
            response = await raw_context.__aenter__()
            await lease.mark_connected()
            if diagnostics is not None:
                diagnostics.capture_response(
                    response,
                    client=self._client._client,
                )
            self._binding = _ResponseLeaseBinding(response, lease)
            _validate_stream_content_encoding(response)
            if response.is_closed:
                await response.aclose()
            return response
        except BaseException as exc:
            if diagnostics is not None:
                diagnostics.observe_exception(
                    exc,
                    client=self._client._client,
                )
            try:
                if response is not None:
                    transport_isolated = await (
                        force_close_response_httpcore_stream_chain_safely(
                        response,
                        label="upstream stream enter response",
                        )
                    )
                    raw_exit = raw_context.__aexit__(None, None, None)
                    if transport_isolated:
                        await await_isolated_transport_cleanup_safely(
                            raw_exit,
                            label="upstream stream enter context",
                        )
                    else:
                        await await_stream_cleanup_safely(
                            raw_exit,
                            label="upstream stream enter context",
                        )
            finally:
                await lease.release()
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None:
        if self._lease is None or self._raw_context is None:
            raise RuntimeError("upstream stream context was not entered")
        if self._exit_task is None:
            self._exit_task = asyncio.create_task(
                self._exit_once(exc_type, exc, traceback)
            )
        exit_task = self._exit_task
        try:
            return await asyncio.shield(exit_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(exit_task)
            raise

    async def _exit_once(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None:
        try:
            transport_isolated = False
            if self._binding is not None:
                transport_isolated = await (
                    force_close_response_httpcore_stream_chain_safely(
                    self._binding.response,
                    label="upstream managed stream response",
                    )
                )
            raw_exit = self._raw_context.__aexit__(exc_type, exc, traceback)
            if transport_isolated:
                await await_isolated_transport_cleanup_safely(
                    raw_exit,
                    label="upstream managed stream context",
                )
            else:
                await await_stream_cleanup_safely(
                    raw_exit,
                    label="upstream managed stream context",
                )
            # HTTPX's stream context never suppresses the caller's exception;
            # preserve its concrete falsey context-manager return contract.
            return False
        finally:
            await self._lease.release()


class _ManagedAsyncClient:
    """The supported HTTPX surface with admission at each network operation."""

    def __init__(self, pool: ClientPool, client_key: str, client: httpx.AsyncClient) -> None:
        self._pool = pool
        self._client_key = client_key
        self._client = client

    async def _acquire(self) -> _UpstreamLease:
        return await self._pool._acquire_upstream(self._client_key)

    async def _non_stream_request(
        self,
        method: str,
        url: Any,
        *,
        reservation_multiplier: int | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        diagnostics = current_upstream_transport_diagnostics()
        if diagnostics is not None:
            diagnostics.bind_client(self._client)
        request_kwargs = inject_transport_trace(
            _force_identity_accept_encoding(kwargs),
            diagnostics,
        )
        lease = await self._acquire()
        try:
            async with self._client.stream(method, url, **request_kwargs) as response:
                await lease.mark_connected()
                if diagnostics is not None:
                    diagnostics.capture_response(response, client=self._client)
                return await self._buffer_bounded_response(
                    response,
                    reservation_multiplier=reservation_multiplier,
                )
        except BaseException as exc:
            if diagnostics is not None:
                diagnostics.observe_exception(exc, client=self._client)
            raise
        finally:
            await lease.release()

    @staticmethod
    async def _buffer_bounded_response(
        response: httpx.Response,
        *,
        reservation_multiplier: int | None = None,
    ) -> httpx.Response:
        success = 200 <= response.status_code < 300
        limit = (
            upstream_success_body_max_bytes()
            if success
            else upstream_error_body_max_bytes()
        )
        # Call sites sometimes parse JSON even when an upstream omits or lies
        # about Content-Type.  Generic buffered responses therefore get a
        # token-aware materialization budget; explicit trusted binary paths
        # pass multiplier=1 and skip structural accounting.
        structured_json_budget = reservation_multiplier is None
        if reservation_multiplier is None:
            reservation_multiplier = 4
        else:
            # read_limited_response_body materializes bytearray + final bytes
            # concurrently even for trusted binary responses.
            reservation_multiplier = max(2, reservation_multiplier)
        if reservation_multiplier <= 0:
            raise ValueError("response reservation multiplier must be positive")
        request_lease = get_request_admission_lease()
        pending_reservation = (
            await request_lease.reserve_temporary_response_bytes(0)
            if request_lease is not None
            else None
        )
        try:
            limited = await read_limited_response_body(
                response,
                max_bytes=limit,
                reserve_bytes=(
                    pending_reservation.reserve
                    if pending_reservation is not None
                    else None
                ),
                reservation_multiplier=reservation_multiplier,
            )
            if limited.truncated and success:
                raise UpstreamResponseTooLarge(limit)
            content = limited.body
            if limited.truncated:
                if pending_reservation is not None:
                    await pending_reservation.reserve(
                        len(_TRUNCATED_ERROR_SUFFIX) * reservation_multiplier
                    )
                content += _TRUNCATED_ERROR_SUFFIX

            if structured_json_budget and not success and content:
                # Error bodies must preserve their real upstream status even
                # when a text/plain diagnostic happens to begin with ``{``.
                # Reserve a syntax-independent worst-case JSON/container
                # envelope before later provider error handling can parse it.
                # The 256 KiB error-body cap keeps this conservative charge
                # finite without trusting Content-Type or a prefix heuristic.
                target = (
                    len(content) * _ERROR_RESPONSE_MATERIALIZATION_MULTIPLIER
                )
                additional = max(
                    0,
                    target - len(content) * reservation_multiplier,
                )
                if pending_reservation is not None and additional:
                    await pending_reservation.reserve(additional)
                elif target > 256 * 1024 * 1024:
                    raise UpstreamResponseTooLarge(limit)
            elif structured_json_budget and success and content:
                prefix_kind = (
                    await run_json_cpu(_json_prefix_kind, content)
                    if len(content) >= 64 * 1024
                    else _json_prefix_kind(content)
                )
                if prefix_kind == "unsupported":
                    raise UpstreamResponseJSONEncodingUnsupported()
                looks_like_json = prefix_kind == "json"
            else:
                looks_like_json = False

            if structured_json_budget and success and content and looks_like_json:
                try:
                    estimate = await run_json_cpu(
                        estimate_json_memory_bytes,
                        content,
                        raw_memory_multiplier=4,
                        token_memory_bytes=128,
                        max_estimated_bytes=2 * 1024 * 1024 * 1024,
                    )
                except JSONMemoryComplexityError as exc:
                    raise UpstreamResponseTooLarge(limit) from exc
                additional = max(
                    0,
                    estimate.estimated_bytes
                    - len(content) * reservation_multiplier,
                )
                if pending_reservation is not None and additional:
                    # Raw bytes were reserved incrementally before retention.
                    # Add the token-aware materialization envelope before any
                    # caller can run json.loads/Pydantic.
                    await pending_reservation.reserve(additional)
                elif estimate.estimated_bytes > 256 * 1024 * 1024:
                    raise UpstreamResponseTooLarge(limit)
            elif structured_json_budget and success and content:
                # Invalid/non-JSON bodies cannot expand into a container graph,
                # but may still coexist as wire bytes, decoded text, and an
                # error/detail copy.  Preserve their real upstream status.
                # The four-copy envelope was already reserved incrementally.
                pass
            if pending_reservation is not None:
                await pending_reservation.commit()
        except BaseException:
            if pending_reservation is not None:
                await pending_reservation.release()
            raise
        headers = [
            (name, value)
            for name, value in response.headers.raw
            if name.lower()
            not in {b"content-length", b"content-encoding", b"transfer-encoding"}
        ]
        extensions = {
            key: value
            for key, value in response.extensions.items()
            if key != "network_stream"
        }
        extensions["uni_api_body_already_bounded"] = True
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=content,
            request=response.request,
            extensions=extensions,
        )

    async def request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request(method, url, **kwargs)

    async def get(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("GET", url, **kwargs)

    async def post(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("POST", url, **kwargs)

    async def post_buffered_binary(self, url: Any, **kwargs: Any) -> httpx.Response:
        """Buffer a trusted binary response without applying JSON expansion."""

        return await self._non_stream_request(
            "POST",
            url,
            reservation_multiplier=1,
            **kwargs,
        )

    async def put(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("PUT", url, **kwargs)

    async def delete(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("DELETE", url, **kwargs)

    async def patch(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("PATCH", url, **kwargs)

    async def head(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("HEAD", url, **kwargs)

    async def options(self, url: Any, **kwargs: Any) -> httpx.Response:
        return await self._non_stream_request("OPTIONS", url, **kwargs)

    def build_request(self, method: str, url: Any, **kwargs: Any) -> httpx.Request:
        # Building a request allocates no connection and must not consume a slot.
        return self._client.build_request(method, url, **kwargs)

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        request.headers["Accept-Encoding"] = "identity"
        diagnostics = current_upstream_transport_diagnostics()
        if diagnostics is not None:
            diagnostics.bind_client(self._client)
        combined = inject_transport_trace(
            {"extensions": request.extensions},
            diagnostics,
        )
        request.extensions = combined.get("extensions", request.extensions)
        lease = await self._acquire()
        try:
            response = await self._client.send(request, stream=True, **kwargs)
            await lease.mark_connected()
            if diagnostics is not None:
                diagnostics.capture_response(response, client=self._client)
        except BaseException as exc:
            if diagnostics is not None:
                diagnostics.observe_exception(exc, client=self._client)
            await lease.release()
            raise

        if not stream:
            try:
                try:
                    return await self._buffer_bounded_response(response)
                except BaseException as exc:
                    if diagnostics is not None:
                        diagnostics.observe_exception(
                            exc,
                            client=self._client,
                        )
                    raise
            finally:
                try:
                    await response.aclose()
                finally:
                    await lease.release()

        binding = _ResponseLeaseBinding(response, lease)
        try:
            _validate_stream_content_encoding(response)
        except BaseException:
            try:
                await force_close_response_httpcore_stream_chain_safely(
                    response,
                    label="upstream send streaming response",
                )
            finally:
                await lease.release()
            raise
        if response.is_closed:
            await response.aclose()
        return response

    def stream(self, *args: Any, **kwargs: Any) -> _ManagedStreamContext:
        return _ManagedStreamContext(self, args, kwargs)


class ClientPool:
    def __init__(
        self,
        pool_size: int | None = None,
        *,
        waiter_limit: int | None = None,
        wait_timeout_seconds: float = 5.0,
        pool_timeout_seconds: float = 1.0,
        sweep_client: Callable[[httpx.AsyncClient], Awaitable[int]] | None = None,
        current_trace: Callable[[], Any] | None = None,
        begin_upstream_pool: Callable[[Any], Any] | None = None,
        end_upstream_pool: Callable[[], Any] | None = None,
        record_upstream_wait: Callable[[float], Any] | None = None,
        network_governor: AdaptiveNetworkGovernor | None = None,
    ) -> None:
        if pool_size is not None and pool_size <= 0:
            raise ValueError("pool_size must be greater than zero")
        resolved_waiter_limit = (
            pool_size if pool_size is not None and waiter_limit is None
            else waiter_limit
        )
        if resolved_waiter_limit is not None and resolved_waiter_limit < 0:
            raise ValueError("waiter_limit cannot be negative")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be greater than zero")
        if pool_timeout_seconds <= 0:
            raise ValueError("pool_timeout_seconds must be greater than zero")

        self.pool_size = pool_size
        self.waiter_limit = resolved_waiter_limit
        self.resource_mode = pool_size is None
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self.pool_timeout_seconds = float(pool_timeout_seconds)
        self.clients: dict[str, httpx.AsyncClient] = {}
        self._managed_clients: dict[str, _ManagedAsyncClient] = {}
        self._admission_gates: dict[str, BoundedAdmissionGate] = {}
        self._admission_stats: dict[str, _AdmissionStats] = {}
        self._client_labels: dict[str, str] = {}
        self._network_governor = network_governor or AdaptiveNetworkGovernor()
        self._active_operations = 0
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._maintenance_task: Optional[asyncio.Task] = None
        self._last_sweep_closed_connections = 0
        self._last_sweep_error: Optional[str] = None
        self._last_sweep_at: Optional[datetime] = None
        self._sweep_client = sweep_client
        self._current_trace = current_trace or (lambda: None)
        self._begin_upstream_pool = begin_upstream_pool or (lambda trace: None)
        self._end_upstream_pool = end_upstream_pool or (lambda: None)
        self._record_upstream_wait = record_upstream_wait or (lambda wait_ms: None)

    async def init(self, default_config: dict[str, Any]) -> None:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("cannot initialize a closed upstream client pool")
            self.default_config = default_config
            if self._maintenance_task is None:
                self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            await self.sweep_idle_connections()

    async def sweep_idle_connections(self) -> int:
        closed = 0
        errors: list[str] = []
        if self._sweep_client is None:
            return 0
        for key, client in list(self.clients.items()):
            try:
                closed += await self._sweep_client(client)
            except Exception as exc:
                errors.append(f"{self._client_labels.get(key, 'unknown')}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "Failed to sweep upstream HTTP client idle connections: key=%s",
                    self._client_labels.get(key, "unknown"),
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        self._last_sweep_closed_connections = closed
        self._last_sweep_error = "; ".join(errors)[:512] if errors else None
        self._last_sweep_at = datetime.now(timezone.utc)
        return closed

    def snapshot(self) -> dict[str, Any]:
        per_key: dict[str, dict[str, Any]] = {}
        rejected: Counter[str] = Counter()
        active = self._active_operations if self.resource_mode else 0
        network_snapshot = self._network_governor.snapshot()
        waiters = (
            network_snapshot.waiting_connection_attempts
            if self.resource_mode
            else 0
        )
        acquired_total = 0
        cancelled_total = (
            network_snapshot.cancelled_total if self.resource_mode else 0
        )
        wait_ms_total = 0.0
        wait_ms_max = 0.0
        attempts_observed = 0

        for client_key, stats in self._admission_stats.items():
            label = self._client_labels[client_key]
            gate = self._admission_gates.get(client_key)
            gate_snapshot = gate.snapshot() if gate is not None else {
                "mode": "weighted_resources",
                "active": None,
                "waiters": None,
                "capacity": None,
                "waiter_limit": None,
                "wait_timeout_seconds": None,
                "acquired_total": stats.attempts_observed,
                "cancelled_total": 0,
                "rejected": {},
            }
            acquired = int(gate_snapshot["acquired_total"])
            key_snapshot = {
                **gate_snapshot,
                "wait_ms_last": round(stats.wait_ms_last, 3),
                "wait_ms_total": round(stats.wait_ms_total, 3),
                "wait_ms_max": round(stats.wait_ms_max, 3),
                "wait_ms_avg": round(stats.wait_ms_total / stats.attempts_observed, 3)
                if stats.attempts_observed
                else 0.0,
                "attempts_observed": stats.attempts_observed,
            }
            per_key[label] = key_snapshot
            if not self.resource_mode:
                active += int(gate_snapshot["active"])
                waiters += int(gate_snapshot["waiters"])
            acquired_total += acquired
            if not self.resource_mode:
                cancelled_total += int(gate_snapshot["cancelled_total"])
            rejected.update(gate_snapshot["rejected"])
            wait_ms_total += stats.wait_ms_total
            wait_ms_max = max(wait_ms_max, stats.wait_ms_max)
            attempts_observed += stats.attempts_observed

        admission = {
            "mode": (
                "weighted_resources" if self.resource_mode else "legacy_count"
            ),
            "active": active,
            "waiters": waiters,
            "capacity_per_key": self.pool_size,
            "waiter_limit_per_key": self.waiter_limit,
            "acquired_total": acquired_total,
            "cancelled_total": cancelled_total,
            "rejected": dict(rejected),
            "wait_ms_total": round(wait_ms_total, 3),
            "wait_ms_max": round(wait_ms_max, 3),
            "wait_ms_avg": round(wait_ms_total / attempts_observed, 3)
            if attempts_observed
            else 0.0,
            "attempts_observed": attempts_observed,
            "network_resources": asdict(network_snapshot),
        }
        return {
            "client_count": len(self.clients),
            "pool_size": self.pool_size,
            "resource_mode": self.resource_mode,
            "closing": self._closing,
            "closed": self._closed,
            "pool_timeout_seconds": self.pool_timeout_seconds,
            "waiter_limit": self.waiter_limit,
            "wait_timeout_seconds": self.wait_timeout_seconds,
            "admission": admission,
            "per_key": per_key,
            "last_sweep_closed_connections": self._last_sweep_closed_connections,
            "last_sweep_at": self._last_sweep_at.isoformat() if self._last_sweep_at else None,
            "last_sweep_error": self._last_sweep_error,
            "network_resources": asdict(network_snapshot),
        }

    @asynccontextmanager
    async def get_client(
        self,
        base_url: str,
        proxy: str | None = None,
        http2: Optional[bool] = None,
    ):
        client_key = self._client_key(base_url, proxy, http2)
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise self._pool_closed_rejection(client_key)
            if client_key not in self.clients:
                timeout = httpx.Timeout(
                    connect=15.0,
                    read=None,
                    write=30.0,
                    pool=self.pool_timeout_seconds,
                )
                # ``None`` removes HTTPX's request-count pool cap. Live FD and
                # ephemeral-port headroom gate only connection establishment.
                limits = httpx.Limits(max_connections=self.pool_size)
                client_config = {
                    **self.default_config,
                    "timeout": timeout,
                    "limits": limits,
                }
                client_config = get_proxy(proxy, client_config)
                if http2 is not None:
                    client_config["http2"] = bool(http2)
                raw_client = httpx.AsyncClient(**client_config)
                install_transport_phase_observability(raw_client)
                self.clients[client_key] = raw_client
                if not self.resource_mode:
                    assert self.pool_size is not None
                    assert self.waiter_limit is not None
                    self._admission_gates[client_key] = BoundedAdmissionGate(
                        self.pool_size,
                        waiter_limit=self.waiter_limit,
                        wait_timeout_seconds=self.wait_timeout_seconds,
                    )
                self._admission_stats[client_key] = _AdmissionStats()
                self._client_labels[client_key] = self._client_label(
                    base_url,
                    proxy,
                    http2,
                    client_key,
                )
                self._managed_clients[client_key] = _ManagedAsyncClient(
                    self,
                    client_key,
                    raw_client,
                )
            managed_client = self._managed_clients[client_key]

        yield managed_client

    async def _acquire_upstream(self, client_key: str) -> _UpstreamLease:
        trace = self._safe_current_trace()
        started_at = monotonic()
        self._safe_trace_mark(trace, "client_pool_acquire_start")
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise self._pool_closed_rejection(client_key)
            gate = self._admission_gates.get(client_key)
            stats = self._admission_stats[client_key]
            client_label = self._client_labels[client_key]
        try:
            if self.resource_mode:
                gate_lease: AdmissionLease | NetworkResourceLease = (
                    await self._network_governor.acquire(
                        abort=lambda: self._closing or self._closed,
                    )
                )
            else:
                if gate is None:
                    raise RuntimeError("legacy upstream admission gate is missing")
                gate_lease = await gate.acquire()
        except asyncio.CancelledError:
            wait_ms = (monotonic() - started_at) * 1000.0
            stats.record(wait_ms)
            self._safe_trace_mark(trace, "client_pool_acquire_end")
            self._record_trace_wait(trace, wait_ms)
            self._safe_record_upstream_wait(wait_ms)
            raise
        except NetworkGovernorClosed:
            wait_ms = (monotonic() - started_at) * 1000.0
            stats.record(wait_ms)
            self._safe_trace_mark(trace, "client_pool_acquire_end")
            self._record_trace_wait(trace, wait_ms)
            self._safe_record_upstream_wait(wait_ms)
            raise self._pool_closed_rejection(
                client_key,
                client_label=client_label,
            ) from None
        except AdmissionRejected as exc:
            wait_ms = (monotonic() - started_at) * 1000.0
            stats.record(wait_ms)
            self._safe_trace_mark(trace, "client_pool_acquire_end")
            self._record_trace_wait(trace, wait_ms)
            self._safe_record_upstream_wait(wait_ms)
            pool_closed = self._closing or self._closed
            if pool_closed:
                raise self._pool_closed_rejection(
                    client_key,
                    client_label=client_label,
                ) from None
            reason = {
                "queue_full": "upstream_queue_full",
                "wait_timeout": "upstream_wait_timeout",
            }.get(exc.reason, f"upstream_{exc.reason}")
            raise UpstreamAdmissionRejected(
                reason,
                retry_after_seconds=ceil(self.wait_timeout_seconds),
                client_key_id=client_label,
            ) from None

        wait_ms = gate_lease.wait_ms
        stats.record(wait_ms)
        pool_closed = self._closing or self._closed
        if pool_closed:
            self._safe_trace_mark(trace, "client_pool_acquire_end")
            self._record_trace_wait(trace, wait_ms)
            self._safe_record_upstream_wait(wait_ms)
            await gate_lease.release()
            raise self._pool_closed_rejection(
                client_key,
                client_label=client_label,
            )

        callback_started = self._safe_begin_upstream_pool(trace)
        if self.resource_mode:
            self._active_operations += 1

        def release_active_operation() -> None:
            if self.resource_mode:
                self._active_operations = max(
                    0,
                    self._active_operations - 1,
                )
            self._safe_end_upstream_pool(callback_started)

        self._safe_trace_mark(trace, "client_pool_acquire_end")
        self._safe_trace_mark(trace, "client_pool_acquired")
        self._record_trace_wait(trace, wait_ms)
        self._safe_record_upstream_wait(wait_ms)
        return _UpstreamLease(
            gate_lease,
            release_on_connected=self.resource_mode,
            on_release=release_active_operation,
        )

    def _pool_closed_rejection(
        self,
        client_key: str,
        *,
        client_label: str | None = None,
    ) -> UpstreamAdmissionRejected:
        label = client_label or self._client_labels.get(client_key)
        if not label:
            label = f"closed|id={sha256(client_key.encode('utf-8')).hexdigest()[:12]}"
        return UpstreamAdmissionRejected(
            "upstream_pool_closed",
            retry_after_seconds=1,
            client_key_id=label,
        )

    def _safe_current_trace(self) -> Any:
        try:
            return self._current_trace()
        except Exception:
            logger.warning("Failed to resolve upstream admission trace", exc_info=True)
            return None

    @staticmethod
    def _safe_trace_mark(trace: Any, stage: str) -> None:
        if trace is None:
            return
        try:
            trace.mark(stage)
        except Exception:
            logger.warning("Failed to record upstream admission trace mark", exc_info=True)

    @staticmethod
    def _record_trace_wait(trace: Any, wait_ms: float) -> None:
        if trace is None:
            return
        try:
            spans = getattr(trace, "spans", None)
            previous_total = 0.0
            previous_max = 0.0
            previous_count = 0.0
            if isinstance(spans, dict):
                previous_total = float(spans.get("upstream_pool_wait_ms_total", 0) or 0)
                previous_max = float(spans.get("upstream_pool_wait_ms_max", 0) or 0)
                previous_count = float(spans.get("upstream_pool_admission_count", 0) or 0)
            trace.add_ms("upstream_pool_wait_ms", wait_ms)
            trace.add_ms("upstream_pool_wait_ms_total", previous_total + wait_ms)
            trace.add_ms("upstream_pool_wait_ms_max", max(previous_max, wait_ms))
            trace.add_ms("upstream_pool_admission_count", previous_count + 1)
        except Exception:
            logger.warning("Failed to record upstream admission wait", exc_info=True)

    def _safe_begin_upstream_pool(self, trace: Any) -> bool:
        try:
            self._begin_upstream_pool(trace)
            return True
        except Exception:
            logger.warning("Failed to increment upstream active gauge", exc_info=True)
            return False

    def _safe_end_upstream_pool(self, callback_started: bool) -> None:
        if not callback_started:
            return
        try:
            self._end_upstream_pool()
        except Exception:
            logger.warning("Failed to decrement upstream active gauge", exc_info=True)

    def _safe_record_upstream_wait(self, wait_ms: float) -> None:
        try:
            self._record_upstream_wait(max(0.0, float(wait_ms)))
        except Exception:
            logger.warning("Failed to record upstream admission wait gauge", exc_info=True)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_once())
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await _finish_cleanup_despite_cancellation(close_task)
            raise

    async def _close_once(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True
            maintenance_task = self._maintenance_task
            self._maintenance_task = None
            clients = list(self.clients.values())

        if maintenance_task is not None:
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "Upstream HTTP client maintenance task failed during close",
                    exc_info=True,
                )

        for client in clients:
            shutdown_trackers = []
            try:
                transport = getattr(client, "_transport", None)
                pool = getattr(transport, "_pool", None)
                connections = getattr(pool, "_connections", None)
                if isinstance(connections, list):
                    for connection in list(connections):
                        for tracker in observe_client_pool_shutdown_connection(
                            connection
                        ):
                            if all(
                                existing is not tracker
                                for existing in shutdown_trackers
                            ):
                                shutdown_trackers.append(tracker)
                await client.aclose()
            except Exception:
                observe_client_pool_shutdown_completed(
                    shutdown_trackers,
                    succeeded=False,
                )
                logger.warning("Failed to close upstream HTTP client", exc_info=True)
            else:
                observe_client_pool_shutdown_completed(
                    shutdown_trackers,
                    succeeded=True,
                )

        async with self._lifecycle_lock:
            self.clients.clear()
            self._managed_clients.clear()
            self._closing = False
            self._closed = True

            # A close racing active/waiting operations must retain their gate
            # objects until those tasks release or abandon ownership. An idle
            # pool can discard all per-key admission state immediately.
            gate_snapshots = [
                gate.snapshot() for gate in self._admission_gates.values()
            ]
            if all(
                snapshot["active"] == 0 and snapshot["waiters"] == 0
                for snapshot in gate_snapshots
            ):
                self._admission_gates.clear()
                self._admission_stats.clear()
                self._client_labels.clear()

    @staticmethod
    def _client_key(base_url: str, proxy: str | None, http2: Optional[bool]) -> str:
        parsed_url = urlparse(base_url)
        client_key = f"{parsed_url.netloc}"
        if proxy:
            client_key += f"_{proxy.replace('socks5h://', 'socks5://')}"
        if http2 is not None:
            client_key += f"_http2_{int(bool(http2))}"
        return client_key

    @staticmethod
    def _client_label(
        base_url: str,
        proxy: str | None,
        http2: Optional[bool],
        client_key: str,
    ) -> str:
        parsed = urlparse(base_url)
        host = parsed.hostname or "unknown"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        http2_label = "default" if http2 is None else str(int(bool(http2)))
        key_id = sha256(client_key.encode("utf-8")).hexdigest()[:12]
        return f"{host}|proxy={int(bool(proxy))}|http2={http2_label}|id={key_id}"
