from __future__ import annotations

import asyncio
import hashlib
import re
from collections import deque
from collections.abc import Callable
from inspect import isawaitable
from time import monotonic
from typing import Any
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from uni_api.admission import (
    AdmissionRejected,
    RequestBodyObservation,
    RequestAdmissionController,
    bind_request_admission_lease,
    reset_request_admission_lease,
)
from uni_api.admission.json_memory import DEFAULT_JSON_RAW_MEMORY_MULTIPLIER
from core.log_config import logger
from uni_api.disconnect import DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY
from uni_api.http_content import is_json_media_type
from uni_api.middleware.request_decompression import (
    BODY_BYTES_RESERVATION_SCOPE_KEY,
    BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY,
    BODY_REJECTION_RECORDER_SCOPE_KEY,
    initialize_request_body_observation,
    observe_request_wire_bytes,
    request_body_observation_from_scope,
)


RESERVE_BODY_BYTES_STATE_KEY = BODY_BYTES_RESERVATION_SCOPE_KEY
RELEASE_BODY_BYTES_STATE_KEY = "uni_api_release_body_bytes"
ADMISSION_LEASE_STATE_KEY = "uni_api_admission_lease"
ADMISSION_WAIT_MS_STATE_KEY = "uni_api_admission_wait_ms"
ADMISSION_REQUEST_ID_STATE_KEY = "uni_api_admission_request_id"
ADMISSION_TRACE_ID_STATE_KEY = "uni_api_admission_trace_id"
ADMISSION_PREBUFFER_MESSAGE_HIGH_WATERMARK = 16
_SAFE_CORRELATION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_W3C_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_W3C_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_W3C_BYTE_RE = re.compile(r"^[0-9a-f]{2}$")


def _bounded_observation_text(value: Any, *, max_length: int = 160) -> str | None:
    text = str(value or "").strip()
    return text[:max_length] if text else None


def _safe_correlation_id(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if _SAFE_CORRELATION_ID_RE.fullmatch(text) else None


def _safe_w3c_trace_id(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if _W3C_TRACE_ID_RE.fullmatch(text) is None or text == "0" * 32:
        return None
    return text


def _valid_declared_content_length(scope: Scope) -> int | None:
    values = [
        value
        for name, value in (scope.get("headers") or [])
        if name.lower() == b"content-length"
    ]
    if len(values) != 1:
        return None
    try:
        decoded = values[0].decode("ascii")
        if not decoded or not decoded.isdigit():
            return None
        return int(decoded)
    except (UnicodeDecodeError, ValueError):
        return None


def _admission_fairness_key(scope: Scope) -> str:
    headers = {
        name.lower(): value
        for name, value in (scope.get("headers") or [])
    }
    identity = b"\x00".join(
        headers.get(name, b"")
        for name in (
            b"openai-organization",
            b"x-organization-id",
            b"authorization",
            b"x-api-key",
        )
    )
    if not identity.strip(b"\x00"):
        client = scope.get("client")
        client_identity = (
            client[0] if isinstance(client, tuple) and client else "anonymous"
        )
        identity = str(client_identity).encode("utf-8", errors="replace")
    return hashlib.sha256(identity).hexdigest()[:24]


def _declared_memory_target(scope: Scope, content_length: int) -> int:
    headers = {
        name.lower(): value.decode("latin-1", errors="replace").strip().lower()
        for name, value in (scope.get("headers") or [])
    }
    encoding = headers.get(b"content-encoding", "")
    content_type = headers.get(b"content-type", "")
    identity_encoded = encoding in {"", "identity"}
    json_body = is_json_media_type(content_type)
    multiplier = (
        DEFAULT_JSON_RAW_MEMORY_MULTIPLIER
        if identity_encoded and json_body
        else 1
    )
    return max(0, int(content_length)) * max(1, int(multiplier))


def _incoming_observation_identifiers(scope: Scope) -> tuple[str, str]:
    state = scope.setdefault("state", {})
    if isinstance(state, dict):
        saved_request_id = _safe_correlation_id(
            state.get(ADMISSION_REQUEST_ID_STATE_KEY)
        )
        saved_trace_id = _safe_w3c_trace_id(
            state.get(ADMISSION_TRACE_ID_STATE_KEY)
        )
        if saved_request_id is not None and saved_trace_id is not None:
            return saved_request_id, saved_trace_id

    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in (scope.get("headers") or [])
    }
    request_id = _safe_correlation_id(headers.get("x-request-id"))
    trace_id = None
    traceparent = str(headers.get("traceparent") or "").strip().lower()
    fields = traceparent.split("-")
    if (
        len(fields) == 4
        and _W3C_BYTE_RE.fullmatch(fields[0]) is not None
        and fields[0] != "ff"
        and _safe_w3c_trace_id(fields[1]) is not None
        and _W3C_SPAN_ID_RE.fullmatch(fields[2]) is not None
        and fields[2] != "0" * 16
        and _W3C_BYTE_RE.fullmatch(fields[3]) is not None
    ):
        trace_id = fields[1]
    if request_id is None and trace_id is not None:
        request_id = trace_id
    if trace_id is None:
        generated = uuid4().hex
        request_id = request_id or generated
        trace_id = generated
    if isinstance(state, dict):
        state[ADMISSION_REQUEST_ID_STATE_KEY] = request_id
        state[ADMISSION_TRACE_ID_STATE_KEY] = trace_id
    return request_id, trace_id


def _request_body_observation(scope: Scope) -> RequestBodyObservation:
    raw = request_body_observation_from_scope(scope)
    request_id, trace_id = _incoming_observation_identifiers(scope)
    state = scope.get("state")
    if isinstance(state, dict):
        current_info = state.get("uni_api_request_info")
        if isinstance(current_info, dict):
            request_id = (
                _safe_correlation_id(current_info.get("request_id")) or request_id
            )
            trace_id = (
                _safe_correlation_id(current_info.get("trace_id")) or trace_id
            )
    return RequestBodyObservation(
        request_id=request_id,
        trace_id=trace_id,
        method=_bounded_observation_text(scope.get("method"), max_length=16),
        path=_bounded_observation_text(scope.get("path")),
        declared_content_length_bytes=raw.get("declared_content_length_bytes"),
        wire_bytes=raw.get("wire_bytes", 0),
        decoded_bytes=raw.get("decoded_bytes", 0),
        decoder_workspace_bytes=raw.get("decoder_workspace_bytes", 0),
        json_raw_bytes=raw.get("json_raw_bytes"),
        json_structural_item_count=raw.get("json_structural_item_count"),
        json_depth=raw.get("json_depth"),
        json_peak_depth=raw.get("json_peak_depth"),
        json_scalar_bytes=raw.get("json_scalar_bytes"),
        json_estimated_bytes=raw.get("json_estimated_bytes"),
        json_raw_memory_multiplier=raw.get("json_raw_memory_multiplier"),
        json_structural_item_memory_bytes=raw.get(
            "json_structural_item_memory_bytes"
        ),
    )


async def _finish_ownership_cleanup(
    cleanup_task: asyncio.Task[None],
) -> None:
    """Run a complete ownership transaction despite repeated cancellation."""

    pending_cancel: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            pending_cancel = pending_cancel or exc
    cleanup_task.result()
    if pending_cancel is not None:
        raise pending_cancel


class RequestAdmissionMiddleware:
    """Bound requests before they can allocate application-owned resources.

    This is deliberately a pure ASGI middleware. Awaiting the downstream app
    therefore covers the complete response body lifecycle, including a
    streaming response and its disconnect cleanup.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        controller: RequestAdmissionController,
        bypass: Callable[[Scope], bool] | None = None,
        resource_only_body_admission: Callable[[Scope], bool] | None = None,
        retry_after_seconds: int = 1,
        on_rejection: Callable[[Scope, AdmissionRejected, float], Any] | None = None,
        on_early_response: Callable[[Scope, int, str], Any] | None = None,
        on_rejection_response_write: Callable[
            [Scope, AdmissionRejected, bool], Any
        ]
        | None = None,
    ) -> None:
        if retry_after_seconds <= 0:
            raise ValueError("retry_after_seconds must be greater than zero")
        self.app = app
        self.controller = controller
        self.bypass = bypass or (lambda scope: False)
        self.resource_only_body_admission = (
            resource_only_body_admission or (lambda scope: False)
        )
        self.retry_after_seconds = retry_after_seconds
        self.on_rejection = on_rejection
        self.on_early_response = on_early_response
        self.on_rejection_response_write = on_rejection_response_write

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self.bypass(scope):
            # A bypassed request owns no admission lease.  Explicitly mask an
            # inherited ContextVar value so health/observability work spawned
            # from another request cannot charge a lease that is already in
            # release.  Reset afterwards to preserve the caller's context.
            bypass_context_token = bind_request_admission_lease(None)
            try:
                await self.app(scope, receive, send)
            finally:
                reset_request_admission_lease(bypass_context_token)
            return

        response_started = False
        response_completed = False
        _incoming_observation_identifiers(scope)
        initialize_request_body_observation(scope)

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            nonlocal response_completed
            if message["type"] == "http.response.start":
                response_started = True
            final_response_body = bool(
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            )
            await send(message)
            if final_response_body:
                response_completed = True
                if lease is not None:
                    lease.finish_pending_response_attempt(
                        outcome="stream_completed"
                    )

        lease = None
        lease_context_token = None
        replayed_messages: deque[tuple[Message, int]] = deque()
        handed_off_receive_task: asyncio.Task[Message] | None = None
        available_replayed_credit = 0
        pre_reserved_wire_credit = 0
        admission_started_at = monotonic()
        release_reason = "request_completed"
        resource_only_body_admission = bool(
            self.resource_only_body_admission(scope)
        )

        def no_response_expected_disconnect() -> bool:
            state = scope.get("state")
            disconnect_event = (
                state.get(DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY)
                if isinstance(state, dict)
                else None
            )
            return (
                release_reason == "request_body_disconnected"
                and not response_started
                and isinstance(disconnect_event, asyncio.Event)
                and disconnect_event.is_set()
            )

        try:
            if resource_only_body_admission:
                lease = await self.controller.try_acquire(
                    resource_only_body_admission=True,
                )
            else:
                lease = await self.controller.try_acquire()
            if lease is None:
                (
                    lease,
                    replayed_messages,
                    handed_off_receive_task,
                    disconnected_while_queued,
                ) = await self._acquire_queued_request(
                    scope,
                    receive,
                    resource_only_body_admission=(
                        resource_only_body_admission
                    ),
                )
                if disconnected_while_queued:
                    state = scope.setdefault("state", {})
                    state[ADMISSION_WAIT_MS_STATE_KEY] = (
                        monotonic() - admission_started_at
                    ) * 1000.0
                    await self._notify_early_response(
                        scope,
                        499,
                        "disconnected_while_queued",
                    )
                    return
                assert lease is not None
            lease_context_token = bind_request_admission_lease(lease)
            state: dict[str, Any] = scope.setdefault("state", {})
            state[ADMISSION_LEASE_STATE_KEY] = lease
            state[ADMISSION_WAIT_MS_STATE_KEY] = lease.wait_ms
            lease.set_fairness_key(_admission_fairness_key(scope))
            lease.observe_body(_request_body_observation(scope))
            declared_content_length = _valid_declared_content_length(scope)
            if declared_content_length is not None:
                already_reserved = lease.reserved_body_bytes
                declared_memory_target = _declared_memory_target(
                    scope,
                    declared_content_length,
                )
                additional_declared = max(
                    0,
                    declared_memory_target - already_reserved,
                )
                if additional_declared:
                    await lease.reserve_body_bytes(
                        additional_declared,
                        observation=_request_body_observation(scope),
                    )
                pre_reserved_wire_credit = max(
                    0,
                    declared_content_length - already_reserved,
                )
                available_replayed_credit += max(
                    0,
                    declared_memory_target - declared_content_length,
                )

            async def reserve_body_bytes(additional_bytes: int) -> int:
                nonlocal available_replayed_credit
                if int(additional_bytes) < 0:
                    raise ValueError("additional_bytes cannot be negative")
                credited = min(
                    max(0, int(additional_bytes)),
                    available_replayed_credit,
                )
                available_replayed_credit -= credited
                remaining = int(additional_bytes) - credited
                observation = _request_body_observation(scope)
                if remaining == 0:
                    lease.observe_body(observation)
                    return lease.reserved_body_bytes
                return await lease.reserve_body_bytes(
                    remaining,
                    observation=observation,
                )

            state[RESERVE_BODY_BYTES_STATE_KEY] = reserve_body_bytes

            async def release_body_bytes(released_bytes: int) -> int:
                return await lease.release_body_bytes(int(released_bytes))

            state[RELEASE_BODY_BYTES_STATE_KEY] = release_body_bytes
            state[BODY_REJECTION_RECORDER_SCOPE_KEY] = (
                self.controller.record_rejection
            )

            async def observe_body_early_response(
                status_code: int,
                reason: str,
            ) -> None:
                nonlocal release_reason
                normalized_reason = str(reason or "request_body_early_response")
                release_reason = normalized_reason
                # A terminal parse/size decision may update wire/decoded/JSON
                # facts without adding another weighted byte. Refresh holder
                # identity and facts before its eventual release event.
                lease.observe_body(_request_body_observation(scope))
                await self._notify_early_response(scope, status_code, reason)

            state[BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY] = (
                observe_body_early_response
            )

            async def replay_receive() -> Message:
                nonlocal available_replayed_credit
                nonlocal handed_off_receive_task
                nonlocal pre_reserved_wire_credit
                if replayed_messages:
                    message, credited_bytes = replayed_messages.popleft()
                    available_replayed_credit += credited_bytes
                    return message
                if handed_off_receive_task is not None:
                    task = handed_off_receive_task
                    handed_off_receive_task = None
                    message = await task
                else:
                    message = await receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"") or b""
                    if body:
                        observe_request_wire_bytes(scope, len(body))
                        credited = min(len(body), pre_reserved_wire_credit)
                        pre_reserved_wire_credit -= credited
                        additional_wire = len(body) - credited
                        if additional_wire:
                            await lease.reserve_body_bytes(
                                additional_wire,
                                observation=_request_body_observation(scope),
                            )
                        available_replayed_credit += len(body)
                    if (
                        not message.get("more_body", False)
                        and pre_reserved_wire_credit
                    ):
                        unused_credit = pre_reserved_wire_credit
                        pre_reserved_wire_credit = 0
                        await lease.release_body_bytes(unused_credit)
                return message

            await self.app(scope, replay_receive, tracking_send)
        except AdmissionRejected as exc:
            rejection_reason = str(exc.reason or "admission_rejected")
            release_reason = rejection_reason
            if response_started:
                raise
            wait_ms = (
                lease.wait_ms
                if lease is not None
                else (monotonic() - admission_started_at) * 1000.0
            )
            await self._notify_rejection(scope, exc, wait_ms)
            try:
                write_completed = await self._send_rejection(
                    scope,
                    receive,
                    send,
                    exc,
                )
            except BaseException:
                release_reason = f"{rejection_reason}_response_write_failed"
                raise
            release_reason = (
                f"{rejection_reason}_response_written"
                if write_completed
                else f"{rejection_reason}_response_incomplete"
            )
        except asyncio.CancelledError:
            if release_reason == "request_completed":
                release_reason = "request_cancelled"
            elif not no_response_expected_disconnect():
                release_reason = f"{release_reason}_response_write_cancelled"
            raise
        except BaseException:
            if release_reason == "request_completed":
                release_reason = "request_error"
            elif not no_response_expected_disconnect():
                release_reason = f"{release_reason}_response_write_failed"
            raise
        finally:
            state = scope.get("state")
            disconnect_event = (
                state.get(DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY)
                if isinstance(state, dict)
                else None
            )
            if (
                release_reason == "request_completed"
                and isinstance(disconnect_event, asyncio.Event)
                and disconnect_event.is_set()
            ):
                release_reason = "downstream_disconnected"
            abandoned_receive = handed_off_receive_task
            handed_off_receive_task = None

            async def cleanup_request_ownership() -> None:
                try:
                    if abandoned_receive is not None:
                        if not abandoned_receive.done():
                            abandoned_receive.cancel()
                        try:
                            abandoned_message = await abandoned_receive
                            abandoned_message = None
                        except (asyncio.CancelledError, Exception):
                            pass
                finally:
                    # Drop body aliases and callbacks before returning their
                    # byte/accounting owner.  The state mapping can outlive the
                    # downstream app until the ASGI scope itself is discarded.
                    replayed_messages.clear()
                    request_state = scope.get("state")
                    if isinstance(request_state, dict):
                        request_state.pop(ADMISSION_LEASE_STATE_KEY, None)
                        request_state.pop(RESERVE_BODY_BYTES_STATE_KEY, None)
                        request_state.pop(RELEASE_BODY_BYTES_STATE_KEY, None)
                        request_state.pop(BODY_REJECTION_RECORDER_SCOPE_KEY, None)
                        request_state.pop(
                            BODY_EARLY_RESPONSE_OBSERVER_SCOPE_KEY,
                            None,
                        )
                    if lease is not None:
                        if response_completed:
                            pending_outcome = "stream_completed"
                        elif (
                            isinstance(disconnect_event, asyncio.Event)
                            and disconnect_event.is_set()
                        ):
                            pending_outcome = "downstream_disconnected"
                        elif "cancel" in release_reason:
                            pending_outcome = "cancelled_or_consumer_closed"
                        else:
                            pending_outcome = "stream_failed"
                        lease.finish_pending_response_attempt(
                            outcome=pending_outcome
                        )
                        await lease.release(reason=release_reason)

            cleanup_task = asyncio.create_task(cleanup_request_ownership())
            try:
                await _finish_ownership_cleanup(cleanup_task)
            finally:
                if lease_context_token is not None:
                    reset_request_admission_lease(lease_context_token)

    async def _acquire_queued_request(
        self,
        scope: Scope,
        receive: Receive,
        *,
        resource_only_body_admission: bool = False,
    ) -> tuple[
        Any,
        deque[tuple[Message, int]],
        asyncio.Task[Message] | None,
        bool,
    ]:
        """Observe a queued peer without allowing an unbounded body prebuffer."""

        # Establish control-memory ownership before starting any receive. A
        # request waiting in legacy mode must not allocate body chunks outside
        # its bounded queue; weighted mode normally resolves this immediately.
        if resource_only_body_admission:
            acquire_task = await self.controller.begin_acquire(
                resource_only_body_admission=True,
            )
        else:
            acquire_task = await self.controller.begin_acquire()
        receive_task: asyncio.Task[Message] | None = (
            None if acquire_task.done() else asyncio.create_task(receive())
        )
        state = scope.get("state")
        disconnect_event = (
            state.get(DOWNSTREAM_DISCONNECT_EVENT_SCOPE_KEY)
            if isinstance(state, dict)
            else None
        )
        disconnect_task: asyncio.Task[bool] | None = (
            asyncio.create_task(disconnect_event.wait())
            if isinstance(disconnect_event, asyncio.Event)
            and not acquire_task.done()
            else None
        )
        buffered: deque[tuple[Message, int]] = deque()
        pending = self.controller.pending_body_reservation(
            fairness_key=_admission_fairness_key(scope),
            resource_only_body_admission=resource_only_body_admission,
        )
        lease = None
        queued_body_complete = False

        async def cancel_acquire_and_release_result(release_reason: str) -> None:
            # Cancellation can lose the race with a just-granted lease.  Always
            # consume the task result and release any transferred ownership;
            # otherwise a same-turn peer disconnect strands an active slot.
            if not acquire_task.done():
                acquire_task.cancel()
            try:
                granted_lease = await acquire_task
            except (asyncio.CancelledError, AdmissionRejected):
                return
            await granted_lease.release(reason=release_reason)

        async def cancel_disconnect_observer(
            task: asyncio.Task[bool],
        ) -> None:
            async def cancel_and_consume() -> None:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            cleanup_task = asyncio.create_task(cancel_and_consume())
            await _finish_ownership_cleanup(cleanup_task)

        async def cleanup_queued_ownership(
            receive_to_cleanup: asyncio.Task[Message] | None,
            disconnect_to_cleanup: asyncio.Task[bool] | None,
            known_lease: Any,
            release_reason: str,
        ) -> None:
            """Release every queued owner even if an earlier step fails."""

            try:
                if receive_to_cleanup is not None:
                    if not receive_to_cleanup.done():
                        receive_to_cleanup.cancel()
                    try:
                        abandoned_message = await receive_to_cleanup
                        abandoned_message = None
                    except (asyncio.CancelledError, Exception):
                        pass
            finally:
                try:
                    if disconnect_to_cleanup is not None:
                        if not disconnect_to_cleanup.done():
                            disconnect_to_cleanup.cancel()
                        try:
                            await disconnect_to_cleanup
                        except (asyncio.CancelledError, Exception):
                            pass
                finally:
                    try:
                        await cancel_acquire_and_release_result(release_reason)
                    finally:
                        try:
                            if known_lease is not None:
                                await known_lease.release(reason=release_reason)
                        finally:
                            buffered.clear()
                            await pending.release(reason=release_reason)

        async def finish_queued_ownership_cleanup(
            receive_to_cleanup: asyncio.Task[Message] | None,
            disconnect_to_cleanup: asyncio.Task[bool] | None,
            known_lease: Any,
            release_reason: str,
        ) -> None:
            cleanup_task = asyncio.create_task(
                cleanup_queued_ownership(
                    receive_to_cleanup,
                    disconnect_to_cleanup,
                    known_lease,
                    release_reason,
                )
            )
            await _finish_ownership_cleanup(cleanup_task)

        try:
            while True:
                wait_tasks: set[asyncio.Task[Any]] = {acquire_task}
                if receive_task is not None:
                    wait_tasks.add(receive_task)
                if disconnect_task is not None:
                    wait_tasks.add(disconnect_task)
                done, _ = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # A transport close owns a same-turn race with admission.  If
                # the gate already granted, cleanup consumes and releases that
                # lease before returning the disconnected outcome.
                if disconnect_task is not None and disconnect_task in done:
                    disconnected_wait = disconnect_task
                    disconnect_task = None
                    abandoned_receive = receive_task
                    receive_task = None
                    await finish_queued_ownership_cleanup(
                        abandoned_receive,
                        disconnected_wait,
                        lease,
                        "downstream_disconnected_while_queued",
                    )
                    return None, buffered, None, True

                if acquire_task in done:
                    lease = acquire_task.result()
                    if receive_task is not None:
                        if receive_task.done():
                            message = await receive_task
                            receive_task = None
                            if message.get("type") == "http.disconnect":
                                message = None
                                await finish_queued_ownership_cleanup(
                                    None,
                                    disconnect_task,
                                    lease,
                                    "downstream_disconnected_while_queued",
                                )
                                disconnect_task = None
                                lease = None
                                return None, buffered, None, True
                            if not queued_body_complete:
                                await self._buffer_pending_message(
                                    scope,
                                    pending,
                                    buffered,
                                    message,
                                )
                            receive_task = None
                    await pending.transfer_to(lease)
                    if (
                        isinstance(disconnect_event, asyncio.Event)
                        and disconnect_event.is_set()
                    ):
                        abandoned_receive = receive_task
                        receive_task = None
                        await finish_queued_ownership_cleanup(
                            abandoned_receive,
                            disconnect_task,
                            lease,
                            "downstream_disconnected_while_queued",
                        )
                        disconnect_task = None
                        lease = None
                        return None, buffered, None, True
                    if disconnect_task is not None:
                        await cancel_disconnect_observer(disconnect_task)
                        disconnect_task = None
                    # Cancellation of event.wait() yields once.  Recheck the
                    # transport-owned event so a close in that turn still owns
                    # the admission race.
                    if (
                        isinstance(disconnect_event, asyncio.Event)
                        and disconnect_event.is_set()
                    ):
                        abandoned_receive = receive_task
                        receive_task = None
                        await finish_queued_ownership_cleanup(
                            abandoned_receive,
                            None,
                            lease,
                            "downstream_disconnected_while_queued",
                        )
                        lease = None
                        return None, buffered, None, True
                    return lease, buffered, receive_task, False

                assert receive_task is not None and receive_task in done
                message = receive_task.result()
                receive_task = None
                if message.get("type") == "http.disconnect":
                    message = None
                    await finish_queued_ownership_cleanup(
                        None,
                        disconnect_task,
                        lease,
                        "downstream_disconnected_while_queued",
                    )
                    disconnect_task = None
                    return None, buffered, None, True

                if not queued_body_complete:
                    await self._buffer_pending_message(
                        scope,
                        pending,
                        buffered,
                        message,
                    )
                if (
                    not queued_body_complete
                    and
                    message.get("type") == "http.request"
                    and not message.get("more_body", False)
                ):
                    queued_body_complete = True
                if (
                    not queued_body_complete
                    and len(buffered)
                    >= ADMISSION_PREBUFFER_MESSAGE_HIGH_WATERMARK
                ):
                    # ASGI body frame boundaries are transport details, not a
                    # measure of request size or memory pressure.  Pause the
                    # receive side at a bounded message high-water mark instead
                    # of rejecting an otherwise valid request.  The queued
                    # acquisition task still enforces its normal wait timeout;
                    # once admitted, replay_receive drains these frames and
                    # resumes the remaining body under the active lease.
                    continue
                # Even after the final body frame, keep exactly one receive
                # pending so a peer that disconnects while still queued is
                # observed before the application is admitted.
                receive_task = asyncio.create_task(receive())
        except BaseException as exc:
            abandoned_receive = receive_task
            receive_task = None
            abandoned_disconnect = disconnect_task
            disconnect_task = None
            if isinstance(exc, asyncio.CancelledError):
                queued_release_reason = "queued_request_cancelled"
            elif isinstance(exc, AdmissionRejected):
                queued_release_reason = f"queued_{exc.reason}"
            else:
                queued_release_reason = "queued_request_error"
            await finish_queued_ownership_cleanup(
                abandoned_receive,
                abandoned_disconnect,
                lease,
                queued_release_reason,
            )
            lease = None
            raise

    @staticmethod
    async def _buffer_pending_message(
        scope: Scope,
        pending: Any,
        buffered: deque[tuple[Message, int]],
        message: Message,
    ) -> None:
        if message.get("type") != "http.request":
            return
        body = message.get("body", b"") or b""
        if body:
            observe_request_wire_bytes(scope, len(body))
        observation = _request_body_observation(scope)
        if body:
            await pending.reserve(len(body), observation=observation)
        else:
            pending.observe_body(observation)
        buffered.append((message, len(body)))

    async def _notify_rejection(
        self,
        scope: Scope,
        rejection: AdmissionRejected,
        wait_ms: float,
    ) -> None:
        if self.on_rejection is None:
            return
        try:
            result = self.on_rejection(scope, rejection, max(0.0, wait_ms))
            if isawaitable(result):
                await result
        except Exception:
            # Telemetry must never turn a deliberate 413/503 into a 500.
            logger.exception("Failed to record request admission rejection")

    async def _notify_early_response(
        self,
        scope: Scope,
        status_code: int,
        reason: str,
    ) -> None:
        if self.on_early_response is None:
            return
        try:
            result = self.on_early_response(scope, int(status_code), str(reason))
            if isawaitable(result):
                await result
        except Exception:
            logger.exception("Failed to record early request-body response")

    async def _send_rejection(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        rejection: AdmissionRejected,
    ) -> bool:
        status_code = int(rejection.status_code)
        reason = rejection.reason
        if status_code == 413:
            message = "Request body exceeds the configured local limit"
            error_type = "request_too_large"
        else:
            message = "Service is at its bounded local capacity; retry later"
            error_type = "local_overload"

        headers = {"x-uni-api-admission-reason": reason}
        if status_code == 503:
            headers["retry-after"] = str(self.retry_after_seconds)
            headers["x-uni-api-status-origin"] = "ember_local_admission"
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": reason,
                }
            },
            headers=headers,
        )
        terminal_body_written = False

        async def tracking_rejection_send(message: Message) -> None:
            nonlocal terminal_body_written
            await send(message)
            if (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                terminal_body_written = True

        try:
            await response(scope, receive, tracking_rejection_send)
        except BaseException:
            if status_code == 503:
                await self._notify_rejection_response_write(
                    scope,
                    rejection,
                    completed=False,
                )
            raise
        if status_code == 503:
            await self._notify_rejection_response_write(
                scope,
                rejection,
                completed=terminal_body_written,
            )
        return terminal_body_written

    async def _notify_rejection_response_write(
        self,
        scope: Scope,
        rejection: AdmissionRejected,
        *,
        completed: bool,
    ) -> None:
        callback = self.on_rejection_response_write
        if callback is None:
            return
        try:
            result = callback(scope, rejection, bool(completed))
            if isawaitable(result):
                await result
        except Exception:
            # A post-write telemetry failure must never replace the deliberate
            # admission response or make a successful ASGI write look failed.
            logger.exception("Failed to record admission response write outcome")
