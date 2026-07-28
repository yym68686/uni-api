from __future__ import annotations

import asyncio
import errno
import os
from contextlib import suppress
from datetime import datetime, timezone
from time import time
from typing import Any, Awaitable, Callable, Optional

from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from core.log_config import logger
from uni_api.admission import AdmissionRejected, get_request_admission_lease
from uni_api.admission.json_parsing import parse_owned_json_value
from uni_api.observability.spans import merge_timing_spans
from uni_api.observability.responses_stream import (
    ResponsesStreamDiagnostics,
    safe_responses_event_type,
)
from uni_api.http_content import is_json_media_type
from uni_api.serialization import json
from uni_api.streaming.cleanup import call_cleanup_safely
from uni_api.streaming.bounded_queue import ObservedStreamChunk
from uni_api.streaming.error_text import bounded_stream_error_text
from uni_api.streaming.sse import (
    DEFAULT_MAX_PENDING_BYTES,
    IncrementalSSEParser,
    SSEIncompleteEventError,
    SSEProtocolError,
    parse_owned_sse_event,
)
from uni_api.streaming.usage import (
    StreamUsageSnapshot,
    parse_usage_count,
    stream_usage_snapshot,
    stream_usage_snapshot_from_payload,
)
from uni_api.upstream.responses_errors import ResponsesSemanticError


AsyncCloseCallback = Callable[[dict[str, Any]], Awaitable[None]]
_MAX_JSON_USAGE_TELEMETRY_BYTES = 64 * 1024
_MAX_STATS_TEXT_BYTES = 64 * 1024
_MAX_STATS_FIELD_BYTES = 4 * 1024
_STATS_SNAPSHOT_FIELDS = frozenset(
    {
        "request_id",
        "trace_id",
        "endpoint",
        "client_ip",
        "process_time",
        "first_response_time",
        "provider",
        "model",
        "api_key",
        "is_flagged",
        "text",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_price",
        "completion_price",
        "timing_spans",
        "success",
    }
)


def _truncate_utf8_field(value: str, limit_bytes: int) -> str:
    # Slice before encoding so an attacker-sized field never creates an
    # attacker-sized temporary bytes object.  At most four UTF-8 bytes can be
    # emitted per Python character.
    candidate = value[:limit_bytes]
    encoded = candidate.encode("utf-8")
    if len(encoded) <= limit_bytes:
        return candidate
    return encoded[:limit_bytes].decode("utf-8", errors="ignore")


def _bounded_stats_snapshot(current_info: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in _STATS_SNAPSHOT_FIELDS:
        if key not in current_info:
            continue
        value = current_info[key]
        if key == "timing_spans" and isinstance(value, dict):
            try:
                value = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                value = "{}"
        if isinstance(value, str):
            value = _truncate_utf8_field(
                value,
                _MAX_STATS_TEXT_BYTES if key == "text" else _MAX_STATS_FIELD_BYTES,
            )
        elif not isinstance(value, (int, float, bool, type(None))):
            # RequestStat columns are scalar.  Do not detach an arbitrary
            # request-owned graph behind a timed-out database task.
            value = _truncate_utf8_field(str(value), _MAX_STATS_FIELD_BYTES)
        snapshot[key] = value
    return snapshot


class DownstreamWriteTimeout(TimeoutError):
    pass


class DownstreamDisconnected(ConnectionError):
    """A socket-like error observed specifically while calling ASGI send."""


class DownstreamSendError(RuntimeError):
    """A non-disconnect exception raised at the downstream ASGI boundary."""


def _positive_timeout_from_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(0.001, value)


def _positive_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1, value)


try:
    _TIMED_OUT_IO_TASK_LIMIT = max(
        1,
        int(os.getenv("TIMED_OUT_IO_TASK_LIMIT", "128")),
    )
except (TypeError, ValueError):
    _TIMED_OUT_IO_TASK_LIMIT = 128
_TIMED_OUT_IO_TASKS: set[asyncio.Task[Any]] = set()


def timed_out_io_task_snapshot() -> dict[str, int]:
    return {
        "pending": sum(not task.done() for task in _TIMED_OUT_IO_TASKS),
        "total": len(_TIMED_OUT_IO_TASKS),
        "capacity": _TIMED_OUT_IO_TASK_LIMIT,
    }


def _consume_timed_out_task(task: asyncio.Task[Any], *, label: str) -> None:
    _TIMED_OUT_IO_TASKS.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except BaseException as exc:
        logger.debug(
            "%s finished after timeout with %s",
            label,
            type(exc).__name__,
        )


async def _cancel_or_bound_detach(task: asyncio.Task[Any], *, label: str) -> None:
    if not task.done():
        task.cancel()
    if task.done():
        _consume_timed_out_task(task, label=label)
        return
    if len(_TIMED_OUT_IO_TASKS) < _TIMED_OUT_IO_TASK_LIMIT:
        _TIMED_OUT_IO_TASKS.add(task)
        task.add_done_callback(
            lambda completed: _consume_timed_out_task(
                completed,
                label=label,
            )
        )
        return
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    _consume_timed_out_task(task, label=label)


async def _await_with_hard_deadline(
    awaitable: Awaitable[Any],
    *,
    timeout: float,
    label: str,
) -> Any:
    task = asyncio.create_task(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        await _cancel_or_bound_detach(task, label=label)
        raise
    if task in done:
        return task.result()

    await _cancel_or_bound_detach(task, label=label)
    raise asyncio.TimeoutError(f"{label} exceeded {timeout:g} seconds")


async def await_with_hard_deadline(
    awaitable: Awaitable[Any],
    *,
    timeout: float,
    label: str,
) -> Any:
    return await _await_with_hard_deadline(
        awaitable,
        timeout=timeout,
        label=label,
    )


class _HardDeadlineASGIWriter:
    """Serialize ASGI writes through one detachable task per response.

    A separate task is still required for the hard-deadline contract: an ASGI
    adapter may suppress cancellation while blocked in a socket write.  Reuse
    that task for the whole response instead of allocating a task and an
    ``asyncio.wait`` waiter for every SSE frame.
    """

    def __init__(
        self,
        send: Send,
        *,
        timeout: float,
        label: str,
    ) -> None:
        self._send = send
        self._timeout = timeout
        self._label = label
        self._queue: asyncio.Queue[
            tuple[dict[str, Any], asyncio.Future[None]] | None
        ] = asyncio.Queue(maxsize=1)
        self._task = asyncio.create_task(
            self._run(),
            name="uni-api-downstream-asgi-writer",
        )
        self._closed = False
        self._aborted = False

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            message, completion = item
            try:
                await self._send(message)
            except asyncio.CancelledError:
                if not completion.done():
                    completion.cancel()
                return
            except BaseException as exc:
                if not completion.done():
                    completion.set_exception(exc)
                return
            if completion.done():
                # The caller timed out or was cancelled while the underlying
                # adapter ignored cancellation.  Do not leave a detached
                # writer waiting forever for another queue item.
                return
            completion.set_result(None)

    @staticmethod
    def _expire(completion: asyncio.Future[None], message: str) -> None:
        if not completion.done():
            completion.set_exception(asyncio.TimeoutError(message))

    async def write(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("downstream ASGI writer is closed")
        if self._task.done():
            self._task.result()
            raise RuntimeError("downstream ASGI writer stopped unexpectedly")

        loop = asyncio.get_running_loop()
        completion: asyncio.Future[None] = loop.create_future()
        deadline = loop.call_later(
            self._timeout,
            self._expire,
            completion,
            f"{self._label} exceeded {self._timeout:g} seconds",
        )
        self._queue.put_nowait((message, completion))
        try:
            await completion
        except BaseException:
            self._aborted = True
            await _cancel_or_bound_detach(self._task, label=self._label)
            raise
        finally:
            deadline.cancel()

    async def close(self) -> asyncio.CancelledError | None:
        if self._closed:
            return None
        self._closed = True
        if self._aborted:
            return None
        if not self._task.done():
            self._queue.put_nowait(None)

        pending_cancel: asyncio.CancelledError | None = None
        while not self._task.done():
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError as exc:
                pending_cancel = pending_cancel or exc
        self._task.result()
        return pending_cancel


class LoggingStreamingResponse(Response):
    """A streaming response whose observability lifetime matches the ASGI body.

    Starlette cannot change an HTTP status after ``http.response.start``.  Stream
    failures are therefore represented by ``stream_outcome``/``error_type`` and,
    for SSE responses only, an in-band error event.  Binary responses are never
    modified with text error frames.
    """

    def __init__(
        self,
        content,
        status_code=200,
        headers=None,
        media_type=None,
        current_info=None,
        *,
        mark_first_byte_observed: Optional[Callable[[dict[str, Any]], None]] = None,
        emit_request_observability: Optional[Callable[[dict[str, Any]], None]] = None,
        update_stats: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        trace_type: Optional[type] = None,
        debug: bool = False,
        disconnect_event: Optional[asyncio.Event] = None,
        lifecycle_close: Optional[AsyncCloseCallback] = None,
        observe_usage: bool = True,
        usage_buffer_limit_bytes: int = DEFAULT_MAX_PENDING_BYTES,
        downstream_write_timeout_seconds: Optional[float] = None,
        stats_write_timeout_seconds: Optional[float] = None,
        downstream_chunk_bytes: Optional[int] = None,
    ):
        had_explicit_content_length = bool(
            headers is not None
            and any(str(name).lower() == "content-length" for name in headers)
        )
        super().__init__(content=None, status_code=status_code, headers=headers, media_type=media_type)
        if usage_buffer_limit_bytes <= 0:
            raise ValueError("usage_buffer_limit_bytes must be greater than zero")

        self.body_iterator = content
        self.current_info = current_info or {}
        responses_tracker = self.current_info.get(
            "_responses_stream_diagnostics_tracker"
        )
        self._responses_diagnostics_tracker = (
            responses_tracker
            if isinstance(responses_tracker, ResponsesStreamDiagnostics)
            else None
        )
        request_kind = str(self.current_info.get("request_kind") or "")
        endpoint = str(self.current_info.get("endpoint") or "")
        self._responses_tracker_may_attach = bool(
            self._responses_diagnostics_tracker is not None
            or request_kind.rstrip("/")
            in {"/v1/responses", "/v1/responses/compact"}
            or endpoint.rstrip("/").endswith(
                ("/v1/responses", "/v1/responses/compact")
            )
        )
        self._asgi_phase_tracker = (
            self._responses_diagnostics_tracker
            if self._responses_diagnostics_tracker is not None
            and self._responses_diagnostics_tracker.phase_sampled
            else None
        )
        if self._asgi_phase_tracker is not None:
            self._send_with_deadline = self._send_with_deadline_profiled
        elif self._responses_diagnostics_tracker is None and (
            self._responses_tracker_may_attach
        ):
            self._send_with_deadline = self._send_with_deadline_resolving
        else:
            self._send_with_deadline = self._send_with_deadline_unprofiled
        self._mark_first_byte_observed = mark_first_byte_observed or (lambda current_info: None)
        self._emit_request_observability = emit_request_observability or (lambda current_info: None)
        self._update_stats = update_stats
        self._trace_type = trace_type
        self._debug = debug
        self._disconnect_event = disconnect_event
        self._lifecycle_close = lifecycle_close
        self._usage_observation_enabled = bool(observe_usage)
        self._usage_buffer_limit_bytes = usage_buffer_limit_bytes
        self._usage_json_buffer_limit_bytes = min(
            usage_buffer_limit_bytes,
            _MAX_JSON_USAGE_TELEMETRY_BYTES,
        )
        self._downstream_write_timeout_seconds = (
            _positive_timeout_from_env(
                "DOWNSTREAM_WRITE_TIMEOUT_SECONDS",
                30.0,
            )
            if downstream_write_timeout_seconds is None
            else max(0.001, float(downstream_write_timeout_seconds))
        )
        self._stats_write_timeout_seconds = (
            _positive_timeout_from_env(
                "REQUEST_STATS_WRITE_TIMEOUT_SECONDS",
                5.0,
            )
            if stats_write_timeout_seconds is None
            else max(0.001, float(stats_write_timeout_seconds))
        )
        self._downstream_chunk_bytes = (
            _positive_int_from_env(
                "DOWNSTREAM_WRITE_CHUNK_BYTES",
                256 * 1024,
            )
            if downstream_chunk_bytes is None
            else max(1, int(downstream_chunk_bytes))
        )

        self._is_sse_response = self._content_type().lower().startswith("text/event-stream")
        self._is_json_response = is_json_media_type(self._content_type())
        self._usage_sse_parser = (
            IncrementalSSEParser(
                max_pending_bytes=usage_buffer_limit_bytes,
                max_event_bytes=usage_buffer_limit_bytes,
            )
            if self._usage_observation_enabled and self._is_sse_response
            else None
        )
        self._usage_json_buffer = bytearray()
        self._usage_json_reservation = None
        self._usage_parser_disabled = False
        self._wire_sse_boundary_known = self._is_sse_response
        self._wire_sse_at_event_boundary = self._is_sse_response
        self._metadata_sse_event_in_progress = False
        self._metadata_sse_pending_bytes = 0

        self._body_closed = False
        self._body_close_lock = asyncio.Lock()
        self._body_close_task: Optional[asyncio.Task] = None
        self._finalized = False
        self._finalize_lock = asyncio.Lock()
        self._stream_task: Optional[asyncio.Task] = None
        self._downstream_writer: _HardDeadlineASGIWriter | None = None
        diagnostics = self.current_info.get("responses_stream_diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["downstream_usage_observer_status"] = (
                "active" if self._usage_observation_enabled else "not_applicable"
            )
            if not self._usage_observation_enabled:
                diagnostics["downstream_usage_observer_reason"] = "disabled_by_policy"
            diagnostics.setdefault("downstream_usage_object_seen", False)
            diagnostics.setdefault("downstream_usage_counters_seen", False)
            diagnostics.setdefault("downstream_usage_input_known", False)
            diagnostics.setdefault("downstream_usage_output_known", False)
            diagnostics.setdefault("downstream_usage_total_known", False)
            diagnostics.setdefault("downstream_usage_seen", False)
            diagnostics.setdefault("response_start_asgi_write_attempted", False)
            diagnostics.setdefault("response_start_asgi_write_completed", False)
            diagnostics.setdefault("downstream_final_body_attempted", False)
            diagnostics.setdefault("downstream_final_body_completed", False)
            diagnostics.setdefault(
                "downstream_final_body_outcome",
                "not_attempted",
            )

        # Response(content=None) synthesizes Content-Length: 0. Remove only
        # that synthetic value for genuinely streaming bodies; preserve an
        # explicit upstream/application length for fixed JSON, binary, and
        # HEAD semantics.
        if not had_explicit_content_length and "content-length" in self.headers:
            del self.headers["content-length"]
        # ASGI servers choose HTTP/1.1 chunking themselves. Emitting this
        # hop-by-hop header here is invalid when the downstream uses HTTP/2.
        if "transfer-encoding" in self.headers:
            del self.headers["transfer-encoding"]

    def _resolve_responses_diagnostics_tracker(
        self,
    ) -> ResponsesStreamDiagnostics | None:
        tracker = self._responses_diagnostics_tracker
        if tracker is not None or not self._responses_tracker_may_attach:
            return tracker
        candidate = self.current_info.get(
            "_responses_stream_diagnostics_tracker"
        )
        if isinstance(candidate, ResponsesStreamDiagnostics):
            self._responses_diagnostics_tracker = candidate
            self._asgi_phase_tracker = (
                candidate if candidate.phase_sampled else None
            )
            self._send_with_deadline = (
                self._send_with_deadline_profiled
                if self._asgi_phase_tracker is not None
                else self._send_with_deadline_unprofiled
            )
            return candidate
        return None

    def _content_type(self) -> str:
        return self.headers.get("content-type", self.media_type or "")

    def _is_trace(self, value: Any) -> bool:
        return self._trace_type is not None and isinstance(value, self._trace_type)

    @staticmethod
    def _parse_usage_count(value: Any) -> tuple[int, bool]:
        return parse_usage_count(value)

    def _record_usage(self, usage_obj: Any) -> bool:
        return self._record_usage_snapshot(stream_usage_snapshot(usage_obj))

    def _record_usage_snapshot(
        self,
        snapshot: StreamUsageSnapshot | None,
    ) -> bool:
        if snapshot is None:
            return False
        diagnostics = self.current_info.get("responses_stream_diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["downstream_usage_object_seen"] = True
        if not snapshot.counters_seen:
            return False
        if isinstance(diagnostics, dict):
            diagnostics["downstream_usage_counters_seen"] = True
        if isinstance(diagnostics, dict):
            diagnostics["downstream_usage_input_known"] = snapshot.input_known
            diagnostics["downstream_usage_output_known"] = snapshot.output_known
            diagnostics["downstream_usage_total_known"] = snapshot.total_known
            diagnostics["downstream_usage_values_valid"] = snapshot.values_valid
            diagnostics["downstream_usage_alias_consistent"] = (
                snapshot.alias_consistent
            )
        if snapshot.parse_error is not None:
            self.current_info["usage_parse_error"] = snapshot.parse_error
            return False

        if snapshot.input_known and snapshot.prompt_tokens is not None:
            self.current_info["prompt_tokens"] = snapshot.prompt_tokens
        if snapshot.output_known and snapshot.completion_tokens is not None:
            self.current_info["completion_tokens"] = snapshot.completion_tokens
        if snapshot.total_known and snapshot.total_tokens is not None:
            self.current_info["total_tokens"] = snapshot.total_tokens
        if not snapshot.complete:
            if isinstance(diagnostics, dict):
                diagnostics["downstream_usage_completeness"] = "incomplete"
            return False
        self.current_info["usage_seen"] = True
        if isinstance(diagnostics, dict):
            diagnostics["downstream_usage_seen"] = True
            diagnostics["downstream_usage_completeness"] = "complete"
        return True

    def _record_usage_from_payload(self, payload: Any) -> bool:
        return self._record_usage_snapshot(
            stream_usage_snapshot_from_payload(payload)
        )

    async def _record_usage_from_data(self, data: str) -> bool:
        data = data.strip()
        if not data or data.startswith("[DONE]") or data.startswith("OK") or "\"usage\"" not in data:
            return False
        if data.startswith("data:"):
            data = data.removeprefix("data:").lstrip()
        if not (data.startswith("{") or data.startswith("[")):
            return False
        owner = None
        payload = None
        try:
            owner = await parse_owned_json_value(data)
            payload = owner.value
            return self._record_usage_from_payload(payload)
        except Exception:
            return False
        finally:
            payload = None
            if owner is not None:
                await owner.aclose()

    async def _record_sse_usage_event(self, raw_event: str) -> None:
        # Token deltas dominate stream event volume.  Avoid JSON parsing and a
        # thread hop unless the bounded SSE frame can actually contain usage.
        if '"usage"' not in raw_event:
            return
        owner = None
        payload = None
        try:
            owner = await parse_owned_sse_event(
                raw_event,
                max_event_bytes=self._usage_buffer_limit_bytes,
            )
            payload = owner.payload
            if isinstance(payload, dict):
                self._record_usage_from_payload(payload)
        finally:
            payload = None
            if owner is not None:
                await owner.aclose()

    async def _release_usage_json_reservation(self) -> None:
        reservation = self._usage_json_reservation
        self._usage_json_reservation = None
        if reservation is not None:
            await reservation.release()

    async def _disable_usage_parser(self, exc: BaseException) -> None:
        if self._usage_parser_disabled:
            return
        self._usage_parser_disabled = True
        self._usage_sse_parser = None
        self._usage_json_buffer.clear()
        await self._release_usage_json_reservation()
        self.current_info["usage_parse_error"] = type(exc).__name__
        diagnostics = self.current_info.get("responses_stream_diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["downstream_usage_observer_status"] = "disabled"
            diagnostics["downstream_usage_observer_error_type"] = type(exc).__name__
        logger.warning(
            "Disabled bounded streaming usage parser: %s: %s",
            type(exc).__name__,
            exc,
        )

    def _record_downstream_sse_event_sent(
        self,
        event_type: str | None,
        *,
        semantic_outcome: str | None,
    ) -> None:
        if not event_type and not semantic_outcome:
            return
        diagnostics = self.current_info.get("responses_stream_diagnostics")
        if not isinstance(diagnostics, dict):
            return
        if event_type:
            event_type = safe_responses_event_type(event_type)
            diagnostics["downstream_declared_terminal_type"] = event_type
        if semantic_outcome == "completed":
            diagnostics["downstream_terminal_seen"] = True
            diagnostics["downstream_semantic_status"] = "completed"
        elif semantic_outcome == "incomplete":
            diagnostics["downstream_terminal_seen"] = True
            diagnostics["downstream_semantic_status"] = "incomplete"
        elif semantic_outcome == "failed":
            diagnostics["downstream_terminal_seen"] = True
            diagnostics["downstream_semantic_status"] = "failed"
        if event_type == "error" or semantic_outcome == "error":
            diagnostics["error_event_seen"] = True
        if diagnostics.get("downstream_terminal_seen") or semantic_outcome == "error":
            tracker = self._resolve_responses_diagnostics_tracker()
            if tracker is not None and tracker.facts.get(
                "upstream_terminal_seen"
            ):
                tracker.mark_terminal_asgi_write_completed()
            else:
                diagnostics["downstream_terminal_asgi_write_completed"] = True
                diagnostics["downstream_terminal_asgi_write_completed_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )

    async def _observe_usage_chunk(self, chunk: bytes) -> None:
        if not self._usage_observation_enabled or self._usage_parser_disabled:
            return

        if self._usage_sse_parser is not None:
            try:
                for raw_event in self._usage_sse_parser.feed(chunk):
                    await self._record_sse_usage_event(raw_event)
            except (
                AdmissionRejected,
                SSEProtocolError,
                UnicodeError,
                RuntimeError,
            ) as exc:
                # Usage extraction is telemetry-only.  A malformed/oversized
                # telemetry frame must not corrupt an otherwise valid response.
                await self._disable_usage_parser(exc)
            return

        if not self._is_json_response:
            return
        observed = len(self._usage_json_buffer) + len(chunk)
        if observed > self._usage_json_buffer_limit_bytes:
            await self._disable_usage_parser(
                ValueError(
                    f"streaming JSON usage buffer exceeded "
                    f"{self._usage_json_buffer_limit_bytes} bytes"
                )
            )
            return
        try:
            if self._usage_json_reservation is None:
                request_lease = get_request_admission_lease()
                if request_lease is not None:
                    self._usage_json_reservation = (
                        await request_lease.reserve_temporary_response_bytes(0)
                    )
            if self._usage_json_reservation is not None:
                # Cover bytearray slack plus the final bytes/text copies before
                # the separately-owned JSON graph is materialized.
                await self._usage_json_reservation.reserve(len(chunk) * 8)
            self._usage_json_buffer.extend(chunk)
        except (AdmissionRejected, RuntimeError) as exc:
            # Usage parsing is telemetry-only and must never alter the wire.
            # A response running without request admission (or finishing
            # during cancellation) can observe a lease whose release already
            # started.  Treat that lifecycle race exactly like a telemetry
            # budget rejection instead of truncating a committed response.
            await self._disable_usage_parser(exc)
        return

    async def _finish_usage_observation(self) -> None:
        if not self._usage_observation_enabled or self._usage_parser_disabled:
            return
        if self._usage_sse_parser is not None:
            if self._metadata_sse_event_in_progress:
                await self._disable_usage_parser(
                    SSEIncompleteEventError(
                        pending_bytes=self._metadata_sse_pending_bytes
                    )
                )
                return
            try:
                for raw_event in self._usage_sse_parser.finish():
                    await self._record_sse_usage_event(raw_event)
            except (
                AdmissionRejected,
                SSEProtocolError,
                UnicodeError,
                RuntimeError,
            ) as exc:
                await self._disable_usage_parser(exc)
            else:
                diagnostics = self.current_info.get(
                    "responses_stream_diagnostics"
                )
                if isinstance(diagnostics, dict):
                    diagnostics["downstream_usage_observer_status"] = "completed"
            return
        if self._usage_json_buffer:
            data = None
            try:
                data = bytes(self._usage_json_buffer).decode("utf-8", errors="strict")
                self._usage_json_buffer.clear()
                await self._record_usage_from_data(data)
            except UnicodeError as exc:
                await self._disable_usage_parser(exc)
            finally:
                data = None
                self._usage_json_buffer.clear()
                await self._release_usage_json_reservation()
        diagnostics = self.current_info.get("responses_stream_diagnostics")
        if isinstance(diagnostics, dict) and not self._usage_parser_disabled:
            diagnostics["downstream_usage_observer_status"] = "completed"

    async def _listen_for_disconnect(self, receive: Receive) -> None:
        if self._disconnect_event is not None:
            await self._disconnect_event.wait()
            return
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            # Some test ASGI receivers return immediately.  Do not let a
            # non-disconnect message create a tight loop that starves streaming.
            await asyncio.sleep(0)

    @staticmethod
    def _is_disconnect_error(exc: BaseException) -> bool:
        # ``TimeoutError`` is an ``OSError`` subclass on CPython.  A local
        # queue/backpressure timeout is therefore not evidence that the
        # downstream socket disappeared.
        if isinstance(exc, TimeoutError):
            return False
        if isinstance(exc, OSError):
            return exc.errno is None or exc.errno in {
                errno.EPIPE,
                errno.ECONNABORTED,
                errno.ECONNRESET,
                errno.ENOTCONN,
                errno.ESHUTDOWN,
            }
        return type(exc).__name__ in {
            "BrokenResourceError",
            "ClosedResourceError",
            "ClientDisconnect",
            "EndOfStream",
        }

    def _record_stream_failure(
        self,
        *,
        outcome: str,
        error: BaseException | None = None,
        downstream_disconnected: bool = False,
    ) -> None:
        existing_outcome = str(self.current_info.get("stream_outcome") or "")
        preserve_existing = (
            existing_outcome
            and existing_outcome not in {"completed", "error", "cancelled"}
            and outcome in {"error", "cancelled"}
        )
        if not preserve_existing:
            self.current_info["stream_outcome"] = outcome
        self.current_info["success"] = False
        if downstream_disconnected:
            self.current_info["downstream_disconnected"] = True
            self.current_info["error_type"] = "downstream_disconnect"
        elif error is not None and not (
            preserve_existing and self.current_info.get("error_type")
        ):
            self.current_info["error_type"] = type(error).__name__
        self.current_info["stream_error_after_response_start"] = bool(
            self.current_info.get("response_committed")
        )
        diagnostics = self.current_info.get("responses_stream_diagnostics")
        if isinstance(diagnostics, dict):
            if diagnostics.get("downstream_usage_observer_status") == "aborted":
                diagnostics["downstream_usage_observer_abort_reason"] = outcome
            diagnosis_by_outcome = {
                "downstream_disconnected": "responses_downstream_disconnect",
                "downstream_write_timeout": "responses_downstream_write_timeout",
                "downstream_send_error": "responses_downstream_send_error",
            }
            diagnosis = diagnosis_by_outcome.get(outcome)
            if diagnosis:
                diagnostics["diagnosis"] = diagnosis
                diagnostics["failure_stage"] = "downstream"
                diagnostics["downstream_failure_outcome"] = outcome
                diagnostics["downstream_failure_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
            if not self.current_info.get("response_committed"):
                diagnostics["response_start_asgi_write_completed"] = False
                diagnostics["response_start_asgi_write_outcome"] = outcome
                diagnostics["response_start_asgi_write_error_type"] = (
                    type(error).__name__
                    if error is not None
                    else "DownstreamDisconnected"
                    if downstream_disconnected
                    else "unknown"
                )
                diagnostics["response_start_asgi_write_error_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )

    async def _send_sse_error(
        self,
        send: Send,
        exc: BaseException,
        *,
        error_summary: str | None = None,
    ) -> None:
        if not self._is_sse_response:
            return
        if (
            not self._wire_sse_boundary_known
            or not self._wire_sse_at_event_boundary
        ):
            # Bytes already written to a socket cannot be rolled back.  Adding
            # an error event after a partial data/event field would glue two
            # protocol frames together and misrepresent the wire.  End the
            # response without fabricating an in-band terminal instead.
            self.current_info["sse_error_event_suppressed"] = (
                "partial_or_unknown_frame_boundary"
            )
            return
        if isinstance(exc, ResponsesSemanticError):
            self.current_info["stream_error_status_code"] = exc.status_code
            self.current_info["stream_error_code"] = exc.error_code
            self.current_info["stream_error_type"] = exc.error_type
            self.current_info["stream_error_event_type"] = exc.event_type
            error_data = json.dumps(exc.sse_payload, ensure_ascii=False)
        else:
            summary = error_summary or bounded_stream_error_text(exc)
            error_data = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": bounded_stream_error_text(
                            f"Streaming error: {summary}"
                        ),
                        "type": "stream_error",
                    }
                }
            )
        self._wire_sse_boundary_known = False
        await self._send_with_deadline(
            send,
            {
                "type": "http.response.body",
                "body": f"event: error\ndata: {error_data}\n\n".encode("utf-8"),
                "more_body": True,
            },
        )
        self._record_downstream_sse_event_sent(
            "error",
            semantic_outcome="error",
        )
        self._wire_sse_boundary_known = True
        self._wire_sse_at_event_boundary = True

    async def _send_with_deadline_unprofiled(
        self,
        send: Send,
        message: dict[str, Any],
    ) -> None:
        try:
            if self._downstream_writer is None:
                await _await_with_hard_deadline(
                    send(message),
                    timeout=self._downstream_write_timeout_seconds,
                    label="downstream ASGI write",
                )
            else:
                await self._downstream_writer.write(message)
        except asyncio.TimeoutError as exc:
            raise DownstreamWriteTimeout(
                "downstream write exceeded "
                f"{self._downstream_write_timeout_seconds:g} seconds"
            ) from exc
        except Exception as exc:
            if self._is_disconnect_error(exc):
                raise DownstreamDisconnected(str(exc)) from exc
            raise DownstreamSendError(type(exc).__name__) from exc

    async def _send_with_deadline_profiled(
        self,
        send: Send,
        message: dict[str, Any],
    ) -> None:
        tracker = self._asgi_phase_tracker
        assert tracker is not None
        phase = tracker.begin_phase("asgi_write")
        body = message.get("body")
        body_size = (
            len(body)
            if isinstance(body, (bytes, bytearray, memoryview))
            else 0
        )
        try:
            await self._send_with_deadline_unprofiled(send, message)
        finally:
            tracker.finish_phase(
                "asgi_write",
                phase,
                bytes_count=body_size,
                events=1,
            )

    async def _send_with_deadline_resolving(
        self,
        send: Send,
        message: dict[str, Any],
    ) -> None:
        tracker = self._resolve_responses_diagnostics_tracker()
        if tracker is not None and tracker.phase_sampled:
            await self._send_with_deadline_profiled(send, message)
            return
        await self._send_with_deadline_unprofiled(send, message)

    async def _send_with_deadline(self, send: Send, message: dict[str, Any]) -> None:
        await self._send_with_deadline_unprofiled(send, message)

    async def _stream_response_body(self, send: Send) -> None:
        try:
            await self._stream_response_body_inner(send)
        finally:
            # Usage extraction is telemetry-only, but its partial buffers are
            # still real request-owned memory.  Every disconnect, write
            # timeout, cancellation, and parser failure must release them.
            parser = self._usage_sse_parser
            self._usage_sse_parser = None
            if parser is not None:
                parser.discard()
            self._usage_json_buffer.clear()
            await self._release_usage_json_reservation()
            diagnostics = self.current_info.get(
                "responses_stream_diagnostics"
            )
            if (
                isinstance(diagnostics, dict)
                and diagnostics.get("downstream_usage_observer_status")
                == "active"
            ):
                diagnostics["downstream_usage_observer_status"] = "aborted"
                diagnostics["downstream_usage_observer_abort_reason"] = str(
                    self.current_info.get("stream_outcome")
                    or "body_stream_terminated_before_observer_finish"
                )
                diagnostics["downstream_usage_observer_aborted_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )

    async def _stream_response_body_inner(self, send: Send) -> None:
        async for chunk in self.body_iterator:
            segment = None
            text = None
            observed_event_type = None
            observed_semantic_outcome = None
            observed_terminal_timeline_event = False
            observed_final_event_segment = False
            observed_sse_metadata_complete = False
            observed_usage_snapshot = None
            try:
                self._mark_first_byte_observed(self.current_info)
                if isinstance(chunk, ObservedStreamChunk):
                    observed_event_type = chunk.event_type
                    observed_semantic_outcome = chunk.semantic_outcome
                    observed_terminal_timeline_event = (
                        chunk.terminal_timeline_event
                    )
                    observed_final_event_segment = chunk.final_event_segment
                    observed_sse_metadata_complete = (
                        chunk.sse_metadata_complete
                    )
                    observed_usage_snapshot = chunk.usage_snapshot
                    chunk = chunk.data
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                elif isinstance(chunk, memoryview):
                    chunk = chunk.tobytes()
                elif not isinstance(chunk, bytes):
                    chunk = bytes(chunk)

                if self._debug and not str(
                    self.current_info.get("endpoint") or ""
                ).endswith("/v1/audio/speech"):
                    try:
                        if len(chunk) <= self._downstream_chunk_bytes:
                            text = chunk.decode("utf-8", errors="replace")
                            logger.info(
                                text.encode("utf-8").decode("unicode_escape")
                            )
                        else:
                            logger.info("stream chunk bytes=%s", len(chunk))
                    except Exception:
                        logger.info("stream chunk bytes=%s", len(chunk))

                if chunk:
                    offsets = range(0, len(chunk), self._downstream_chunk_bytes)
                else:
                    offsets = (0,)
                for offset in offsets:
                    if offset == 0 and len(chunk) <= self._downstream_chunk_bytes:
                        # Keep the metadata-carrying bytes object itself on the
                        # overwhelmingly common one-frame/one-write path.
                        segment = chunk
                    else:
                        segment = chunk[
                            offset : offset + self._downstream_chunk_bytes
                        ]
                    metadata_event_complete = bool(
                        observed_final_event_segment
                        and offset + len(segment) >= len(chunk)
                    )
                    parser = self._usage_sse_parser
                    use_sse_metadata = bool(
                        observed_sse_metadata_complete
                        and (
                            self._metadata_sse_event_in_progress
                            or (
                                parser is not None
                                and parser.pending_bytes == 0
                            )
                            or (
                                parser is None
                                and self._wire_sse_boundary_known
                                and self._wire_sse_at_event_boundary
                            )
                        )
                    )
                    if use_sse_metadata:
                        self._metadata_sse_pending_bytes += len(segment)
                        if (
                            self._usage_observation_enabled
                            and not self._usage_parser_disabled
                            and metadata_event_complete
                            and observed_usage_snapshot is not None
                        ):
                            self._record_usage_snapshot(
                                observed_usage_snapshot
                            )
                    else:
                        await self._observe_usage_chunk(segment)
                    candidate_sse_boundary: bool | None = None
                    if self._is_sse_response:
                        if use_sse_metadata:
                            candidate_sse_boundary = metadata_event_complete
                        elif (
                            not self._usage_parser_disabled
                            and parser is not None
                        ):
                            candidate_sse_boundary = parser.pending_bytes == 0
                        # An ASGI adapter may raise after a partial socket
                        # write.  Until the await returns successfully the
                        # resulting wire boundary is unknowable.
                        self._wire_sse_boundary_known = False
                    if observed_terminal_timeline_event:
                        tracker = self._resolve_responses_diagnostics_tracker()
                        if tracker is not None:
                            tracker.mark_terminal_asgi_write_attempted()
                    await self._send_with_deadline(
                        send,
                        {
                            "type": "http.response.body",
                            "body": segment,
                            "more_body": True,
                        },
                    )
                    if use_sse_metadata:
                        self._metadata_sse_event_in_progress = (
                            not metadata_event_complete
                        )
                        if metadata_event_complete:
                            self._metadata_sse_pending_bytes = 0
                    if (
                        observed_final_event_segment
                        and offset + len(segment) >= len(chunk)
                    ):
                        self._record_downstream_sse_event_sent(
                            observed_event_type,
                            semantic_outcome=observed_semantic_outcome,
                        )
                    if self._is_sse_response:
                        if candidate_sse_boundary is None:
                            self._wire_sse_boundary_known = False
                        else:
                            self._wire_sse_boundary_known = True
                            self._wire_sse_at_event_boundary = (
                                candidate_sse_boundary
                            )
                    segment = None
            finally:
                # ByteBoundedQueue releases an in-flight item when the body
                # iterator is advanced.  Drop every local alias before that
                # next __anext__ call so the queue gauge and the real object
                # lifetime cannot diverge while waiting for more data.
                segment = None
                text = None
                chunk = None
                observed_event_type = None
                observed_semantic_outcome = None
                observed_terminal_timeline_event = False
                observed_final_event_segment = False
                observed_sse_metadata_complete = False
                observed_usage_snapshot = None
        await self._finish_usage_observation()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        trace = self.current_info.get("trace") if isinstance(self.current_info, dict) else None
        # This is the immutable wire status emitted below. Post-commit failures
        # belong in stream_error_status_code/stream_outcome, never here.
        self.current_info["status_code"] = self.status_code
        self.current_info["wire_status_code"] = self.status_code
        self.current_info["response_committed"] = False
        if self._is_trace(trace):
            trace.mark("downstream_response_start")
            merge_timing_spans(self.current_info, trace.snapshot())

        started = False
        should_send_final_body = True
        pending_cancel: asyncio.CancelledError | None = None
        disconnect_listener: Optional[asyncio.Task] = None
        self._downstream_writer = _HardDeadlineASGIWriter(
            send,
            timeout=self._downstream_write_timeout_seconds,
            label="downstream ASGI write",
        )
        try:
            diagnostics = self.current_info.get("responses_stream_diagnostics")
            if isinstance(diagnostics, dict):
                diagnostics["response_start_asgi_write_attempted"] = True
                diagnostics["response_start_asgi_write_attempted_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )
            await self._send_with_deadline(
                send,
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                },
            )
            started = True
            self.current_info["response_committed"] = True
            if isinstance(diagnostics, dict):
                diagnostics["response_start_asgi_write_completed"] = True
                diagnostics["response_start_asgi_write_completed_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )
            self._stream_task = asyncio.create_task(self._stream_response_body(send))

            # BaseHTTPMiddleware supplies a receive wrapper that may itself
            # wait for response completion.  Never start a second consumer on
            # it.  The Stats middleware provides a single sticky disconnect
            # event when monitoring is safe; otherwise send errors/deadlines
            # still terminate the stream.
            if self._disconnect_event is not None:
                disconnect_listener = asyncio.create_task(self._listen_for_disconnect(receive))
                done, _ = await asyncio.wait(
                    {self._stream_task, disconnect_listener},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Prefer a completed stream if both tasks become ready in the
                # same loop turn.  This avoids reclassifying a normal EOF as a
                # disconnect merely because the peer closed immediately after
                # receiving the final application byte.
                if self._stream_task in done:
                    await self._stream_task
                elif disconnect_listener in done:
                    should_send_final_body = False
                    self._record_stream_failure(
                        outcome="downstream_disconnected",
                        downstream_disconnected=True,
                    )
                    self._stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._stream_task
            else:
                await self._stream_task

            if "stream_outcome" not in self.current_info:
                self.current_info["stream_outcome"] = "completed"
        except asyncio.CancelledError as exc:
            should_send_final_body = False
            pending_cancel = exc
            disconnected = bool(
                self._disconnect_event is not None and self._disconnect_event.is_set()
            )
            self._record_stream_failure(
                outcome="downstream_disconnected" if disconnected else "cancelled",
                error=None if disconnected else exc,
                downstream_disconnected=disconnected,
            )
        except DownstreamWriteTimeout as exc:
            should_send_final_body = False
            self._record_stream_failure(
                outcome="downstream_write_timeout",
                error=exc,
            )
            logger.warning("Streaming response downstream write timed out")
        except DownstreamDisconnected:
            should_send_final_body = False
            self._record_stream_failure(
                outcome="downstream_disconnected",
                downstream_disconnected=True,
            )
        except DownstreamSendError as exc:
            should_send_final_body = False
            self.current_info["sse_error_event_suppressed"] = (
                "downstream_send_failed_boundary_unknown"
            )
            self._record_stream_failure(
                outcome="downstream_send_error",
                error=exc,
            )
            logger.warning(
                "Streaming response downstream send failed: %s",
                bounded_stream_error_text(exc),
            )
        except Exception as exc:
            disconnected = bool(
                self._disconnect_event is not None and self._disconnect_event.is_set()
            )
            should_send_final_body = not disconnected
            self._record_stream_failure(
                outcome=(
                    "downstream_disconnected"
                    if disconnected
                    else "upstream_failure_terminal"
                    if isinstance(exc, ResponsesSemanticError)
                    else "error"
                ),
                error=None if disconnected else exc,
                downstream_disconnected=disconnected,
            )
            error_summary = bounded_stream_error_text(exc)
            logger.error("Error in streaming response: %s", error_summary)
            if self._debug:
                import traceback

                traceback.print_tb(exc.__traceback__, limit=20)
            if started and not disconnected:
                try:
                    await self._send_sse_error(
                        send,
                        exc,
                        error_summary=error_summary,
                    )
                except DownstreamWriteTimeout as send_error:
                    should_send_final_body = False
                    self._record_stream_failure(
                        outcome="downstream_write_timeout",
                        error=send_error,
                    )
                    logger.warning("SSE error-event downstream write timed out")
                except DownstreamDisconnected:
                    should_send_final_body = False
                    self._record_stream_failure(
                        outcome="downstream_disconnected",
                        downstream_disconnected=True,
                    )
                except Exception as send_error:
                    should_send_final_body = False
                    self._record_stream_failure(
                        outcome="downstream_send_error",
                        error=send_error,
                    )
                    logger.warning(
                        "Error sending SSE stream error: %s",
                        bounded_stream_error_text(send_error),
                    )
        finally:
            if disconnect_listener is not None and not disconnect_listener.done():
                disconnect_listener.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_listener
            if self._stream_task is not None and not self._stream_task.done():
                self._stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._stream_task
            cleanup_cancel = await self._close_body_iterator_once()
            pending_cancel = pending_cancel or cleanup_cancel

            if started and should_send_final_body:
                diagnostics = self.current_info.get(
                    "responses_stream_diagnostics"
                )
                if isinstance(diagnostics, dict):
                    diagnostics["downstream_final_body_attempted"] = True
                    diagnostics["downstream_final_body_outcome"] = "attempting"
                    diagnostics["downstream_final_body_attempted_at"] = (
                        datetime.now(timezone.utc).isoformat()
                    )
                try:
                    await self._send_with_deadline(
                        send,
                        {
                            "type": "http.response.body",
                            "body": b"",
                            "more_body": False,
                        },
                    )
                    if isinstance(diagnostics, dict):
                        diagnostics["downstream_final_body_completed"] = True
                        diagnostics["downstream_final_body_outcome"] = "completed"
                        diagnostics["downstream_final_body_completed_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                except asyncio.CancelledError as exc:
                    pending_cancel = pending_cancel or exc
                    self._record_stream_failure(outcome="cancelled", error=exc)
                    if isinstance(diagnostics, dict):
                        diagnostics["downstream_final_body_error_type"] = type(
                            exc
                        ).__name__
                        diagnostics["downstream_final_body_outcome"] = "cancelled"
                        diagnostics["downstream_final_body_error_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                except DownstreamWriteTimeout as exc:
                    self._record_stream_failure(
                        outcome="downstream_write_timeout",
                        error=exc,
                    )
                    if isinstance(diagnostics, dict):
                        diagnostics["downstream_final_body_error_type"] = type(
                            exc
                        ).__name__
                        diagnostics["downstream_final_body_outcome"] = (
                            "downstream_write_timeout"
                        )
                        diagnostics["downstream_final_body_error_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                    logger.warning("Final downstream streaming write timed out")
                except DownstreamDisconnected:
                    self._record_stream_failure(
                        outcome="downstream_disconnected",
                        downstream_disconnected=True,
                    )
                    if isinstance(diagnostics, dict):
                        diagnostics["downstream_final_body_error_type"] = (
                            "DownstreamDisconnected"
                        )
                        diagnostics["downstream_final_body_outcome"] = (
                            "downstream_disconnected"
                        )
                        diagnostics["downstream_final_body_error_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                    logger.warning("Final downstream streaming peer disconnected")
                except DownstreamSendError as exc:
                    self._record_stream_failure(
                        outcome="downstream_send_error",
                        error=exc,
                    )
                    if isinstance(diagnostics, dict):
                        diagnostics["downstream_final_body_error_type"] = type(
                            exc
                        ).__name__
                        diagnostics["downstream_final_body_outcome"] = (
                            "downstream_send_error"
                        )
                        diagnostics["downstream_final_body_error_at"] = (
                            datetime.now(timezone.utc).isoformat()
                        )
                    logger.warning(
                        "Error sending final streaming response body: %s",
                        bounded_stream_error_text(exc),
                    )
            else:
                diagnostics = self.current_info.get(
                    "responses_stream_diagnostics"
                )
                if isinstance(diagnostics, dict):
                    diagnostics["downstream_final_body_skip_reason"] = (
                        "response_start_not_completed"
                        if not started
                        else str(
                            self.current_info.get("stream_outcome")
                            or "prior_stream_failure"
                        )
                    )

            downstream_writer = self._downstream_writer
            self._downstream_writer = None
            if downstream_writer is not None:
                writer_cancel = await downstream_writer.close()
                pending_cancel = pending_cancel or writer_cancel

            finalize_cancel = await self._finalize_once()
            pending_cancel = pending_cancel or finalize_cancel
            if pending_cancel is not None:
                raise pending_cancel

    async def _close_body_iterator_once(self) -> asyncio.CancelledError | None:
        if self._body_close_task is None:
            self._body_close_task = asyncio.create_task(
                self._close_body_iterator(),
                name="uni-api-close-stream-body",
            )
        return await self._await_cleanup_task(self._body_close_task)

    @staticmethod
    async def _await_cleanup_task(
        task: asyncio.Task,
    ) -> asyncio.CancelledError | None:
        pending_cancel: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                pending_cancel = pending_cancel or exc
        task.result()
        return pending_cancel

    async def _close_body_iterator(self) -> None:
        async with self._body_close_lock:
            if self._body_closed:
                return
            if hasattr(self.body_iterator, "aclose"):
                await call_cleanup_safely(
                    self.body_iterator.aclose,
                    label="Downstream streaming body iterator",
                )
            self._body_closed = True

    async def _default_lifecycle_close(self) -> None:
        self.current_info["process_time"] = time() - self.current_info.get("start_time", time())
        final_trace = self.current_info.get("trace") if isinstance(self.current_info, dict) else None
        if self._is_trace(final_trace):
            final_trace.mark("stream_end")
            final_trace.mark("usage_recorded")
            merge_timing_spans(self.current_info, final_trace.snapshot())
            logger.info(
                "trace_span trace_id=%s request_id=%s endpoint=%s spans=%s",
                self.current_info.get("trace_id"),
                self.current_info.get("request_id"),
                self.current_info.get("endpoint"),
                self.current_info.get("timing_spans"),
            )
        try:
            self._emit_request_observability(self.current_info)
        except Exception:
            logger.exception("Failed to emit streaming request observability")

    async def _finalize_once(self) -> asyncio.CancelledError | None:
        async with self._finalize_lock:
            if self._finalized:
                return None
            pending_cancel: asyncio.CancelledError | None = None
            # Stabilize the fields needed by persistence before either the DB
            # write or the one-shot observability/lifecycle callback.
            self.current_info["process_time"] = time() - self.current_info.get(
                "start_time",
                time(),
            )
            final_trace = self.current_info.get("trace")
            if self._is_trace(final_trace):
                final_trace.mark("stream_end")
                final_trace.mark("usage_recorded")
                merge_timing_spans(self.current_info, final_trace.snapshot())

            if self._update_stats is not None:
                stats_snapshot = _bounded_stats_snapshot(self.current_info)
                try:
                    persisted = await _await_with_hard_deadline(
                        self._update_stats(stats_snapshot),
                        timeout=self._stats_write_timeout_seconds,
                        label="request stats write",
                    )
                    if persisted is False:
                        self.current_info["stats_write_failed"] = True
                        logger.error("Streaming request stats write failed")
                except asyncio.TimeoutError:
                    self.current_info["stats_write_timeout"] = True
                    logger.error(
                        "Streaming request stats write exceeded %.3f seconds",
                        self._stats_write_timeout_seconds,
                    )
                except asyncio.CancelledError as exc:
                    pending_cancel = pending_cancel or exc
                except Exception:
                    self.current_info["stats_write_failed"] = True
                    logger.exception("Failed to update streaming request stats")
                finally:
                    for price_key in ("prompt_price", "completion_price"):
                        if price_key in stats_snapshot:
                            self.current_info[price_key] = stats_snapshot[price_key]

            # Emit only after the persistence result is known so timeout/fail
            # fields are present in the single request summary.
            lifecycle_completed = False
            while not lifecycle_completed:
                try:
                    if self._lifecycle_close is not None:
                        await self._lifecycle_close(self.current_info)
                    else:
                        await self._default_lifecycle_close()
                    lifecycle_completed = True
                except asyncio.CancelledError as exc:
                    pending_cancel = pending_cancel or exc
                except Exception:
                    logger.exception("Failed to finalize streaming request lifecycle")
                    lifecycle_completed = True
            self._finalized = True
            return pending_cancel

    async def close(self):
        stream_task = self._stream_task
        if stream_task is not None and not stream_task.done() and stream_task is not asyncio.current_task():
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task
        pending_cancel = await self._close_body_iterator_once()
        if "stream_outcome" not in self.current_info:
            self._record_stream_failure(outcome="cancelled", error=asyncio.CancelledError())
        finalize_cancel = await self._finalize_once()
        pending_cancel = pending_cancel or finalize_cancel
        if pending_cancel is not None:
            raise pending_cancel
