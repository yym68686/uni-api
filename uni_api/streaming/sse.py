from __future__ import annotations

import asyncio
import codecs
import hashlib
import os
import threading
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from uni_api.admission import AdmissionRejected, get_request_admission_lease
from uni_api.admission.memory import AdaptiveMemoryGovernor, process_memory_governor
from uni_api.admission.json_memory import JSONMemoryComplexityError
from uni_api.admission.json_parsing import (
    OwnedJSONValue,
    ReusableJSONParseWorkspace,
    parse_owned_json_value,
)
from uni_api.serialization import json


# Image-generation Responses events can legitimately carry multi-megabyte
# base64 payloads. These are conservative defaults for generic SSE consumers;
# Responses proxy paths disable the fixed byte limits while aggregate parser
# ownership remains charged to the cgroup-aware memory governor below.
DEFAULT_MAX_PENDING_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_EVENT_BYTES = 8 * 1024 * 1024
# A zero byte limit disables only that fixed limit. Retained parser bytes and
# JSON materialization remain charged to the cgroup-aware memory governor.
UNLIMITED_SSE_BYTES = 0
DEFAULT_MAX_EVENTS_PER_FEED = 4096
DEFAULT_MAX_LINES_PER_FEED = 4096
DEFAULT_MAX_FIELDS_PER_EVENT = 4096
DEFAULT_MAX_FEED_BYTES = DEFAULT_MAX_EVENT_BYTES + 64 * 1024
_INTERNAL_FEED_CHUNK_UNITS = 256 * 1024
_JSON_PARSE_OFFLOAD_THRESHOLD_BYTES = 64 * 1024
_SSE_JSON_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
try:
    _STREAM_PARSER_RETAINED_BUDGET_BYTES = max(
        1,
        int(
            os.getenv(
                "STREAM_PARSER_RETAINED_BUDGET_BYTES",
                str(process_memory_governor.maximum_capacity_bytes()),
            )
        ),
    )
except (TypeError, ValueError):
    _STREAM_PARSER_RETAINED_BUDGET_BYTES = (
        process_memory_governor.maximum_capacity_bytes()
    )


class StreamParserBufferBudgetExhausted(AdmissionRejected):
    local_admission_rejection = True

    def __init__(self) -> None:
        super().__init__("stream_parser_buffer_budget_exhausted", status_code=503)


class _StreamParserRetainedBudget:
    def __init__(
        self,
        capacity_bytes: int,
        *,
        memory_governor: AdaptiveMemoryGovernor | None = None,
        memory_category: str = "stream_parser",
    ) -> None:
        self.capacity_bytes = int(capacity_bytes)
        self.memory_governor = memory_governor
        self.memory_category = str(memory_category or "stream_parser")
        self.used_bytes = 0
        self.peak_bytes = 0
        self.rejected = 0
        # Releasing the last _RetainedTextFrame reference can invoke its
        # finalizer while snapshot() already holds this lock.
        self._lock = threading.RLock()

    def reserve(self, size: int) -> None:
        if size <= 0:
            return
        parent_reserved = False
        if self.memory_governor is not None:
            parent_reserved = self.memory_governor.reserve_nowait(
                self.memory_category,
                size,
            )
            if not parent_reserved:
                with self._lock:
                    self.rejected += 1
                raise StreamParserBufferBudgetExhausted()
        with self._lock:
            if self.used_bytes + size > self.capacity_bytes:
                self.rejected += 1
                if parent_reserved and self.memory_governor is not None:
                    self.memory_governor.release(self.memory_category, size)
                raise StreamParserBufferBudgetExhausted()
            self.used_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def release(self, size: int) -> None:
        if size <= 0:
            return
        with self._lock:
            if size > self.used_bytes:
                raise RuntimeError("stream parser retained-byte budget underflow")
            self.used_bytes -= size
        if self.memory_governor is not None:
            self.memory_governor.release(self.memory_category, size)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            effective_capacity = self.capacity_bytes
            if self.memory_governor is not None:
                effective_capacity = min(
                    effective_capacity,
                    self.memory_governor.snapshot().capacity_bytes,
                )
            return {
                "used_bytes": self.used_bytes,
                "peak_bytes": self.peak_bytes,
                "capacity_bytes": effective_capacity,
                "rejected": self.rejected,
            }


_STREAM_PARSER_RETAINED_BUDGET = _StreamParserRetainedBudget(
    _STREAM_PARSER_RETAINED_BUDGET_BYTES,
    memory_governor=process_memory_governor,
)


def stream_parser_retained_budget_snapshot() -> dict[str, int]:
    return _STREAM_PARSER_RETAINED_BUDGET.snapshot()


class StreamParserRetainedLease:
    """Explicit ownership for custom protocol accumulators."""

    def __init__(self) -> None:
        self._retained_budget = _STREAM_PARSER_RETAINED_BUDGET
        self.size = 0
        self._released = False

    def grow(self, additional_bytes: int) -> None:
        additional_bytes = int(additional_bytes)
        if additional_bytes < 0:
            raise ValueError("parser reservation cannot grow by a negative size")
        if self._released:
            raise RuntimeError("parser reservation is released")
        self._retained_budget.reserve(additional_bytes)
        self.size += additional_bytes

    def shrink(self, released_bytes: int) -> None:
        released_bytes = int(released_bytes)
        if released_bytes < 0 or released_bytes > self.size:
            raise ValueError("invalid parser reservation shrink")
        if self._released:
            raise RuntimeError("parser reservation is released")
        self.size -= released_bytes
        self._retained_budget.release(released_bytes)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        retained = self.size
        self.size = 0
        self._retained_budget.release(retained)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


async def _finish_sse_cleanup_despite_cancellation(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def _release_sse_owned_resources(
    json_owner: OwnedJSONValue | None,
    workspace_reservation: Any | None,
    raw_event: str | None,
) -> None:
    try:
        if json_owner is not None:
            await json_owner.aclose()
    finally:
        if workspace_reservation is not None:
            await workspace_reservation.release()
        # Raw parser ownership follows the actual str object's lifetime.  A
        # producer generator or feed-result list may still alias the frame
        # after this consumer closes; explicit counter release here would make
        # the global budget under-report live memory.  Clearing this owner's
        # reference lets _RetainedTextFrame.__del__ release exactly when the
        # final alias disappears.


async def _release_sse_owned_resources_safely(
    json_owner: OwnedJSONValue | None,
    workspace_reservation: Any | None,
    raw_event: str | None,
) -> None:
    cleanup_task = asyncio.create_task(
        _release_sse_owned_resources(
            json_owner,
            workspace_reservation,
            raw_event,
        )
    )
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        await _finish_sse_cleanup_despite_cancellation(cleanup_task)
        raise


class _RetainedTextFrame(str):
    """A parsed frame keeps its process-wide raw-byte ownership until GC."""

    def __new__(
        cls,
        value: str,
        retained_bytes: int,
        retained_budget: _StreamParserRetainedBudget,
    ):
        instance = str.__new__(cls, value)
        instance._retained_bytes = int(retained_bytes)
        instance._retained_budget = retained_budget
        instance._released = False
        return instance

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._retained_budget.release(self._retained_bytes)
        self._retained_bytes = 0

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def retain_joined_parser_text(
    parts: list[str],
    *,
    retained_bytes: int,
) -> str:
    """Reserve the joined copy before materializing it."""

    retained_bytes = int(retained_bytes)
    retained_budget = _STREAM_PARSER_RETAINED_BUDGET
    retained_budget.reserve(retained_bytes)
    try:
        return _RetainedTextFrame(
            "".join(parts),
            retained_bytes,
            retained_budget,
        )
    except BaseException:
        retained_budget.release(retained_bytes)
        raise


class OwnedSSEEvent:
    """One SSE frame whose raw and materialized bytes have explicit owners."""

    def __init__(
        self,
        *,
        raw_event: str,
        event_name: str,
        declared_event_name: str,
        payload: Any,
        json_owner: OwnedJSONValue | None,
        workspace_reservation: Any | None,
        is_comment: bool,
        has_event_field: bool,
        has_data_field: bool,
    ) -> None:
        self._raw_event: str | None = raw_event
        self._event_name: str | None = event_name
        self._declared_event_name: str | None = declared_event_name
        self._payload = payload
        self._json_owner = json_owner
        self._workspace_reservation = workspace_reservation
        self._is_comment = bool(is_comment)
        self._has_event_field = bool(has_event_field)
        self._has_data_field = bool(has_data_field)
        self._payload_reservation_transferred = False
        self._closed = False
        self._closing = False
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def raw_event(self) -> str:
        if self._closed or self._closing or self._raw_event is None:
            raise RuntimeError("owned SSE event is closed")
        return self._raw_event

    @property
    def event_name(self) -> str:
        if self._closed or self._closing or self._event_name is None:
            raise RuntimeError("owned SSE event is closed")
        return self._event_name

    @property
    def declared_event_name(self) -> str:
        """Return only the event name present on the wire, if any."""

        if self._closed or self._closing or self._declared_event_name is None:
            raise RuntimeError("owned SSE event is closed")
        return self._declared_event_name

    @property
    def payload(self) -> Any:
        if self._closed or self._closing:
            raise RuntimeError("owned SSE event is closed")
        if self._json_owner is not None:
            return self._json_owner.value
        return self._payload

    @property
    def is_comment(self) -> bool:
        return self._is_comment

    @property
    def has_event_field(self) -> bool:
        """Whether the wire block explicitly contained an ``event`` field."""

        return self._has_event_field

    @property
    def has_data_field(self) -> bool:
        """Whether this SSE block contained at least one ``data`` field.

        Presence is intentionally distinct from a joined data value of ``""``.
        WHATWG dispatch ignores a block with no data field, while an explicit
        empty ``data:`` field still dispatches an event with an empty payload.
        """

        return self._has_data_field

    def take_payload_reservation(self):
        """Transfer parsed-graph ownership to a persistent consumer once."""

        if self._closed or self._closing:
            raise RuntimeError("owned SSE event is closed")
        if self._payload_reservation_transferred:
            raise RuntimeError("owned SSE payload reservation was already transferred")
        self._payload_reservation_transferred = True
        if self._json_owner is None:
            return None
        return self._json_owner.take_reservation()

    def _close_nowait(self) -> bool:
        if (
            self._closed
            or self._closing
            or self._workspace_reservation is not None
            or self._close_task is not None
            or self._lock.locked()
        ):
            return False
        json_owner = self._json_owner
        if json_owner is not None and not json_owner.can_close_nowait:
            return False
        self._closing = True
        self._payload = None
        self._event_name = None
        self._declared_event_name = None
        self._raw_event = None
        self._json_owner = None
        self._workspace_reservation = None
        if json_owner is not None:
            if not json_owner.close_nowait():
                raise RuntimeError("unreserved JSON owner close became blocking")
        self._closed = True
        self._closing = False
        return True

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._close_nowait():
            return
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_once())
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await _finish_sse_cleanup_despite_cancellation(close_task)
            raise

    async def _close_once(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            raw_event = self._raw_event
            json_owner = self._json_owner
            workspace_reservation = self._workspace_reservation
            # Clear graph/raw aliases owned by this object before returning
            # either budget.  Callers follow the same rule for local aliases.
            self._payload = None
            self._event_name = None
            self._declared_event_name = None
            self._raw_event = None
            self._json_owner = None
            self._workspace_reservation = None
        await _release_sse_owned_resources(
            json_owner,
            workspace_reservation,
            raw_event,
        )

    async def __aenter__(self) -> OwnedSSEEvent:
        if self._closed or self._closing:
            raise RuntimeError("owned SSE event is closed")
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


def _bounded_utf8_size(text: str, *, limit_bytes: int) -> int:
    """Measure only until a byte limit is known to be exceeded."""

    observed = 0
    for offset in range(0, len(text), _INTERNAL_FEED_CHUNK_UNITS):
        observed += len(
            text[offset : offset + _INTERNAL_FEED_CHUNK_UNITS].encode("utf-8")
        )
        if observed > limit_bytes:
            break
    return observed


class SSEProtocolError(ValueError):
    """The upstream byte stream cannot be parsed as a valid SSE stream."""


class SSEBufferOverflowError(SSEProtocolError):
    """An SSE frame exceeded one of the parser's configured byte limits."""

    def __init__(
        self,
        *,
        buffer_name: str,
        limit_bytes: int,
        observed_bytes: int,
    ) -> None:
        self.buffer_name = buffer_name
        self.limit_bytes = limit_bytes
        self.observed_bytes = observed_bytes
        super().__init__(
            f"SSE {buffer_name} exceeded {limit_bytes} bytes "
            f"(observed {observed_bytes} bytes)"
        )


class SSEOutputLimitError(SSEProtocolError):
    """One input chunk would produce an unbounded number of records."""

    def __init__(
        self,
        *,
        output_name: str,
        limit: int,
        observed: int,
    ) -> None:
        self.output_name = output_name
        self.limit = limit
        self.observed = observed
        super().__init__(
            f"SSE {output_name} per feed exceeded {limit} "
            f"(observed at least {observed})"
        )


class SSEIncompleteEventError(SSEProtocolError):
    """The SSE stream ended before its trailing event was terminated."""

    def __init__(self, *, pending_bytes: int) -> None:
        self.pending_bytes = pending_bytes
        super().__init__(
            f"SSE stream ended with an incomplete trailing event "
            f"({pending_bytes} pending bytes)"
        )


class IncrementalSSEParser:
    """Incrementally split a resource-accounted UTF-8 SSE stream into frames."""

    def __init__(
        self,
        *,
        max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_events_per_feed: int = DEFAULT_MAX_EVENTS_PER_FEED,
        max_feed_bytes: int = DEFAULT_MAX_FEED_BYTES,
    ) -> None:
        if max_pending_bytes < 0:
            raise ValueError("max_pending_bytes cannot be negative")
        if max_event_bytes < 0:
            raise ValueError("max_event_bytes cannot be negative")
        if max_events_per_feed <= 0:
            raise ValueError("max_events_per_feed must be greater than zero")
        if max_feed_bytes < 0:
            raise ValueError("max_feed_bytes cannot be negative")

        self.max_pending_bytes = max_pending_bytes
        self.max_event_bytes = max_event_bytes
        self.max_events_per_feed = max_events_per_feed
        self.max_feed_bytes = max_feed_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._retained_budget = _STREAM_PARSER_RETAINED_BUDGET
        # Retain only the current event.  A bytearray lets feed() append each
        # decoded fragment once instead of repeatedly copying/rescanning the
        # entire partial event when upstream sends very small chunks.
        self._event = bytearray()
        self._pending_budget_bytes = 0
        self._trailing_cr = False
        self._finished = False
        self._failed = False
        self._at_stream_start = True
        self._failure_pending_diagnostics: dict[str, Any] | None = None

    def __del__(self) -> None:
        try:
            self._release_pending_budget()
        except Exception:
            pass

    def discard(self) -> None:
        """Drop an abandoned partial frame and release its global budget."""

        self._failed = True
        self._finished = True
        self._clear_pending_event()

    @property
    def pending_text(self) -> str:
        return self._event.decode("utf-8")

    @property
    def pending_bytes(self) -> int:
        return self._pending_size_bytes()

    @property
    def can_bypass_complete_frame(self) -> bool:
        """Whether an external complete frame can leave parser state unchanged."""

        undecoded_bytes, _ = self._decoder.getstate()
        return bool(
            not self._finished
            and not self._failed
            and not self._at_stream_start
            and not self._event
            and not undecoded_bytes
            and not self._trailing_cr
        )

    @property
    def pending_data(self) -> bytes:
        """Return the normalized partial event plus undecoded UTF-8 bytes.

        This is intended for transparent proxies which commit after parsing a
        complete event but have already received the start of the next event.
        """

        undecoded_bytes, _ = self._decoder.getstate()
        pending = bytes(self._event)
        if self._trailing_cr and pending.endswith(b"\n"):
            # Newline normalization records a trailing CR as LF immediately,
            # while retaining enough state to swallow a following LF.  A
            # proxy which hands this partial frame to a fresh parser must keep
            # the raw CR; otherwise a split CRLF becomes two line endings and
            # can prematurely terminate the event.
            pending = pending[:-1] + b"\r"
        return pending + bytes(undecoded_bytes)

    def pending_diagnostics(self) -> dict[str, Any]:
        """Return only size/hash for the normalized incomplete event.

        Decoded CR/CRLF line endings are represented as LF. Any trailing
        incomplete UTF-8 bytes are appended unchanged and explicitly covered
        by the ``normalized_prefix_plus_utf8_tail_v1`` scope.
        """

        undecoded_bytes, _ = self._decoder.getstate()
        pending_bytes = len(self._event) + len(undecoded_bytes)
        result: dict[str, Any] = {
            "bytes": pending_bytes,
            "scope": "normalized_prefix_plus_utf8_tail_v1",
        }
        if pending_bytes:
            digest = hashlib.sha256()
            # hashlib accepts the existing bytearray/buffer directly.  Avoid a
            # full bytes copy and concatenation at the failure/EOF snapshot.
            digest.update(self._event)
            digest.update(undecoded_bytes)
            result["sha256"] = digest.hexdigest()
        return result

    @property
    def failure_pending_diagnostics(self) -> dict[str, Any] | None:
        if self._failure_pending_diagnostics is None:
            return None
        return dict(self._failure_pending_diagnostics)

    def feed(self, chunk: str | bytes | bytearray) -> list[str]:
        if self._finished:
            raise SSEProtocolError("cannot feed an SSE parser after finish()")
        if self._failed:
            raise SSEProtocolError("cannot feed an SSE parser after a parse failure")

        try:
            self._validate_feed_size(chunk)
            events: list[str] = []
            for piece in self._feed_pieces(chunk):
                decoded_chunk = self._strip_initial_bom(self._decode_chunk(piece))
                normalized_chunk = self._normalize_newlines(decoded_chunk, final=False)
                extracted = self._extract_events(normalized_chunk)
                if len(events) + len(extracted) > self.max_events_per_feed:
                    self._failed = True
                    raise SSEOutputLimitError(
                        output_name="events",
                        limit=self.max_events_per_feed,
                        observed=len(events) + len(extracted),
                    )
                events.extend(extracted)
                self._validate_pending_size()
            return events
        except BaseException:
            self._failed = True
            self._capture_failure_pending_diagnostics()
            self._clear_pending_event()
            raise

    def _validate_feed_size(self, chunk: str | bytes | bytearray) -> None:
        if not isinstance(chunk, (str, bytes, bytearray)):
            raise TypeError("SSE chunks must be str, bytes, or bytearray")
        if not self.max_feed_bytes:
            return
        try:
            observed = (
                _bounded_utf8_size(chunk, limit_bytes=self.max_feed_bytes)
                if isinstance(chunk, str)
                else len(chunk)
            )
        except UnicodeEncodeError as exc:
            self._failed = True
            raise SSEProtocolError(
                "SSE text contains an invalid Unicode scalar"
            ) from exc
        if self.max_feed_bytes and observed > self.max_feed_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="input chunk",
                limit_bytes=self.max_feed_bytes,
                observed_bytes=observed,
            )

    @staticmethod
    def _feed_pieces(chunk: str | bytes | bytearray):
        for offset in range(0, len(chunk), _INTERNAL_FEED_CHUNK_UNITS):
            yield chunk[offset : offset + _INTERNAL_FEED_CHUNK_UNITS]

    def finish(self) -> list[str]:
        """Finish the byte stream and reject any unterminated trailing event."""
        if self._failed:
            raise SSEProtocolError("cannot finish an SSE parser after a parse failure")
        if self._finished:
            return []

        self._finished = True
        try:
            decoded_tail = self._decoder.decode(b"", final=True)
            normalized_tail = self._normalize_newlines(
                self._strip_initial_bom(decoded_tail, final=True),
                final=True,
            )
            events = self._extract_events(normalized_tail)
            self._validate_pending_size()
            if self._event:
                self._failed = True
                raise SSEIncompleteEventError(pending_bytes=self._pending_size_bytes())
            return events
        except UnicodeDecodeError as exc:
            self._failed = True
            self._capture_failure_pending_diagnostics()
            self._clear_pending_event()
            raise SSEProtocolError(
                "SSE stream ended with an incomplete UTF-8 sequence"
            ) from exc
        except BaseException:
            self._failed = True
            self._capture_failure_pending_diagnostics()
            self._clear_pending_event()
            raise

    def _decode_chunk(self, chunk: str | bytes | bytearray) -> str:
        if isinstance(chunk, str):
            undecoded_bytes, _ = self._decoder.getstate()
            if undecoded_bytes:
                self._failed = True
                raise SSEProtocolError(
                    "cannot feed text while a UTF-8 byte sequence is incomplete"
                )
            return chunk
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("SSE chunks must be str, bytes, or bytearray")

        try:
            return self._decoder.decode(bytes(chunk), final=False)
        except UnicodeDecodeError as exc:
            self._failed = True
            raise SSEProtocolError("SSE stream contains invalid UTF-8") from exc

    def _normalize_newlines(self, text: str, *, final: bool) -> str:
        # A CR is a complete SSE line ending on its own, so it is normalized
        # immediately. If the next chunk begins with LF, swallow that LF as
        # the second half of the already-emitted CRLF line ending.
        if self._trailing_cr and (text or final):
            self._trailing_cr = False
            if text.startswith("\n"):
                text = text[1:]

        if not final and text.endswith("\r"):
            self._trailing_cr = True

        # These replacements scan only this newly decoded chunk.  They avoid
        # a per-character temporary list while preserving CR, LF, and CRLF as
        # the three protocol-defined SSE line endings.
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _strip_initial_bom(self, text: str, *, final: bool = False) -> str:
        if not self._at_stream_start:
            return text
        if not text and not final:
            return text
        self._at_stream_start = False
        return text.removeprefix("\ufeff")

    def _extract_events(self, normalized_chunk: str) -> list[str]:
        events: list[str] = []
        if not normalized_chunk:
            return events
        try:
            normalized_bytes = normalized_chunk.encode("utf-8")
        except UnicodeEncodeError as exc:
            self._failed = True
            raise SSEProtocolError(
                "SSE text contains an invalid Unicode scalar"
            ) from exc

        # A feed returns every complete frame at once, so all non-separator
        # bytes in this decoded chunk remain retained either by a returned
        # frame or by this parser. Reserve the same total in one governor call
        # and transfer it incrementally instead of locking the global budget
        # once for every SSE field and newline.
        reserved_credit = len(normalized_bytes)
        self._retained_budget.reserve(reserved_credit)
        normalized_view = memoryview(normalized_bytes)
        cursor = 0
        try:
            # Search only normalized_bytes. The one pending newline needed to
            # recognize a separator across chunk boundaries is already the
            # last byte in _event, so old input is never concatenated or
            # scanned again.
            while True:
                newline_index = normalized_bytes.find(b"\n", cursor)
                segment_end = (
                    len(normalized_bytes)
                    if newline_index < 0
                    else newline_index
                )
                segment_size = segment_end - cursor
                if segment_size:
                    self._pending_budget_bytes += segment_size
                    reserved_credit -= segment_size
                    try:
                        self._event.extend(
                            normalized_view[cursor:segment_end]
                        )
                    except BaseException:
                        self._pending_budget_bytes -= segment_size
                        reserved_credit += segment_size
                        raise
                    if self.max_event_bytes and len(self._event) > self.max_event_bytes:
                        self._failed = True
                        raise SSEBufferOverflowError(
                            buffer_name="event",
                            limit_bytes=self.max_event_bytes,
                            observed_bytes=len(self._event),
                        )
                if newline_index < 0:
                    break

                cursor = newline_index + 1
                if self._event.endswith(b"\n"):
                    # The previous LF and this LF are the SSE blank-line
                    # separator. Neither belongs to the raw event frame. The
                    # current LF remains in reserved_credit and is released
                    # once when this decoded chunk is fully processed.
                    self._event.pop()
                    self._release_pending_bytes(1)
                    self._validate_complete_event()
                    if len(events) >= self.max_events_per_feed:
                        self._failed = True
                        raise SSEOutputLimitError(
                            output_name="events",
                            limit=self.max_events_per_feed,
                            observed=self.max_events_per_feed + 1,
                        )
                    retained_bytes = self._pending_budget_bytes
                    try:
                        frame = _RetainedTextFrame(
                            self._event.decode("utf-8"),
                            retained_bytes,
                            self._retained_budget,
                        )
                    except BaseException:
                        self._clear_pending_event()
                        raise
                    self._pending_budget_bytes = 0
                    self._event = bytearray()
                    events.append(frame)
                else:
                    self._pending_budget_bytes += 1
                    reserved_credit -= 1
                    try:
                        self._event.append(0x0A)
                    except BaseException:
                        self._pending_budget_bytes -= 1
                        reserved_credit += 1
                        raise
        finally:
            normalized_view.release()
            if reserved_credit:
                self._retained_budget.release(reserved_credit)

        return events

    def _append_event_text(self, text: str) -> None:
        """Append parser-owned text outside the batched feed reservation.

        Kept as a private compatibility hook for focused tests and extensions;
        the normal feed path reserves a whole decoded chunk in _extract_events.
        """
        if not text:
            return
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            self._failed = True
            raise SSEProtocolError(
                "SSE text contains an invalid Unicode scalar"
            ) from exc
        self._reserve_pending_bytes(len(encoded))
        try:
            self._event.extend(encoded)
        except BaseException:
            self._release_pending_bytes(len(encoded))
            raise

        # A delimiter cannot remove non-newline event data, so enforcing the
        # event bound here prevents a single incomplete frame from growing
        # past the limit even within one large feed().  max_pending_bytes is a
        # retained-across-feeds limit and is checked once the feed completes.
        if self.max_event_bytes and len(self._event) > self.max_event_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="event",
                limit_bytes=self.max_event_bytes,
                observed_bytes=len(self._event),
            )

    def _validate_complete_event(self) -> None:
        event_bytes = len(self._event)
        if self.max_event_bytes and event_bytes > self.max_event_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="event",
                limit_bytes=self.max_event_bytes,
                observed_bytes=event_bytes,
            )

    def _validate_pending_size(self) -> None:
        pending_bytes = self._pending_size_bytes()
        if self.max_pending_bytes and pending_bytes > self.max_pending_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="pending buffer",
                limit_bytes=self.max_pending_bytes,
                observed_bytes=pending_bytes,
            )

        if self.max_event_bytes and pending_bytes > self.max_event_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="event",
                limit_bytes=self.max_event_bytes,
                observed_bytes=pending_bytes,
            )

    def _pending_size_bytes(self) -> int:
        undecoded_bytes, _ = self._decoder.getstate()
        return len(self._event) + len(undecoded_bytes)

    def _reserve_pending_bytes(self, size: int) -> None:
        self._retained_budget.reserve(size)
        self._pending_budget_bytes += size

    def _release_pending_bytes(self, size: int) -> None:
        if size > self._pending_budget_bytes:
            raise RuntimeError("SSE parser pending-byte budget underflow")
        self._pending_budget_bytes -= size
        self._retained_budget.release(size)

    def _release_pending_budget(self) -> None:
        retained = self._pending_budget_bytes
        self._pending_budget_bytes = 0
        if retained:
            self._retained_budget.release(retained)

    def _clear_pending_event(self) -> None:
        self._event = bytearray()
        self._release_pending_budget()

    def _capture_failure_pending_diagnostics(self) -> None:
        if self._failure_pending_diagnostics is not None:
            return
        try:
            self._failure_pending_diagnostics = self.pending_diagnostics()
        except BaseException:
            self._failure_pending_diagnostics = {
                "bytes": self._pending_size_bytes(),
                "scope": "normalized_prefix_plus_utf8_tail_v1",
                "sha256_unavailable": True,
            }


class IncrementalLineParser:
    """Strict UTF-8 line framing with a hard retained-byte limit."""

    def __init__(
        self,
        *,
        max_line_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_lines_per_feed: int = DEFAULT_MAX_LINES_PER_FEED,
        max_feed_bytes: int = DEFAULT_MAX_FEED_BYTES,
    ) -> None:
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be greater than zero")
        if max_lines_per_feed <= 0:
            raise ValueError("max_lines_per_feed must be greater than zero")
        if max_feed_bytes <= 0:
            raise ValueError("max_feed_bytes must be greater than zero")
        self.max_line_bytes = max_line_bytes
        self.max_lines_per_feed = max_lines_per_feed
        self.max_feed_bytes = max_feed_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._retained_budget = _STREAM_PARSER_RETAINED_BUDGET
        self._line = bytearray()
        self._pending_budget_bytes = 0
        self._pending_cr = False
        self._finished = False
        self._failed = False
        self._at_stream_start = True

    def __del__(self) -> None:
        try:
            self._release_line_budget()
        except Exception:
            pass

    def discard(self) -> None:
        """Drop an abandoned partial line and release its global budget."""

        self._failed = True
        self._finished = True
        self._clear_pending_line()

    @property
    def pending_bytes(self) -> int:
        undecoded, _ = self._decoder.getstate()
        return (
            len(self._line)
            + (1 if self._pending_cr else 0)
            + len(undecoded)
        )

    def feed(self, chunk: str | bytes | bytearray) -> list[str]:
        if self._finished:
            raise SSEProtocolError("cannot feed a line parser after finish()")
        if self._failed:
            raise SSEProtocolError("cannot feed a line parser after a parse failure")
        try:
            if not isinstance(chunk, (str, bytes, bytearray)):
                raise TypeError("line chunks must be str, bytes, or bytearray")
            observed_feed_bytes = (
                _bounded_utf8_size(chunk, limit_bytes=self.max_feed_bytes)
                if isinstance(chunk, str)
                else len(chunk)
            )
            if observed_feed_bytes > self.max_feed_bytes:
                self._failed = True
                raise SSEBufferOverflowError(
                    buffer_name="line input chunk",
                    limit_bytes=self.max_feed_bytes,
                    observed_bytes=observed_feed_bytes,
                )
            lines: list[str] = []
            for offset in range(0, len(chunk), _INTERNAL_FEED_CHUNK_UNITS):
                piece = chunk[offset : offset + _INTERNAL_FEED_CHUNK_UNITS]
                text = self._strip_initial_bom(self._decode(piece, final=False))
                normalized = self._normalize_newlines(text, final=False)
                extracted = self._extract_lines(normalized)
                if len(lines) + len(extracted) > self.max_lines_per_feed:
                    self._failed = True
                    raise SSEOutputLimitError(
                        output_name="lines",
                        limit=self.max_lines_per_feed,
                        observed=len(lines) + len(extracted),
                    )
                lines.extend(extracted)
                self._validate_pending()
            return lines
        except UnicodeEncodeError as exc:
            self._failed = True
            self._clear_pending_line()
            raise SSEProtocolError(
                "line text contains an invalid Unicode scalar"
            ) from exc
        except BaseException:
            self._failed = True
            self._clear_pending_line()
            raise

    def finish(self) -> list[str]:
        if self._failed:
            raise SSEProtocolError("cannot finish a line parser after a parse failure")
        if self._finished:
            return []
        self._finished = True
        try:
            text = self._strip_initial_bom(self._decode(b"", final=True), final=True)
            normalized = self._normalize_newlines(text, final=True)
            lines = self._extract_lines(normalized)
            if self._line:
                self._validate_complete_line()
                lines.append(self._transfer_line_frame())
            return lines
        except BaseException:
            self._failed = True
            self._clear_pending_line()
            raise

    def _decode(self, chunk: str | bytes | bytearray, *, final: bool) -> str:
        if isinstance(chunk, str):
            undecoded, _ = self._decoder.getstate()
            if undecoded:
                self._failed = True
                raise SSEProtocolError(
                    "cannot feed text while a UTF-8 byte sequence is incomplete"
                )
            return chunk
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("line chunks must be str, bytes, or bytearray")
        try:
            return self._decoder.decode(bytes(chunk), final=final)
        except UnicodeDecodeError as exc:
            self._failed = True
            raise SSEProtocolError("line stream contains invalid UTF-8") from exc

    def _normalize_newlines(self, text: str, *, final: bool) -> str:
        prefix = ""
        if self._pending_cr and (text or final):
            prefix = "\n"
            self._pending_cr = False
            if text.startswith("\n"):
                text = text[1:]

        # Unlike the SSE event parser, the generic line parser delays a final
        # CR until the next feed so CRLF split across chunks emits one line.
        if not final and text.endswith("\r"):
            text = text[:-1]
            self._pending_cr = True

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return prefix + normalized

    def _strip_initial_bom(self, text: str, *, final: bool = False) -> str:
        if not self._at_stream_start:
            return text
        if not text and not final:
            return text
        self._at_stream_start = False
        return text.removeprefix("\ufeff")

    def _extract_lines(self, text: str) -> list[str]:
        lines: list[str] = []
        cursor = 0
        while True:
            newline_index = text.find("\n", cursor)
            if newline_index < 0:
                self._append_line_text(text[cursor:])
                break

            self._append_line_text(text[cursor:newline_index])
            cursor = newline_index + 1
            self._validate_complete_line()
            if len(lines) >= self.max_lines_per_feed:
                self._failed = True
                raise SSEOutputLimitError(
                    output_name="lines",
                    limit=self.max_lines_per_feed,
                    observed=self.max_lines_per_feed + 1,
                )
            lines.append(self._transfer_line_frame())
        return lines

    def _append_line_text(self, text: str) -> None:
        if not text:
            return
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            self._failed = True
            raise SSEProtocolError(
                "line text contains an invalid Unicode scalar"
            ) from exc
        self._retained_budget.reserve(len(encoded))
        self._pending_budget_bytes += len(encoded)
        try:
            self._line.extend(encoded)
        except BaseException:
            self._pending_budget_bytes -= len(encoded)
            self._retained_budget.release(len(encoded))
            raise
        self._validate_complete_line()

    def _validate_pending(self) -> None:
        observed = self.pending_bytes
        if observed > self.max_line_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="line",
                limit_bytes=self.max_line_bytes,
                observed_bytes=observed,
            )

    def _validate_complete_line(self) -> None:
        observed = len(self._line)
        if observed > self.max_line_bytes:
            self._failed = True
            raise SSEBufferOverflowError(
                buffer_name="line",
                limit_bytes=self.max_line_bytes,
                observed_bytes=observed,
            )

    def _transfer_line_frame(self) -> str:
        retained = self._pending_budget_bytes
        try:
            frame = _RetainedTextFrame(
                self._line.decode("utf-8"),
                retained,
                self._retained_budget,
            )
        except BaseException:
            self._clear_pending_line()
            raise
        self._pending_budget_bytes = 0
        self._line = bytearray()
        return frame

    def _release_line_budget(self) -> None:
        retained = self._pending_budget_bytes
        self._pending_budget_bytes = 0
        if retained:
            self._retained_budget.release(retained)

    def _clear_pending_line(self) -> None:
        self._line = bytearray()
        self._release_line_budget()


async def iter_sse_events(
    chunks: AsyncIterable[str | bytes | bytearray],
    *,
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    max_events_per_feed: int = DEFAULT_MAX_EVENTS_PER_FEED,
    max_feed_bytes: int = DEFAULT_MAX_FEED_BYTES,
) -> AsyncIterator[str]:
    """Yield validated SSE frames and validate the final EOF exactly once."""

    parser = IncrementalSSEParser(
        max_pending_bytes=max_pending_bytes,
        max_event_bytes=max_event_bytes,
        max_events_per_feed=max_events_per_feed,
        max_feed_bytes=max_feed_bytes,
    )
    try:
        async for chunk in chunks:
            try:
                raw_events = parser.feed(chunk)
            finally:
                chunk = None
            try:
                for index in range(len(raw_events)):
                    raw_event = raw_events[index]
                    try:
                        yield raw_event
                    finally:
                        raw_events[index] = ""
                        raw_event = None
            finally:
                raw_events.clear()
                raw_events = None
        raw_events = parser.finish()
        try:
            for index in range(len(raw_events)):
                raw_event = raw_events[index]
                try:
                    yield raw_event
                finally:
                    raw_events[index] = ""
                    raw_event = None
        finally:
            raw_events.clear()
            raw_events = None
    finally:
        parser.discard()


async def iter_lines(
    chunks: AsyncIterable[str | bytes | bytearray],
    *,
    max_line_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    max_lines_per_feed: int = DEFAULT_MAX_LINES_PER_FEED,
    max_feed_bytes: int = DEFAULT_MAX_FEED_BYTES,
) -> AsyncIterator[str]:
    parser = IncrementalLineParser(
        max_line_bytes=max_line_bytes,
        max_lines_per_feed=max_lines_per_feed,
        max_feed_bytes=max_feed_bytes,
    )
    try:
        async for chunk in chunks:
            try:
                lines = parser.feed(chunk)
            finally:
                chunk = None
            try:
                for index in range(len(lines)):
                    line = lines[index]
                    try:
                        yield line
                    finally:
                        lines[index] = ""
                        line = None
            finally:
                lines.clear()
                lines = None
        lines = parser.finish()
        try:
            for index in range(len(lines)):
                line = lines[index]
                try:
                    yield line
                finally:
                    lines[index] = ""
                    line = None
        finally:
            lines.clear()
            lines = None
    finally:
        parser.discard()


def is_sse_comment_frame(raw_event: str) -> bool:
    has_line = False
    for line in _iter_bounded_event_lines(raw_event):
        if not line:
            continue
        has_line = True
        if not line.startswith(":"):
            return False
    return has_line


def parse_sse_event(raw_event: str) -> tuple[str, Any]:
    event_name, data_str, _has_event_field, _has_data_field = (
        _extract_sse_event_fields(raw_event)
    )
    return _parse_sse_event_data(event_name, data_str)


def sse_event_has_data_field(raw_event: str) -> bool:
    """Return field presence without materializing the event payload.

    Diagnostics call this before the owned parser has reserved its temporary
    workspace.  Keep the scan allocation-free with respect to field values so
    an attacker-sized ``data`` line is not copied outside admission accounting.
    """

    start = 0
    observed = 0
    raw_length = len(raw_event)
    while True:
        newline = raw_event.find("\n", start)
        line_end = raw_length if newline < 0 else newline
        observed += 1
        if observed > DEFAULT_MAX_FIELDS_PER_EVENT:
            raise SSEOutputLimitError(
                output_name="fields per event",
                limit=DEFAULT_MAX_FIELDS_PER_EVENT,
                observed=observed,
            )

        colon = raw_event.find(":", start, line_end)
        field_end = line_end if colon < 0 else colon
        if field_end - start == 4 and raw_event.startswith("data", start):
            return True

        if newline < 0:
            return False
        start = newline + 1


def validate_sse_event_type_consistency(
    declared_event_name: str,
    payload: Any,
    *,
    protocol_name: str,
    has_event_field: bool,
    require_event_name: bool = False,
) -> None:
    """Reject an ambiguous protocol event when both type sources disagree.

    SSE itself does not assign meaning to a JSON ``type`` member, so callers
    opt in only for protocols such as Responses that define both fields as the
    same semantic event type.
    """

    if require_event_name and not isinstance(payload, dict):
        raise SSEProtocolError(
            f"{protocol_name} SSE data must be a JSON object"
        )
    if has_event_field and not declared_event_name:
        raise SSEProtocolError(
            f"{protocol_name} SSE event field must not be empty"
        )
    if declared_event_name:
        if len(declared_event_name) > 256:
            raise SSEProtocolError(
                f"{protocol_name} SSE event type exceeds 256 bytes"
            )
        try:
            declared_event_name_bytes = declared_event_name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SSEProtocolError(
                f"{protocol_name} SSE event type is invalid"
            ) from exc
        if len(declared_event_name_bytes) > 256:
            raise SSEProtocolError(
                f"{protocol_name} SSE event type exceeds 256 bytes"
            )

    payload_event_name = ""
    if isinstance(payload, dict) and "type" in payload:
        raw_payload_event_name = payload.get("type")
        if not isinstance(raw_payload_event_name, str):
            raise SSEProtocolError(
                f"{protocol_name} SSE data.type must be a non-empty string"
            )
        payload_event_name = raw_payload_event_name
        if len(payload_event_name) > 256:
            raise SSEProtocolError(
                f"{protocol_name} SSE data.type is invalid"
            )
        if not payload_event_name or payload_event_name.strip() != payload_event_name:
            raise SSEProtocolError(
                f"{protocol_name} SSE data.type must be a non-empty string"
            )
        try:
            payload_event_name_bytes = payload_event_name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise SSEProtocolError(
                f"{protocol_name} SSE data.type is invalid"
            ) from exc
        if len(payload_event_name_bytes) > 256 or "\r" in payload_event_name or "\n" in payload_event_name:
            raise SSEProtocolError(
                f"{protocol_name} SSE data.type is invalid"
            )

    if (
        declared_event_name
        and payload_event_name
        and declared_event_name != payload_event_name
    ):
        raise SSEProtocolError(
            f"{protocol_name} SSE event field conflicts with data.type"
        )
    if require_event_name and not (declared_event_name or payload_event_name):
        raise SSEProtocolError(f"{protocol_name} SSE event type is missing")


def _extract_sse_event_fields(raw_event: str) -> tuple[str, str, bool, bool]:
    event_name = ""
    data_lines: list[str] = []
    has_event_field = False
    has_data_field = False
    for line in _iter_bounded_event_lines(raw_event):
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "event":
            has_event_field = True
            event_name = value
        elif field == "data":
            has_data_field = True
            data_lines.append(value)

    return event_name, "\n".join(data_lines), has_event_field, has_data_field


def _parse_sse_event_data(event_name: str, data_str: str) -> tuple[str, Any]:
    if data_str == "[DONE]":
        return "[DONE]", "[DONE]"

    parsed_payload: Any = data_str
    if data_str:
        try:
            parsed_payload = json.loads(data_str)
        except Exception:
            parsed_payload = data_str

    if not event_name and isinstance(parsed_payload, dict):
        payload_event_name = parsed_payload.get("type")
        if isinstance(payload_event_name, str) and len(payload_event_name) <= 256:
            event_name = payload_event_name.strip()

    return event_name, parsed_payload


def _iter_bounded_event_lines(raw_event: str):
    """Iterate SSE fields without materializing an attacker-sized split list."""

    start = 0
    observed = 0
    while True:
        newline = raw_event.find("\n", start)
        if newline < 0:
            line = raw_event[start:]
        else:
            line = raw_event[start:newline]
        observed += 1
        if observed > DEFAULT_MAX_FIELDS_PER_EVENT:
            raise SSEOutputLimitError(
                output_name="fields per event",
                limit=DEFAULT_MAX_FIELDS_PER_EVENT,
                observed=observed,
            )
        yield line
        if newline < 0:
            break
        start = newline + 1


async def parse_owned_sse_event(
    raw_event: str,
    *,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
    max_json_estimated_bytes: int = _SSE_JSON_MAX_ESTIMATED_BYTES,
    workspace: ReusableJSONParseWorkspace | None = None,
) -> OwnedSSEEvent:
    """Parse one resource-accounted event with transferable ownership."""

    if max_event_bytes < 0:
        raise ValueError("max_event_bytes cannot be negative")
    if max_json_estimated_bytes <= 0:
        raise ValueError("max_json_estimated_bytes must be greater than zero")
    request_lease = get_request_admission_lease()
    # Field splitting can temporarily retain line/value slices plus the joined
    # data string.  Reserve a conservative copy workspace before performing
    # those allocations.  The fixed component covers the bounded field list.
    workspace_bytes = len(raw_event) * 8 + 64 * 1024
    workspace_reservation = None
    json_owner: OwnedJSONValue | None = None
    try:
        if max_event_bytes and len(raw_event) > max_event_bytes:
            raise SSEBufferOverflowError(
                buffer_name="event",
                limit_bytes=max_event_bytes,
                observed_bytes=len(raw_event),
            )
        if workspace is not None:
            await workspace.ensure(workspace_bytes)
        elif request_lease is not None:
            workspace_reservation = (
                await request_lease.reserve_temporary_response_bytes(
                    workspace_bytes
                )
            )
        if max_event_bytes:
            observed_bytes = _bounded_utf8_size(
                raw_event,
                limit_bytes=max_event_bytes,
            )
            if observed_bytes > max_event_bytes:
                raise SSEBufferOverflowError(
                    buffer_name="event",
                    limit_bytes=max_event_bytes,
                    observed_bytes=observed_bytes,
                )
        if is_sse_comment_frame(raw_event):
            return OwnedSSEEvent(
                raw_event=raw_event,
                event_name="",
                declared_event_name="",
                payload=None,
                json_owner=None,
                workspace_reservation=workspace_reservation,
                is_comment=True,
                has_event_field=False,
                has_data_field=False,
            )

        (
            declared_event_name,
            data_str,
            has_event_field,
            has_data_field,
        ) = _extract_sse_event_fields(raw_event)
        event_name = declared_event_name
        if data_str == "[DONE]":
            return OwnedSSEEvent(
                raw_event=raw_event,
                event_name="[DONE]",
                declared_event_name=declared_event_name,
                payload="[DONE]",
                json_owner=None,
                workspace_reservation=workspace_reservation,
                is_comment=False,
                has_event_field=has_event_field,
                has_data_field=has_data_field,
            )
        if not data_str:
            return OwnedSSEEvent(
                raw_event=raw_event,
                event_name=event_name,
                declared_event_name=declared_event_name,
                payload=data_str,
                json_owner=None,
                workspace_reservation=workspace_reservation,
                is_comment=False,
                has_event_field=has_event_field,
                has_data_field=has_data_field,
            )

        json_owner = await parse_owned_json_value(
            data_str,
            max_estimated_bytes=max_json_estimated_bytes,
            allow_invalid=True,
            workspace=workspace,
            workspace_extra_bytes=workspace_bytes,
        )
        parsed_payload = json_owner.value
        if not event_name and isinstance(parsed_payload, dict):
            payload_event_name = parsed_payload.get("type")
            if isinstance(payload_event_name, str) and len(payload_event_name) <= 256:
                event_name = payload_event_name.strip()
        return OwnedSSEEvent(
            raw_event=raw_event,
            event_name=event_name,
            declared_event_name=declared_event_name,
            payload=None,
            json_owner=json_owner,
            workspace_reservation=workspace_reservation,
            is_comment=False,
            has_event_field=has_event_field,
            has_data_field=has_data_field,
        )
    except JSONMemoryComplexityError as exc:
        await _release_sse_owned_resources_safely(
            json_owner,
            workspace_reservation,
            raw_event,
        )
        raise SSEProtocolError(
            f"SSE JSON materialization exceeds local limit: {exc}"
        ) from exc
    except BaseException:
        await _release_sse_owned_resources_safely(
            json_owner,
            workspace_reservation,
            raw_event,
        )
        raise


async def parse_sse_event_async(raw_event: str) -> OwnedSSEEvent:
    """Return an owned event; callers must use it as an async context manager."""

    return await parse_owned_sse_event(raw_event)


@asynccontextmanager
async def parsed_sse_event(raw_event: str):
    """Parse one event while its materialized object remains memory-accounted."""

    owner = await parse_owned_sse_event(raw_event)
    async with owner:
        yield owner.event_name, owner.payload
