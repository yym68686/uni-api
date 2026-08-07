from __future__ import annotations

import asyncio
import errno as errno_module
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import struct
import sys
import weakref
from datetime import datetime, timezone
from itertools import count
from time import perf_counter_ns, thread_time_ns, time_ns
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - production is Linux
    fcntl = None

from uni_api.upstream.transport_errors import classify_httpx_transport_error


_SCHEMA_VERSION = 4
_MAX_CAUSE_DEPTH = 16
_MAX_TRACE_EVENTS = 32
_MAX_CLEANUP_ACTIONS = 8
_MAX_EVENT_FACT_CACHE = 64
_MAX_EVENT_TYPE_BYTES = 128
_MAX_USAGE_COUNTER = (1 << 63) - 1
_ENDPOINT_HMAC_KEY = secrets.token_bytes(32)
_TrackerRef = weakref.ReferenceType[Any]
_SOCKET_TRACKERS: dict[str, set[_TrackerRef]] = {}
_NETWORK_STREAM_TRACKERS: dict[int, set[_TrackerRef]] = {}
_SOCKET_UNREAD_SAMPLE_RATE = 128
_SOCKET_UNREAD_SAMPLE_SEQUENCE = count()
_LINUX_FIONREAD = 0x541B
_PERFORMANCE_PHASES = frozenset(
    {
        "socket_receive",
        "sse_frame",
        "json_parse",
        "observer_hash",
        "queue_put",
        "asgi_write",
    }
)
_TERMINAL_TIMELINE_POINTS = (
    "received",
    "parse_completed",
    "observer_completed",
    "semantic_classified",
    "queue_handoff_completed",
    "asgi_write_attempted",
    "asgi_write_completed",
)

OAIX_TERMINAL_FLUSH_MARKER_HEADER = "x-oaix-terminal-flush-marker"
OAIX_TERMINAL_FLUSH_MARKER_CONTRACT = "sse-comment-v1"
OAIX_TERMINAL_FLUSH_MARKER_PREFIX = ": oaix-terminal-flush-v1 "
_OAIX_TERMINAL_FLUSH_MARKER_MAX_BYTES = 1024
_TERMINAL_HOP_MAX_MS = 5 * 60 * 1000
_TERMINAL_HOP_NEGATIVE_TOLERANCE_MS = 5.0

_EVENT_LINE_RE = re.compile(r"(?m)^event:[ \t]?(?P<event>[^\r\n]*)$")
_SAFE_RESPONSES_EVENT_TYPES = frozenset(
    {
        "error",
        "keepalive",
        "ping",
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.completed",
        "response.incomplete",
        "response.failed",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        "response.image_generation_call.partial_image",
        "response.image_generation_call.completed",
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call.completed",
        "response.mcp_list_tools.in_progress",
        "response.mcp_list_tools.completed",
        "response.mcp_list_tools.failed",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.in_progress",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "response.audio.delta",
        "response.audio.done",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
    }
)
_TERMINAL_EVENT_TYPES = frozenset(
    {
        "error",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
)
_AUTH_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[^\s,;]+")
_API_KEY_RE = re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}")
_URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/@\s]+@")
_URL_QUERY_RE = re.compile(r"(?i)(https?://[^?\s]+)\?[^\s]*")
_USAGE_COUNTER_FIELDS = frozenset(
    {
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "total_tokens",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any, *, max_bytes: int = 512) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value[:max_bytes].decode("utf-8", errors="replace")
    elif isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        text = type(value).__name__
    text = text.strip()
    if not text:
        return None
    candidate = text[:max_bytes]
    encoded = candidate.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        candidate = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return candidate


def is_oaix_terminal_flush_marker(raw_event: str) -> bool:
    if not isinstance(raw_event, str):
        return False
    return raw_event.startswith(OAIX_TERMINAL_FLUSH_MARKER_PREFIX)


def _parse_oaix_terminal_flush_marker(raw_event: str) -> dict[str, Any] | None:
    if not is_oaix_terminal_flush_marker(raw_event):
        return None
    encoded = raw_event.encode("utf-8", errors="replace")
    if len(encoded) > _OAIX_TERMINAL_FLUSH_MARKER_MAX_BYTES:
        return None
    if "\n" in raw_event or "\r" in raw_event:
        return None
    payload_text = raw_event[len(OAIX_TERMINAL_FLUSH_MARKER_PREFIX) :].strip()
    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        schema_version = int(payload.get("schema_version") or 0)
    except (TypeError, ValueError):
        return None
    if schema_version != 1:
        return None
    digest = str(payload.get("terminal_wire_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    try:
        flush_attempted_ns = int(payload.get("flush_attempted_unix_nano"))
        flush_completed_ns = int(payload.get("flush_completed_unix_nano"))
    except (TypeError, ValueError):
        return None
    if flush_attempted_ns <= 0 or flush_completed_ns < flush_attempted_ns:
        return None
    connection_id = _safe_text(payload.get("connection_id"), max_bytes=256)
    return {
        "schema_version": 1,
        "connection_id": connection_id,
        "terminal_wire_sha256": digest,
        "flush_attempted_unix_nano": flush_attempted_ns,
        "flush_completed_unix_nano": flush_completed_ns,
    }


def _redacted_error_text(exc: BaseException) -> str | None:
    detail: Any = exc.args[0] if exc.args else None
    text = _safe_text(detail, max_bytes=512)
    if not text:
        return None
    text = _AUTH_RE.sub("[redacted-auth]", text)
    text = _API_KEY_RE.sub("[redacted-api-key]", text)
    text = _URL_USERINFO_RE.sub(r"\1[redacted]@", text)
    return _URL_QUERY_RE.sub(r"\1?[redacted]", text)


def _errno_value(exc: BaseException) -> int | None:
    value = getattr(exc, "errno", None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _exception_chain(exc: BaseException) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    relation = "raised"
    while current is not None and id(current) not in seen and len(rows) < _MAX_CAUSE_DEPTH:
        seen.add(id(current))
        error_number = _errno_value(current)
        row: dict[str, Any] = {
            "relation": relation,
            "type": type(current).__name__,
            "module": type(current).__module__,
        }
        if error_number is not None:
            row["errno"] = error_number
            row["errno_name"] = errno_module.errorcode.get(error_number)
        message = _redacted_error_text(current)
        if message:
            row["message_sha256"] = hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest()
        rows.append(row)
        if current.__cause__ is not None:
            current = current.__cause__
            relation = "cause"
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            relation = "context"
        else:
            current = None
    return rows, current is not None


def _socket_inode(sock: Any) -> str | None:
    fileno = getattr(sock, "fileno", None)
    if not callable(fileno):
        return None
    try:
        fd = int(fileno())
        target = os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, TypeError, ValueError):
        return None
    match = re.fullmatch(r"socket:\[(\d+)]", target)
    return match.group(1) if match else None


def _socket_file_descriptor(sock: Any) -> int | None:
    fileno = getattr(sock, "fileno", None)
    if not callable(fileno):
        return None
    try:
        fd = int(fileno())
    except (OSError, TypeError, ValueError):
        return None
    return fd if fd >= 0 else None


def _kernel_socket_unread_bytes(fd: int, expected_inode: str) -> int | None:
    """Return Linux receive-queue bytes only if the fd still owns this socket."""

    if fcntl is None or not sys.platform.startswith("linux"):
        return None
    try:
        target = os.readlink(f"/proc/self/fd/{int(fd)}")
        if target != f"socket:[{expected_inode}]":
            return None
        encoded = fcntl.ioctl(
            int(fd),
            _LINUX_FIONREAD,
            struct.pack("I", 0),
        )
        return max(0, int(struct.unpack("I", encoded[:4])[0]))
    except (OSError, TypeError, ValueError, struct.error):
        return None


def _endpoint_parts(value: Any) -> tuple[str | None, int | None, str | None]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        address = _safe_text(value[0], max_bytes=256)
        raw_port = value[1]
        port = raw_port if isinstance(raw_port, int) and not isinstance(raw_port, bool) else None
        family = "ipv6" if address and ":" in address else "ipv4"
        return address, port, family
    address = _safe_text(value, max_bytes=256)
    return address, None, "other" if address else None


def _endpoint_hmac(address: str | None, port: int | None) -> str | None:
    if not address:
        return None
    canonical = f"{address}\x00{port if port is not None else ''}".encode("utf-8")
    return hmac.new(_ENDPOINT_HMAC_KEY, canonical, hashlib.sha256).hexdigest()


def safe_responses_event_type(value: Any) -> str:
    event_name = str(value or "").strip().lower()
    if event_name == "[done]":
        return "[DONE]"
    if event_name in _SAFE_RESPONSES_EVENT_TYPES:
        return event_name
    return "other" if event_name else "message"


def _event_type(raw_event: str) -> str:
    # SSE uses the last event field in a frame.  Mirror the parser semantics,
    # but never export an arbitrary upstream-controlled field as a cardinality
    # dimension or payload side channel.
    event_name = ""
    for match in _EVENT_LINE_RE.finditer(raw_event):
        event_name = (match.group("event") or "").strip().lower()
    if event_name:
        return safe_responses_event_type(event_name)
    if raw_event.strip() == "data: [DONE]":
        return "[DONE]"
    return "message"


def _register_tracker(
    registry: dict[Any, set[_TrackerRef]],
    key: Any,
    tracker: ResponsesStreamDiagnostics,
) -> None:
    references = registry.setdefault(key, set())
    for reference in tuple(references):
        candidate = reference()
        if candidate is tracker:
            return
        if candidate is None:
            references.discard(reference)

    tracker_reference: _TrackerRef

    def remove_dead(reference: _TrackerRef) -> None:
        current = registry.get(key)
        if current is None:
            return
        current.discard(reference)
        if not current:
            registry.pop(key, None)

    tracker_reference = weakref.ref(tracker, remove_dead)
    references.add(tracker_reference)


def _registered_trackers(
    registry: dict[Any, set[_TrackerRef]],
    key: Any,
) -> list[ResponsesStreamDiagnostics]:
    references = registry.get(key)
    if references is None:
        return []
    live: list[ResponsesStreamDiagnostics] = []
    for reference in tuple(references):
        tracker = reference()
        if tracker is None:
            references.discard(reference)
        else:
            live.append(tracker)
    if not live:
        registry.pop(key, None)
    return live


def _sequence_number(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("sequence_number")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _usage_counter_value_valid(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= _MAX_USAGE_COUNTER
    if isinstance(value, float):
        return (
            math.isfinite(value)
            and 0 <= value <= _MAX_USAGE_COUNTER
            and value.is_integer()
        )
    if isinstance(value, str):
        rendered = value.strip()
        if not rendered or len(rendered) > 20 or not rendered.isdigit():
            return False
        try:
            return int(rendered) <= _MAX_USAGE_COUNTER
        except (ValueError, OverflowError):
            return False
    return False


def _normalized_usage_counter(value: Any) -> int | None:
    if not _usage_counter_value_valid(value):
        return None
    try:
        return int(str(value).strip()) if isinstance(value, str) else int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _usage_observation(
    payload: Any,
) -> tuple[bool, bool, bool, bool, bool, bool | None, bool | None]:
    if not isinstance(payload, dict):
        return False, False, False, False, False, None, None
    usage = payload.get("usage")
    response = payload.get("response")
    if not isinstance(usage, dict) and isinstance(response, dict):
        usage = response.get("usage")
    if not isinstance(usage, dict):
        return False, False, False, False, False, None, None
    observed_fields = [field for field in _USAGE_COUNTER_FIELDS if field in usage]
    if not observed_fields:
        return True, False, False, False, False, None, None
    values_valid = all(
        _usage_counter_value_valid(usage[field]) for field in observed_fields
    )
    input_known = "prompt_tokens" in usage or "input_tokens" in usage
    output_known = "completion_tokens" in usage or "output_tokens" in usage
    # Total is deterministically derivable only when both components exist.
    total_known = "total_tokens" in usage or (input_known and output_known)
    alias_consistent = True
    for first, second in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
    ):
        if first in usage and second in usage:
            first_value = _normalized_usage_counter(usage[first])
            second_value = _normalized_usage_counter(usage[second])
            alias_consistent = alias_consistent and (
                first_value is not None
                and second_value is not None
                and first_value == second_value
            )
    return (
        True,
        True,
        input_known,
        output_known,
        total_known,
        values_valid,
        alias_consistent,
    )


def _transport_error_code(exc: BaseException) -> tuple[str, str]:
    """Return a bounded deterministic transport code without retaining text."""

    current: BaseException | None = exc
    seen: set[int] = set()
    fallback_type = type(exc).__name__
    while current is not None and id(current) not in seen and len(seen) < _MAX_CAUSE_DEPTH:
        seen.add(id(current))
        number = _errno_value(current)
        if number is not None:
            name = errno_module.errorcode.get(number, f"ERRNO_{number}")
            return f"errno_{name.lower()}", "errno"

        error_type = type(current).__name__
        rendered = str(current).lower()
        if error_type == "RemoteProtocolError":
            if "incomplete chunked read" in rendered:
                return "peer_closed_incomplete_chunked_body", "known_message_pattern"
            if "incomplete message body" in rendered:
                if "expected" in rendered and "received" in rendered:
                    return "peer_closed_incomplete_fixed_body", "known_message_pattern"
                return "peer_closed_incomplete_message_body", "known_message_pattern"
            if "without sending a response" in rendered:
                return "peer_closed_before_response_headers", "known_message_pattern"
            return "remote_protocol_error_unspecified", "exception_type"

        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        else:
            current = None

    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", fallback_type).lower()
    return f"{normalized}_unspecified", "exception_type"


def _terminal_semantics_consistency(
    event_type: str,
    payload: Any,
    semantic_outcome: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    payload_type = ""
    response_status = ""
    if isinstance(payload, dict):
        payload_type = str(payload.get("type") or "").strip()
        response = payload.get("response")
        if isinstance(response, dict):
            response_status = str(response.get("status") or "").strip().lower()
    if payload_type and payload_type != event_type:
        reasons.append("payload_type_mismatch")
    declared_outcome = {
        "response.completed": "completed",
        "response.incomplete": "incomplete",
        "response.failed": "failed",
        "error": "failed",
    }.get(event_type)
    if declared_outcome and semantic_outcome != declared_outcome:
        reasons.append("declared_outcome_mismatch")
    if (
        semantic_outcome in {"completed", "incomplete", "failed"}
        and response_status in {"completed", "incomplete", "failed"}
        and response_status != semantic_outcome
    ):
        reasons.append("response_status_mismatch")
    return not reasons, reasons


class ResponsesStreamDiagnostics:
    """Bounded, payload-free facts for one Responses upstream attempt."""

    def __init__(
        self,
        *,
        current_info: dict[str, Any],
        attempt_index: int | None,
        logical_authority: str | None,
        proxy_configured: bool,
        sse_chunk_observer: Callable[[int], Any] | None = None,
        sse_event_observer: Callable[[int], Any] | None = None,
        terminal_hop_observer: Callable[[dict[str, Any]], Any] | None = None,
        terminal_marker_missing_observer: Callable[[], Any] | None = None,
        phase_sampled: bool = False,
        phase_sample_rate: int = 128,
        phase_observer: Callable[..., Any] | None = None,
        socket_unread_observer: Callable[[int | None], Any] | None = None,
        wall_time_ns: Callable[[], int] = time_ns,
        monotonic_ns: Callable[[], int] = perf_counter_ns,
        thread_cpu_ns: Callable[[], int] = thread_time_ns,
    ) -> None:
        self._socket_inode: str | None = None
        self._socket_fd: int | None = None
        self._network_stream_id: int | None = None
        self._cleanup_claimed = False
        self._cleanup_owner_claimed = False
        self._observed_event_facts: dict[int, tuple[int, int, str]] = {}
        self._sse_chunk_observer = sse_chunk_observer
        self._sse_event_observer = sse_event_observer
        self._terminal_hop_observer = terminal_hop_observer
        self._terminal_marker_missing_observer = (
            terminal_marker_missing_observer
        )
        self._phase_sampled = bool(phase_sampled)
        self._phase_sample_rate = max(1, int(phase_sample_rate))
        self._phase_observer = phase_observer
        self._socket_unread_observer = socket_unread_observer
        self._wall_time_ns = wall_time_ns
        self._monotonic_ns = monotonic_ns
        self._thread_cpu_ns = thread_cpu_ns
        self._request_id = _safe_text(current_info.get("request_id"), max_bytes=256)
        self._trace_id = _safe_text(current_info.get("trace_id"), max_bytes=256)
        self._declared_terminal_received_at: datetime | None = None
        self._oaix_terminal_flush_marker: dict[str, Any] | None = None
        self._terminal_hop_emitted = False
        self._terminal_marker_missing_recorded = False
        self._terminal_received_monotonic_ns: int | None = None
        self._facts: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "hash_scope": "ember_normalized_sse_event_lf_v1",
            "event_hash_policy": "terminal_or_error_only_v1",
            "terminal_timeline_schema_version": 1,
            "terminal_timeline_clock": "unix_nano_plus_process_monotonic_v1",
            "terminal_received_semantics": "upstream_iterator_yield_completed_v1",
            "phase_sample_rate": self._phase_sample_rate,
            "phase_sampled": self._phase_sampled,
            "phase_cpu_semantics": "calling_thread_elapsed_inclusive_v1",
            "phase_bytes_semantics": "known_wire_bytes_only_v1",
            "socket_unread_sample_rate": _SOCKET_UNREAD_SAMPLE_RATE,
            "socket_unread_semantics": (
                "linux_connection_receive_queue_after_chunk_v1"
            ),
            "partial_hash_scope": "normalized_prefix_plus_utf8_tail_v1",
            "transport_peer_semantics": "physical_peer_may_be_proxy",
            "logical_authority": _safe_text(logical_authority, max_bytes=256),
            "explicit_proxy_configured": bool(proxy_configured),
            "upstream_body_bytes": 0,
            "upstream_chunk_count": 0,
            "complete_event_count": 0,
            "ignored_no_data_event_count": 0,
            "canonicalized_data_only_event_count": 0,
            "provider_error_to_response_failed_count": 0,
            "upstream_eof_seen": False,
            "terminal_frame_seen": False,
            "upstream_terminal_seen": False,
            "upstream_terminal_validated": False,
            "usage_object_seen": False,
            "usage_counters_seen": False,
            "usage_input_known": False,
            "usage_output_known": False,
            "usage_total_known": False,
            "usage_seen": False,
            "downstream_terminal_seen": False,
            "downstream_terminal_asgi_write_completed": False,
            "oaix_terminal_flush_marker_expected": False,
            "oaix_terminal_flush_marker_seen": False,
            "oaix_terminal_flush_to_ember_receive_observed": False,
            "error_event_seen": False,
            "diagnosis": "responses_stream_in_progress",
            "phase": "upstream_headers",
        }
        attempts = current_info.get("upstream_attempts")
        if (
            isinstance(attempts, list)
            and isinstance(attempt_index, int)
            and 0 <= attempt_index < len(attempts)
            and isinstance(attempts[attempt_index], dict)
        ):
            attempts[attempt_index]["stream_diagnostics"] = self._facts
        # This is deliberately the most recent attempt. A precommit failure
        # followed by a successful retry must not contaminate request_summary.
        current_info["responses_stream_diagnostics"] = self._facts
        current_info["_responses_stream_diagnostics_tracker"] = self

    @property
    def facts(self) -> dict[str, Any]:
        return self._facts

    @property
    def expects_oaix_terminal_flush_marker(self) -> bool:
        return bool(self._facts.get("oaix_terminal_flush_marker_expected"))

    @property
    def terminal_hop_observed(self) -> bool:
        return bool(
            self._facts.get(
                "oaix_terminal_flush_to_ember_receive_observed"
            )
        )

    @property
    def phase_sampled(self) -> bool:
        return self._phase_sampled

    def begin_phase(self, phase: str) -> tuple[int, int] | None:
        if not self._phase_sampled or phase not in _PERFORMANCE_PHASES:
            return None
        try:
            return self._monotonic_ns(), self._thread_cpu_ns()
        except Exception as exc:
            self._facts["phase_sampler_error"] = type(exc).__name__
            return None

    def finish_phase(
        self,
        phase: str,
        token: tuple[int, int] | None,
        *,
        bytes_count: int = 0,
        events: int = 0,
    ) -> None:
        if token is None or phase not in _PERFORMANCE_PHASES:
            return
        try:
            wall_ns = max(0, self._monotonic_ns() - int(token[0]))
            cpu_ns = max(0, self._thread_cpu_ns() - int(token[1]))
            normalized_bytes = max(0, int(bytes_count))
            normalized_events = max(0, int(events))
            prefix = f"phase_{phase}"
            self._facts[f"{prefix}_samples"] = int(
                self._facts.get(f"{prefix}_samples") or 0
            ) + 1
            self._facts[f"{prefix}_wall_ns"] = int(
                self._facts.get(f"{prefix}_wall_ns") or 0
            ) + wall_ns
            self._facts[f"{prefix}_cpu_ns"] = int(
                self._facts.get(f"{prefix}_cpu_ns") or 0
            ) + cpu_ns
            self._facts[f"{prefix}_bytes"] = int(
                self._facts.get(f"{prefix}_bytes") or 0
            ) + normalized_bytes
            self._facts[f"{prefix}_events"] = int(
                self._facts.get(f"{prefix}_events") or 0
            ) + normalized_events
            observer = self._phase_observer
            if observer is not None:
                observer(
                    phase,
                    wall_ns=wall_ns,
                    cpu_ns=cpu_ns,
                    bytes_count=normalized_bytes,
                    events=normalized_events,
                )
        except Exception as exc:
            self._facts["phase_sampler_error"] = type(exc).__name__

    @staticmethod
    def _iso_from_unix_nano(value: int) -> str:
        return datetime.fromtimestamp(
            int(value) / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat()

    def _record_terminal_timeline_point(
        self,
        point: str,
        *,
        unix_nano: int | None = None,
        monotonic_nano: int | None = None,
    ) -> None:
        if point not in _TERMINAL_TIMELINE_POINTS:
            return
        unix_key = f"terminal_{point}_unix_nano"
        monotonic_key = f"terminal_{point}_monotonic_nano"
        if unix_key in self._facts and monotonic_key in self._facts:
            return
        try:
            observed_unix = (
                self._wall_time_ns()
                if unix_nano is None
                else max(1, int(unix_nano))
            )
            observed_monotonic = (
                self._monotonic_ns()
                if monotonic_nano is None
                else max(1, int(monotonic_nano))
            )
        except Exception as exc:
            self._facts["terminal_timeline_error"] = type(exc).__name__
            return
        self._facts.setdefault(unix_key, observed_unix)
        self._facts.setdefault(monotonic_key, observed_monotonic)
        if point == "received" and self._terminal_received_monotonic_ns is None:
            self._terminal_received_monotonic_ns = observed_monotonic
        received_monotonic = self._terminal_received_monotonic_ns
        if received_monotonic is not None:
            self._facts[f"terminal_{point}_from_receive_us"] = (
                observed_monotonic - received_monotonic
            ) / 1000.0
        self._refresh_terminal_timeline_durations()

    def _refresh_terminal_timeline_durations(self) -> None:
        transitions = (
            ("received", "parse_completed"),
            ("parse_completed", "observer_completed"),
            ("observer_completed", "semantic_classified"),
            ("semantic_classified", "queue_handoff_completed"),
            ("queue_handoff_completed", "asgi_write_attempted"),
            ("asgi_write_attempted", "asgi_write_completed"),
        )
        for start, end in transitions:
            start_value = self._facts.get(
                f"terminal_{start}_monotonic_nano"
            )
            end_value = self._facts.get(f"terminal_{end}_monotonic_nano")
            if isinstance(start_value, int) and isinstance(end_value, int):
                self._facts[f"terminal_{start}_to_{end}_us"] = (
                    end_value - start_value
                ) / 1000.0
        if all(
            f"terminal_{point}_monotonic_nano" in self._facts
            for point in _TERMINAL_TIMELINE_POINTS
        ):
            self._facts["terminal_timeline_complete"] = True

    def mark_terminal_parse_completed(
        self,
        event_type: str,
        *,
        received_at: datetime | None = None,
        received_unix_nano: int | None = None,
        received_monotonic_nano: int | None = None,
    ) -> None:
        if event_type not in _TERMINAL_EVENT_TYPES:
            return
        if received_unix_nano is None and received_at is not None:
            received_unix_nano = int(received_at.timestamp() * 1_000_000_000)
        self._record_terminal_timeline_point(
            "received",
            unix_nano=received_unix_nano,
            monotonic_nano=received_monotonic_nano,
        )
        self._record_terminal_timeline_point("parse_completed")

    def mark_terminal_asgi_write_attempted(self) -> None:
        if not self._facts.get("upstream_terminal_seen"):
            return
        self._record_terminal_timeline_point("asgi_write_attempted")

    def mark_terminal_asgi_write_completed(self) -> None:
        if not self._facts.get("upstream_terminal_seen"):
            return
        self._record_terminal_timeline_point("asgi_write_completed")
        unix_nano = self._facts.get("terminal_asgi_write_completed_unix_nano")
        if isinstance(unix_nano, int):
            self._facts["downstream_terminal_asgi_write_completed_at"] = (
                self._iso_from_unix_nano(unix_nano)
            )
        self._facts["downstream_terminal_asgi_write_completed"] = True

    def mark_terminal_flush_marker_missing(self, reason: str) -> None:
        if not self.expects_oaix_terminal_flush_marker:
            return
        if self.terminal_hop_observed or self._terminal_marker_missing_recorded:
            return
        self._terminal_marker_missing_recorded = True
        self._facts["oaix_terminal_flush_marker_missing"] = True
        self._facts["oaix_terminal_flush_marker_missing_reason"] = (
            _safe_text(reason, max_bytes=80) or "unknown"
        )
        self._facts["oaix_terminal_flush_marker_missing_at"] = _utc_now()
        observer = self._terminal_marker_missing_observer
        if observer is not None:
            try:
                observer()
            except Exception as exc:
                self._facts["terminal_marker_missing_observer_error"] = (
                    type(exc).__name__
                )

    def capture_response(self, response: Any) -> None:
        try:
            extensions = getattr(response, "extensions", None)
            if not isinstance(extensions, dict):
                extensions = {}
            raw_version = extensions.get("http_version")
            if isinstance(raw_version, bytes):
                version = raw_version.decode("ascii", errors="replace")
            else:
                version = _safe_text(raw_version, max_bytes=32)
            if version:
                self._facts["http_version"] = version
            stream_id = extensions.get("stream_id")
            if isinstance(stream_id, int) and not isinstance(stream_id, bool):
                self._facts["httpcore_stream_id"] = stream_id

            headers = getattr(response, "headers", None)
            connection_id = None
            terminal_flush_marker_contract = None
            if headers is not None:
                try:
                    connection_id = headers.get("x-oaix-connection-id")
                except Exception:
                    connection_id = None
                try:
                    terminal_flush_marker_contract = headers.get(
                        OAIX_TERMINAL_FLUSH_MARKER_HEADER
                    )
                except Exception:
                    terminal_flush_marker_contract = None
                if connection_id is None:
                    try:
                        connection_id = next(
                            value
                            for name, value in headers.items()
                            if str(name).lower() == "x-oaix-connection-id"
                        )
                    except (AttributeError, StopIteration, TypeError):
                        connection_id = None
                if terminal_flush_marker_contract is None:
                    try:
                        terminal_flush_marker_contract = next(
                            value
                            for name, value in headers.items()
                            if str(name).lower()
                            == OAIX_TERMINAL_FLUSH_MARKER_HEADER
                        )
                    except (AttributeError, StopIteration, TypeError):
                        terminal_flush_marker_contract = None
            connection_id = _safe_text(connection_id, max_bytes=256)
            if connection_id:
                self._facts["oaix_connection_id"] = connection_id
            marker_contract = _safe_text(
                terminal_flush_marker_contract,
                max_bytes=64,
            )
            if marker_contract:
                self._facts["oaix_terminal_flush_marker_contract"] = (
                    marker_contract
                )
            if marker_contract == OAIX_TERMINAL_FLUSH_MARKER_CONTRACT:
                self._facts["oaix_terminal_flush_marker_expected"] = True

            network_stream = extensions.get("network_stream")
            get_extra_info = getattr(network_stream, "get_extra_info", None)
            if not callable(get_extra_info):
                self._facts["transport_metadata_available"] = False
                return
            self._facts["transport_metadata_available"] = True
            self._network_stream_id = id(network_stream)
            _register_tracker(
                _NETWORK_STREAM_TRACKERS,
                self._network_stream_id,
                self,
            )
            try:
                local = get_extra_info("client_addr")
            except Exception:
                local = None
            try:
                peer = get_extra_info("server_addr")
            except Exception:
                peer = None
            local_address, local_port, local_family = _endpoint_parts(local)
            peer_address, peer_port, peer_family = _endpoint_parts(peer)
            local_hmac = _endpoint_hmac(local_address, local_port)
            peer_hmac = _endpoint_hmac(peer_address, peer_port)
            if local_hmac:
                self._facts["transport_local_endpoint_hmac"] = local_hmac
                self._facts["transport_local_family"] = local_family
            if peer_hmac:
                self._facts["transport_peer_endpoint_hmac"] = peer_hmac
                self._facts["transport_peer_family"] = peer_family
            if local_hmac and peer_hmac:
                four_tuple = f"{local_hmac}\x00{peer_hmac}".encode("ascii")
                self._facts["transport_four_tuple_hmac"] = hmac.new(
                    _ENDPOINT_HMAC_KEY, four_tuple, hashlib.sha256
                ).hexdigest()
            try:
                sock = get_extra_info("socket")
            except Exception:
                sock = None
            inode = _socket_inode(sock)
            if inode:
                self._socket_inode = inode
                self._socket_fd = _socket_file_descriptor(sock)
                self._facts["transport_socket_hmac"] = hmac.new(
                    _ENDPOINT_HMAC_KEY,
                    inode.encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
                _register_tracker(_SOCKET_TRACKERS, inode, self)
        except Exception as exc:
            self._facts["transport_metadata_error"] = type(exc).__name__

    def set_phase(self, phase: str) -> None:
        normalized = _safe_text(phase, max_bytes=40)
        if normalized:
            if normalized != self._facts.get("phase"):
                self._observed_event_facts.clear()
            self._facts["phase"] = normalized

    def observe_upstream_chunk(self, chunk: Any) -> None:
        try:
            size = len(chunk)
        except Exception:
            return
        self._facts["upstream_body_bytes"] = int(
            self._facts.get("upstream_body_bytes") or 0
        ) + max(0, int(size))
        self._facts["upstream_chunk_count"] = int(
            self._facts.get("upstream_chunk_count") or 0
        ) + 1
        self._facts.setdefault("first_upstream_body_at", _utc_now())
        observer = self._sse_chunk_observer
        if observer is not None:
            try:
                observer(max(0, int(size)))
            except Exception as exc:
                self._facts["sse_chunk_observer_error"] = type(exc).__name__
        if (
            self._socket_fd is not None
            and self._socket_inode is not None
            and next(_SOCKET_UNREAD_SAMPLE_SEQUENCE)
            % _SOCKET_UNREAD_SAMPLE_RATE
            == 0
        ):
            unread = _kernel_socket_unread_bytes(
                self._socket_fd,
                self._socket_inode,
            )
            if unread is None:
                self._facts["socket_unread_sample_failures"] = int(
                    self._facts.get("socket_unread_sample_failures") or 0
                ) + 1
            else:
                self._facts["socket_unread_samples"] = int(
                    self._facts.get("socket_unread_samples") or 0
                ) + 1
                self._facts["socket_unread_bytes_last"] = unread
                self._facts["socket_unread_bytes_max"] = max(
                    unread,
                    int(self._facts.get("socket_unread_bytes_max") or 0),
                )
                self._facts["socket_unread_bytes_total"] = int(
                    self._facts.get("socket_unread_bytes_total") or 0
                ) + unread
            if self._socket_unread_observer is not None:
                try:
                    self._socket_unread_observer(unread)
                except Exception as exc:
                    self._facts["socket_unread_observer_error"] = type(
                        exc
                    ).__name__

    def observe_complete_event(
        self,
        raw_event: str,
        *,
        has_data_field: bool = True,
        event_type: str | None = None,
        wire_bytes: bytes | None = None,
        received_at: datetime | None = None,
        received_unix_nano: int | None = None,
        received_monotonic_nano: int | None = None,
        force_hash: bool = False,
    ) -> None:
        try:
            if is_oaix_terminal_flush_marker(raw_event):
                self._observe_oaix_terminal_flush_marker(raw_event)
                return
            if wire_bytes is None:
                wire_bytes = raw_event.encode("utf-8") + b"\n\n"
            ordinal = int(self._facts.get("complete_event_count") or 0) + 1
            if event_type is None:
                event_type = _event_type(raw_event)
            observer = self._sse_event_observer
            if observer is not None:
                try:
                    observer(len(wire_bytes))
                except Exception as exc:
                    self._facts["sse_event_observer_error"] = type(exc).__name__
            should_hash = bool(
                force_hash
                or (has_data_field and event_type in _TERMINAL_EVENT_TYPES)
            )
            digest_hex = None
            received_at_text = None
            if should_hash:
                digest_hex = hashlib.sha256(wire_bytes).hexdigest()
                if received_unix_nano is not None and received_at is None:
                    received_at = datetime.fromtimestamp(
                        int(received_unix_nano) / 1_000_000_000,
                        tz=timezone.utc,
                    )
                if received_at is None:
                    received_at = datetime.now(timezone.utc)
                received_at_text = received_at.isoformat()
            if has_data_field and digest_hex is not None:
                if len(self._observed_event_facts) >= _MAX_EVENT_FACT_CACHE:
                    self._observed_event_facts.pop(
                        next(iter(self._observed_event_facts)),
                        None,
                    )
                self._observed_event_facts[id(raw_event)] = (
                    ordinal,
                    len(wire_bytes),
                    digest_hex,
                )
            self._facts.update(
                {
                    "complete_event_count": ordinal,
                    "last_event_ordinal": ordinal,
                    "last_event_type": event_type,
                    "last_event_bytes": len(wire_bytes),
                }
            )
            if digest_hex is None:
                # A stale digest must never describe a newer non-terminal
                # frame. Only terminal/error frames retain a wire identity.
                self._facts.pop("last_event_sha256", None)
                self._facts.pop("last_event_received_at", None)
            else:
                self._facts["last_event_sha256"] = digest_hex
                self._facts["last_event_received_at"] = received_at_text
            if has_data_field and event_type in _TERMINAL_EVENT_TYPES:
                assert digest_hex is not None
                assert received_at is not None
                assert received_at_text is not None
                self._facts["terminal_frame_seen"] = True
                self._facts["declared_terminal_type"] = event_type
                self._facts["declared_terminal_ordinal"] = ordinal
                self._facts["declared_terminal_bytes"] = len(wire_bytes)
                self._facts["declared_terminal_sha256"] = digest_hex
                self._facts["declared_terminal_received_at"] = received_at_text
                self._declared_terminal_received_at = received_at
                if received_unix_nano is None:
                    received_unix_nano = int(
                        received_at.timestamp() * 1_000_000_000
                    )
                self._record_terminal_timeline_point(
                    "received",
                    unix_nano=received_unix_nano,
                    monotonic_nano=received_monotonic_nano,
                )
                self._try_emit_terminal_hop_observation()
                self._refresh_diagnosis()
                self._record_terminal_timeline_point("observer_completed")
        except Exception as exc:
            self._facts["event_observer_error"] = type(exc).__name__

    def _observe_oaix_terminal_flush_marker(self, raw_event: str) -> None:
        self._facts["oaix_terminal_flush_marker_seen"] = True
        self._facts["oaix_terminal_flush_marker_received_at"] = _utc_now()
        marker = _parse_oaix_terminal_flush_marker(raw_event)
        if marker is None:
            self._facts["oaix_terminal_flush_marker_valid"] = False
            self._facts["oaix_terminal_flush_marker_invalid_reason"] = (
                "invalid_payload"
            )
            return
        expected_connection = _safe_text(
            self._facts.get("oaix_connection_id"),
            max_bytes=256,
        )
        marker_connection = _safe_text(
            marker.get("connection_id"),
            max_bytes=256,
        )
        if expected_connection and marker_connection != expected_connection:
            self._facts["oaix_terminal_flush_marker_valid"] = False
            self._facts["oaix_terminal_flush_marker_invalid_reason"] = (
                "connection_id_mismatch"
            )
            return
        self._facts["oaix_terminal_flush_marker_valid"] = True
        self._facts["oaix_terminal_flush_marker_schema_version"] = 1
        self._facts["oaix_terminal_flush_marker_wire_sha256"] = marker[
            "terminal_wire_sha256"
        ]
        self._facts["oaix_terminal_flush_attempted_unix_nano"] = marker[
            "flush_attempted_unix_nano"
        ]
        self._facts["oaix_terminal_flush_completed_unix_nano"] = marker[
            "flush_completed_unix_nano"
        ]
        self._oaix_terminal_flush_marker = marker
        self._try_emit_terminal_hop_observation()

    def _try_emit_terminal_hop_observation(self) -> None:
        if self._terminal_hop_emitted:
            return
        marker = self._oaix_terminal_flush_marker
        terminal_received_at = self._declared_terminal_received_at
        if marker is None or terminal_received_at is None:
            return
        marker_digest = str(marker.get("terminal_wire_sha256") or "")
        terminal_digest = str(
            self._facts.get("declared_terminal_sha256") or ""
        )
        if not terminal_digest or marker_digest != terminal_digest:
            self._facts["oaix_terminal_flush_marker_hash_matched"] = False
            self._facts["oaix_terminal_flush_marker_invalid_reason"] = (
                "terminal_hash_mismatch"
            )
            return
        self._facts["oaix_terminal_flush_marker_hash_matched"] = True
        attempted_ns = int(marker["flush_attempted_unix_nano"])
        completed_ns = int(marker["flush_completed_unix_nano"])
        attempted_at = datetime.fromtimestamp(
            attempted_ns / 1_000_000_000,
            tz=timezone.utc,
        )
        completed_at = datetime.fromtimestamp(
            completed_ns / 1_000_000_000,
            tz=timezone.utc,
        )
        signed_lag_ms = (
            terminal_received_at - completed_at
        ).total_seconds() * 1000.0
        attempted_lag_ms = (
            terminal_received_at - attempted_at
        ).total_seconds() * 1000.0
        flush_duration_ms = (completed_at - attempted_at).total_seconds() * 1000.0
        self._facts["oaix_terminal_flush_attempted_at"] = attempted_at.isoformat()
        self._facts["oaix_terminal_flush_completed_at"] = completed_at.isoformat()
        self._facts["oaix_terminal_flush_duration_ms"] = flush_duration_ms
        self._facts[
            "oaix_terminal_flush_to_ember_receive_signed_ms"
        ] = signed_lag_ms
        self._facts[
            "oaix_terminal_flush_attempt_to_ember_receive_ms"
        ] = attempted_lag_ms
        if signed_lag_ms < -_TERMINAL_HOP_NEGATIVE_TOLERANCE_MS:
            self._facts["oaix_terminal_flush_marker_invalid_reason"] = (
                "negative_cross_clock_lag"
            )
            return
        if signed_lag_ms > _TERMINAL_HOP_MAX_MS:
            self._facts["oaix_terminal_flush_marker_invalid_reason"] = (
                "lag_exceeds_bound"
            )
            return
        lag_ms = max(0.0, signed_lag_ms)
        if signed_lag_ms < 0:
            self._facts["oaix_terminal_flush_lag_clamped_for_clock_order"] = True
        self._facts[
            "oaix_terminal_flush_to_ember_receive_ms"
        ] = lag_ms
        self._facts[
            "oaix_terminal_flush_to_ember_receive_observed"
        ] = True
        self._terminal_hop_emitted = True
        observation = {
            "schema_version": 1,
            "request_id": self._request_id,
            "trace_id": self._trace_id,
            "oaix_connection_id": self._facts.get("oaix_connection_id"),
            "terminal_wire_sha256": terminal_digest,
            "flush_attempted_at": attempted_at.isoformat(),
            "flush_completed_at": completed_at.isoformat(),
            "terminal_received_at": terminal_received_at.isoformat(),
            "flush_duration_ms": flush_duration_ms,
            "lag_ms": lag_ms,
            "signed_lag_ms": signed_lag_ms,
        }
        observer = self._terminal_hop_observer
        if observer is not None:
            try:
                observer(observation)
            except Exception as exc:
                self._facts["terminal_hop_observer_error"] = type(exc).__name__

    def observe_normalization(self, rule: str, event_type: Any) -> None:
        safe_event_type = safe_responses_event_type(event_type)
        if rule == "ignored_no_data_event_block":
            key = "ignored_no_data_event_count"
        elif rule == "canonicalized_data_only_event":
            key = "canonicalized_data_only_event_count"
        elif rule == "provider_error_to_response_failed":
            key = "provider_error_to_response_failed_count"
        else:
            self._facts["normalization_observer_error"] = "unknown_rule"
            return
        self._facts[key] = int(self._facts.get(key) or 0) + 1
        self._facts["last_normalization_rule"] = rule
        self._facts["last_normalized_event_type"] = safe_event_type
        self._facts["normalization_applied"] = True

    def observe_parsed_event(
        self,
        raw_event: str,
        event_type: str,
        payload: Any,
        *,
        semantic_outcome: str,
        wire_bytes: bytes | None = None,
        received_at: datetime | None = None,
    ) -> None:
        declared_terminal = event_type in _TERMINAL_EVENT_TYPES
        cached_event = self._observed_event_facts.pop(id(raw_event), None)
        if not declared_terminal and semantic_outcome == "nonterminal":
            return
        safe_event_type = safe_responses_event_type(event_type)
        if cached_event is None:
            if wire_bytes is None:
                wire_bytes = raw_event.encode("utf-8") + b"\n\n"
            event_ordinal = int(self._facts.get("complete_event_count") or 0)
            event_bytes = len(wire_bytes)
            event_sha256 = hashlib.sha256(wire_bytes).hexdigest()
            if received_at is None:
                received_at = datetime.now(timezone.utc)
            if self._facts.get("last_event_ordinal") == event_ordinal:
                self._facts["last_event_sha256"] = event_sha256
                self._facts["last_event_received_at"] = received_at.isoformat()
        else:
            event_ordinal, event_bytes, event_sha256 = cached_event
        if declared_terminal:
            self._facts["terminal_frame_seen"] = True
            # A precommit data-only frame may be replayed after canonicalizing
            # it with an event field.  Preserve the first, upstream-wire
            # identity instead of overwriting it with that local replay.
            self._facts.setdefault("declared_terminal_type", safe_event_type)
            self._facts.setdefault("declared_terminal_ordinal", event_ordinal)
            self._facts.setdefault("declared_terminal_bytes", event_bytes)
            self._facts.setdefault("declared_terminal_sha256", event_sha256)
            if "declared_terminal_received_at" not in self._facts:
                if received_at is None:
                    received_at = datetime.now(timezone.utc)
                self._facts["declared_terminal_received_at"] = (
                    received_at.isoformat()
                )
                self._declared_terminal_received_at = received_at
            self._facts["terminal_frame_structured"] = True
            self._facts.setdefault("terminal_frame_structured_at", _utc_now())
            self._facts["terminal_frame_semantic_outcome"] = semantic_outcome
            consistent, reasons = _terminal_semantics_consistency(
                event_type,
                payload,
                semantic_outcome,
            )
            self._facts["terminal_semantics_consistent"] = consistent
            self._facts["terminal_consistency_status"] = (
                "consistent" if consistent else "inconsistent"
            )
            if reasons:
                self._facts["terminal_semantics_inconsistency"] = reasons
            else:
                self._facts.pop("terminal_semantics_inconsistency", None)

        if semantic_outcome in {"completed", "incomplete", "failed"}:
            now = _utc_now()
            self._facts.update(
                {
                    "upstream_terminal_seen": True,
                    "upstream_terminal_validated": bool(
                        self._facts.get("terminal_semantics_consistent")
                    ),
                }
            )
            self._facts.setdefault("semantic_terminal_type", safe_event_type)
            self._facts.setdefault("semantic_terminal_outcome", semantic_outcome)
            self._facts.setdefault("semantic_terminal_bytes", event_bytes)
            self._facts.setdefault("semantic_terminal_sha256", event_sha256)
            self._facts.setdefault("semantic_terminal_classified_at", now)
            self._facts.setdefault(
                "transport_end_trigger",
                f"semantic_{semantic_outcome}",
            )
            if semantic_outcome == "completed":
                self._facts["response_completed_validated"] = bool(
                    self._facts.get("terminal_semantics_consistent")
                )
            elif semantic_outcome == "incomplete":
                self._facts["response_incomplete_validated"] = bool(
                    self._facts.get("terminal_semantics_consistent")
                )
            else:
                self._facts["failure_terminal_validated"] = bool(
                    self._facts.get("terminal_semantics_consistent")
                )
            sequence = _sequence_number(payload)
            if sequence is not None:
                self._facts["semantic_terminal_sequence_number"] = sequence
            (
                usage_object_seen,
                usage_counters_seen,
                usage_input_known,
                usage_output_known,
                usage_total_known,
                usage_values_valid,
                usage_alias_consistent,
            ) = _usage_observation(payload)
            if usage_object_seen:
                self._facts["usage_object_seen"] = True
            if usage_counters_seen:
                self._facts["usage_counters_seen"] = True
                self._facts["usage_input_known"] = usage_input_known
                self._facts["usage_output_known"] = usage_output_known
                self._facts["usage_total_known"] = usage_total_known
                self._facts["usage_values_valid"] = bool(usage_values_valid)
                self._facts["usage_alias_consistent"] = bool(
                    usage_alias_consistent
                )
            if (
                usage_counters_seen
                and usage_input_known
                and usage_output_known
                and usage_total_known
                and usage_values_valid
                and usage_alias_consistent
            ):
                self._facts["usage_seen"] = True
            if semantic_outcome == "failed":
                self._facts["failure_stage"] = (
                    self._facts.get("phase") or "unknown"
                )
        self._refresh_diagnosis()
        if semantic_outcome in {"completed", "incomplete", "failed"}:
            self._record_terminal_timeline_point("semantic_classified")

    def mark_terminal_queue_handoff_completed(self) -> None:
        self._facts["ember_queue_terminal_handoff_completed"] = True
        self._record_terminal_timeline_point("queue_handoff_completed")
        unix_nano = self._facts.get(
            "terminal_queue_handoff_completed_unix_nano"
        )
        self._facts["ember_queue_terminal_handoff_completed_at"] = (
            self._iso_from_unix_nano(unix_nano)
            if isinstance(unix_nano, int)
            else _utc_now()
        )
        self._refresh_diagnosis()

    def observe_partial_event(self, pending_data: bytes) -> None:
        try:
            self.observe_partial_diagnostics(
                {
                    "bytes": len(pending_data),
                    "sha256": hashlib.sha256(pending_data).hexdigest()
                    if pending_data
                    else None,
                    "scope": "caller_supplied_pending_bytes_v1",
                }
            )
        except Exception as exc:
            self._facts["partial_observer_error"] = type(exc).__name__

    def observe_partial_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        try:
            size = max(0, int(diagnostics.get("bytes") or 0))
            self._facts["partial_event_bytes"] = size
            digest = _safe_text(diagnostics.get("sha256"), max_bytes=128)
            if size and digest:
                self._facts["partial_event_sha256"] = digest
            elif not size:
                self._facts.pop("partial_event_sha256", None)
            scope = _safe_text(diagnostics.get("scope"), max_bytes=128)
            if scope:
                self._facts["partial_hash_scope"] = scope
        except Exception as exc:
            self._facts["partial_observer_error"] = type(exc).__name__

    def observe_upstream_eof(self) -> None:
        self._facts["upstream_eof_seen"] = True
        self._facts["upstream_eof_at"] = _utc_now()
        self._facts.setdefault("transport_end_trigger", "upstream_http_body_eof")
        self._refresh_diagnosis()

    def observe_exception(self, exc: BaseException, *, origin: str) -> None:
        if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
            return
        if self._facts.get("exception_type"):
            return
        chain, truncated = _exception_chain(exc)
        self._facts.update(
            {
                "exception_origin": _safe_text(origin, max_bytes=80),
                "exception_type": type(exc).__name__,
                "exception_at": _utc_now(),
                "exception_chain": chain,
                "exception_chain_depth": len(chain),
                "exception_chain_truncated": truncated,
                "failure_stage": self._facts.get("phase") or "unknown",
            }
        )
        transport_error_code, transport_error_code_source = _transport_error_code(
            exc
        )
        self._facts["transport_error_code"] = transport_error_code
        self._facts["transport_error_code_source"] = transport_error_code_source
        transport_failure = classify_httpx_transport_error(
            exc,
            failure_stage=self._facts.get("phase"),
            first_byte_observed=bool(
                int(self._facts.get("upstream_chunk_count") or 0)
            ),
        )
        if transport_failure is not None:
            self._facts.update(transport_failure.observability_facts())
        if chain:
            deepest_errno = next(
                (row for row in reversed(chain) if row.get("errno") is not None),
                None,
            )
            if deepest_errno is not None:
                self._facts["exception_errno"] = deepest_errno["errno"]
                if deepest_errno.get("errno_name"):
                    self._facts["exception_errno_name"] = deepest_errno[
                        "errno_name"
                    ]
        self._refresh_diagnosis()

    async def httpcore_trace(self, name: str, info: dict[str, Any]) -> None:
        """HTTPcore trace callback; it must never affect the request."""

        try:
            # Successful body reads are high-volume and do not help attribute a
            # close.  Keeping them would also consume the bounded ledger before
            # a late read failure.  Preserve only the failure and close-order
            # transitions needed for a deterministic transport diagnosis.
            interesting = (
                name.endswith("receive_response_body.failed")
                or "response_closed" in name
            )
            if not interesting:
                return
            row: dict[str, Any] = {"name": _safe_text(name, max_bytes=128), "at": _utc_now()}
            exc = info.get("exception") if isinstance(info, dict) else None
            if isinstance(exc, BaseException):
                chain, truncated = _exception_chain(exc)
                row["exception_chain"] = chain
                row["exception_chain_truncated"] = truncated
            events = self._facts.setdefault("httpcore_events", [])
            if isinstance(events, list) and len(events) < _MAX_TRACE_EVENTS:
                events.append(row)
            else:
                self._facts["httpcore_events_truncated"] = True
            if name.endswith("receive_response_body.failed"):
                if isinstance(exc, asyncio.CancelledError):
                    self._facts["httpcore_body_read_cancelled_at"] = row["at"]
                    self._facts["httpcore_body_read_cancellation_had_local_claim"] = (
                        self._cleanup_claimed
                    )
                    self._facts.setdefault(
                        "transport_end_trigger",
                        "ember_local_cleanup"
                        if self._cleanup_claimed
                        else "httpcore_body_read_cancellation_origin_unknown",
                    )
                else:
                    self._facts["httpcore_body_read_failed_at"] = row["at"]
                    self._facts.setdefault(
                        "transport_end_trigger",
                        "httpcore_body_read_failure",
                    )
                    self._facts[
                        "local_cleanup_claimed_before_body_read_failure"
                    ] = self._cleanup_claimed
                    if isinstance(exc, BaseException):
                        self._facts["httpcore_body_read_failure_type"] = type(
                            exc
                        ).__name__
            elif name.endswith("response_closed.started"):
                self._facts["httpcore_response_close_started_at"] = row["at"]
                if self._cleanup_claimed:
                    trigger = "ember_explicit_cleanup"
                elif self._facts.get("httpcore_body_read_failed_at"):
                    trigger = "httpcore_reactive_close_after_body_read_failure"
                elif self._facts.get("httpcore_body_read_cancelled_at"):
                    trigger = "httpcore_reactive_close_after_body_read_cancellation"
                elif self._facts.get("pool_sweeper_close_started_at"):
                    trigger = "pool_sweeper"
                else:
                    trigger = "httpcore_close_trigger_unknown"
                self._facts["httpcore_response_close_trigger"] = trigger
            elif name.endswith("response_closed.complete"):
                self._facts["httpcore_response_close_completed_at"] = row["at"]
            elif name.endswith("response_closed.failed"):
                self._facts["httpcore_response_close_failed_at"] = row["at"]
        except Exception:
            # HTTPcore treats trace callback failures as transport failures.
            # Observability is therefore deliberately fail-open here.
            return

    def mark_downstream_disconnect(self, *, stage: str) -> None:
        self._facts["downstream_disconnected"] = True
        self._facts["downstream_disconnect_stage"] = _safe_text(stage, max_bytes=80)
        self._facts["downstream_disconnect_at"] = _utc_now()
        self._facts["failure_stage"] = "downstream"
        self._facts.setdefault("transport_end_trigger", "downstream_disconnect")
        self._refresh_diagnosis()

    def mark_local_end(self, *, origin: str) -> None:
        self._facts.setdefault("local_end_origin", _safe_text(origin, max_bytes=80))
        self._facts.setdefault("local_end_at", _utc_now())
        self._refresh_diagnosis()

    def mark_postcommit_stream_terminal_suppressed(self) -> None:
        """Record an incomplete post-commit stream without a semantic terminal."""

        self._facts["postcommit_stream_terminal_suppressed"] = True
        self._refresh_diagnosis()

    def mark_cleanup_intent(self, *, owner: str, trigger: str) -> None:
        """Record a local cancellation intent before it reaches HTTPcore.

        This is deliberately separate from ownership of the actual transport
        close. Queue/body cancellation must be visible to the HTTPcore trace,
        but the later proxy finalizer is the component that executes cleanup.
        """

        self._cleanup_claimed = True
        self._facts.setdefault("cleanup_intent_owner", _safe_text(owner, max_bytes=80))
        self._facts.setdefault(
            "cleanup_intent_trigger",
            _safe_text(trigger, max_bytes=80),
        )
        self._facts.setdefault("cleanup_intent_at", _utc_now())
        self._facts.setdefault("transport_end_trigger", "ember_local_cleanup")

    def begin_cleanup(self, *, owner: str, trigger: str) -> bool:
        attempts = int(self._facts.get("cleanup_attempt_count") or 0) + 1
        self._facts["cleanup_attempt_count"] = attempts
        if self._cleanup_owner_claimed:
            return False
        self._cleanup_claimed = True
        self._cleanup_owner_claimed = True
        self._facts.setdefault("transport_end_trigger", "ember_local_cleanup")
        self._facts.update(
            {
                "cleanup_owner": _safe_text(owner, max_bytes=80),
                "cleanup_trigger": _safe_text(trigger, max_bytes=80),
                "cleanup_started_at": _utc_now(),
                "cleanup_started_after_body_read_failure": bool(
                    self._facts.get("httpcore_body_read_failed_at")
                ),
            }
        )
        return True

    def record_cleanup_action(
        self,
        *,
        actor: str,
        trigger: str,
        outcome: dict[str, Any],
        transport_safe: bool,
        context_exit_succeeded: bool | None,
    ) -> None:
        actions = self._facts.setdefault("cleanup_actions", [])
        if not isinstance(actions, list) or len(actions) >= _MAX_CLEANUP_ACTIONS:
            self._facts["cleanup_actions_truncated"] = True
            return
        row: dict[str, Any] = {
            "actor": _safe_text(actor, max_bytes=80),
            "trigger": _safe_text(trigger, max_bytes=80),
            "transport_safe": bool(transport_safe),
            "completed_at": _utc_now(),
        }
        if context_exit_succeeded is not None:
            row["context_exit_succeeded"] = bool(context_exit_succeeded)
        for key in (
            "method",
            "cooperative_close_started",
            "cooperative_close_completed",
            "transport_evicted",
            "transport_isolated",
            "detached_cleanup",
            "won_cleanup_claim",
        ):
            if key in outcome:
                value = outcome[key]
                row[key] = (
                    _safe_text(value, max_bytes=80)
                    if key == "method"
                    else bool(value)
                )
        actions.append(row)

    def observe_cleanup_transport_outcome(
        self,
        outcome: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> None:
        normalized_actor = _safe_text(actor, max_bytes=80) if actor else None
        existing_actor = self._facts.get("cleanup_transport_action_actor")
        if normalized_actor and existing_actor and existing_actor != normalized_actor:
            return
        if normalized_actor and not existing_actor:
            self._facts["cleanup_transport_action_actor"] = normalized_actor
        for key in (
            "method",
            "cooperative_close_started",
            "cooperative_close_completed",
            "transport_evicted",
            "transport_isolated",
            "detached_cleanup",
        ):
            if key in outcome:
                fact_key = f"cleanup_{key}"
                if key in {
                    "transport_evicted",
                    "transport_isolated",
                    "detached_cleanup",
                }:
                    self._facts[fact_key] = bool(
                        self._facts.get(fact_key)
                    ) or bool(outcome[key])
                else:
                    self._facts.setdefault(fact_key, outcome[key])

    def finish_cleanup(
        self,
        *,
        transport_safe: bool,
        context_exit_succeeded: bool | None,
        actor: str | None = None,
    ) -> None:
        if "cleanup_transport_safe" not in self._facts:
            self._facts["cleanup_transport_safe"] = bool(transport_safe)
            if actor:
                self._facts["cleanup_transport_result_actor"] = _safe_text(
                    actor,
                    max_bytes=80,
                )
        if (
            context_exit_succeeded is not None
            and "cleanup_context_exit_succeeded" not in self._facts
        ):
            self._facts["cleanup_context_exit_succeeded"] = bool(
                context_exit_succeeded
            )
            if actor:
                self._facts["cleanup_context_exit_actor"] = _safe_text(
                    actor,
                    max_bytes=80,
                )
        transport_result = self._facts.get("cleanup_transport_safe")
        context_result = self._facts.get("cleanup_context_exit_succeeded")
        if transport_result is False or context_result is False:
            self._facts.setdefault("cleanup_completed_at", _utc_now())
            self._facts["cleanup_result"] = "incomplete"
            self._facts["cleanup_failure"] = True
            self._facts["cleanup_failure_stage"] = "cleanup"
            # Preserve the stage of an earlier transport/protocol failure.
            # Cleanup is a second failure dimension, not a rewrite of cause.
            self._facts.setdefault("failure_stage", "cleanup")
            self._unregister_socket()
        elif transport_result is True and context_result is True:
            self._facts.setdefault("cleanup_completed_at", _utc_now())
            self._facts["cleanup_result"] = "succeeded"
            self._unregister_socket()
        elif transport_result is True:
            self._facts["cleanup_result"] = "transport_safe_context_pending"

    def mark_downstream_sse_events_sent(self, event_types: set[str]) -> None:
        if not event_types:
            return
        if "response.completed" in event_types:
            self._facts["downstream_terminal_seen"] = True
            self._facts["downstream_semantic_status"] = "completed"
        elif "response.incomplete" in event_types:
            self._facts["downstream_terminal_seen"] = True
            self._facts["downstream_semantic_status"] = "incomplete"
        elif "response.failed" in event_types:
            self._facts["downstream_terminal_seen"] = True
            self._facts["downstream_semantic_status"] = "failed"
        if "error" in event_types:
            self._facts["error_event_seen"] = True
        if self._facts.get("downstream_terminal_seen") or "error" in event_types:
            if self._facts.get("upstream_terminal_seen"):
                self.mark_terminal_asgi_write_completed()
            else:
                self._facts["downstream_terminal_asgi_write_completed"] = True
                self._facts["downstream_terminal_asgi_write_completed_at"] = (
                    _utc_now()
                )

    def snapshot_json(self) -> str:
        return json.dumps(
            self._facts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def observe_pool_sweeper_close(self, *, trigger: str) -> None:
        self._facts.setdefault("pool_sweeper_close_started_at", _utc_now())
        self._facts.setdefault("pool_sweeper_trigger", _safe_text(trigger, max_bytes=80))
        self._facts["pool_sweeper_close_observed"] = True
        if trigger == "kernel_close_wait":
            self._facts.setdefault(
                "transport_end_trigger",
                "kernel_close_wait_observed",
            )
        else:
            self._facts.setdefault(
                "transport_end_trigger",
                f"pool_sweeper_{trigger}",
            )
        self.mark_local_end(origin="pool_sweeper_close")
        claimed = self.begin_cleanup(owner="pool_sweeper", trigger=trigger)
        self._facts.setdefault("pool_sweeper_won_cleanup_claim", claimed)

    def observe_pool_sweeper_close_completed(self, *, succeeded: bool) -> None:
        self._facts["pool_sweeper_close_completed_at"] = _utc_now()
        self._facts["pool_sweeper_close_succeeded"] = bool(succeeded)
        outcome = {
            "method": "pool_sweeper_connection_close",
            "cooperative_close_started": False,
            "cooperative_close_completed": False,
            "transport_evicted": bool(succeeded),
            "transport_isolated": bool(succeeded),
            "detached_cleanup": False,
        }
        self.record_cleanup_action(
            actor="pool_sweeper",
            trigger=str(self._facts.get("pool_sweeper_trigger") or "unknown"),
            outcome=outcome,
            transport_safe=bool(succeeded),
            context_exit_succeeded=None,
        )
        self.observe_cleanup_transport_outcome(outcome, actor="pool_sweeper")
        self.finish_cleanup(
            transport_safe=bool(succeeded),
            context_exit_succeeded=None,
            actor="pool_sweeper",
        )

    def _refresh_diagnosis(self) -> None:
        facts = self._facts
        semantic_outcome = str(facts.get("semantic_terminal_outcome") or "")
        queue_handoff_completed = bool(
            facts.get("ember_queue_terminal_handoff_completed")
        )
        local_end_origin = str(facts.get("local_end_origin") or "")
        if facts.get("downstream_disconnected"):
            diagnosis = "responses_downstream_disconnect"
            semantic_status = "unknown"
        elif local_end_origin == "local_backpressure_abort":
            diagnosis = "responses_local_backpressure_abort"
            semantic_status = "unknown"
        elif facts.get("exception_type"):
            error_type = str(facts.get("exception_type") or "")
            transport_error_kind = str(
                facts.get("transport_error_kind") or ""
            )
            transport_error_phase = str(
                facts.get("transport_error_phase") or ""
            )
            if facts.get("local_overload"):
                diagnosis = "responses_local_overload"
            elif transport_error_kind == "connect_timeout":
                diagnosis = "responses_connect_timeout"
            elif (
                transport_error_kind == "read_timeout"
                and transport_error_phase == "before_first_byte"
            ):
                diagnosis = "responses_read_timeout_before_first_byte"
            elif error_type in {"ReadError", "RemoteProtocolError"}:
                diagnosis = "responses_read_error"
            elif error_type == "ReadTimeout":
                diagnosis = "responses_read_timeout"
            elif error_type == "SSEProtocolError" or error_type.startswith("SSE"):
                diagnosis = "responses_sse_protocol_error"
            else:
                diagnosis = "responses_stream_error"
            if int(facts.get("partial_event_bytes") or 0) > 0:
                diagnosis = "responses_partial_event_abort"
            semantic_status = (
                "transport_error"
                if facts.get("postcommit_stream_terminal_suppressed")
                else "error"
            )
        elif facts.get("terminal_semantics_consistent") is False:
            # Consistency remains an independent fact.  A later confirmed
            # transport/protocol exception is the primary stream failure and
            # therefore takes precedence in the single diagnosis field.
            diagnosis = "responses_terminal_semantics_inconsistent"
            semantic_status = (
                semantic_outcome
                if queue_handoff_completed
                or (semantic_outcome == "failed" and facts.get("phase") == "precommit")
                else "unknown"
            )
        elif semantic_outcome == "failed" and facts.get("phase") == "precommit":
            # A provider failure rejected before the HTTP 200 stream commit is
            # already the final semantic result; there is no Ember queue handoff
            # on this branch.
            diagnosis = "responses_failure_terminal"
            semantic_status = "failed"
        elif semantic_outcome == "completed" and queue_handoff_completed:
            diagnosis = (
                "responses_completed_with_usage"
                if facts.get("usage_seen")
                else "responses_completed_without_usage"
            )
            semantic_status = "completed"
        elif semantic_outcome == "incomplete" and queue_handoff_completed:
            diagnosis = "responses_incomplete_terminal"
            semantic_status = "incomplete"
        elif semantic_outcome == "failed" and queue_handoff_completed:
            diagnosis = "responses_failure_terminal"
            semantic_status = "failed"
        elif local_end_origin:
            diagnosis = "responses_local_close_before_terminal"
            semantic_status = "unknown"
        elif semantic_outcome:
            diagnosis = "responses_terminal_pending_queue_handoff"
            semantic_status = "unknown"
        elif facts.get("upstream_eof_seen"):
            diagnosis = "responses_eof_before_terminal"
            semantic_status = "unknown"
        elif facts.get("terminal_frame_seen"):
            diagnosis = "responses_terminal_event_unvalidated"
            semantic_status = "unknown"
        else:
            diagnosis = "responses_stream_in_progress"
            semantic_status = "unknown"
        facts["diagnosis"] = diagnosis
        facts["semantic_status"] = semantic_status

    def _unregister_socket(self) -> None:
        inode = self._socket_inode
        self._socket_fd = None
        if inode:
            references = _SOCKET_TRACKERS.get(inode)
            if references is not None:
                for reference in tuple(references):
                    if reference() is None or reference() is self:
                        references.discard(reference)
                if not references:
                    _SOCKET_TRACKERS.pop(inode, None)
        network_stream_id = self._network_stream_id
        if network_stream_id is not None:
            references = _NETWORK_STREAM_TRACKERS.get(network_stream_id)
            if references is not None:
                for reference in tuple(references):
                    if reference() is None or reference() is self:
                        references.discard(reference)
                if not references:
                    _NETWORK_STREAM_TRACKERS.pop(network_stream_id, None)


class ObservedResponseByteIterator:
    """Count bytes and capture transport EOF/errors without changing chunks."""

    def __init__(self, inner: Any, diagnostics: ResponsesStreamDiagnostics) -> None:
        self._inner = inner
        self._diagnostics = diagnostics

    def __aiter__(self) -> ObservedResponseByteIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._inner.__anext__()
        except StopAsyncIteration:
            self._diagnostics.observe_upstream_eof()
            raise
        except BaseException as exc:
            self._diagnostics.observe_exception(exc, origin="upstream_body_iterator")
            raise
        self._diagnostics.observe_upstream_chunk(chunk)
        return chunk


def observe_pool_sweeper_socket_close(
    socket_inode: str | None,
    *,
    trigger: str,
) -> list[ResponsesStreamDiagnostics]:
    if not socket_inode:
        return []
    trackers = _registered_trackers(_SOCKET_TRACKERS, socket_inode)
    for tracker in trackers:
        tracker.observe_pool_sweeper_close(trigger=trigger)
    return trackers


def observe_pool_sweeper_connection_close(
    connection: Any,
    *,
    trigger: str,
    socket_inode: str | None = None,
) -> list[ResponsesStreamDiagnostics]:
    current = connection
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        network_stream = getattr(current, "_network_stream", None)
        trackers = _registered_trackers(
            _NETWORK_STREAM_TRACKERS,
            id(network_stream),
        )
        if trackers:
            for tracker in trackers:
                tracker.observe_pool_sweeper_close(trigger=trigger)
            return trackers
        next_connection = getattr(current, "_connection", None)
        if next_connection is None or next_connection is current:
            break
        current = next_connection
    return observe_pool_sweeper_socket_close(socket_inode, trigger=trigger)


def observe_client_pool_shutdown_connection(
    connection: Any,
) -> list[ResponsesStreamDiagnostics]:
    current = connection
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        network_stream = getattr(current, "_network_stream", None)
        trackers = _registered_trackers(
            _NETWORK_STREAM_TRACKERS,
            id(network_stream),
        )
        if trackers:
            for tracker in trackers:
                tracker.mark_local_end(origin="client_pool_shutdown")
                tracker.begin_cleanup(
                    owner="client_pool_shutdown",
                    trigger="client_pool_shutdown",
                )
                tracker.facts.setdefault("client_pool_shutdown_started_at", _utc_now())
            return trackers
        next_connection = getattr(current, "_connection", None)
        if next_connection is None or next_connection is current:
            break
        current = next_connection
    return []


def observe_client_pool_shutdown_completed(
    trackers: list[ResponsesStreamDiagnostics],
    *,
    succeeded: bool,
) -> None:
    outcome = {
        "method": "client_pool_aclose",
        "cooperative_close_started": True,
        "cooperative_close_completed": bool(succeeded),
        "transport_evicted": False,
        "transport_isolated": bool(succeeded),
        "detached_cleanup": False,
    }
    for tracker in trackers:
        tracker.facts["client_pool_shutdown_completed_at"] = _utc_now()
        tracker.facts["client_pool_shutdown_succeeded"] = bool(succeeded)
        tracker.record_cleanup_action(
            actor="client_pool_shutdown",
            trigger="client_pool_shutdown",
            outcome=outcome,
            transport_safe=bool(succeeded),
            context_exit_succeeded=None,
        )
        tracker.observe_cleanup_transport_outcome(
            outcome,
            actor="client_pool_shutdown",
        )
        tracker.finish_cleanup(
            transport_safe=bool(succeeded),
            context_exit_succeeded=None,
            actor="client_pool_shutdown",
        )
